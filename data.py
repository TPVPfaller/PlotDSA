import math
import os, sys

# fix lsl for single file .exe
if getattr(sys, "frozen", False):
    base = sys._MEIPASS if hasattr(sys, "_MEIPASS") else os.path.dirname(sys.executable)
    os.environ["PYLSL_LIB"] = os.path.join(base, "pylsl", "lib", "lsl.dll")

from pylsl import StreamInlet, resolve_byprop
import csv
import config
import numpy as np
from calculations import DSACalculator
import time
from datetime import datetime as dt
import datetime

class DSABuffer:
    """Ring buffer for DSA frames (time x frequency) with gap-filling and wrap-around.

    Stores PSD columns aligned to a fixed time grid defined by `config.TIME_RESOLUTION`.
    Read-only methods return windows sized for the current view.
    """

    def __init__(self):
        self.max_frames = int(config.DISPLAY_MINUTES_BOUNDS[1] * 60 / config.TIME_RESOLUTION)
        # Check if we should limit pre-allocation to a sane value if bounds are very large
        # 12 hours at 1s resolution is ~43200 frames.
        # Let's keep it pre-allocated but maybe not for a whole year by default if not needed.
        # However, to support "no limit" as requested, we'll stick to the config for now.
        # Class B: Ensure memory allocation success or handle failure.
        try:
            self._reset()
        except MemoryError:
            print("Failed to allocate DSABuffer. Reducing size to 24 hours.")
            self.max_frames = int(24 * 60 * 60 / config.TIME_RESOLUTION)
            self._reset()

    def append(self, ts, psd):
        if psd is None or len(psd) == 0:
            psd = np.full(len(config.FREQ_BINS), np.nan, dtype=np.float32)

        # Initialize time grid
        if self.t0 is None:
            self.t0 = ts
            slot = 0
        else:
            slot = self._timestamp_to_slot(ts)

        # Rewind if timestamp is older than newest
        if self.last_slot is not None and slot < self.last_slot:
            # Clear data for slots that are being "overwritten" by the rewind
            # If buffer is full, clear max_frames back from last_slot
            start_clear = slot
            end_clear = self.last_slot
            
            for s in range(start_clear, end_clear + 1):
                self.data[s % self.max_frames] = np.nan
                self.timestamps[s % self.max_frames] = np.nan
            
            self.last_slot = slot - 1 if slot > 0 else None
            if self.last_slot is None:
                self.t0 = ts
                slot = 0

        idx = slot % self.max_frames

        # Fill gaps with NaNs
        if self.last_slot is not None and slot > self.last_slot + 1:
            for s in range(self.last_slot + 1, slot):
                self.data[s % self.max_frames] = np.nan
                self.timestamps[s % self.max_frames] = self.t0 + s * config.TIME_RESOLUTION

        # Store data
        self.data[idx] = psd
        self.timestamps[idx] = ts

        # Mark buffer full if we wrapped
        if self.last_slot is not None:
            if slot - self.last_slot >= self.max_frames:
                self.full = True
            elif slot >= self.max_frames:
                self.full = True

        self.last_slot = max(self.last_slot, slot) if self.last_slot is not None else slot

    def _timestamp_to_slot(self, ts):
        return int(math.floor((ts - self.t0) / config.TIME_RESOLUTION + 0.5))

    def get_oldest_timestamp(self):
        if self.last_slot is None:
            return time.time()

        if not self.full:
            return self.t0

        oldest_slot = self.last_slot - self.max_frames + 1
        return self.t0 + oldest_slot * config.TIME_RESOLUTION

    def get_newest_timestamp(self):
        if self.last_slot is None:
            return time.time()
        return self.timestamps[self.last_slot % self.max_frames]

    def get_view_at(self, width, height, pan_sec):
        """Return a frame starting at pan_sec."""
        if self.t0 is None:
            return time.time(), np.full((width, height), np.nan, dtype=np.float32)

        height = min(height, self.data.shape[1])
        width = min(width, self.max_frames)

        # Convert pan offset to slot index
        slot_start = self._timestamp_to_slot(pan_sec)
        slot_end = slot_start + width - 1

        # Clamp to available data
        if self.last_slot is not None:
            # We cannot show slots beyond last_slot
            if slot_end > self.last_slot:
                slot_end = self.last_slot
                slot_start = max(0, slot_end - width + 1)

            # If buffer is not full, we cannot show slots before 0
            if not self.full:
                slot_start = max(0, slot_start)
            else:
                # If buffer is full, we can show up to max_frames back from last_slot
                min_available_slot = self.last_slot - self.max_frames + 1
                slot_start = max(min_available_slot, slot_start)

        idxs = (np.arange(width) + slot_start) % self.max_frames

        t_start = self.t0 + slot_start * config.TIME_RESOLUTION
        frame = self.data[idxs, :height]

        view = self.empty_buffer[:width, :height]
        view[:] = np.nan
        view[:len(frame)] = frame

        return float(t_start), view

    def _reset(self):
        self.data = np.full((self.max_frames, len(config.FREQ_BINS)), np.nan, dtype=np.float32)
        self.empty_buffer = np.empty_like(self.data)
        self.empty_buffer[:] = np.nan

        self.timestamps = np.full(self.max_frames, np.nan)

        self.t0 = None
        self.last_slot = None
        self.full = False


class EEGBuffer:
    """Buffers raw EEG samples and emits DSA-ready PSD columns using a sliding window.

    - Maintains continuity by resetting on missing/invalid samples or timestamp gaps.
    - Uses DSACalculator to compute a PSD for each full window and advances by `hop_len` samples.
    """

    def __init__(self, window_sec, overlap):
        self.window_sec = window_sec
        self.timestamps = []
        self.eeg_values = []
        self.time_delta = 1000.0 / float(config.SAMPLE_RATE_HZ)
        self.last_ts = None
        self.processor = DSACalculator(window_sec)
        self.window_len = int(window_sec * config.SAMPLE_RATE_HZ)
        self.hop_len = max(1, int(self.window_len * (1.0 - overlap)))

    def _get_ts_diff(self, timestamp, value):
        if self.last_ts is not None:
            expected = self.last_ts + datetime.timedelta(milliseconds=self.time_delta)
            return abs((timestamp - expected).total_seconds())
        if value is None or np.isnan(value):
            print(f"Invalid sample: {timestamp}, {value}")
            return config.DSA_TIME_DIFF_TOLERANCE + config.EEG_TIME_DIFF_TOLERANCE
        return 0.0

    def get_dsa_columns(self, data, method='multitaper'):
        if data is None or len(data) == 0:
            return [], []

        output_dsa = []
        samples = []

        for ts, eeg in data:
            # continuity check
            diff = self._get_ts_diff(ts, eeg)
            if diff > config.EEG_TIME_DIFF_TOLERANCE:
                samples.append(np.nan)
                # We accept a few missing values
                if diff > config.DSA_TIME_DIFF_TOLERANCE:
                    print("Timestamp difference")
                    self.eeg_values.clear()
                    self.timestamps.clear()
                    self.last_ts = None
                    continue

            self.eeg_values.append(eeg)
            self.timestamps.append(ts)
            self.last_ts = ts

            samples.append(float(eeg))

            # while enough data exists for a window
            while len(self.eeg_values) >= self.window_len:
                window = np.asarray(self.eeg_values[:self.window_len], dtype=np.float32)
                window_ts = self.timestamps[self.window_len - 1]

                psd = self.processor.compute_psd_column(window, method=method)

                output_dsa.append((window_ts.timestamp() - self.window_sec, psd))

                # SLIDE the window forward
                del self.eeg_values[:self.hop_len]
                del self.timestamps[:self.hop_len]

        return output_dsa, samples

    def apply_config(self, window_sec, overlap):
        self.window_sec = window_sec
        self.window_len = int(window_sec * config.SAMPLE_RATE_HZ)
        self.hop_len = int(self.window_len * (1.0 - overlap))
        if self.hop_len < 1:
            self.hop_len = 1
        self.processor.update_config(window_sec)


class EEGStream:
    def __init__(self):
        self.receiving = False
        self._inlet = None
        self.connect()

    def connect(self):
        try:
            streams = resolve_byprop("name", config.LSL_STREAM_NAME, timeout=2)
            if streams:
                self._inlet = StreamInlet(streams[0])
                print(f"Connected to: {streams[0].name()} (uid: {streams[0].uid()})")
                self.receiving = True
            else:
                self.receiving = False
        except Exception as e:
            print(f"LSL Connection error: {e}")
            self.receiving = False

    def read_samples(self):
        if not self.receiving or self._inlet is None:
            return []

        samples = []
        while True:
            sample, _ = self._inlet.pull_sample(timeout=0)
            if sample is None:
                break
            try:
                timestamp, eeg_str = sample[0].split(",")

                value = float(eeg_str)
                try:
                    timestamp = dt.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
                except ValueError:
                    # Fallback if only time is provided (HH:MM:SS.f)
                    # We assume it's today's date
                    time_part = dt.strptime(timestamp, "%H:%M:%S.%f").time()
                    timestamp = dt.combine(dt.now().date(), time_part)

                if not np.isfinite(value):
                    value = np.nan
                elif value < config.EEG_BOUNDS[0] or value > config.EEG_BOUNDS[1]:
                    print(f"Out of bounds: {value}")
                    value = np.nan

                samples.append((timestamp, value))

            except Exception as e:
                # Invalid sample discarded
                print("Invalid sample: ", e)
                continue

        return samples


class Output:
    @staticmethod
    def _build_filename(base_dir, start_time=None):
        if start_time is None:
            start_time = dt.now()
        else:
            start_time = dt.fromtimestamp(start_time)
        ts = start_time.strftime("%Y-%m-%d")

        filename = f"dsa_{ts}Hz.csv"
        return os.path.join(base_dir, filename)

    @staticmethod
    def load_psd_from_time(start_time_dt):
        """
        Loads PSD data starting from a specific datetime.
        Returns a list of (timestamp, duration, psd) tuples.
        """
        threshold = start_time_dt
        now = dt.now()

        # Determine unique dates to check (from threshold until now)
        num_days = (now.date() - threshold.date()).days
        unique_dates = [threshold.date() + datetime.timedelta(days=i) for i in range(num_days + 1)]

        loaded_data = []

        for date in unique_dates:
            ts_str = date.strftime("%Y-%m-%d")
            filename = f"dsa_{ts_str}Hz.csv"
            filepath = os.path.join(config.BASE_DIR, filename)

            if not os.path.exists(filepath):
                continue

            try:
                with open(filepath, "r", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if not header:
                        continue

                    has_duration = "duration" in header
                    expected_cols = (2 if has_duration else 1) + len(config.FREQ_BINS)

                    if len(header) != expected_cols:
                        continue

                    for row in reader:
                        if not row:
                            continue
                        try:
                            ts_str = row[0]
                            try:
                                # Try parsing as full ISO format first (for backward compatibility)
                                row_dt = dt.fromisoformat(ts_str)
                            except ValueError:
                                # Fallback: assume HH:MM:SS.mmm and use the date from the filename
                                time_part = dt.strptime(ts_str, "%H:%M:%S.%f").time()
                                row_dt = dt.combine(date, time_part)

                            if row_dt < threshold:
                                continue

                            if has_duration:
                                duration = float(row[1])
                                psd_start_idx = 2
                            else:
                                duration = config.TIME_RESOLUTION
                                psd_start_idx = 1

                            psd = []
                            for val_str in row[psd_start_idx:]:
                                if val_str == "" or val_str.lower() == "nan":
                                    psd.append(np.nan)
                                else:
                                    psd.append(float(val_str))

                            loaded_data.append((row_dt.timestamp(), duration, np.array(psd, dtype=np.float32)))
                        except (ValueError, IndexError):
                            continue
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        # Sort by timestamp just in case
        loaded_data.sort(key=lambda x: x[0])
        return loaded_data

    @staticmethod
    def save_psd_to_csv(timestamp, duration, psd):
        psd = np.asarray(psd).ravel()

        if timestamp is None:
            ts_dt = dt.now()
        else:
            if isinstance(timestamp, dt):
                ts_dt = timestamp
            elif isinstance(timestamp, (int, float)):
                ts_dt = dt.fromtimestamp(timestamp)
            else:
                try:
                    ts_dt = dt.fromisoformat(str(timestamp))
                except ValueError:
                    ts_dt = dt.now() # Fallback

        ts_str = ts_dt.strftime("%H:%M:%S.%f")[:-3] # HH:MM:SS.mmm

        filepath = Output._build_filename(config.BASE_DIR, ts_dt.timestamp())
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0

        # If appending to an existing file, validate column count and check for existing timestamp
        if not write_header:
            try:
                with open(filepath, "r", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header is None:
                        write_header = True
                    else:
                        expected_cols = 2 + len(config.FREQ_BINS)  # timestamp + duration + N frequencies
                        if len(header) != expected_cols:
                            raise ValueError(
                                f"Existing CSV has {len(header)} columns but expected {expected_cols}. "
                                "Frequency bins must remain identical across saves."
                            )
                        
                        # Optimization: check if timestamp is already in the file
                        # For efficiency, we read the whole file once, but since this is called for each save
                        # we might want to be careful. However, 86k lines is manageable.
                        # For Class B, it's safer to check the whole file.
                        for row in reader:
                            if row and row[0] == ts_str:
                                return  # Already exists, skip saving
            except FileNotFoundError:
                print(f"File not found: {filepath} creating new file")
                write_header = True
        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                freq_headers = [f"f_{freq:.1f}_Hz" for freq in config.FREQ_BINS]
                writer.writerow(["timestamp", "duration"] + freq_headers)

            row_values = [np.round(x, 4) if np.isfinite(x) else "nan" for x in psd]
            writer.writerow([ts_str, f"{duration:.3f}"] + row_values)
