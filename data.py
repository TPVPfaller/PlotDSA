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
    """Multi-resolution Ring buffer for DSA frames.

    Stores PSD columns aligned to a fixed time grid at multiple resolutions
    to improve performance when viewing large time ranges.
    """

    RESOLUTIONS = [1, 10, 60, 300]  # seconds per frame

    def __init__(self):
        self.max_minutes = config.DISPLAY_MINUTES_BOUNDS[1]
        self.n_freqs = len(config.FREQ_BINS)
        
        # We'll use a dictionary to store buffers for each resolution
        self.buffers = {}
        # Each level will have: data, timestamps, last_slot, full, max_frames
        
        try:
            self._reset()
        except MemoryError:
            print("Failed to allocate DSABuffer. Reducing size.")
            self.max_minutes = 24 * 60  # 24 hours fallback
            self._reset()

    def _reset(self):
        self.buffers = {}
        self.t0 = None

        for res in self.RESOLUTIONS:
            max_frames = int(self.max_minutes * 60 / res)
            self.buffers[res] = {
                'data': np.full((max_frames, self.n_freqs), np.nan, dtype=np.float32),
                'timestamps': np.full(max_frames, np.nan, dtype=np.float64),
                'last_slot': None,
                'full': False,
                'max_frames': max_frames,
                'counts': np.zeros(max_frames, dtype=np.int32)
            }

    def append(self, ts, psd):
        if psd is None or len(psd) == 0:
            psd = np.full(self.n_freqs, np.nan, dtype=np.float32)

        if self.t0 is None:
            self.t0 = ts

        # Update base resolution (Level 0: 1s)
        self._append_to_res(1, ts, psd)

        # Update higher resolutions
        # We can either use a sliding average or just average non-overlapping blocks.
        # Given the "slot" nature, non-overlapping blocks aligned to t0 is easier and consistent.
        for res in self.RESOLUTIONS:
            if res == 1:
                continue
            
            # For higher resolutions, we want to average the base data.
            # However, append() is called with 1s data. 
            # We can use an accumulator for each resolution.
            buf = self.buffers[res]
            slot = int(math.floor((ts - self.t0) / res))
            
            if buf['last_slot'] is not None and slot < buf['last_slot']:
                # Rewind detected, clear higher res too
                self._clear_from_slot(res, slot)
            
            # Simple approach: average every 'res' samples from the 1s resolution.
            # But append might not be called every second or might be called out of order.
            # For simplicity and robust Class B behavior, let's just use the slot 
            # and update the slot value (e.g. running average or just overwrite)
            # Actually, to be accurate, Level N should be the mean of Level 0 frames.
            # For now, let's do a simple update:
            self._update_higher_res(res, ts, psd)

    def _append_to_res(self, res, ts, psd):
        buf = self.buffers[res]
        slot = int(math.floor((ts - self.t0) / res + 0.5))
        max_f = buf['max_frames']

        if buf['last_slot'] is not None and slot < buf['last_slot']:
            self._clear_from_slot(res, slot)
            buf['last_slot'] = slot - 1 if slot > 0 else None
            if res == 1 and buf['last_slot'] is None:
                self.t0 = ts
                slot = 0

        idx = slot % max_f

        # Fill gaps with NaNs
        if buf['last_slot'] is not None and slot > buf['last_slot'] + 1:
            for s in range(buf['last_slot'] + 1, slot):
                buf['data'][s % max_f] = np.nan
                buf['timestamps'][s % max_f] = self.t0 + s * res

        buf['data'][idx] = psd
        buf['timestamps'][idx] = ts

        if buf['last_slot'] is not None:
            if slot - buf['last_slot'] >= max_f or slot >= max_f:
                buf['full'] = True
        
        buf['last_slot'] = max(buf['last_slot'], slot) if buf['last_slot'] is not None else slot

    def _update_higher_res(self, res, ts, psd):
        buf = self.buffers[res]
        slot = int(math.floor((ts - self.t0) / res))
        max_f = buf['max_frames']
        idx = slot % max_f

        if buf['last_slot'] is not None and slot > buf['last_slot']:
             # New slot in higher res, initialize it
             buf['data'][idx] = psd
             buf['timestamps'][idx] = self.t0 + slot * res
             buf['last_slot'] = slot
             buf['counts'][idx] = 1
             if slot >= max_f:
                 buf['full'] = True
        elif buf['last_slot'] == slot:
            # Update existing slot (running mean)
            count = buf['counts'][idx]
            existing = buf['data'][idx]
            if np.isnan(existing).all():
                buf['data'][idx] = psd
                buf['counts'][idx] = 1
            else:
                # Welford's algorithm or simple incremental mean
                buf['data'][idx] = existing + (psd - existing) / (count + 1)
                buf['counts'][idx] = count + 1
        else:
            # Older slot, just ignore or update?
            if buf['last_slot'] is None or (buf['full'] and slot > buf['last_slot'] - max_f) or (not buf['full'] and slot >= 0):
                buf['data'][idx] = psd
                buf['timestamps'][idx] = self.t0 + slot * res
                buf['counts'][idx] = 1
                if buf['last_slot'] is None:
                    buf['last_slot'] = slot

    def _clear_from_slot(self, res, slot):
        buf = self.buffers[res]
        max_f = buf['max_frames']
        if buf['last_slot'] is None: return
        
        start_clear = slot
        end_clear = buf['last_slot']
        for s in range(start_clear, end_clear + 1):
            buf['data'][s % max_f] = np.nan
            buf['timestamps'][s % max_f] = np.nan
            buf['counts'][s % max_f] = 0

    def apply_config(self, display_minutes):
        """Update buffer configuration and reset data."""
        self.max_minutes = display_minutes
        self._reset()

    def get_oldest_timestamp(self):
        buf = self.buffers[1]
        if buf['last_slot'] is None:
            return time.time()
        if not buf['full']:
            return self.t0
        return self.t0 + (buf['last_slot'] - buf['max_frames'] + 1) * 1.0

    def get_newest_timestamp(self):
        buf = self.buffers[1]
        if buf['last_slot'] is None:
            return time.time()
        return buf['timestamps'][buf['last_slot'] % buf['max_frames']]

    def get_view_at(self, width, height, pan_sec, target_resolution=None):
        """Return a frame starting at pan_sec with optimal resolution."""
        if self.t0 is None:
            return float(time.time()), np.full((width, height), np.nan, dtype=np.float32), 1

        # Select resolution
        res = 1
        if target_resolution is not None:
            # Find closest available resolution
            res = min(self.RESOLUTIONS, key=lambda x: abs(x - target_resolution))

        buf = self.buffers[res]
        max_f = buf['max_frames']

        effective_width = max(1, int(width / res))
        
        slot_start = int(math.floor((pan_sec - self.t0) / res))
        slot_end = slot_start + effective_width - 1

        if buf['last_slot'] is not None:
            if slot_end > buf['last_slot']:
                slot_end = buf['last_slot']
                slot_start = max(0, slot_end - effective_width + 1)
            
            if not buf['full']:
                slot_start = max(0, slot_start)
            else:
                min_available = buf['last_slot'] - max_f + 1
                slot_start = max(min_available, slot_start)
        
        actual_width = slot_end - slot_start + 1
        if actual_width <= 0:
            return pan_sec, np.full((1, height), np.nan, dtype=np.float32), 1

        idxs = (np.arange(actual_width) + slot_start) % max_f
        t_start = self.t0 + slot_start * res
        
        frame = buf['data'][idxs, :height]
        
        return float(t_start), frame, res


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
