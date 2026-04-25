import math
import os, sys
from collections import deque

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

    Stores only populated PSD columns aligned to a fixed time grid at multiple
    resolutions, then reconstructs NaN-padded views on demand.
    """

    RESOLUTIONS = [1, 10, 40, 100, 300, 600]  # seconds per frame

    def __init__(self):
        self.max_minutes = config.DISPLAY_MINUTES_BOUNDS[1]
        self.n_freqs = len(config.FREQ_BINS)
        try:
            self._reset()
        except MemoryError:
            print("Failed to allocate DSABuffer. Reducing size.")
            self.max_minutes = 24 * 60
            self._reset()

    def _reset(self):
        self.t0 = None
        self.latest_timestamp = None
        self.buffers = {
            res: {
                'data': {},
                'order': deque(),
                'last_slot': None,
                'max_frames': int(self.max_minutes * 60 / res),
                'counts': {}
            }
            for res in self.RESOLUTIONS
        }

    def append(self, ts, psd):
        has_data = psd is not None and len(psd) and not np.isnan(psd).all()
        if has_data:
            psd = np.array(psd, dtype=np.float32, copy=True)
        else:
            psd = None

        if self.t0 is None:
            self.t0 = ts
        self.latest_timestamp = ts

        for res, buf in self.buffers.items():
            self._append_to_res(buf, self._get_slot(ts, res), psd, res)

    def _get_slot(self, ts, res):
        offset = (ts - self.t0) / res
        return int(offset + 0.5) if res == 1 else int(math.floor(offset))

    def _append_to_res(self, buf, slot, psd, res):
        last_slot = buf['last_slot']
        if last_slot is None or slot > last_slot:
            self._trim_expired_slots(buf, slot)
            buf['last_slot'] = slot

        if psd is None:
            return

        existing = buf['data'].get(slot)
        if existing is None or res == 1:
            buf['data'][slot] = psd
            if existing is None:
                buf['counts'][slot] = 1
                buf['order'].append(slot)
        else:
            count = buf['counts'][slot]
            buf['data'][slot] = existing + (psd - existing) / (count + 1)
            buf['counts'][slot] = count + 1


    def _trim_expired_slots(self, buf, latest_slot):
        oldest_kept = max(0, latest_slot - buf['max_frames'] + 1)
        while buf['order'] and buf['order'][0] < oldest_kept:
            expired_slot = buf['order'].popleft()
            del buf['data'][expired_slot]
            del buf['counts'][expired_slot]

    def _get_oldest_slot(self, res):
        if self.t0 is None:
            return None
        buf = self.buffers[res]
        if buf['last_slot'] is None:
            return 0
        return max(0, buf['last_slot'] - buf['max_frames'] + 1)

    def get_oldest_timestamp(self):
        oldest_slot = self._get_oldest_slot(1)
        return time.time() if oldest_slot is None else self.t0 + oldest_slot

    def get_newest_timestamp(self):
        return time.time() if self.latest_timestamp is None else self.latest_timestamp

    def get_view_at(self, width, height, pan_sec, target_resolution):
        """Return a frame starting at pan_sec with optimal resolution."""
        res = min(self.RESOLUTIONS, key=lambda x: abs(x - target_resolution))
        effective_width = max(1, int(width / res))
        if self.t0 is None:
            return float(time.time()), np.full((effective_width, height), np.nan, dtype=np.float32), res

        pan_sec = max(pan_sec, self.t0)
        buf = self.buffers[res]
        data = buf['data']
        slot_start = int(math.floor((pan_sec - self.t0) / res))
        slot_start = max(self._get_oldest_slot(res) or 0, slot_start)
        slot_end = min(slot_start + effective_width - 1, buf['last_slot'])
        actual_width = slot_end - slot_start + 1
        t_start = self.t0 + slot_start * res
        frame = np.full((actual_width, height), np.nan, dtype=np.float32)

        for frame_idx, slot in enumerate(range(slot_start, slot_end + 1)):
            column = data.get(slot)
            if column is not None:
                frame[frame_idx] = column[:height]

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
        existing_rows = []
        timestamp_exists = False
        truncated = False

        # If appending to an existing file, validate column count and truncate any newer rows.
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

                        for row in reader:
                            if not row:
                                continue
                            row_ts = row[0]
                            if row_ts > ts_str:
                                truncated = True
                                continue
                            existing_rows.append(row)
                            if row_ts == ts_str:
                                timestamp_exists = True
            except FileNotFoundError:
                print(f"File not found: {filepath} creating new file")
                write_header = True

        if truncated:
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                freq_headers = [f"f_{freq:.1f}_Hz" for freq in config.FREQ_BINS]
                writer.writerow(["timestamp", "duration"] + freq_headers)
                writer.writerows(existing_rows)
            write_header = False

        if timestamp_exists:
            return

        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                freq_headers = [f"f_{freq:.1f}_Hz" for freq in config.FREQ_BINS]
                writer.writerow(["timestamp", "duration"] + freq_headers)

            row_values = [np.round(x, 4) if np.isfinite(x) else "nan" for x in psd]
            writer.writerow([ts_str, f"{duration:.3f}"] + row_values)
