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

    def get_dsa_columns(self, data):
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

                psd = self.processor.compute_psd_column(window)

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
                timestamp = dt.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")

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
    def save_psd_to_csv(timestamp, psd_db):
        psd_db = np.asarray(psd_db).ravel()

        if timestamp is None:
            ts_str = dt.now().isoformat(timespec="milliseconds")
        else:
            if isinstance(timestamp, dt):
                ts_str = timestamp.isoformat(timespec="milliseconds")
            elif isinstance(timestamp, (int, float)):
                ts_str = dt.fromtimestamp(timestamp).isoformat(timespec="milliseconds")
            else:
                ts_str = str(timestamp)
        filepath = Output._build_filename(config.BASE_DIR, timestamp)
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0

        # If appending to an existing file, validate column count
        if not write_header:
            try:
                with open(filepath, "r", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header is None:
                        write_header = True
                    else:
                        expected_cols = 1 + len(config.FREQ_BINS)  # timestamp + N frequencies
                        if len(header) != expected_cols:
                            raise ValueError(
                                f"Existing CSV has {len(header)} columns but expected {expected_cols}. "
                                "Frequency bins must remain identical across saves."
                            )
            except FileNotFoundError:
                print(f"File not found: {filepath} creating new file")
                write_header = True
        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                freq_headers = [f"f_{freq:.2f}_Hz" for freq in config.FREQ_BINS]
                writer.writerow(["timestamp"] + freq_headers)

            row_values = [int(np.round(x, 0)) if np.isfinite(x) else "" for x in psd_db]
            writer.writerow([ts_str] + row_values)
