import os, sys

# fix lsl for single file .exe
if getattr(sys, "frozen", False):
    base = sys._MEIPASS if hasattr(sys, "_MEIPASS") else os.path.dirname(sys.executable)
    os.environ["PYLSL_LIB"] = os.path.join(base, "pylsl", "lib", "lsl.dll")

from pylsl import StreamInlet, resolve_byprop
import csv
from datetime import datetime as dt
from config import SystemConfig
import numpy as np
import datetime
from calculations import DSACalculator


class DSABuffer:
    """Ring buffer for DSA frames (time x frequency) with gap-filling and wrap-around.

    Stores PSD columns aligned to a fixed time grid defined by `SystemConfig.UPDATE_STEP_SEC`.
    Read-only methods return windows sized for the current view.
    """
    def __init__(self, SEGMENT_SEC):
        self.SEGMENT_SEC = SEGMENT_SEC

        self.max_frames = int(SystemConfig.DISPLAY_MINUTES_BOUNDS[1] * 60 / SystemConfig.UPDATE_STEP_SEC)
        self._reset()

    def append(self, ts, f, psd):
        if len(f) != len(self.freq_bins):
            print("Frequency bins do not match. Resetting DSABuffer.")
            self._reset()
            return

        if psd is None or len(psd)==0:
            psd = np.full(len(self.freq_bins), np.nan, dtype=np.float32)

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
                self.timestamps[s % self.max_frames] = self.t0 + s * SystemConfig.UPDATE_STEP_SEC

        # Store data
        self.data[idx] = psd
        self.timestamps[idx] = ts

        # Mark buffer full if we wrapped
        if self.last_slot is not None and slot - self.last_slot >= self.max_frames:
            self.full = True
        elif slot >= self.max_frames:
            self.full = True

        self.last_slot = max(self.last_slot, slot) if self.last_slot is not None else slot

    def _timestamp_to_slot(self, ts):
        return int(np.round((ts - self.t0) / SystemConfig.UPDATE_STEP_SEC))

    def get_last_timestamp(self):
        if self.last_slot is None:
            return datetime.datetime.now().timestamp()
        return self.timestamps[self.last_slot % self.max_frames]

    def get_frame(self, width, height):
        if self.t0 is None:
            return datetime.datetime.now().timestamp(), np.empty((1, 0), dtype=np.float32)

        height = min(height, self.data.shape[1])
        width = min(width, self.max_frames)

        slot_now = self.last_slot
        if self.full:
            slot_start = slot_now - width + 1
        else:
            slot_start = max(0, slot_now - width + 1)

        slots = np.arange(slot_start, slot_now+1)
        idxs = slots % self.max_frames

        return self.timestamps[idxs[0]], self.data[idxs, :height]

    def get_view(self, width, height):
        dsa_view = np.full((width, height), np.nan, dtype=np.float32)
        t0, frame = self.get_frame(width, height)
        dsa_view[0:len(frame), 0:len(frame[0])] = frame

        return t0, dsa_view

    def get_view_at(self, width, height, pan_offset_sec):
        """Return a frame starting at t0 + pan_offset_sec."""
        if self.t0 is None:
            return datetime.datetime.now().timestamp(), np.full((width, height), np.nan, dtype=np.float32)

        height = min(height, self.data.shape[1])
        width = min(width, self.max_frames)

        # Convert pan offset to slot index
        pan_slots = int(np.round(pan_offset_sec / SystemConfig.UPDATE_STEP_SEC))
        slot_start = pan_slots
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

            slot_end = slot_start + width - 1

        slots = np.arange(slot_start, slot_end + 1)
        idxs = slots % self.max_frames

        frame = self.data[idxs, :height]
        t_start = self.t0 + slot_start * SystemConfig.UPDATE_STEP_SEC

        result = np.full((width, height), np.nan, dtype=np.float32)
        result[:len(frame)] = frame

        return float(t_start), result

    def apply_config(self, SEGMENT_SEC):
        if self.SEGMENT_SEC != SEGMENT_SEC:
            self.SEGMENT_SEC = SEGMENT_SEC
            self._reset()

    def _reset(self):
        nperseg = int(self.SEGMENT_SEC * SystemConfig.SAMPLE_RATE_HZ)
        freq_bins = np.fft.rfftfreq(nperseg, d=1.0 / SystemConfig.SAMPLE_RATE_HZ)
        mask = (freq_bins >= SystemConfig.LOWEST_FREQ_HZ) & (freq_bins <= SystemConfig.MAX_FREQ_HZ_BOUNDS[1])
        self.freq_bins = freq_bins[mask]

        self.data = np.full((self.max_frames, len(self.freq_bins)), np.nan, dtype=np.float32)
        self.timestamps = np.full(self.max_frames, np.nan)

        self.t0 = None
        self.last_slot = None
        self.full = False


class EEGBuffer:
    """Buffers raw EEG samples and emits DSA-ready PSD columns using a sliding window.

    - Maintains continuity by resetting on missing/invalid samples or timestamp gaps.
    - Uses DSACalculator to compute a PSD for each full window and advances by `hop_len` samples.
    - Exposes `last_accepted_samples` so UI (EEGView) can update per-sample with validated data.
    """
    def __init__(self, window_sec, segment_sec, segment_overlap, overlap):
        self.window_sec = window_sec
        self.timestamps = []
        self.eeg_values = []
        self.time_delta = 1000.0/float(SystemConfig.SAMPLE_RATE_HZ)
        self.last_ts = None
        self.processor = DSACalculator(window_sec, segment_sec, segment_overlap)
        self.window_len = int(window_sec * SystemConfig.SAMPLE_RATE_HZ)
        self.hop_len = int(self.window_len * (1.0 - overlap))
        if self.hop_len < 1:
            self.hop_len = 1
        # Will be filled by `get_dsa_columns` with tuples (epoch_seconds, value)
        self.last_accepted_samples = []

    def _is_valid_sample(self, timestamp, value):
        if self.last_ts is not None:
            expected = self.last_ts + datetime.timedelta(milliseconds=self.time_delta)
            dt = abs((timestamp - expected).total_seconds())
            if dt > SystemConfig.TIME_DIFF_TOLERANCE:
                print(f"Timestamp fault:{timestamp}, expected {expected}")
                return False
            if value is None or np.isnan(value):
                print(f"Invalid sample: {timestamp}, {value}")
                return False
        return True

    def get_dsa_columns(self, data):
        if data is None or len(data) == 0:
            return [], []

        output_dsa = []
        samples = []

        for ts, eeg in data:
            # continuity check
            if not self._is_valid_sample(ts, eeg):
                self.eeg_values.clear()
                self.timestamps.clear()
                self.last_ts = None
                samples.append((ts.timestamp(), np.nan))
                continue

            self.eeg_values.append(eeg)
            self.timestamps.append(ts)
            self.last_ts = ts
            samples.append((ts.timestamp(), float(eeg)))

            # while enough data exists for a window
            while len(self.eeg_values) >= self.window_len:
                window = self.eeg_values[:self.window_len]
                window_ts = self.timestamps[self.window_len - 1]

                f, psd = self.processor.compute_psd_column(window.copy())

                output_dsa.append((window_ts.timestamp() - self.window_sec, f, psd))

                # SLIDE the window forward
                self.eeg_values = self.eeg_values[self.hop_len:]
                self.timestamps = self.timestamps[self.hop_len:]

        return output_dsa, samples

    def apply_config(self, window_sec, segment_sec, segment_overlap, overlap):
        self.window_sec = window_sec
        self.window_len = int(window_sec * SystemConfig.SAMPLE_RATE_HZ)
        self.hop_len = int(self.window_len * (1.0 - overlap))
        if self.hop_len < 1:
            self.hop_len = 1
        self.processor.update_config(window_sec, segment_sec, segment_overlap)


class EEGStream:
    def __init__(self):
        self.receiving = False
        self._inlet = None
        self.connect()

    def connect(self):
        try:
            streams = resolve_byprop("name", "EEG_DATA", timeout=2)
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
                elif value < SystemConfig.EEG_BOUNDS[0] or value > SystemConfig.EEG_BOUNDS[1]:
                    print(f"Out of bounds: {value}")
                    value = np.nan

                samples.append((timestamp, value))

            except Exception as e:
                # Invalid sample discarded
                print("Invalid sample: ",  e)
                continue

        return samples


class Output:
    @staticmethod
    def _build_filename(base_dir, freqs, start_time=None):
        freqs = np.asarray(freqs)
        if freqs.size < 2:
            raise ValueError("At least 2 frequency bins required")

        df = round(freqs[1] - freqs[0], 6)

        if start_time is None:
            start_time = dt.now()
        else:
            start_time = dt.fromtimestamp(start_time)
        ts = start_time.strftime("%Y-%m-%d")

        filename = f"dsa_{ts}_df{df:.1f}Hz.csv"
        return os.path.join(base_dir, filename)

    @staticmethod
    def save_psd_to_csv(timestamp, freqs, psd_db):
        """
        Save a single PSD column to a CSV file.

        File format:
        - Header written on first creation: 'timestamp' followed by one column per frequency bin labeled
          like 'f_{freq:.2f}_Hz'.
        - Each call appends a new row with the provided timestamp and PSD values (dB/Hz).

        Args:
            timestamp: Optional timestamp for the row. If a datetime is provided, it will be formatted
                       in ISO 8601 with milliseconds. If None, the current time is used.
            freqs: 1D array-like of frequency bin centers (Hz), length N.
            psd_db: 1D array-like of PSD values in dB/Hz, length N.

        Raises:
            ValueError: If shapes of inputs mismatch or if appending to an existing file with an
                        incompatible number of columns (i.e., different frequency bins).
        """

        # Normalize inputs
        freqs = np.asarray(freqs).ravel()
        psd_db = np.asarray(psd_db).ravel()
        if freqs.shape[0] != psd_db.shape[0]:
            raise ValueError("freqs and psd_db must have the same length")

        # Prepare timestamp string
        if timestamp is None:
            ts_str = dt.now().isoformat(timespec="milliseconds")
        else:
            if isinstance(timestamp, dt):
                ts_str = timestamp.isoformat(timespec="milliseconds")
            elif isinstance(timestamp, (int, float)):
                ts_str = dt.fromtimestamp(timestamp).isoformat(timespec="milliseconds")
            else:
                ts_str = str(timestamp)
        # Ensure directory exists
        filepath = Output._build_filename(SystemConfig.BASE_DIR, freqs, timestamp)
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
                        expected_cols = 1 + len(freqs)  # timestamp + N frequencies
                        if len(header) != expected_cols:
                            raise ValueError(
                                f"Existing CSV has {len(header)} columns but expected {expected_cols}. "
                                "Frequency bins must remain identical across saves."
                            )
            except FileNotFoundError:
                write_header = True
        # Write
        # TODO: Save in a predefined resolution of 0.5 Hz
        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                freq_headers = [f"f_{freq:.2f}_Hz" for freq in freqs]
                writer.writerow(["timestamp"] + freq_headers)

            row_values = [int(np.round(x, 0)) if np.isfinite(x) else "" for x in psd_db]
            writer.writerow([ts_str] + row_values)

