import datetime
import math
import time
from collections import deque

import numpy as np

import config
from calculations import DSACalculator


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
                "data": {},
                "order": deque(),
                "last_slot": None,
                "max_frames": int(self.max_minutes * 60 / res),
                "counts": {},
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
        last_slot = buf["last_slot"]
        if last_slot is None or slot > last_slot:
            self._trim_expired_slots(buf, slot)
            buf["last_slot"] = slot

        if psd is None:
            return

        existing = buf["data"].get(slot)
        if existing is None or res == 1:
            buf["data"][slot] = psd
            if existing is None:
                buf["counts"][slot] = 1
                buf["order"].append(slot)
        else:
            count = buf["counts"][slot]
            buf["data"][slot] = existing + (psd - existing) / (count + 1)
            buf["counts"][slot] = count + 1

    def _trim_expired_slots(self, buf, latest_slot):
        oldest_kept = max(0, latest_slot - buf["max_frames"] + 1)
        while buf["order"] and buf["order"][0] < oldest_kept:
            expired_slot = buf["order"].popleft()
            del buf["data"][expired_slot]
            del buf["counts"][expired_slot]

    def _get_oldest_slot(self, res):
        if self.t0 is None:
            return None
        buf = self.buffers[res]
        if buf["last_slot"] is None:
            return 0
        return max(0, buf["last_slot"] - buf["max_frames"] + 1)

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
        data = buf["data"]
        slot_start = int(math.floor((pan_sec - self.t0) / res))
        slot_start = max(self._get_oldest_slot(res) or 0, slot_start)
        slot_end = min(slot_start + effective_width - 1, buf["last_slot"])
        actual_width = slot_end - slot_start + 1
        t_start = self.t0 + slot_start * res
        frame = np.full((actual_width, height), np.nan, dtype=np.float32)

        for frame_idx, slot in enumerate(range(slot_start, slot_end + 1)):
            column = data.get(slot)
            if column is not None:
                frame[frame_idx] = column[:height]

        return float(t_start), frame, res


class EEGBuffer:
    """Buffers raw EEG samples and emits DSA-ready PSD columns using a sliding window."""

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

    def get_dsa_columns(self, data, method="multitaper"):
        if data is None or len(data) == 0:
            return [], []

        output_dsa = []
        samples = []

        for ts, eeg in data:
            diff = self._get_ts_diff(ts, eeg)
            if diff > config.EEG_TIME_DIFF_TOLERANCE:
                samples.append(np.nan)
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

            while len(self.eeg_values) >= self.window_len:
                window = np.asarray(self.eeg_values[:self.window_len], dtype=np.float32)
                window_ts = self.timestamps[self.window_len - 1]
                psd = self.processor.compute_psd_column(window, method=method)
                output_dsa.append((window_ts.timestamp() - self.window_sec, psd))
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
