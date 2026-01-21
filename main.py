"""
IEC 62304 – Class B

EEG Density Spectral Array Viewer
"""

import numpy as np
import sys
import datetime
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PySide6.QtCore import QTimer

from config import SystemConfig, ConfigWidget
from data import EEGStream, save_psd_to_csv
from calculations import DSACalculator
from views import DSAView, PSDView, EEGView

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

    def extend_and_process(self, data):
        if data is None or len(data) == 0:
            return []

        output_dsa = []

        for ts, eeg in data:
            # continuity check
            if self.last_ts is not None:
                expected = self.last_ts + datetime.timedelta(milliseconds=self.time_delta)
                dt = abs((ts - expected).total_seconds())
                if dt > SystemConfig.TIME_DIFF_TOLERANCE * self.time_delta / 1000.0 or eeg is None or np.isnan(eeg):
                    print(ts, expected)
                    self.eeg_values.clear()
                    self.timestamps.clear()
                    self.last_ts = ts
                    continue

            self.eeg_values.append(eeg)
            self.timestamps.append(ts)
            self.last_ts = ts

            # while enough data exists for a window
            while len(self.eeg_values) >= self.window_len:
                window = self.eeg_values[:self.window_len]
                window_ts = self.timestamps[self.window_len - 1]

                f, psd = self.processor.compute_psd_column(window.copy())

                output_dsa.append((window_ts.timestamp() - self.window_sec / 2.0, f, psd))

                # SLIDE the window forward
                self.eeg_values = self.eeg_values[self.hop_len:]
                self.timestamps = self.timestamps[self.hop_len:]

        return output_dsa

    def update_config(self, window_sec, segment_sec, segment_overlap, overlap):
        self.window_sec = window_sec
        self.window_len = int(window_sec * SystemConfig.SAMPLE_RATE_HZ)
        self.hop_len = int(self.window_len * (1.0 - overlap))
        if self.hop_len < 1:
            self.hop_len = 1
        self.processor.update_config(window_sec, segment_sec, segment_overlap)

class EEGDSAApplication(QMainWindow):
    """Main Qt application wiring together config UI, data stream, processing, and views.

    Periodically pulls samples from `EEGStream`, updates the raw EEG view immediately,
    computes PSD/DSA via `EEGBuffer` and `DSACalculator`, and refreshes DSA/PSD/EEG
    displays on the configured cadence.
    """

    def __init__(self):
        super().__init__()

        self.config = ConfigWidget(self._set_new_config)

        self.setWindowTitle("EEG Density Spectral Array")

        self.eeg_buffer = EEGBuffer(self.config.WINDOW_SEC, self.config.SEGMENT_SEC, self.config.SEGMENT_OVERLAP, self.config.OVERLAP)
        self.dsa_buffer = DSABuffer(self.config.SEGMENT_SEC)

        self.stream = None
        self.start_receive = False
        self.new_config = False

        self.dsa_view = DSAView(self.config)
        self.psd_view = PSDView(self.config.PSD_DB_MIN, self.config.PSD_DB_MAX)
        self.eeg_view = EEGView(self.config.WINDOW_SEC)

        self.io_executor = ThreadPoolExecutor(max_workers=1)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.config)
        layout.addWidget(self.dsa_view)
        layout.addWidget(self.psd_view)
        layout.addWidget(self.eeg_view)
        self.setCentralWidget(container)

        self.last_ts = None
        self.last_freqs = None
        self.update = False
        self.time_since_last_ts = 0.0

        self.timer = QTimer()
        self.timer.setInterval(int(SystemConfig.UPDATE_STEP_SEC*1000))
        self.timer.timeout.connect(self._update_cycle)
        self.timer.start()


    def _update_cycle(self):
        if self.stream is None:
            self.stream = EEGStream()
            if not self.stream.receiving:
                return  # Skip this cycle if still no stream

        new_samples = self.stream.read_samples()
        if not self.stream.receiving:
            self.stream.connect()
            return

        # Update raw EEG view first
        self.eeg_view.append_samples(new_samples)

        dsa_column = self.eeg_buffer.extend_and_process(new_samples)
        for ts, freqs, psd in dsa_column:
            if psd is None or freqs is None or ts is None:
                continue

            # Run disk I/O in a background thread to prevent UI stuttering
            self.io_executor.submit(
                save_psd_to_csv,
                freqs,
                psd,
                "C:\\temp\\VSCaptureWave",
                ts
            )
            self.psd_view.update(freqs, psd)
            steps = int(self.config.WINDOW_SEC*(1.0/SystemConfig.UPDATE_STEP_SEC))
            for i in range(steps):
                self.dsa_buffer.append(ts + i * SystemConfig.UPDATE_STEP_SEC, freqs, psd)

            self.last_freqs = freqs
            self.last_ts = ts
            self.time_since_last_ts = 0.0
            self.update = True
            self.start_receive = True

        if self.update:
            self.dsa_view.update(self.dsa_buffer)
            self.eeg_view.redraw_current()
            self.update = False

        if self.new_config:
            self._update_configs()


    def _set_new_config(self):
        self.new_config = True

    def _update_configs(self):
        self.timer.stop()

        # Deletes old Buffer when SEGMENT_SEC is updated
        self.dsa_buffer.apply_config(self.config.SEGMENT_SEC)
        self.eeg_buffer.update_config(self.config.WINDOW_SEC, self.config.SEGMENT_SEC, self.config.SEGMENT_OVERLAP, self.config.OVERLAP)

        self.dsa_view.update_config(self.config)
        self.psd_view.update_config(self.config.PSD_DB_MIN, self.config.PSD_DB_MAX)
        self.eeg_view.update_config(self.config.WINDOW_SEC)

        self.update = True
        self.new_config = False
        self.timer.start()

def main():
    app = QApplication(sys.argv)
    win = EEGDSAApplication()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
