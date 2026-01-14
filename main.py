"""
IEC 62304 – Class B

EEG Density Spectral Array Viewer
"""
import datetime

import numpy as np
import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout

from PySide6.QtCore import QTimer

import pyqtgraph as pg
from pyqtgraph import ColorBarItem
import datetime as dt

from config import SystemConfig, ConfigWidget
from data import EEGStream, save_psd_to_csv
from calculations import DSACalculator


class DSAView(pg.GraphicsLayoutWidget):
    def __init__(self, config: SystemConfig):
        super().__init__()

        self.config = config

        # --- Layout ---
        self.time_axis = pg.DateAxisItem("bottom")
        self.plot = self.addPlot(row=0, col=0, axisItems={"bottom": self.time_axis})
        self.plot.setLabel("bottom", "Time")
        self.plot.setLabel("left", "Frequency", units="Hz")
        self.plot.setMenuEnabled(False)
        self.plot.showGrid(x=False, y=False)
        self.plot.invertY(False)

        self.setInteractive(False)
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)

        # --- Colormap ---
        self._init_colormap()

        # --- Colorbar ---
        self.colorbar = ColorBarItem(
            values=(self.config.PSD_DB_MIN, self.config.PSD_DB_MAX),
            colorMap=self.cmap,
            label="Power / Frequency (dB/Hz)",
            interactive=False,
        )
        self.colorbar.setImageItem(self.image)
        self.addItem(self.colorbar, row=0, col=1)

        self.ci.layout.setColumnStretchFactor(0, 10)
        self.ci.layout.setColumnStretchFactor(1, 1)

        #self.dsa_buffer = DSABuffer(config)

        self.t0 = datetime.datetime.now().timestamp()
        nperseg = int(self.config.SEGMENT_SEC * self.config.SAMPLE_RATE_HZ)
        freq_bins = np.fft.rfftfreq(nperseg, d=1.0 / self.config.SAMPLE_RATE_HZ)
        mask = (freq_bins >= self.config.LOWEST_FREQ_HZ) & (freq_bins <= self.config.MAX_FREQ_HZ_BOUNDS[1])
        self.n_freq_bins = len(freq_bins[mask])
        self.n_time_bins = int(self.config.DISPLAY_MINUTES * 60.0 / self.config.WINDOW_SEC)

        self.dsa_rect = np.full((self.n_time_bins, self.n_freq_bins), np.nan, dtype=np.float32)

        self.write_index = 0

        # Image is (freq, time) when displayed
        self.image.setImage(
            self.dsa_rect,
            autoLevels=False
        )

        self.image.setLevels((
            self.config.PSD_DB_MIN,
            self.config.PSD_DB_MAX
        ))

        # Set pixel-to-axis mapping
        self.image.setRect(
            (
                self.t0,  # x
                0.0,  # y
                self.config.DISPLAY_MINUTES * 60,  # width
                self.config.MAX_FREQ_HZ  # height
            )
        )

    def _init_colormap(self):
        colors = [
            (36, 24, 111),
            (0, 0, 128),
            (0, 128, 255),
            (0, 255, 0),
            (255, 255, 0),
            (255, 0, 0),
        ]

        pos = np.linspace(0.0, 1.0, len(colors))
        self.cmap = pg.ColorMap(pos, colors)

        self.lut = self.cmap.getLookupTable(nPts=256, mode="byte")

        self.image.setLookupTable(self.lut)

    def update(self, dsa_buffer):
        self.t0, self.dsa_rect = dsa_buffer.get_view(
            width=int(self.config.DISPLAY_MINUTES * 60.0 / self.config.UPDATE_STEP_SEC),
            height=self.n_freq_bins,
        )

        self.image.setImage(
            self.dsa_rect,
            autoLevels=False,
            levels=(self.config.PSD_DB_MIN, self.config.PSD_DB_MAX),
            lut=self.cmap.getLookupTable(),
            nan_policy="omit",
        )

        self.image.setRect(
            (
                self.t0,  # x
                0.0,  # y
                self.config.DISPLAY_MINUTES*60,  # width
                self.config.MAX_FREQ_HZ  # height
            )
        )



class DSABuffer:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.apply_config()

        self.t0 = None
        self.last_slot = 0
        self.full = False

    def append(self, ts, f, psd):

        if len(f) != len(self.freq_bins):
            print("Frequency bins do not match. Resetting DSABuffer.")
            self.apply_config()
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
        #print("index")
        #print(idx, ts)

        # Fill gaps with NaNs
        if self.last_slot is not None and slot > self.last_slot + 1:
            for s in range(self.last_slot + 1, slot):
                self.data[s % self.max_frames] = np.nan
                self.timestamps[s % self.max_frames] = self.t0 + s * self.time_delta

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
        return int(np.round((ts - self.t0) / self.time_delta))

    def get_last_timestamp(self):
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

    def apply_config(self):
        # calculate freq_bins
        nperseg = int(self.config.SEGMENT_SEC * self.config.SAMPLE_RATE_HZ)
        self.freq_bins = np.fft.rfftfreq(nperseg, d=1.0/self.config.SAMPLE_RATE_HZ)
        mask = (self.freq_bins >= self.config.LOWEST_FREQ_HZ) & (self.freq_bins <= self.config.MAX_FREQ_HZ_BOUNDS[1])
        self.freq_bins = self.freq_bins[mask]
        self.max_frames = int(self.config.DISPLAY_MINUTES_BOUNDS[1]*60 / self.config.UPDATE_STEP_SEC)
        self.data = np.full((self.max_frames, len(self.freq_bins)), np.nan, dtype=np.float32)
        self.timestamps = np.full(self.max_frames, np.nan)
        self.time_delta = self.config.UPDATE_STEP_SEC
        self.last_ts = None
        self.full = False


class PSDView(pg.PlotWidget):
    def __init__(self, config: SystemConfig):
        super().__init__()

        self.config = config

        self.setLabel("bottom", "Frequency", units="Hz")
        self.setLabel("left", "Power Spectral Density", units="dB/Hz")
        self.setMenuEnabled(False)
        self.showGrid(x=True, y=True)

        self.curve = self.plot(pen=pg.mkPen("y", width=2))
        self.setInteractive(False)

        self.setYRange(
            self.config.PSD_DB_MIN-15,
            self.config.PSD_DB_MAX+15
        )

    def update(self, freqs, psd_db):
        if freqs is None or psd_db is None:
            return

        self.curve.setData(freqs, psd_db)


class EEGBuffer:
    def __init__(self):
        self.window_sec = SystemConfig.WINDOW_SEC
        self.timestamps = []
        self.eeg_values = []
        self.time_delta = 1.0/float(SystemConfig.SAMPLE_RATE_HZ)
        self.last_ts = None
        self.processor = DSACalculator()

    def extend_and_process(self, data):
        if data is None or len(data) == 0:
            return None
        output_dsa = []
        for ts, eeg in data:
            if self.last_ts is None:
                expected_ts = ts
            else:
                expected_ts = self.last_ts + datetime.timedelta(milliseconds=self.time_delta*1000)
            if expected_ts != ts or eeg is None or np.isnan(eeg): # make sure the window is continuos
                self.timestamps.clear()
                self.eeg_values.clear()
                self.last_ts = ts
                continue
            else:
                self.timestamps.append(ts)
                self.eeg_values.append(eeg)
            self.last_ts = ts

            if len(self.eeg_values) == int(self.window_sec * SystemConfig.SAMPLE_RATE_HZ):
                f, psd = self.processor.compute_psd_column(self.eeg_values.copy())
                output_dsa.append((ts.timestamp()-self.window_sec, f, psd))
                self.timestamps.clear()
                self.eeg_values.clear()

        return output_dsa

    def update_config(self):
        self.window_sec = SystemConfig.WINDOW_SEC
        self.time_delta = 1.0/float(SystemConfig.SAMPLE_RATE_HZ)

class EEGDSAApplication(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("EEG Density Spectral Array")
        self.config = SystemConfig()
        self.old_SEGMENT_SEC = self.config.SEGMENT_SEC

        self.eeg_buffer = EEGBuffer()
        self.dsa_buffer = DSABuffer(self.config)

        self.stream = None
        self.start_receive = False

        self.dsa_view = DSAView(self.config)
        self.psd_view = PSDView(self.config)

        self.config_widget = ConfigWidget(
            self.config,
            self._apply_new_config
        )

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.config_widget)
        layout.addWidget(self.dsa_view)
        layout.addWidget(self.psd_view)
        self.setCentralWidget(container)

        self.last_ts = None
        self.last_freqs = None
        self.time_since_last_ts = 0.0

        self.timer = QTimer()
        self.timer.setInterval(int(self.config.UPDATE_STEP_SEC*1000))
        self.timer.timeout.connect(self._update_cycle)
        self.timer.start()


    def _update_cycle(self):
        if self.stream is None:
            self.stream = EEGStream()
        while not self.stream.receiving:
            print("No stream available")
            print("Looking for LSL Stream...")
            self.stream = EEGStream()

        new_samples = self.stream.read_samples()
        self.time_since_last_ts += self.config.UPDATE_STEP_SEC
        if new_samples is None or len(new_samples) == 0:
            if not self.start_receive:
                return
            if self.time_since_last_ts > self.config.WINDOW_SEC:
                self.dsa_buffer.append(self.dsa_buffer.get_last_timestamp()+self.config.UPDATE_STEP_SEC, self.last_freqs, [])
        else:
            dsa_column = self.eeg_buffer.extend_and_process(new_samples)
            for ts, freqs, psd in dsa_column:
                if psd is None or freqs is None or ts is None:
                    continue
                save_psd_to_csv(freqs, psd, "C:\\temp\\VSCaptureWave")
                self.psd_view.update(freqs, psd)
                for i in range(int(self.config.WINDOW_SEC*(1.0/self.config.UPDATE_STEP_SEC))):
                    self.dsa_buffer.append(ts + i * self.config.UPDATE_STEP_SEC, freqs, psd)
                self.start_receive = True
                self.last_freqs = freqs
                self.last_ts = ts
                self.time_since_last_ts = 0.0

        # TODO: only update when necessary
        self.dsa_view.update(self.dsa_buffer)


    def _apply_new_config(self):
        self.timer.stop()

        self.processor = DSACalculator()
        if self.old_SEGMENT_SEC != self.config.SEGMENT_SEC:
            self.dsa_buffer.apply_config()
        #self.eeg_buffer = EEGBuffer()

        self.timer.start()


def main():
    app = QApplication(sys.argv)
    win = EEGDSAApplication()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
