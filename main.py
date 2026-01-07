"""
IEC 62304 – Class B

EEG Density Spectral Array Viewer
"""

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

        self.dsa = None

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

    def initialize(self, freq_bins, t0):
        self.freq_bins = freq_bins

        self.time_bins = int(
            self.config.DISPLAY_MINUTES*60 / self.config.WINDOW_SEC
        )

        # Internal buffer: time x frequency
        self.dsa = np.full((self.time_bins, len(freq_bins)), np.nan)
        self.write_index = 0

        self.t0 = t0.timestamp()
        self.last_timestamp = t0.timestamp()

        print(self.t0)
        delta_f = freq_bins[1] - freq_bins[0]

        # Image is (freq, time) when displayed
        self.image.setImage(
            self.dsa,
            autoLevels=False
        )

        self.image.setLevels((
            self.config.PSD_DB_MIN,
            self.config.PSD_DB_MAX
        ))

        # Set pixel-to-axis mapping
        self.image.setRect(
            (
                self.t0,             # x
                freq_bins[0],               # y
                self.config.DISPLAY_MINUTES * 60, # width
                len(freq_bins) * delta_f         # height
            )
        )

    def update(self, psd_column, timestamp):
        psd_column = np.asarray(psd_column)
        psd_column[~np.isfinite(psd_column)] = np.nan
        if self.write_index < self.time_bins:
            # Fill from left to right (startup phase)
            self.dsa[self.write_index, :] = psd_column
            self.write_index += 1
        else:
            # Scroll left once full
            self.dsa[:-1, :] = self.dsa[1:, :]
            self.dsa[-1, :] = psd_column

            self.t0 += (timestamp.timestamp() - self.last_timestamp) # TODO: + time_delta of appended timestep
            print(self.t0)

        self.image.setRect(
            (
                self.t0,  # x
                self.freq_bins[0],  # y
                self.config.DISPLAY_MINUTES*60,  # width
                self.freq_bins[-1] - self.freq_bins[0]  # height
            )
        )

        self.image.setImage(
            self.dsa,
            autoLevels=False,
            levels=(self.config.PSD_DB_MIN, self.config.PSD_DB_MAX),
            lut=self.cmap.getLookupTable(),
            nan_policy="omit",
        )

class BoundedDSABuffer:
    def __init__(self, config: SystemConfig, freq_bins):
        self.config = config
        self.max_frames = int(self.config.DISPLAY_MINUTES_BOUNDS[1]*60 / self.config.UPDATE_STEP_SEC)
        self.freq_bins = freq_bins

        self.timestamps = []
        self.data = []

        self.write_index = 0
        self.full = False

    def append(self, timestamp, psd):
        ts = timestamp.timestamp()

        if not self.full:
            self.timestamps.append(ts)
            self.data.append(psd)

            if len(self.timestamps) == self.max_frames:
                self.full = True
                self.write_index = 0
        else:
            self.timestamps[self.write_index] = ts
            self.data[self.write_index] = psd
            self.write_index = (self.write_index + 1) % self.max_frames

    def view(self):
        if not self.full:
            return (
                np.asarray(self.timestamps),
                np.asarray(self.data),
            )

        idx = self.write_index
        ts = (
            self.timestamps[idx:] +
            self.timestamps[:idx]
        )
        data = (
            self.data[idx:] +
            self.data[:idx]
        )

        return (
            np.asarray(ts),
            np.asarray(data),
        )


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

class EEGDSAApplication(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("EEG Density Spectral Array")

        self.config = SystemConfig()
        self.stream = EEGStream()

        self.processor = DSACalculator(self.config)
        self.buffer = []

        self.view = DSAView(self.config)
        self.psd_view = PSDView(self.config)

        self.config_widget = ConfigWidget(
            self.config,
            self._apply_new_config
        )

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.config_widget)
        layout.addWidget(self.view)
        layout.addWidget(self.psd_view)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self._update_timer()
        self.timer.timeout.connect(self._update_cycle)
        self.timer.start()

    def _update_cycle(self):
        new_samples = self.stream.read_samples()
        self.buffer.extend(new_samples)  # buffer: list of (timestamp_sec, value)

        if not self.buffer or len(self.buffer) < self.config.SAMPLE_RATE_HZ*self.config.WINDOW_SEC:
            return

        self.last_time = self.buffer[-1][0]

        # Determine the current window
        window_start = self.last_time - dt.timedelta(seconds=self.config.WINDOW_SEC)
        window_end = self.last_time

        # Select samples inside the current window
        window_samples = [v for t, v in self.buffer if window_start <= t <= window_end]

        # Remove old samples (older than the last full window)
        self.buffer = [s for s in self.buffer if s[0] >= window_start]
        print(window_samples)
        f, psd_db = self.processor.compute_psd_column(np.array(window_samples))
        if psd_db is None:
            return
        save_psd_to_csv(f, psd_db, "C:\\temp\\VSCaptureWave")

        if self.view.dsa is None:
            self.view.initialize(f, self.buffer[0][0])
        else:
            # Reinitialize only if frequency bins changed (e.g., MAX_FREQ_HZ/SEGMENT affect f)
            try:
                if len(f) != len(self.view.freq_bins) or not np.allclose([f[0], f[-1]], [self.view.freq_bins[0], self.view.freq_bins[-1]]):
                    self.view.initialize(f, self.buffer[0][0])
            except Exception:
                # If anything goes wrong comparing, fall back to reinit
                self.view.initialize(f, self.buffer[0][0])

        self.buffer.clear()
        self.view.update(psd_db, self.last_time)
        self.psd_view.update(f, psd_db)

    def _apply_new_config(self):
        self.timer.stop()

        self.processor = DSACalculator(self.config)
        self.buffer.clear()

        self._update_timer()

        self.timer.start()

    def _update_timer(self):
        self.timer.setInterval(
            int(self.config.UPDATE_STEP_SEC * 1000)
        )

def main():
    app = QApplication(sys.argv)
    win = EEGDSAApplication()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
