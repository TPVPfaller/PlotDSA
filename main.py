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
        width = int(self.config.DISPLAY_MINUTES * 60 / self.config.UPDATE_STEP_SEC)
        self.t0 = datetime.datetime.now().timestamp()

        # --- Layout ---
        self.time_axis = pg.DateAxisItem("bottom")
        self.plot = self.addPlot(row=0, col=0, axisItems={"bottom": self.time_axis})
        self.plot.setLabel("bottom", "Time")
        self.plot.setLabel("left", "Frequency", units="Hz")
        self.plot.setMenuEnabled(False)
        self.plot.showGrid(x=False, y=False)
        self.plot.invertY(False)

        self.setInteractive(False)
        self.dsa_view = None
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

        self.dsa_buffer = DSABuffer(config)

        nperseg = int(self.config.SEGMENT_SEC * self.config.SAMPLE_RATE_HZ)
        self.freq_bins = np.fft.rfftfreq(nperseg, d=1.0 / self.config.SAMPLE_RATE_HZ)

        self.time_bins = int(self.config.DISPLAY_MINUTES * 60 / self.config.WINDOW_SEC)

        # Internal buffer: time x frequency
        self.time_bins = int(
            (self.config.DISPLAY_MINUTES * 60) / self.config.UPDATE_STEP_SEC
        )

        self.dsa_view = np.full((self.time_bins, len(self.freq_bins)), np.nan, dtype=np.float32)

        self.write_index = 0

        self.t0 = datetime.datetime.now().timestamp()
        print(self.t0)

        # Image is (freq, time) when displayed
        self.image.setImage(
            self.dsa_view,
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
                self.freq_bins[0],  # y
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
        self.t0, self.dsa_view = dsa_buffer.get_view(
            width=self.config.DISPLAY_MINUTES * 60,
            height=self.config.MAX_FREQ_HZ,
            num_time_bins=self.time_bins,
        )

        self.image.setImage(
            self.dsa_view,
            autoLevels=False,
            levels=(self.config.PSD_DB_MIN, self.config.PSD_DB_MAX),
            lut=self.cmap.getLookupTable(),
            nan_policy="omit",
        )

        self.image.setRect(
            (
                self.t0,  # x
                self.freq_bins[0],  # y
                self.config.DISPLAY_MINUTES*60,  # width
                self.freq_bins[-1] - self.freq_bins[0]  # height
            )
        )



class DSABuffer:
    def __init__(self, config: SystemConfig, freq_bins=None):
        self.config = config
        self.reset(freq_bins)
        self._view_write_index = 0
        self.write_index = 0

    def append(self, timestamp, f, psd):
        if len(f) != len(self.freq_bins):
            print("Frequency bins do not match. Resetting DSABuffer")
            self.reset(f)
        ts = timestamp.timestamp()
        print(ts)
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


    def get_view(self, width, height, num_time_bins):
        """
        Returns a fixed-size DSA view (num_time_bins, n_freqs) for ImageItem.
        Scrolls left→right. Handles buffer larger than view.
        """

        n_freqs = len(self.freq_bins)

        # --- empty buffer ---
        if len(self.data) == 0:
            return datetime.datetime.now().timestamp(), np.full((num_time_bins, n_freqs), np.nan, dtype=np.float32)

        # --- ordered buffer (oldest → newest) ---
        if not self.full:
            dsa_buf = np.asarray(self.data)  # (Nbuf, F)
        else:
            idx = self.write_index
            dsa_buf = np.asarray(self.data[idx:] + self.data[:idx])

        # --- apply frequency height ---
        if height is not None:
            height = min(height, dsa_buf.shape[1])
            dsa_buf = dsa_buf[:, :height]

        n_buf, n_freqs = dsa_buf.shape

        # --- scale factor ---
        scale = num_time_bins/width

        repeat = max(1, int(round(scale)))
        dsa_resampled = np.repeat(dsa_buf, repeat, axis=0)

        n_res = dsa_resampled.shape[0]

        # --- keep track of view write index separately ---
        if not hasattr(self, "_view_write_index"):
            self._view_write_index = 0

        # --- scroll logic ---
        dsa_view = np.full((num_time_bins, n_freqs), np.nan, dtype=np.float32)

        if n_res <= num_time_bins:
            # buffer smaller than view → pad left
            pad = num_time_bins - n_res
            dsa_view[pad:] = dsa_resampled
            self._view_write_index = n_res % num_time_bins
        else:
            # buffer larger than view → take last num_time_bins columns
            dsa_view[:, :] = dsa_resampled[-num_time_bins:]
            self._view_write_index = num_time_bins

        return datetime.datetime.now().timestamp(), dsa_view

    def reset(self, freq_bins=None):
        if freq_bins is None:  # calculate freq_bins
            nperseg = int(self.config.SEGMENT_SEC * self.config.SAMPLE_RATE_HZ)
            self.freq_bins = np.fft.rfftfreq(nperseg, d=1.0 / self.config.SAMPLE_RATE_HZ)
            print(len(self.freq_bins))
        else:
            self.freq_bins = freq_bins
        self.max_frames = int(self.config.DISPLAY_MINUTES_BOUNDS[1]*60 / self.config.UPDATE_STEP_SEC)
        self.timestamps = []
        self.data = []
        self.write_index = 0
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
        self.time_delta = 1000.0/float(SystemConfig.SAMPLE_RATE_HZ)
        self.last_ts = None
        self.processor = DSACalculator()

    def extend_and_process(self, data):
        if data is None or len(data) == 0:
            return None
        output_dsa = []
        if len(self.timestamps) == 0:
            self.last_ts = data[-1][0]
        else:
            self.last_ts = self.timestamps[-1]
        for ts, eeg in data:
            expected_ts = self.last_ts + datetime.timedelta(milliseconds=self.time_delta)
            if expected_ts != ts or eeg is None or np.isnan(eeg): # make sure the window is continuos
                self.timestamps.clear()
                self.eeg_values.clear()
                print(self.last_ts)
                self.last_ts = ts
                print(expected_ts, ts)
                continue
            else:
                self.timestamps.append(ts)
                self.eeg_values.append(eeg)
            self.last_ts = ts

            if len(self.eeg_values) == int(self.window_sec * SystemConfig.SAMPLE_RATE_HZ):
                f, psd = self.processor.compute_psd_column(self.eeg_values.copy())
                output_dsa.append((ts, f, psd))
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
        self.stream = EEGStream()

        self.eeg_buffer = EEGBuffer()
        self.dsa_buffer = DSABuffer(self.config)

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

        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._update_cycle)
        self.timer.start()

    def _update_cycle(self):
        new_samples = self.stream.read_samples()
        print(new_samples)
        if new_samples is None or len(new_samples) == 0:
            if not self.start_receive:
                return

        else:
            self.start_receive = True
            dsa_column = self.eeg_buffer.extend_and_process(new_samples)
            for ts, f, psd in dsa_column:
                if psd is None or f is None or ts is None:
                    continue
                save_psd_to_csv(f, psd, "C:\\temp\\VSCaptureWave")
                self.psd_view.update(f, psd)
                self.dsa_buffer.append(ts, f, psd)

            print(dsa_column)

        self.dsa_view.update(self.dsa_buffer)


    def _apply_new_config(self):
        self.timer.stop()

        self.processor = DSACalculator()
        self.dsa_buffer.reset()
        self.eeg_buffer = EEGBuffer()

        self.timer.start()


def main():
    app = QApplication(sys.argv)
    win = EEGDSAApplication()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
