import sys
import numpy as np
from datetime import datetime
from scipy.signal import spectrogram
from pylsl import StreamInlet, resolve_byprop

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt6.QtCore import QTimer

import pyqtgraph as pg

# -------------------------
# Settings
# -------------------------
SAMPLE_RATE = 400   # Hz
WINDOW_SEC = 2      # 2 seconds window
STEP_SEC = 0.25     # update every 250 ms

WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)
STEP_SAMPLES = int(STEP_SEC * SAMPLE_RATE)


def time_to_seconds(t_str):
    dt = datetime.strptime(t_str, "%H:%M:%S.%f")
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6


class EEGWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Realtime EEG Density Spectral Array (DSA)")
        self.resize(900, 600)

        # -------------------------
        # GUI Layout
        # -------------------------
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # PyQtGraph ImageView for DSA
        self.img = pg.ImageView()
        self.img.ui.roiBtn.hide()
        self.img.ui.menuBtn.hide()
        layout.addWidget(self.img)

        # -------------------------
        # LSL Setup
        # -------------------------
        print("Resolving LSL stream...")
        streams = resolve_byprop("name", "EEG_DATA")
        self.inlet = StreamInlet(streams[0])

        self.buffer = []

        # -------------------------
        # Timer for updates
        # -------------------------
        self.timer = QTimer()
        self.timer.setInterval(int(STEP_SEC * 1000))  # ms
        self.timer.timeout.connect(self.update_data)
        self.timer.start()

    def update_data(self):
        # Pull all available samples
        while True:
            sample = self.inlet.pull_sample(timeout=0)[0]
            if sample is None:
                break

            timestamp_str, eeg_str = sample[0].split(",")
            eeg_value = float(eeg_str) * 10
            self.buffer.append(eeg_value)

        # Keep buffer bounded
        if len(self.buffer) > WINDOW_SAMPLES:
            self.buffer = self.buffer[-WINDOW_SAMPLES:]

        # Need enough data for FFT
        if len(self.buffer) < WINDOW_SAMPLES:
            return

        # Convert to numpy
        data = np.array(self.buffer[-WINDOW_SAMPLES:])

        # Compute DSA
        f, t, Sxx = spectrogram(
            data,
            fs=SAMPLE_RATE,
            nperseg=WINDOW_SAMPLES // 2,
            noverlap=WINDOW_SAMPLES // 2 - STEP_SAMPLES,
            scaling="density",
            mode="psd"
        )

        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-12)

        # Display image
        self.img.setImage(
            Sxx_db,
            autoLevels=True,
            xvals=t
        )
        self.img.setPredefinedGradient("viridis")
        self.img.setColorMap(pg.colormap.get("viridis"))



def main():
    app = QApplication(sys.argv)
    win = EEGWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
