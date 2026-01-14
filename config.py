from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QFormLayout, QLineEdit, QPushButton, QGroupBox, QMessageBox, QGridLayout, QLabel
)


class SystemConfig:
    SAMPLE_RATE_HZ = 400
    UPDATE_STEP_SEC = 0.5
    INTERVAL = 1.1
    NO_DATA_VALUE = -10000.0
    LOWEST_FREQ_HZ = 0.1 # Viktor Bublitz et al. Electroencephalogram-based prediction and detection of responsiveness
                         # to noxious stimulation in critical care patients: a retrospective single-centre analysis

    # to observe a frequency reliably: window >= 1/(frequency resolution)
    DISPLAY_MINUTES_BOUNDS = (0.5, 360.0)
    MAX_FREQ_HZ_BOUNDS = (20, 50)


class ConfigWidget(QGroupBox):
    def __init__(self, on_apply_callback):
        super().__init__("System Configuration")

        self._default_config()
        self.on_apply_callback = on_apply_callback

        layout = QFormLayout(self)

        self.window_sec = QLineEdit(str(self.WINDOW_SEC))
        self.overlap = QLineEdit(str(self.OVERLAP))

        self.segment_sec = QLineEdit(str(self.SEGMENT_SEC))
        self.segment_overlap = QLineEdit(str(self.SEGMENT_OVERLAP))

        self.display_min = QLineEdit(str(self.DISPLAY_MINUTES))
        self.max_freq = QLineEdit(str(self.MAX_FREQ_HZ))

        self.min_db = QLineEdit(str(self.PSD_DB_MIN))
        self.max_db = QLineEdit(str(self.PSD_DB_MAX))


        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply)

        grid = QGridLayout()

        grid.addWidget(QLabel("Window Length (s)"), 0, 0)
        grid.addWidget(self.window_sec, 0, 1)

        grid.addWidget(QLabel("Segment Length (s)"), 0, 2)
        grid.addWidget(self.segment_sec, 0, 3)

        grid.addWidget(QLabel("Overlap (0-1)"), 1, 0)
        grid.addWidget(self.overlap, 1, 1)

        grid.addWidget(QLabel("Segment Overlap (0-1)"), 1, 2)
        grid.addWidget(self.segment_overlap, 1, 3)

        grid.addWidget(QLabel("Display Time (min)"), 2, 0)
        grid.addWidget(self.display_min, 2, 1)

        grid.addWidget(QLabel("Max Frequency (Hz)"), 3, 0)
        grid.addWidget(self.max_freq, 3, 1)

        grid.addWidget(QLabel("Min Power (dB)"), 2, 2)
        grid.addWidget(self.min_db, 2, 3)

        grid.addWidget(QLabel("Max Power (dB)"), 3, 2)
        grid.addWidget(self.max_db, 3, 3)

        layout.addRow(grid)
        layout.addRow(apply_btn)

    def _apply(self):
        try:
            if float(self.segment_sec.text()) > float(self.window_sec.text()):
                raise ValueError("Segment size must be smaller than window size")

            self.WINDOW_SEC = float(self.window_sec.text())
            self.OVERLAP = float(self.overlap.text())

            self.SEGMENT_SEC = float(self.segment_sec.text())
            self.SEGMENT_OVERLAP = float(self.segment_overlap.text())

            self.DISPLAY_MINUTES = float(self.display_min.text())
            self.MAX_FREQ_HZ = int(self.max_freq.text())

            self.PSD_DB_MIN = int(self.min_db.text())
            self.PSD_DB_MAX = int(self.max_db.text())

            self.on_apply_callback()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Invalid Configuration",
                str(e)
            )

    def _default_config(self):
        self.WINDOW_SEC = 4.0
        self.OVERLAP = 0.5
        self.SEGMENT_SEC = 2.0
        self.SEGMENT_OVERLAP = 0.5
        self.DISPLAY_MINUTES = 2.0
        self.MAX_FREQ_HZ = 40
        # Percentage of overlap
        self.PSD_DB_MIN = -25
        self.PSD_DB_MAX = 10