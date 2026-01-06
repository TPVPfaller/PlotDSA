from PySide6.QtWidgets import (
    QFormLayout, QLineEdit, QPushButton, QGroupBox, QMessageBox
)


class SystemConfig:
    SAMPLE_RATE_HZ = 400
    WINDOW_SEC = 4.0
    UPDATE_STEP_SEC = 0.25
    DISPLAY_MINUTES = 2
    MAX_FREQ_HZ = 40
    OVERLAP = 0.75

    PSD_DB_MIN = -40
    PSD_DB_MAX = 10
    NO_DATA_VALUE = -10000.0


class ConfigWidget(QGroupBox):
    def __init__(self, config: SystemConfig, on_apply_callback):
        super().__init__("System Configuration")

        self.config = config
        self.on_apply_callback = on_apply_callback

        layout = QFormLayout(self)

        self.overlap = QLineEdit(str(config.OVERLAP))

        self.window_sec = QLineEdit(str(config.WINDOW_SEC))
        #self.window_sec.setValidator(QDoubleValidator(0.5, 10.0, 2))

        self.UPDATE_STEP_SEC = QLineEdit(str(config.UPDATE_STEP_SEC))
        #self.UPDATE_STEP_SEC.setValidator(QDoubleValidator(0.05, 5.0, 2))

        self.display_min = QLineEdit(str(config.DISPLAY_MINUTES))
        #self.display_min.setValidator(QIntValidator(10, 600))

        self.min_db = QLineEdit(str(config.PSD_DB_MIN))
        self.max_db = QLineEdit(str(config.PSD_DB_MAX))

        self.max_freq = QLineEdit(str(config.MAX_FREQ_HZ))
        #self.max_freq.setValidator(QIntValidator(1, 200))

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply)

        layout.addRow("Overlap", self.overlap)
        layout.addRow("Window Length (s)", self.window_sec)
        layout.addRow("Step Size (s)", self.UPDATE_STEP_SEC)
        layout.addRow("Display Time (min)", self.display_min)
        layout.addRow("Min Power (dB)", self.min_db)
        layout.addRow("Max Power (dB)", self.max_db)
        layout.addRow("Max Frequency (Hz)", self.max_freq)
        layout.addRow(apply_btn)

    def _apply(self):
        try:
            self.config.OVERLAP = float(self.overlap.text())
            self.config.WINDOW_SEC = float(self.window_sec.text())
            self.config.UPDATE_STEP_SEC = float(self.UPDATE_STEP_SEC.text())
            self.config.DISPLAY_MINUTES = int(self.display_min.text())
            self.config.PSD_DB_MIN = int(self.min_db.text())
            self.config.PSD_DB_MAX = int(self.max_db.text())
            self.config.MAX_FREQ_HZ = int(self.max_freq.text())

            if self.config.UPDATE_STEP_SEC >= self.config.WINDOW_SEC:
                raise ValueError("Step size must be smaller than window size")

            self.on_apply_callback()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Invalid Configuration",
                str(e)
            )