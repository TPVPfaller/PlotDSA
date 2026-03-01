"""
IEC 62304 – Class B

EEG Density Spectral Array Viewer
"""

import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PySide6.QtCore import QThread, QTimer

from config import UserConfig
from settings_ui import TopBar, SettingsDialog
from worker import ProcessingWorker
from views import DSAView, PSDView, EEGView

class DSAApplication(QMainWindow):
    """Main Qt application wiring together config UI, data stream, processing, and views.

    Periodically pulls samples from `EEGStream`, updates the raw EEG view immediately,
    computes PSD/DSA via `EEGBuffer` and `DSACalculator`, and refreshes DSA/PSD/EEG
    displays on the configured cadence.
    """

    def __init__(self):
        super().__init__()

        self.user_config = UserConfig()
        self.topbar = TopBar(self.user_config, self._on_config_change)
        self.topbar.settings_btn.clicked.connect(self._open_settings)

        self.setWindowTitle("EEG Density Spectral Array")

        self.thread = QThread()
        self.worker = ProcessingWorker(self.user_config)

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.new_data.connect(self._on_new_data)
        # Per-sample raw EEG updates
        self.worker.new_sample.connect(self._on_new_sample)
        # Stream connection status → TopBar indicator
        if hasattr(self.worker, 'connection_changed'):
            self.worker.connection_changed.connect(self.topbar.set_stream_connected)

        self.thread.start()

        self.dsa_view = DSAView(self.user_config)
        self.dsa_view.on_config_change_callback = self._on_config_change
        
        # Connect jump button in topbar
        self.topbar.live_btn.clicked.connect(self.dsa_view.jump_to_live)
        
        #self.psd_view = PSDView(self.user_config.psd_db_min, self.user_config.psd_db_max)
        self.eeg_view = EEGView(self.user_config.window_sec)

        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._check_status)
        self.status_timer.start(500)  # Check every 500ms

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.topbar)
        layout.addWidget(self.dsa_view)
        #layout.addWidget(self.psd_view)
        layout.addWidget(self.eeg_view)
        self.setCentralWidget(container)

    def closeEvent(self, event):
        """Ensure background threads are stopped before closing."""
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        event.accept()

    def _open_settings(self):
        dialog = SettingsDialog(self.user_config, self._on_config_change, self)
        dialog.exec()

    def _on_new_data(self, dsa_buffer, freqs, psd):
        self.dsa_view.update(dsa_buffer)
        # self.psd_view.update(freqs, psd)
        # DSA cadence updates; raw EEG is updated per-sample via _on_new_sample
        self.topbar.sync_sliders(self.user_config, is_new_data=True)
        self.topbar.update_indicator(self.dsa_view)

    def _check_status(self):
        self.topbar.sync_sliders(self.user_config, is_new_data=False)
        self.topbar.update_indicator(self.dsa_view)

    def _on_new_sample(self, t_epoch, value):
        """Receive each validated EEG sample and update the EEGView immediately."""
        self.eeg_view.append_sample(t_epoch, value)

    def _on_config_change(self, new_config):
        self.user_config = new_config

        self.worker.apply_config(new_config)

        self.dsa_view.apply_config(new_config)
        #self.psd_view.apply_config(new_config.psd_db_min, new_config.psd_db_max)

        self.topbar.sync_sliders(self.user_config)
        self.topbar.update_indicator(self.dsa_view)

def main():
    app = QApplication(sys.argv)
    win = DSAApplication()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
