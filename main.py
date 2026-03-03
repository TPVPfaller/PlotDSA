"""
IEC 62304 – Class B

EEG Density Spectral Array Viewer
"""

import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PySide6.QtCore import QThread, QTimer
from PySide6.QtGui import QAction

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

        self.setWindowTitle("EEG Density Spectral Array")

        # Menu bar: Settings
        settings_menu = self.menuBar().addMenu("&Menu")
        action_settings = QAction("System Settings...", self)
        action_settings.setShortcut("Ctrl+,")
        action_settings.triggered.connect(self._open_settings)
        settings_menu.addAction(action_settings)

        view_menu = self.menuBar().addMenu("&View")

        self.action_show_dsa = QAction("Show DSA", self)
        self.action_show_dsa.setCheckable(True)
        self.action_show_dsa.setChecked(True)
        self.action_show_dsa.toggled.connect(self._toggle_dsa)
        view_menu.addAction(self.action_show_dsa)

        self.action_show_eeg = QAction("Show EEG", self)
        self.action_show_eeg.setCheckable(True)
        self.action_show_eeg.setChecked(True)
        self.action_show_eeg.toggled.connect(self._toggle_eeg)
        view_menu.addAction(self.action_show_eeg)

        self.action_show_psd = QAction("Show PSD", self)
        self.action_show_psd.setCheckable(True)
        self.action_show_psd.setChecked(False)
        self.action_show_psd.toggled.connect(self._toggle_psd)
        view_menu.addAction(self.action_show_psd)


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
        
        self.psd_view = PSDView(self.user_config.psd_db_min, self.user_config.psd_db_max)
        self.eeg_view = EEGView(self.user_config.window_sec)

        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._check_status)
        self.status_timer.start(500)  # Check every 500ms

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.topbar)
        layout.addWidget(self.dsa_view)
        layout.addWidget(self.psd_view)
        layout.addWidget(self.eeg_view)

        # Initialize visibility based on View menu actions
        self.dsa_view.setVisible(self.action_show_dsa.isChecked())
        self.eeg_view.setVisible(self.action_show_eeg.isChecked())
        self.psd_view.setVisible(self.action_show_psd.isChecked())

        self.setCentralWidget(container)

        layout.setStretchFactor(self.topbar, 0)
        layout.setStretchFactor(self.dsa_view, 1)
        layout.setStretchFactor(self.psd_view, 1)
        layout.setStretchFactor(self.eeg_view, 1)

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
        # Update PSD view only if visible to save some work
        try:
            if self.psd_view.isVisible():
                self.psd_view.update(freqs, psd)
        except Exception:
            pass
        # DSA cadence updates; raw EEG is updated per-sample via _on_new_sample
        self.topbar.sync_sliders(self.user_config, is_new_data=True)
        self.topbar.update_indicator(self.dsa_view)

    def _check_status(self):
        self.topbar.sync_sliders(self.user_config, is_new_data=False)
        self.topbar.update_indicator(self.dsa_view)

    def _toggle_dsa(self, checked: bool):
        try:
            self.dsa_view.setVisible(bool(checked))
            self.centralWidget().layout().invalidate()
            self.centralWidget().updateGeometry()
        except Exception:
            pass
    def _toggle_eeg(self, checked: bool):
        try:
            self.eeg_view.setVisible(bool(checked))
            self.centralWidget().layout().invalidate()
            self.centralWidget().updateGeometry()
        except Exception:
            pass

    def _toggle_psd(self, checked: bool):
        try:
            self.psd_view.setVisible(bool(checked))
            self.centralWidget().layout().invalidate()
            self.centralWidget().updateGeometry()
        except Exception:
            pass

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
