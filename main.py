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
    """Main application wiring together UI, processing thread and views."""


    def __init__(self):
        super().__init__()

        self.setWindowTitle("EEG Density Spectral Array")

        self.user_config = UserConfig()
        self._init_ui()
        self._init_worker()
        self._init_timers()

    def _init_ui(self):
        self.topbar = TopBar(self.user_config, self._on_config_change)

        self.dsa_view = DSAView(self.user_config)
        self.dsa_view.on_config_change_callback = self._on_config_change

        self.psd_view = PSDView(
            self.user_config.psd_db_min,
            self.user_config.psd_db_max,
        )

        self.eeg_view = EEGView(self.user_config.window_sec)

        self._create_menu()

        # Layout
        container = QWidget()
        self.layout = QVBoxLayout(container)

        self.layout.addWidget(self.topbar)
        self.layout.addWidget(self.dsa_view)
        self.layout.addWidget(self.psd_view)
        self.layout.addWidget(self.eeg_view)

        self.layout.setStretchFactor(self.topbar, 0)
        self.layout.setStretchFactor(self.dsa_view, 1)
        self.layout.setStretchFactor(self.psd_view, 1)
        self.layout.setStretchFactor(self.eeg_view, 1)

        self.setCentralWidget(container)

        # Initial visibility
        self._sync_view_visibility()

        self.topbar.live_btn.clicked.connect(self.dsa_view.jump_to_live)

    def _create_menu(self):
        menu = self.menuBar().addMenu("&Menu")
        action_settings = QAction("System Settings...", self)
        action_settings.setShortcut("Ctrl+,")
        action_settings.triggered.connect(self._open_settings)
        menu.addAction(action_settings)

        view_menu = self.menuBar().addMenu("&View")

        self.action_show_dsa = self._create_toggle_action(
            view_menu, "Show DSA", True, self.dsa_view
        )
        self.action_show_psd = self._create_toggle_action(
            view_menu, "Show PSD", False, self.psd_view
        )
        self.action_show_eeg = self._create_toggle_action(
            view_menu, "Show EEG", True, self.eeg_view
        )

    def _create_toggle_action(self, menu, text, default, widget):
        action = QAction(text, self)
        action.setCheckable(True)
        action.setChecked(default)
        action.toggled.connect(widget.setVisible)
        menu.addAction(action)
        return action

    def _sync_view_visibility(self):
        self.dsa_view.setVisible(self.action_show_dsa.isChecked())
        self.psd_view.setVisible(self.action_show_psd.isChecked())
        self.eeg_view.setVisible(self.action_show_eeg.isChecked())

    def _init_worker(self):
        self.thread = QThread()
        self.worker = ProcessingWorker(self.user_config)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.new_data.connect(self._on_new_dsa)
        self.worker.new_sample.connect(self._on_new_sample)

        self.thread.start()

    def _init_timers(self):
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(500)

    def _open_settings(self):
        dialog = SettingsDialog(
            self.user_config,
            self._on_config_change,
            self,
        )
        dialog.exec()

    def _on_new_dsa(self, dsa_buffer, freqs, psd):
        self.dsa_view.update(dsa_buffer)

        if self.psd_view.isVisible():
            self.psd_view.update(freqs, psd)

        self._update_status()

    def _on_new_sample(self, t_epoch, value):
        self.eeg_view.append_sample(t_epoch, value)
        self.topbar.reset_last_data_timer()

    def _update_status(self):
        self.topbar.sync_slider(self.user_config)
        self.topbar.update_indicator()
        self.topbar.update_jump_live_btn(self.dsa_view)

    def _on_config_change(self, new_config):
        self.user_config = new_config

        self.worker.apply_config(new_config)
        self.dsa_view.apply_config(new_config)

        self._update_status()

    def closeEvent(self, event):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = DSAApplication()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()