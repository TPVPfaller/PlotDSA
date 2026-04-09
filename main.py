"""
IEC 62304 – Class B

EEG Density Spectral Array Viewer
"""

import sys
sys.argv += ['-platform', 'windows:darkmode=2']
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QMessageBox
import qdarktheme

from PySide6.QtCore import QThread, QTimer
from PySide6.QtGui import QAction
import pyqtgraph as pg

from config import UserConfig
from settings_ui import TopBar, SettingsDialog
from worker import ProcessingWorker
from views import DSAView, PSDView, EEGView

import config
from PySide6.QtCore import Qt


class DSAApplication(QMainWindow):
    """Main application wiring together UI, processing thread and views."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("EEG Density Spectral Array")

        import tkinter as tk

        root = tk.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()

        print("Width:", width)
        print("Height:", height)

        screen_width_mm = root.winfo_screenmmwidth()
        screen_height_mm = root.winfo_screenmmheight()
        print(f"  Größe:     {screen_width_mm} x {screen_height_mm} mm")

        root.destroy()
        self.resize(1000, 650)

        self.user_config = UserConfig()
        self._init_ui()
        self._init_worker()
        self._init_timers()

    def _init_ui(self):
        self.topbar = TopBar(self.user_config, self._on_config_change, self._on_zoom_change)

        self.dsa_view = DSAView(self.user_config, self._on_config_change, self._on_zoom_change)
        self.psd_view = PSDView(self.user_config)
        self.eeg_view = EEGView()

        self._create_menu()

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

        self._sync_view_visibility()

        self.topbar.live_btn.clicked.connect(self.dsa_view.jump_to_live)
        self.topbar.sync_slider(self.user_config.display_minutes)
        pg.setConfigOptions(antialias=False)

    def _init_worker(self):
        self.thread = QThread()
        self.worker = ProcessingWorker(self.user_config)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.new_dsa_column.connect(self._on_new_dsa_column)
        self.worker.new_samples.connect(self._on_new_samples)

        self.thread.start()

    def _init_timers(self):
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(250)

    def _create_menu(self):
        menu = self.menuBar().addMenu("&Menu")

        action_settings = QAction("System Settings...", self)
        action_settings.setShortcut("Ctrl+,")
        action_settings.triggered.connect(self._open_settings)
        menu.addAction(action_settings)

        action_info = QAction("Information", self)
        action_info.setShortcut("Ctrl+H")
        action_info.triggered.connect(self._show_information)
        menu.addAction(action_info)

        view_menu = self.menuBar().addMenu("&View")

        self.action_show_dsa = self._create_toggle_action(view_menu, "Show DSA", True, self.dsa_view)
        self.action_show_psd = self._create_toggle_action(view_menu, "Show PSD", False, self.psd_view)
        self.action_show_eeg = self._create_toggle_action(view_menu, "Show EEG", True, self.eeg_view)

    def _show_information(self):
        text = f"""
        <p style="font-size:12pt;">
            - PSD files are saved in: <code>{config.BASE_DIR}</code>
        </p>
        <p style="font-size:12pt;">
            - Viewer supports up to <b>{int(config.DISPLAY_MINUTES_BOUNDS[1]/60)} hours</b> of EEG data
        </p>
        <p style="font-size:12pt;">
            - EEG should arrive at <b>{config.SAMPLE_RATE_HZ} Hz</b>
        </p>
        <p style="font-size:12pt;">
            - Adjusting segment (s) changes the frequency resolution
        </p>
        <ul style="font-size:12pt;">
            <li>DSA history will be cleared</li>
            <li>New PSD data is written to a new CSV file</li>
        </ul>
        <p style="font-size:12pt;">
            - More info: <a href="https://github.com/TPVPfaller/PlotDSA">GitHub</a>
        </p>
        """

        msg = QMessageBox(self)
        msg.setWindowTitle("Information")
        msg.setTextFormat(Qt.RichText)
        msg.setText(text)
        msg.setIcon(QMessageBox.Information)
        msg.exec()

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

    def _open_settings(self):
        dialog = SettingsDialog(self.user_config, self._on_config_change, self)
        dialog.exec()

    def _on_new_dsa_column(self, ts, psd):
        self.dsa_view.update((ts, psd))

        if self.psd_view.isVisible():
            self.psd_view.update(psd)

        self._update_status()

    def _on_new_samples(self, samples):
        for value in samples:
            self.eeg_view.append_sample(value)
        if samples:
            self.topbar.reset_last_data_timer()

    def _update_status(self):
        self.topbar.update_indicator()
        self.topbar.update_jump_live_btn(self.dsa_view)

    def _on_zoom_change(self, display_minutes):
        self.topbar.sync_slider(display_minutes)
        self.topbar.update_jump_live_btn(self.dsa_view)
        self.dsa_view.apply_config(self.user_config, display_minutes)

    def _on_config_change(self, new_config):
        self.user_config = new_config

        self.topbar.apply_config(new_config)
        self.worker.apply_config(new_config)
        self.dsa_view.apply_config(new_config)
        self.psd_view.apply_config(new_config)

        self._update_status()

    def closeEvent(self, event):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        event.accept()


def main():
    qdarktheme.enable_hi_dpi()

    app = QApplication(sys.argv)
    qdarktheme.setup_theme()

    win = DSAApplication()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
