"""
IEC 62304 – Class B

EEG Density Spectral Array Viewer
"""

import sys
import time

sys.argv += ['-platform', 'windows:darkmode=2']
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QMessageBox,
    QLabel,
)
from PySide6.QtCore import QThread, QTimer, Qt
from PySide6.QtGui import QAction
import qdarktheme
import pyqtgraph as pg

if __package__ in (None, ""):
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from plotdsa.io.output import Output
    from plotdsa.config import UserConfig
    from plotdsa.ui.dsa_settings import SettingsDialog
    from plotdsa.ui.topbar import TopBar
    from plotdsa.ui.time_selection import TimeSelectionDialog
    from plotdsa.app.worker import ProcessingWorker
    from plotdsa.ui.views import DSAView, PSDView, EEGView
    from plotdsa import config
else:
    from ..io.output import Output
    from ..config import UserConfig
    from ..ui.dsa_settings import SettingsDialog
    from ..ui.topbar import TopBar
    from ..ui.time_selection import TimeSelectionDialog
    from .worker import ProcessingWorker
    from ..ui.views import DSAView, PSDView, EEGView
    from .. import config


class DSAApplication(QMainWindow):
    """Main application wiring together UI, processing thread and views."""

    def __init__(self):
        super().__init__()
        self._closing = False

        self.setWindowTitle("EEG Density Spectral Array")
        self.resize(1000, 650)
        self.setMinimumSize(config.MIN_WINDOW_WIDTH, config.MIN_WINDOW_HEIGHT)

        self.user_config = UserConfig()
        self._init_ui()
        # self._load_previous_data()  # Removed: Don't load the data at the start only when set in the menu.
        self._init_worker()
        self._init_timers()

    def _init_ui(self):
        self.dsa_view = DSAView(self.user_config, self._on_config_change, self._on_zoom_change)
        self.psd_view = PSDView(self.user_config)
        self.eeg_view = EEGView(self.user_config, self._on_config_change)

        self.topbar = TopBar(self.user_config, self._on_config_change, self._on_zoom_change, self.dsa_view.pan, self.dsa_view.calibrate)
        self._create_menu()

        container = QWidget()
        self.layout = QVBoxLayout(container)
        self.layout.setSpacing(0)

        self.disclaimer_label = QLabel("Nur zu Lehrzwecken")
        self.disclaimer_label.setStyleSheet("color: red; font-weight: bold; font-size: 22px; margin-bottom: -5px; margin-top: -10px")
        self.disclaimer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.layout.addWidget(self.disclaimer_label)
        self.layout.addWidget(self.topbar)
        self.layout.addWidget(self.dsa_view)
        self.layout.addWidget(self.psd_view)
        self.layout.addWidget(self.eeg_view)

        self.layout.setStretchFactor(self.disclaimer_label, 0)
        self.layout.setStretchFactor(self.topbar, 0)
        self.layout.setStretchFactor(self.dsa_view, 1)
        self.layout.setStretchFactor(self.psd_view, 1)
        self.layout.setStretchFactor(self.eeg_view, 1)

        self.setCentralWidget(container)

        self._sync_view_visibility()

        self.topbar.live_btn.clicked.connect(self.dsa_view.jump_to_live)
        self.topbar.sync_slider(self.user_config.display_minutes)
        pg.setConfigOptions(antialias=False)

    def _on_load_data_clicked(self):
        dialog = TimeSelectionDialog(self)
        if dialog.exec():
            start_time = dialog.selected_datetime()
            self._load_data_from_time(start_time)

    def _load_data_from_time(self, start_time_dt):
        self.dsa_view.clear_data()
        self.eeg_view.clear_data()

        try:
            previous_data = Output.load_psd_from_time(start_time_dt)
        except Exception as e:
            self._show_message("Load Error", f"Failed to load data: {e}")
            return

        if not previous_data:
            self._show_message("Load Data", "No data found for the selected time range.")
            return

        for ts, duration, psd in previous_data:
            self._append_dsa_steps(ts, psd, max(1, int(round(duration / config.TIME_RESOLUTION))))

        self.dsa_view.update()
        self.dsa_view.jump_to_live()

    def _append_dsa_steps(self, ts, psd, steps):
        for i in range(steps):
            self.dsa_view.append(ts + i * config.TIME_RESOLUTION, psd)

    def _show_message(self, title, text, buttons=QMessageBox.StandardButton.Ok, default_button=None):
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(buttons)
        if default_button is not None:
            msg.setDefaultButton(default_button)
        msg.setOption(QMessageBox.Option.DontUseNativeDialog, True)
        return msg.exec()


    def _init_worker(self):
        self.thread = QThread()
        self.worker = ProcessingWorker(self.user_config)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.new_dsa_column.connect(self._on_new_dsa_column)
        self.worker.new_samples.connect(self._on_new_samples)

        self.thread.start()

    def _init_timers(self):
        self._last_data_receive_time = time.time() - 2.0

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(250)

    def _create_menu(self):
        menu = self.menuBar().addMenu("&Menu")

        menu.addSeparator()

        action_settings = QAction("DSA Settings", self)
        action_settings.triggered.connect(self._open_settings)
        menu.addAction(action_settings)

        menu.addSeparator()

        self.action_multitaper = QAction("Use Multitaper", self)
        self.action_multitaper.setCheckable(True)
        self.action_multitaper.setChecked(self.user_config.use_multitaper)
        self.action_multitaper.toggled.connect(self._toggle_multitaper)
        menu.addAction(self.action_multitaper)

        view_menu = self.menuBar().addMenu("&View")

        self.action_show_dsa = self._create_toggle_action(view_menu, "Show DSA", True, self.dsa_view)
        self.action_show_psd = self._create_toggle_action(view_menu, "Show PSD", False, self.psd_view)
        self.action_show_eeg = self._create_toggle_action(view_menu, "Show EEG", True, self.eeg_view)

        action_load_data = QAction("Load Data from Time", self)
        action_load_data.triggered.connect(self._on_load_data_clicked)
        view_menu.addAction(action_load_data)

        action_clear_data = QAction("Clear Data", self)
        action_clear_data.triggered.connect(self._confirm_clear_data)
        view_menu.addAction(action_clear_data)

        help_menu = self.menuBar().addMenu("&Help")
        action_info = QAction("Information", self)
        action_info.setShortcut("Ctrl+H")
        action_info.triggered.connect(self._show_information)
        help_menu.addAction(action_info)

        self.connection_indicator = QLabel("●")
        self.connection_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.connection_indicator.setFixedSize(30, 30)

        self.connection_indicator.setStyleSheet("""
            QLabel {
                color: red;
                font-size: 22px;
            }
        """)

        self.connection_indicator.setToolTip("Disconnected")

        # place it in top-right corner of menu bar
        self.menuBar().setCornerWidget(self.connection_indicator, Qt.Corner.TopRightCorner)

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
            - More info: <a href="https://github.com/TPVPfaller/PlotDSA">GitHub</a>
        </p>
        """
        self._show_message("Information", text)

    def _confirm_clear_data(self):
        reply = self._show_message(
            "Confirm data deletion",
            "Are you sure you want to delete all EEG/DSA data?\nThe data can be reloaded with the 'Load Data from Time' menu item.",
            buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            default_button=QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.dsa_view.clear_data()
            self.eeg_view.clear_data()

    def _toggle_multitaper(self, checked):
        new_config = self.user_config.update(use_multitaper=checked)
        self._on_config_change(new_config)

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


    def _on_new_dsa_column(self, ts, psd, steps):
        self._append_dsa_steps(ts, psd, steps)
        self.dsa_view.update()

        if self.psd_view.isVisible():
            self.psd_view.update(psd)

    def _on_new_samples(self, samples):
        if not self.eeg_view.isVisible():
            return
        for value in samples:
            self.eeg_view.append_sample(value)
        if samples:
            self._last_data_receive_time = time.time()


    def _update_status(self):
        if time.time() - self._last_data_receive_time < 2.0:
            self.connection_indicator.setStyleSheet("QLabel { color: green; font-size: 22px; }")
            self.connection_indicator.setToolTip("Connected")
        else:
            self.connection_indicator.setStyleSheet("QLabel { color: red; font-size: 22px; }")
            self.connection_indicator.setToolTip("Disconnected")

        self.topbar.update_jump_live_btn(self.dsa_view)

    def _on_zoom_change(self, display_minutes):
        self.topbar.sync_slider(display_minutes)
        self.topbar.update_jump_live_btn(self.dsa_view)
        self.dsa_view.apply_zoom(display_minutes)

    def _on_config_change(self, new_config):
        self.user_config = new_config

        if hasattr(self, 'action_multitaper'):
            self.action_multitaper.blockSignals(True)
            self.action_multitaper.setChecked(new_config.use_multitaper)
            self.action_multitaper.blockSignals(False)

        self.topbar.apply_config(new_config)
        self.worker.apply_config(new_config)
        self.dsa_view.apply_config(new_config)
        self.eeg_view.apply_config(new_config)

        self._update_status()

    def _thread_is_running(self):
        return bool(getattr(self.thread, "isRunning", lambda: False)())

    def _finish_close_when_worker_stops(self):
        if not self._closing:
            return
        if self._thread_is_running():
            QTimer.singleShot(50, self._finish_close_when_worker_stops)
            return
        self.close()

    def closeEvent(self, event):
        if self._closing:
            if self._thread_is_running():
                event.ignore()
            else:
                event.accept()
            return

        self._closing = True
        if hasattr(self, "status_timer"):
            self.status_timer.stop()
        self.setEnabled(False)
        self.worker.stop()
        self.thread.quit()

        if self._thread_is_running():
            event.ignore()
            QTimer.singleShot(50, self._finish_close_when_worker_stops)
            return

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
