"""
IEC 62304 – Class B

EEG Density Spectral Array Viewer
"""

from datetime import datetime as dt
import datetime
import sys
import time

sys.argv += ['-platform', 'windows:darkmode=2']
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QMessageBox, QLabel, QDialog, QDateTimeEdit, QPushButton, QFormLayout
import qdarktheme
from data import Output

from PySide6.QtCore import QThread, QTimer
from PySide6.QtGui import QAction
import pyqtgraph as pg

from config import UserConfig
from settings_ui import TopBar, SettingsDialog
from worker import ProcessingWorker
from views import DSAView, PSDView, EEGView

import config
from PySide6.QtCore import Qt


class TimeSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Start Time")
        self.layout = QFormLayout(self)

        self.date_time_edit = QDateTimeEdit(self)
        self.date_time_edit.setDateTime(dt.now() - datetime.timedelta(hours=1))
        self.date_time_edit.setCalendarPopup(True)

        self.layout.addRow("Start Time:", self.date_time_edit)

        self.ok_button = QPushButton("Load", self)
        self.ok_button.clicked.connect(self.accept)
        self.layout.addRow(self.ok_button)

    def selected_datetime(self):
        return self.date_time_edit.dateTime().toPython()


class DSAApplication(QMainWindow):
    """Main application wiring together UI, processing thread and views."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("EEG Density Spectral Array")
        self.resize(1000, 650)

        self.user_config = UserConfig()
        self._init_ui()
        # self._load_previous_data()  # Removed: Don't load the data at the start only when set in the menu.
        self._init_worker()
        self._init_timers()

    def _init_ui(self):
        self.dsa_view = DSAView(self.user_config, self._on_config_change, self._on_zoom_change)
        self.psd_view = PSDView(self.user_config)
        self.eeg_view = EEGView()

        self.topbar = TopBar(self.user_config, self._on_config_change, self._on_zoom_change, self.dsa_view.pan, self.dsa_view.calibrate)
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

    def _on_load_data_clicked(self):
        dialog = TimeSelectionDialog(self)
        if dialog.exec():
            start_time = dialog.selected_datetime()
            self._load_data_from_time(start_time)

    def _load_data_from_time(self, start_time_dt):
        # Clear current data first
        self.dsa_view.clear_data()

        # Load data from specific time
        try:
            previous_data = Output.load_psd_from_time(start_time_dt)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load data: {e}")
            return

        if not previous_data:
            QMessageBox.information(self, "Load Data", "No data found for the selected time range.")
            return

        for ts, duration, psd in previous_data:
            steps = int(duration / config.TIME_RESOLUTION)
            for i in range(steps):
                self.dsa_view.update((ts + i * config.TIME_RESOLUTION, psd))

        self.dsa_view.update(None, force_update=True)
        self.dsa_view.jump_to_live()


    def _init_worker(self):
        self.thread = QThread()
        self.worker = ProcessingWorker(self.user_config)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.new_dsa_column.connect(self._on_new_dsa_column)
        self.worker.new_samples.connect(self._on_new_samples)

        self.thread.start()

    def _init_timers(self):
        self._last_data_receive_time = time.time()

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(250)

    def _create_menu(self):
        menu = self.menuBar().addMenu("&Menu")

        action_load_data = QAction("Load Data from Time...", self)
        action_load_data.triggered.connect(self._on_load_data_clicked)
        menu.addAction(action_load_data)

        menu.addSeparator()

        action_settings = QAction("DSA Settings...", self)
        action_settings.setShortcut("Ctrl+,")
        action_settings.triggered.connect(self._open_settings)
        menu.addAction(action_settings)

        action_info = QAction("Information", self)
        action_info.setShortcut("Ctrl+H")
        action_info.triggered.connect(self._show_information)
        menu.addAction(action_info)

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



        action_clear_data = QAction("Clear Data", self)
        action_clear_data.triggered.connect(self._confirm_clear_data)
        view_menu.addAction(action_clear_data)

        self.connection_indicator = QLabel("●")
        self.connection_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.connection_indicator.setFixedSize(30, 30)

        self.connection_indicator.setStyleSheet("""
            QLabel {
                color: #6b0000;   /* red = disconnected */
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

        QMessageBox.information(
            self,
            "Information",
            text,
            QMessageBox.StandardButton.Ok
        )

    def _confirm_clear_data(self):
        reply = QMessageBox.question(
            self,
            "Confirm data deletion",
            "Are you sure you want to delete all EEG/DSA data?\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.dsa_view.clear_data()

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

    def _on_new_dsa_column(self, ts, psd):
        self.dsa_view.update((ts, psd))

        if self.psd_view.isVisible():
            self.psd_view.update(psd)


    def _on_new_samples(self, samples):
        for value in samples:
            self.eeg_view.append_sample(value)
        if samples:
            self._last_data_receive_time = time.time()


    def _update_status(self):
        if time.time() - self._last_data_receive_time < 2.0:
            self.connection_indicator.setStyleSheet("""
                QLabel {
                    color: #00c000;
                    font-size: 22px;
                }
            """)
            self.connection_indicator.setToolTip("Connected")
        else:
            self.connection_indicator.setStyleSheet("""
                QLabel {
                    color: #6b0000;
                    font-size: 22px;
                }
            """)
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
