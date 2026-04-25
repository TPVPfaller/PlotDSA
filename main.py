"""
IEC 62304 – Class B

EEG Density Spectral Array Viewer
"""

from datetime import datetime as dt
import datetime
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
    QDialog,
    QPushButton,
    QHBoxLayout,
    QCalendarWidget,
    QDialogButtonBox,
    QFrame,
    QSlider,
)
from PySide6.QtCore import QDate, QThread, QTimer, Qt
from PySide6.QtGui import QAction
import qdarktheme
from input_output import Output
import pyqtgraph as pg

from config import UserConfig
from settings_ui import TopBar, SettingsDialog
from worker import ProcessingWorker
from views import DSAView, PSDView, EEGView

import config


class TimeSelectionDialog(QDialog):
    PRESET_OFFSETS = (
        ("1 hour ago", datetime.timedelta(hours=1)),
        ("6 hours ago", datetime.timedelta(hours=6)),
        ("24 hours ago", datetime.timedelta(hours=24)),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Start Time")
        self.setMinimumSize(760, 560)

        self._selected_dt = self._default_datetime()
        self._building_ui = False

        self._build_ui()
        self._apply_datetime(self._selected_dt)

    def _build_ui(self):
        self._building_ui = True

        self.setStyleSheet(f"""
            QDialog {{
                font-size: {config.FONT_SIZE}px;
            }}
            QFrame#pickerSection {{
                border: 1px solid palette(mid);
                border-radius: 12px;
                background: rgba(255, 255, 255, 0.03);
            }}
            QLabel#pickerTitle {{
                font-size: {config.FONT_SIZE + 6}px;
                font-weight: 600;
            }}
            QLabel#pickerSubtitle {{
                color: palette(midlight);
            }}
            QLabel#pickerPreview {{
                border-radius: 12px;
                padding: 12px 16px;
                background: palette(alternate-base);
                font-size: {config.FONT_SIZE + 1}px;
                font-weight: 600;
            }}
            QPushButton#presetButton {{
                min-height: 40px;
                padding: 8px 14px;
                border-radius: 10px;
            }}
            QSlider::groove:vertical {{
                width: 10px;
                border-radius: 5px;
                background: palette(mid);
            }}
            QSlider::sub-page:vertical {{
                border-radius: 5px;
                background: palette(highlight);
            }}
            QSlider::add-page:vertical {{
                border-radius: 5px;
                background: palette(mid);
            }}
            QSlider::handle:vertical {{
                width: 26px;
                height: 26px;
                margin: 0 -8px;
                border-radius: 13px;
                background: palette(window-text);
                border: 1px solid palette(base);
            }}
            QLabel#monthLabel {{
                font-size: {config.FONT_SIZE + 2}px;
                font-weight: 600;
            }}
            QLabel#timeDisplay {{
                border: 1px solid palette(mid);
                border-radius: 10px;
                padding: 10px 16px;
                background: palette(alternate-base);
                font-size: {config.FONT_SIZE + 8}px;
                font-weight: 700;
            }}
        """)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

        title = QLabel("Load DSA from a specific time")
        title.setObjectName("pickerTitle")

        root_layout.addWidget(title)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(8)
        for label, offset in self.PRESET_OFFSETS:
            button = QPushButton(label)
            button.setObjectName("presetButton")
            button.clicked.connect(lambda _, delta=offset: self._apply_preset(delta))
            preset_row.addWidget(button)
        root_layout.addLayout(preset_row)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(14)

        calendar_frame = QFrame()
        calendar_frame.setObjectName("pickerSection")
        calendar_layout = QVBoxLayout(calendar_frame)
        calendar_layout.setContentsMargins(12, 12, 12, 12)
        calendar_layout.setSpacing(8)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(10)

        self.prev_month_btn = QPushButton("Previous")
        self.prev_month_btn.setMinimumHeight(40)
        self.prev_month_btn.clicked.connect(self._show_previous_month)
        nav_row.addWidget(self.prev_month_btn)

        self.month_label = QLabel()
        self.month_label.setObjectName("monthLabel")
        self.month_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_row.addWidget(self.month_label, 1)

        self.next_month_btn = QPushButton("Next")
        self.next_month_btn.setMinimumHeight(40)
        self.next_month_btn.clicked.connect(self._show_next_month)
        nav_row.addWidget(self.next_month_btn)

        calendar_layout.addLayout(nav_row)

        self.calendar = QCalendarWidget()
        self.calendar.setGridVisible(True)
        self.calendar.setNavigationBarVisible(False)
        self.calendar.setVerticalHeaderFormat(QCalendarWidget.VerticalHeaderFormat.NoVerticalHeader)
        self.calendar.setFirstDayOfWeek(Qt.DayOfWeek.Monday)
        self.calendar.setMaximumDate(self._to_qdate(dt.now()))
        self.calendar.selectionChanged.connect(self._sync_preview)
        self.calendar.currentPageChanged.connect(self._update_calendar_header)
        calendar_layout.addWidget(self.calendar)

        body_layout.addWidget(calendar_frame, 3)

        time_frame = QFrame()
        time_frame.setObjectName("pickerSection")
        time_layout = QVBoxLayout(time_frame)
        time_layout.setContentsMargins(12, 12, 12, 12)
        time_layout.setSpacing(10)


        slider_row = QHBoxLayout()
        slider_row.setSpacing(16)

        self.hour_slider = QSlider(Qt.Orientation.Vertical)
        self.hour_slider.setRange(0, 23)
        self.hour_slider.setTickPosition(QSlider.TickPosition.TicksRight)
        self.hour_slider.setTickInterval(1)
        self.hour_slider.setSingleStep(1)
        self.hour_slider.setPageStep(1)
        self.hour_slider.setMinimumHeight(300)
        self.hour_slider.valueChanged.connect(self._sync_preview)
        slider_row.addWidget(self.hour_slider, 0, Qt.AlignmentFlag.AlignCenter)

        self.time_display = QLabel()
        self.time_display.setObjectName("timeDisplay")
        self.time_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slider_row.addWidget(self.time_display, 1, Qt.AlignmentFlag.AlignCenter)

        time_layout.addLayout(slider_row)
        time_layout.addStretch(1)

        body_layout.addWidget(time_frame, 2)
        root_layout.addLayout(body_layout)

        self.preview_label = QLabel()
        self.preview_label.setObjectName("pickerPreview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root_layout.addWidget(self.preview_label)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        self.load_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        self.load_button.setText("Load")
        self.load_button.setMinimumHeight(42)
        button_box.button(QDialogButtonBox.StandardButton.Cancel).setMinimumHeight(42)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        root_layout.addWidget(button_box)

        self._building_ui = False

    def _default_datetime(self):
        return (dt.now() - datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    def _apply_preset(self, offset):
        self._apply_datetime((dt.now() - offset).replace(minute=0, second=0, microsecond=0))

    def _apply_datetime(self, selected_dt):
        normalized = min(selected_dt, dt.now()).replace(minute=0, second=0, microsecond=0)
        self._selected_dt = normalized

        self.calendar.blockSignals(True)
        self.calendar.setSelectedDate(self._to_qdate(normalized))
        self.calendar.blockSignals(False)
        self.calendar.setCurrentPage(normalized.year, normalized.month)

        self.hour_slider.blockSignals(True)
        self.hour_slider.setValue(normalized.hour)
        self.hour_slider.blockSignals(False)

        self._update_calendar_header()
        self._update_time_display(normalized.hour)
        self._refresh_preview(normalized)

    def _sync_preview(self, *_):
        if self._building_ui:
            return

        selected_dt = self.selected_datetime()
        if selected_dt > dt.now():
            selected_dt = dt.now().replace(minute=0, second=0, microsecond=0)
            self._apply_datetime(selected_dt)
            return

        self._selected_dt = selected_dt
        self._update_time_display(selected_dt.hour)
        self._refresh_preview(selected_dt)

    def _refresh_preview(self, selected_dt):
        delta = max(datetime.timedelta(), dt.now() - selected_dt)
        total_minutes = int(delta.total_seconds() // 60)
        hours, minutes = divmod(total_minutes, 60)

        if total_minutes < 1:
            relative = "just now"
        elif hours:
            relative = f"{hours}h {minutes:02d}m ago"
        else:
            relative = f"{minutes}m ago"

        self.preview_label.setText(selected_dt.strftime(f"%a, %d %b %Y  %H:%M  |  {relative}"))

    def selected_datetime(self):
        selected_date = self.calendar.selectedDate()
        hour = self.hour_slider.value() if hasattr(self, "hour_slider") else self._selected_dt.hour

        return dt(
            selected_date.year(),
            selected_date.month(),
            selected_date.day(),
            hour,
            0,
        )

    @staticmethod
    def _to_qdate(selected_dt):
        return QDate(selected_dt.year, selected_dt.month, selected_dt.day)

    def _update_time_display(self, hour):
        self.time_display.setText(f"{hour:02d}:00")

    def _update_calendar_header(self, *_):
        shown_year = self.calendar.yearShown()
        shown_month = self.calendar.monthShown()
        self.month_label.setText(QDate(shown_year, shown_month, 1).toString("MMMM yyyy"))

        current_page = QDate(shown_year, shown_month, 1)
        this_month = QDate.currentDate()
        self.next_month_btn.setEnabled(current_page < QDate(this_month.year(), this_month.month(), 1))

    def _show_previous_month(self):
        self.calendar.showPreviousMonth()
        self._update_calendar_header()

    def _show_next_month(self):
        shown_year = self.calendar.yearShown()
        shown_month = self.calendar.monthShown()
        current_page = QDate(shown_year, shown_month, 1)
        this_month = QDate.currentDate()
        if current_page < QDate(this_month.year(), this_month.month(), 1):
            self.calendar.showNextMonth()
            self._update_calendar_header()


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
                self.dsa_view.append(ts + i * config.TIME_RESOLUTION, psd)

        self.dsa_view.update()
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

        menu.addSeparator()

        action_settings = QAction("Settings", self)
        action_settings.setShortcut("Ctrl+,")
        action_settings.triggered.connect(self._open_settings)
        menu.addAction(action_settings)

        action_load_data = QAction("Load Data from Time...", self)
        action_load_data.triggered.connect(self._on_load_data_clicked)
        menu.addAction(action_load_data)

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

    def _on_new_dsa_column(self, ts, psd, steps):
        for i in range(steps):
            self.dsa_view.append(ts + i * config.TIME_RESOLUTION, psd)
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
