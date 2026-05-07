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
    QDialogButtonBox,
    QFrame,
    QSlider,
    QStyle,
)
from PySide6.QtCore import QThread, QTimer, Qt
from PySide6.QtGui import QAction
import qdarktheme
from input_output import Output
import pyqtgraph as pg

from config import UserConfig
from settings_ui import TopBar, SettingsDialog, EEGSettingsDialog
from worker import ProcessingWorker
from views import DSAView, PSDView, EEGView

import config


class TimeSelectionDialog(QDialog):
    PRESET_OFFSETS = (
        ("2 hours ago", datetime.timedelta(hours=2)),
        ("8 hours ago", datetime.timedelta(hours=8)),
        ("24 hours ago", datetime.timedelta(hours=24)),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Start Time")
        self.setMinimumSize(600, 450)

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
                background: palette(alternate-base);
            }}
            QLabel#pickerPreview {{
                border-radius: 12px;
                padding: 12px 16px;
                background: palette(alternate-base);
                font-size: {config.FONT_SIZE + 4}px;
                font-weight: 600;
            }}
            QPushButton#presetButton {{
                min-height: 60px;
                padding: 8px 16px;
                border-radius: 10px;
                color: palette(button-text);
                font-size: {config.FONT_SIZE + 2}px;
            }}
            QPushButton#presetButton:hover {{
                background: palette(highlight);
                color: palette(highlighted-text);
            }}
            QSlider {{
                background: transparent;
            }}
            QSlider::groove:horizontal {{
                height: 20px;
                border-radius: 10px;
                background: palette(mid);
            }}
            QSlider::sub-page:horizontal {{
                border-radius: 10px;
                background: palette(highlight);
            }}
            QSlider::handle:horizontal {{
                width: 44px;
                height: 44px;
                margin: -12px 0;
                border-radius: 22px;
                background: palette(window-text);
                border: 2px solid palette(base);
            }}
            QLabel#timeDisplay {{
                padding: 10px;
                font-size: {config.FONT_SIZE + 16}px;
                font-weight: 700;
            }}
            QLabel#sectionLabel {{
                font-size: {config.FONT_SIZE + 2}px;
                font-weight: 600;
                color: palette(window-text);
            }}
        """)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(20)


        preset_grid = QHBoxLayout()
        preset_grid.setSpacing(12)
        
        # Split presets into two rows if needed, but for touchscreen let's use a flow or just one big row/grid
        # Given 6 presets, let's do 2 rows of 3
        v_presets = QVBoxLayout()
        row1 = QHBoxLayout()
        row2 = QHBoxLayout()
        for i, (label, offset) in enumerate(self.PRESET_OFFSETS):
            button = QPushButton(label)
            button.setObjectName("presetButton")
            button.clicked.connect(lambda _, delta=offset: self._apply_preset(delta))
            if i < 3:
                row1.addWidget(button)
            else:
                row2.addWidget(button)
        v_presets.addLayout(row1)
        v_presets.addLayout(row2)
        root_layout.addLayout(v_presets)

        # Slider Section
        slider_frame = QFrame()
        slider_frame.setObjectName("pickerSection")
        slider_layout = QVBoxLayout(slider_frame)
        slider_layout.setContentsMargins(20, 20, 20, 20)
        slider_layout.setSpacing(15)

        slider_header = QLabel("Adjust Hours Ago:")
        slider_header.setObjectName("sectionLabel")
        slider_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slider_layout.addWidget(slider_header)

        self.time_display = QLabel()
        self.time_display.setObjectName("timeDisplay")
        self.time_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slider_layout.addWidget(self.time_display)

        self.ago_slider = QSlider(Qt.Orientation.Horizontal)
        self.ago_slider.setRange(0, 24)
        self.ago_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.ago_slider.setTickInterval(1)
        self.ago_slider.setSingleStep(1)
        self.ago_slider.setPageStep(4)
        self.ago_slider.valueChanged.connect(self._sync_from_slider)
        slider_layout.addWidget(self.ago_slider)
        
        root_layout.addWidget(slider_frame)

        # Preview
        self.preview_label = QLabel()
        self.preview_label.setObjectName("pickerPreview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root_layout.addWidget(self.preview_label)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        self.load_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        self.load_button.setText("Load Data")
        self.load_button.setMinimumHeight(60)
        self.load_button.setMinimumWidth(150)
        
        cancel_button = button_box.button(QDialogButtonBox.StandardButton.Cancel)
        cancel_button.setMinimumHeight(60)
        cancel_button.setMinimumWidth(150)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        root_layout.addWidget(button_box)

        self._building_ui = False

    def _default_datetime(self):
        return (dt.now() - datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    def _apply_preset(self, offset):
        self._apply_datetime((dt.now() - offset).replace(minute=0, second=0, microsecond=0))

    def _apply_datetime(self, selected_dt):
        now = dt.now()
        # Limit to 24h ago
        limit = now - datetime.timedelta(hours=24)
        if selected_dt < limit:
            selected_dt = limit
        
        normalized = min(selected_dt, now).replace(minute=0, second=0, microsecond=0)
        self._selected_dt = normalized

        # Calculate hours ago for the slider
        diff = now - normalized
        hours_ago = round(diff.total_seconds() / 3600)
        
        self.ago_slider.blockSignals(True)
        self.ago_slider.setValue(hours_ago)
        self.ago_slider.blockSignals(False)

        self._update_time_display(hours_ago)
        self._refresh_preview(normalized)

    def _sync_from_slider(self, hours_ago):
        if self._building_ui:
            return

        selected_dt = (dt.now() - datetime.timedelta(hours=hours_ago)).replace(minute=0, second=0, microsecond=0)
        self._selected_dt = selected_dt
        self._update_time_display(hours_ago)
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
        return self._selected_dt

    def _update_time_display(self, hours_ago):
        if hours_ago == 0:
            text = "Now"
        else:
            text = f"{hours_ago} hours ago"
        self.time_display.setText(text)


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
        self.eeg_view = EEGView(self.user_config)

        self.topbar = TopBar(self.user_config, self._on_config_change, self._on_zoom_change, self.dsa_view.pan, self.dsa_view.calibrate)
        self._create_menu()

        container = QWidget()
        self.layout = QVBoxLayout(container)

        self.disclaimer_label = QLabel("Nur zu Lehrzwecken")
        self.disclaimer_label.setStyleSheet("color: red; font-weight: bold; font-size: 20px; margin-bottom: 5px;")
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
        # Clear current data first
        self.dsa_view.clear_data()
        self.eeg_view.clear_data()

        # Load data from specific time
        try:
            previous_data = Output.load_psd_from_time(start_time_dt)
        except Exception as e:
            msg = QMessageBox(self)
            msg.setWindowTitle("Load Error")
            msg.setText(f"Failed to load data: {e}")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.setOption(QMessageBox.Option.DontUseNativeDialog, True)
            msg.exec()
            return

        if not previous_data:
            msg = QMessageBox(self)
            msg.setWindowTitle("Load Data")
            msg.setText("No data found for the selected time range.")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.setOption(QMessageBox.Option.DontUseNativeDialog, True)
            msg.exec()
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

        action_settings = QAction("DSA Settings", self)
        action_settings.triggered.connect(self._open_settings)
        menu.addAction(action_settings)

        action_eeg_settings = QAction("EEG Sweep Speed", self)
        action_eeg_settings.triggered.connect(self._open_eeg_settings)
        menu.addAction(action_eeg_settings)

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

        msg = QMessageBox(self)
        msg.setWindowTitle("Information")
        msg.setText(text)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.setOption(QMessageBox.Option.DontUseNativeDialog, True)
        
        # Use a custom label for the icon to avoid system sounds associated with QMessageBox.Icon
        label = QLabel()
        pixmap = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation).pixmap(32, 32)
        label.setPixmap(pixmap)
        
        # We need to use a layout to mimic QMessageBox if we don't set the icon via setIcon
        # Or better, just don't set the icon and the sound won't play on Windows.
        # However, the user wants it to look the same but without sound.
        
        msg.exec()

    def _confirm_clear_data(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Confirm data deletion")
        msg.setText("Are you sure you want to delete all EEG/DSA data?\nThis cannot be undone.")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.setDefaultButton(QMessageBox.StandardButton.No)
        msg.setOption(QMessageBox.Option.DontUseNativeDialog, True)
        
        reply = msg.exec()
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

    def _open_eeg_settings(self):
        dialog = EEGSettingsDialog(self.user_config, self._on_config_change, self)
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
