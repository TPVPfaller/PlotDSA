from datetime import datetime as dt
import datetime
import sys

sys.argv += ['-platform', 'windows:darkmode=2']
from PySide6.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QDialog,
    QPushButton,
    QHBoxLayout,
    QDialogButtonBox,
    QFrame,
    QSlider,
)
from PySide6.QtCore import Qt

if __package__ in (None, ""):
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from plotdsa import config
else:
    from .. import config


class TimeSelectionDialog(QDialog):
    PRESET_OFFSETS = (
        ("2 hours ago", datetime.timedelta(hours=2)),
        ("8 hours ago", datetime.timedelta(hours=8)),
        ("24 hours ago", datetime.timedelta(hours=24)),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Start Time")
        self.setMinimumSize(400, 350)

        self._selected_dt = self._default_datetime()
        self._building_ui = False

        self._build_ui()
        self._apply_datetime(self._selected_dt)

    def _build_ui(self):
        self._building_ui = True

        self.setStyleSheet(f"""
            QDialog {{
                font-size: {config.FONT_SIZE}px;
                color: white;
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
                color: white;
            }}
            QPushButton#presetButton {{
                min-height: 48px;
                padding: 8px 16px;
                border-radius: 10px;
                color: white;
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
                height: 24px;
                border-radius: 12px;
                background: palette(mid);
            }}
            QSlider::sub-page:horizontal {{
                height: 24px;
                border-radius: 12px;
                background: palette(highlight);
            }}
            QSlider::handle:horizontal {{
                width: 48px;
                height: 48px;
                margin: -12px 0;
                border-radius: 24px;
                background: palette(window-text);
                border: 2px solid palette(base);
            }}
            QLabel#timeDisplay {{
                padding: 10px;
                font-size: {config.FONT_SIZE + 14}px;
                font-weight: 700;
                color: white;
            }}
        """)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(12)

        # Split presets into two rows if needed, but for touchscreen let's use a flow or just one big row/grid
        # Given 6 presets, let's do 2 rows of 3
        v_presets = QVBoxLayout()
        row1 = QHBoxLayout()
        for i, (label, offset) in enumerate(self.PRESET_OFFSETS):
            button = QPushButton(label)
            button.setObjectName("presetButton")
            button.clicked.connect(lambda _, delta=offset: self._apply_preset(delta))

            row1.addWidget(button)

        v_presets.addLayout(row1)
        root_layout.addLayout(v_presets)

        # Slider Section
        slider_frame = QFrame()
        slider_frame.setObjectName("pickerSection")
        slider_layout = QVBoxLayout(slider_frame)
        slider_layout.setContentsMargins(20, 20, 20, 20)
        slider_layout.setSpacing(15)

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
        self.load_button.setMinimumHeight(40)
        self.load_button.setMinimumWidth(150)

        cancel_button = button_box.button(QDialogButtonBox.StandardButton.Cancel)
        cancel_button.setMinimumHeight(40)
        cancel_button.setMinimumWidth(150)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        root_layout.addWidget(button_box)

        self._building_ui = False

    def _default_datetime(self):
        return self._normalize_datetime(dt.now() - datetime.timedelta(hours=1))

    def _apply_preset(self, offset):
        self._apply_datetime(dt.now() - offset)

    def _normalize_datetime(self, value):
        return value.replace(minute=0, second=0, microsecond=0)

    def _hours_ago(self, selected_dt, now):
        return round((now - selected_dt).total_seconds() / 3600)

    def _set_hours_ago(self, hours_ago):
        self.ago_slider.blockSignals(True)
        self.ago_slider.setValue(hours_ago)
        self.ago_slider.blockSignals(False)

    def _apply_datetime(self, selected_dt):
        now = dt.now()
        limit = now - datetime.timedelta(hours=24)
        if selected_dt < limit:
            selected_dt = limit

        normalized = self._normalize_datetime(min(selected_dt, now))
        self._selected_dt = normalized
        hours_ago = self._hours_ago(normalized, now)
        self._set_hours_ago(hours_ago)

        self._update_time_display(hours_ago)
        self._refresh_preview(normalized)

    def _sync_from_slider(self, hours_ago):
        if self._building_ui:
            return

        selected_dt = self._normalize_datetime(dt.now() - datetime.timedelta(hours=hours_ago))
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

        self.preview_label.setText(selected_dt.strftime(f"%a, %d %b %H:%M  |  {relative}"))

    def selected_datetime(self):
        return self._selected_dt

    def _update_time_display(self, hours_ago):
        if hours_ago == 0:
            text = "Now"
        elif hours_ago == 1:
            text = "1 hour ago"
        else:
            text = f"{hours_ago} hours ago"
        self.time_display.setText(text)