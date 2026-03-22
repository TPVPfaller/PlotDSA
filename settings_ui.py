import time
import math
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QScrollArea, QHBoxLayout, QPushButton, QSlider,
    QLabel, QGridLayout, QFrame, QSizePolicy, QCheckBox, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics

from config import SystemConfig

FONT_SIZE = 15

class TopBar(QWidget):
    def __init__(self, config, on_config_change, on_zoom_change):
        super().__init__()
        self.config = config
        self.on_config_change = on_config_change
        self.on_zoom_change = on_zoom_change

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # --- Zoom slider ---
        self.zoom_label = QLabel("Zoom:")
        self.zoom_label.setMinimumHeight(60)
        self.zoom_label.setStyleSheet(f"font-size: {FONT_SIZE}px;")
        layout.addWidget(self.zoom_label)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(1)
        self.zoom_slider.setMaximum(100)
        self.zoom_slider.setValue(
            int(SystemConfig.DISPLAY_MINUTES / SystemConfig.DISPLAY_MINUTES_BOUNDS[1] * 100)
        )
        self.zoom_slider.setFixedHeight(30)
        self.zoom_slider.valueChanged.connect(self._zoom_changed)
        layout.addWidget(self.zoom_slider)
        layout.setAlignment(self.zoom_label, Qt.AlignVCenter)
        layout.setAlignment(self.zoom_slider, Qt.AlignVCenter)

        # --- Live indicator --- # TODO: smaller
        self.connection_indicator = QLabel("DISCONNECTED")
        self.connection_indicator.setStyleSheet(f"""
            QLabel {{
                color: white;
                background-color: #6b0000;
                padding: 8px 12px;
                border-radius: 6px;
                font-weight: bold;
                font-size: {FONT_SIZE}px;
            }}
        """)
        self.connection_indicator.setMinimumHeight(60)
        self.connection_indicator.setMinimumWidth(140)
        self.connection_indicator.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.connection_indicator)
        layout.setAlignment(self.connection_indicator, Qt.AlignVCenter)

        # --- Live button ---
        self.live_btn = QPushButton("▶ Live")
        self.live_btn.setMinimumHeight(60)
        self.live_btn.setMinimumWidth(100)
        self.live_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: palette(button);
                color: palette(button-text);
                border: 1px solid palette(mid);
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: bold;
                font-size: {FONT_SIZE}px;
            }}
            QPushButton:hover {{ background-color: palette(midlight); }}
            QPushButton:pressed {{ background-color: palette(mid); }}
        """)
        policy = self.live_btn.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.live_btn.setSizePolicy(policy)
        self.live_btn.hide()
        layout.addWidget(self.live_btn)
        layout.setAlignment(self.live_btn, Qt.AlignVCenter)

        self._last_data_receive_time = time.time()

        # --- PSD Normalization Checkbox ---
        self.norm_checkbox = QCheckBox("Relative PSD")
        self.norm_checkbox.setChecked(self.config.normalize_psd)
        self.norm_checkbox.setMinimumHeight(70)
        self.norm_checkbox.setStyleSheet(f"""
            QCheckBox {{
                font-size: {FONT_SIZE}pt;
            }}
            QCheckBox::indicator {{
                width: 30px;   /* width of the box */
                height: 30px;  /* height of the box */
            }}
        """)
        self.norm_checkbox.toggled.connect(self._normalize_toggled)
        layout.addWidget(self.norm_checkbox)
        layout.setAlignment(self.norm_checkbox, Qt.AlignVCenter)

    def _normalize_toggled(self, checked):
        new_config = self.config.update(normalize_psd=bool(checked))
        self.on_config_change(new_config)

    def _zoom_changed(self, value):
        min_minutes, max_minutes = SystemConfig.DISPLAY_MINUTES_BOUNDS
        t = 1.0 - ((1.0 - (value - 1) / 99.0) ** 2)
        new_display_minutes = max_minutes - t * (max_minutes - min_minutes)
        self.on_zoom_change(new_display_minutes)

    def sync_slider(self, display_minutes):
        min_min, max_min = SystemConfig.DISPLAY_MINUTES_BOUNDS
        t = (max_min - display_minutes) / (max_min - min_min) if max_min != min_min else 0
        val_zoom = 1 + 99.0 * (1.0 - np.sqrt(max(0, 1.0 - t)))
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(np.round(val_zoom)))
        self.zoom_slider.blockSignals(False)

    def reset_last_data_timer(self):
        self._last_data_receive_time = time.time()

    def update_indicator(self):
        if time.time() - self._last_data_receive_time < 2.0:
            self.connection_indicator.setText("CONNECTED")
            self.connection_indicator.setStyleSheet(f"""
                QLabel {{
                    color: white;
                    background-color: #034003;
                    padding: 8px 12px;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: {FONT_SIZE}px;
                }}
            """)
        else:
            self.connection_indicator.setText("DISCONNECTED")
            self.connection_indicator.setStyleSheet(f"""
                QLabel {{
                    color: white;
                    background-color: #6b0000;
                    padding: 8px 12px;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: {FONT_SIZE}px;
                }}
            """)

    def update_jump_live_btn(self, dsa_view):
        if dsa_view.dsa_buffer.t0 is None or dsa_view.live_mode:
            self.live_btn.hide()
        else:
            self.live_btn.show()

    def apply_config(self, config):
        self.config = config


class SettingsDialog(QDialog):
    def __init__(self, config, on_config_change, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System Settings")
        self.setMinimumSize(540, 260)

        self.config = config
        self.on_config_change = on_config_change

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 10, 10, 10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        grid = QGridLayout(container)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)

        self.sliders = {}
        row_idx = 0

        def add_slider(name, bounds, value, scale=1, unit: str = "", display_factor: float = 1.0,
                       decimals_override=None):
            nonlocal row_idx
            row_h = 36  # taller for touch

            name_label = QLabel(name)
            name_label.setStyleSheet("font-size: 18px;")
            name_label.setFixedHeight(row_h)
            name_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            name_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(bounds[0] * scale))
            slider.setMaximum(int(bounds[1] * scale))
            slider.setValue(int(value * scale))
            slider.setFixedHeight(row_h)
            slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            slider.setStyleSheet("""
                QSlider::groove:horizontal { height: 6px; background: palette(mid); border-radius: 3px; }
                QSlider::sub-page:horizontal { background: palette(highlight); border-radius: 3px; }
                QSlider::add-page:horizontal { background: palette(mid); border-radius: 3px; }
                QSlider::handle:horizontal { width: 18px; height: 18px; margin: -6px 0; border-radius: 9px; background: palette(window-text); border: 1px solid palette(base); }
            """)

            decimals = decimals_override if decimals_override is not None else max(0, int(round(math.log10(scale))) if scale > 1 else 0)

            def fmt_number(x: float) -> str:
                return f"{x:.{decimals}f}"

            def fmt_with_unit(x: float) -> str:
                return f"{fmt_number(x)}{unit}"

            value_label = QLabel(fmt_with_unit(value * display_factor))
            value_label.setStyleSheet("font-size: 18px;")
            value_label.setFixedHeight(row_h)
            value_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            value_label.setAlignment(Qt.AlignLeft)

            fm = QFontMetrics(value_label.font())
            b0_txt = fmt_with_unit(bounds[0] * display_factor)
            b1_txt = fmt_with_unit(bounds[1] * display_factor)
            max_w = max(fm.horizontalAdvance(b0_txt), fm.horizontalAdvance(b1_txt)) + 25
            value_label.setFixedWidth(max_w)

            slider.valueChanged.connect(lambda val: value_label.setText(fmt_with_unit(val / scale * display_factor)))

            grid.addWidget(name_label, row_idx, 0, alignment=Qt.AlignVCenter | Qt.AlignRight)
            grid.addWidget(slider, row_idx, 1, alignment=Qt.AlignVCenter)
            grid.addWidget(value_label, row_idx, 2, alignment=Qt.AlignLeft)

            self.sliders[name] = (slider, scale)
            row_idx += 1

        # --- Add sliders ---
        add_slider("Window (s)", SystemConfig.WINDOW_SEC_BOUNDS, config.window_sec, 1, unit=" s")
        add_slider("Segment (s)", SystemConfig.SEGMENT_SEC_BOUNDS, config.segment_sec, 10, unit=" s")
        add_slider("Window Overlap", SystemConfig.WINDOW_OVERLAP_BOUNDS, config.window_overlap, 100, unit=" %", display_factor=100.0, decimals_override=0)
        add_slider("Segment Overlap", SystemConfig.SEGMENT_OVERLAP_BOUNDS, config.segment_overlap, 100, unit=" %", display_factor=100.0, decimals_override=0)
        add_slider("Max Frequency (Hz)", SystemConfig.MAX_FREQ_HZ_BOUNDS, config.max_freq_hz, 1, unit=" Hz")

        # --- Buttons row ---
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setMinimumHeight(50)
        reset_btn.setStyleSheet("font-size: 18px;")
        reset_btn.clicked.connect(self._reset_to_defaults)

        apply_btn = QPushButton("Apply and Close")
        apply_btn.setMinimumHeight(50)
        apply_btn.setStyleSheet("font-size: 18px;")
        apply_btn.clicked.connect(self._apply)

        button_bar = QWidget()
        bar_layout = QHBoxLayout(button_bar)
        bar_layout.setContentsMargins(0, 6, 0, 0)
        bar_layout.setSpacing(10)
        bar_layout.addWidget(reset_btn, alignment=Qt.AlignLeft)
        bar_layout.addStretch(1)
        bar_layout.addWidget(apply_btn, alignment=Qt.AlignRight)

        grid.addWidget(button_bar, row_idx, 0, 1, 3)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def _reset_to_defaults(self):
        resp = QMessageBox.question(
            self,
            "Reset Settings",
            "Reset all settings to default values?\n(Changes are not saved until you click 'Apply and Close'.)",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if resp != QMessageBox.Yes:
            return

        mapping = {
            "Window (s)": SystemConfig.WINDOW_SEC,
            "Segment (s)": SystemConfig.SEGMENT_SEC,
            "Window Overlap": SystemConfig.WINDOW_OVERLAP,
            "Segment Overlap": SystemConfig.SEGMENT_OVERLAP,
            "Max Frequency (Hz)": SystemConfig.MAX_FREQ_HZ,
        }
        for name, (slider, scale) in self.sliders.items():
            if name in mapping:
                val = mapping[name]
                slider.blockSignals(True)
                slider.setValue(int(round(val * scale)))
                slider.blockSignals(False)
                slider.valueChanged.emit(slider.value())

    def _apply(self):
        try:
            proposed_window_sec = self.sliders["Window (s)"][0].value()
            proposed_segment_sec = self.sliders["Segment (s)"][0].value() / 10
            proposed_window_overlap = self.sliders["Window Overlap"][0].value() / 100
            proposed_segment_overlap = self.sliders["Segment Overlap"][0].value() / 100
            proposed_max_freq_hz = self.sliders["Max Frequency (Hz)"][0].value()

            if abs(proposed_segment_sec - float(self.config.segment_sec)) > 1e-9:
                resp = QMessageBox.question(
                    self,
                    "Change Segment Length",
                    "Changing the segment length will clear the current DSA view/history.\n\nDo you want to proceed?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if resp != QMessageBox.Yes:
                    return

            new_config = self.config.update(
                window_sec=proposed_window_sec,
                segment_sec=proposed_segment_sec,
                window_overlap=proposed_window_overlap,
                segment_overlap=proposed_segment_overlap,
                max_freq_hz=proposed_max_freq_hz,
            )
            self.on_config_change(new_config)
            self.accept()
        except ValueError as e:
            QMessageBox.critical(self, "Invalid Configuration", str(e))