import time
import math
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QScrollArea, QHBoxLayout, QPushButton, QSlider,
    QLabel, QGridLayout, QFrame, QSizePolicy, QCheckBox, QMessageBox, QStyle
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
import config


# ------------------ TopBar ------------------ #
class TopBar(QWidget):
    def __init__(self, user_config, on_config_change, on_zoom_change, on_pan):
        super().__init__()
        self.user_config = user_config
        self.on_config_change = on_config_change
        self.on_zoom_change = on_zoom_change
        self.on_pan = on_pan

        self.GAMMA = 2.5 # Zoom shape parameter

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # --- Pan Left Button ---
        self.left_btn = QPushButton()
        self.left_btn.setMinimumHeight(40)
        self.left_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowLeft))
        self.left_btn.setToolTip("Pan backward")
        self.left_btn.clicked.connect(lambda: self._pan(-1))
        layout.addWidget(self.left_btn)

        # --- Pan Right Button ---
        self.right_btn = QPushButton()
        self.right_btn.setMinimumHeight(40)
        self.right_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        self.right_btn.setToolTip("Pan forward")
        self.right_btn.clicked.connect(lambda: self._pan(1))
        layout.addWidget(self.right_btn)

        # --- Zoom slider ---
        self.zoom_label = QLabel("Zoom:")
        self.zoom_label.setStyleSheet(f"font-size: {config.FONT_SIZE}px;")
        layout.addWidget(self.zoom_label)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(1)
        self.zoom_slider.setMaximum(100)
        self.zoom_slider.valueChanged.connect(self._zoom_changed)

        layout.addWidget(self.zoom_slider)

        # --- Live button ---
        self.live_btn = QPushButton("▶ Live")
        self.live_btn.setMinimumHeight(40)
        self.live_btn.setMinimumWidth(90)
        self.live_btn.setStyleSheet(f"""
            QPushButton {{
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: bold;
                font-size: {config.FONT_SIZE}px;
            }}
        """)
        policy = self.live_btn.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.live_btn.setSizePolicy(policy)
        self.live_btn.hide()
        layout.addWidget(self.live_btn)

        # --- PSD Normalization Checkbox ---
        self.norm_checkbox = QCheckBox("Relative PSD")
        self.norm_checkbox.setChecked(self.user_config.normalize_psd)
        self.norm_checkbox.setStyleSheet(f"font-size: {config.FONT_SIZE}px;")
        self.norm_checkbox.toggled.connect(self._normalize_toggled)
        layout.addWidget(self.norm_checkbox)

    # ------------------ New actions ------------------ #
    def _pan(self, direction: int):
        """direction: -1 = left, +1 = right"""
        step = 0.25 # percent of display width
        self.on_pan(direction * step)

    def _normalize_toggled(self, checked):
        new_config = self.user_config.update(normalize_psd=bool(checked))
        self.on_config_change(new_config)

    def _zoom_changed(self, value):
        min_minutes, max_minutes = config.DISPLAY_MINUTES_BOUNDS

        t = (value - 1) / 99.0  # 0..1 linear

        # invert gamma effect for desired behavior
        t = 1.0 - (1.0 - t) ** self.GAMMA

        display_minutes = max_minutes - t * (max_minutes - min_minutes)
        self.on_zoom_change(display_minutes)

    def sync_slider(self, display_minutes):
        min_minutes, max_minutes = config.DISPLAY_MINUTES_BOUNDS

        t = (max_minutes - display_minutes) / (max_minutes - min_minutes)
        t = max(0.0, min(1.0, t))

        # inverse of forward curve
        t = 1.0 - (1.0 - t) ** (1.0 / self.GAMMA)

        val_zoom = 1 + 99.0 * t

        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(round(val_zoom)))
        self.zoom_slider.blockSignals(False)

    def update_jump_live_btn(self, dsa_view):
        if dsa_view.dsa_buffer.t0 is None or dsa_view.live_mode:
            self.live_btn.hide()
        else:
            self.live_btn.show()

    def apply_config(self, user_config):
        self.user_config = user_config


class SettingsDialog(QDialog):
    def __init__(self, user_config, on_config_change, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System Settings")
        self.setMinimumSize(640, 260)

        self.user_config = user_config
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
            name_label.setStyleSheet(f"font-size: {config.FONT_SIZE}px;")
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
            value_label.setStyleSheet(f"font-size: {config.FONT_SIZE}px;")
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
        add_slider("Window (s)", config.WINDOW_SEC_BOUNDS, user_config.window_sec, 1, unit=" s")
        add_slider("Window Overlap", config.WINDOW_OVERLAP_BOUNDS, user_config.window_overlap, 100, unit=" %", display_factor=100.0, decimals_override=0)
        add_slider("Max Frequency (Hz)", config.MAX_FREQ_HZ_BOUNDS, user_config.max_freq_hz, 1, unit=" Hz")
        add_slider("PSD Min (dB)", config.PSD_DB_MIN_BOUNDS, user_config.psd_db_min, 1, unit=" dB")
        add_slider("PSD Max (dB)", config.PSD_DB_MAX_BOUNDS, user_config.psd_db_max, 1, unit=" dB")

        psd_min_slider, _ = self.sliders["PSD Min (dB)"]
        psd_max_slider, _ = self.sliders["PSD Max (dB)"]

        def psd_min_changed(val):
            if val >= psd_max_slider.value():
                val = psd_max_slider.value() - 1
                psd_min_slider.blockSignals(True)
                psd_min_slider.setValue(val)
                psd_min_slider.blockSignals(False)

        def psd_max_changed(val):
            if val <= psd_min_slider.value():
                val = psd_min_slider.value() + 1
                psd_max_slider.blockSignals(True)
                psd_max_slider.setValue(val)
                psd_max_slider.blockSignals(False)

        psd_min_slider.valueChanged.connect(psd_min_changed)
        psd_max_slider.valueChanged.connect(psd_max_changed)

        # --- Buttons row ---
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setMinimumHeight(50)
        reset_btn.setStyleSheet(f"font-size: {config.FONT_SIZE}px;")
        reset_btn.clicked.connect(self._reset_to_defaults)

        apply_btn = QPushButton("Apply and Close")
        apply_btn.setMinimumHeight(50)
        apply_btn.setStyleSheet(f"font-size: {config.FONT_SIZE}px;")
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
            "Window (s)": config.WINDOW_SEC,
            "Window Overlap": config.WINDOW_OVERLAP,
            "Max Frequency (Hz)": config.MAX_FREQ_HZ,
            "PSD Min (dB)": config.PSD_DB_MIN,
            "PSD Max (dB)": config.PSD_DB_MAX,
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
            proposed_window_overlap = self.sliders["Window Overlap"][0].value() / 100
            proposed_max_freq_hz = self.sliders["Max Frequency (Hz)"][0].value()
            proposed_psd_min = self.sliders["PSD Min (dB)"][0].value()
            proposed_psd_max = self.sliders["PSD Max (dB)"][0].value()

            new_config = self.user_config.update(
                window_sec=proposed_window_sec,
                window_overlap=proposed_window_overlap,
                max_freq_hz=proposed_max_freq_hz,
                psd_db_min=proposed_psd_min,
                psd_db_max=proposed_psd_max,
            )
            self.on_config_change(new_config)
            self.accept()
        except ValueError as e:
            QMessageBox.critical(self, "Invalid Configuration", str(e))