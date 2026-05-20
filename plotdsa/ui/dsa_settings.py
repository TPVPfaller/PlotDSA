import math

from PySide6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QScrollArea, QHBoxLayout, QPushButton, QSlider,
    QLabel, QGridLayout, QFrame, QSizePolicy, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from .. import config


class SettingsDialog(QDialog):
    def __init__(self, user_config, on_config_change, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DSA Settings")
        self.setMinimumSize(640, 170)

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
            row_h = 44  # taller for touch

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
            slider.setStyleSheet(f"""
                QSlider::groove:horizontal {{ height: 8px; background: palette(mid); border-radius: 4px; }}
                QSlider::sub-page:horizontal {{ background: palette(highlight); border-radius: 4px; }}
                QSlider::add-page:horizontal {{ height: 8px; background: palette(mid); border-radius: 4px; }}
                QSlider::handle:horizontal {{ width: 22px; height: 22px; margin: -7px 0; border-radius: 11px; background: palette(window-text); border: 1px solid palette(base); }}
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

        # --- Buttons row ---
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setMinimumHeight(42)
        reset_btn.setStyleSheet(f"font-size: {config.FONT_SIZE}px; color: white;")
        reset_btn.clicked.connect(self._reset_to_defaults)

        apply_btn = QPushButton("Apply and Close")
        apply_btn.setMinimumHeight(42)
        apply_btn.setStyleSheet(f"font-size: {config.FONT_SIZE}px; color: white;")
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
        msg = QMessageBox(self)
        msg.setWindowTitle("Reset Settings")
        msg.setText("Reset all settings to default values?\n(Changes are not saved until you click 'Apply and Close'.)")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        msg.setOption(QMessageBox.Option.DontUseNativeDialog, True)

        resp = msg.exec()
        if resp != QMessageBox.Yes:
            return

        mapping = {
            "Window (s)": config.WINDOW_SEC,
            "Window Overlap": config.WINDOW_OVERLAP,
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

            new_config = self.user_config.update(
                window_sec=proposed_window_sec,
                window_overlap=proposed_window_overlap,
            )
            self.on_config_change(new_config)
            self.accept()
        except ValueError as e:
            msg = QMessageBox(self)
            msg.setWindowTitle("Invalid Configuration")
            msg.setText(str(e))
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.setOption(QMessageBox.Option.DontUseNativeDialog, True)
            msg.exec()

