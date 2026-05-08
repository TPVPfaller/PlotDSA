import math

from PySide6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QScrollArea, QHBoxLayout, QPushButton, QSlider,
    QLabel, QGridLayout, QFrame, QSizePolicy, QMessageBox, QRadioButton, QButtonGroup
)
from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtGui import QColor, QFontMetrics, QIcon, QPainter, QPixmap, QPolygon
import config


# ------------------ TopBar ------------------ #
class TopBar(QWidget):
    BUTTON_HEIGHT = 35

    def __init__(self, user_config, on_config_change, on_zoom_change, on_pan, on_calibrate):
        super().__init__()
        self.user_config = user_config
        self.on_config_change = on_config_change
        self.on_zoom_change = on_zoom_change
        self.on_pan = on_pan
        self.on_calibrate = on_calibrate

        self.GAMMA = 2.5 # Zoom shape parameter

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self.left_btn = self._create_arrow_button("left")
        self.left_btn.clicked.connect(lambda: self._pan(-1))
        layout.addWidget(self.left_btn)

        self.right_btn = self._create_arrow_button("right")
        self.right_btn.clicked.connect(lambda: self._pan(1))
        layout.addWidget(self.right_btn)

        self.zoom_label = QLabel("Zoom:")
        self.zoom_label.setStyleSheet(f"font-size: {config.FONT_SIZE}px;")
        layout.addWidget(self.zoom_label)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(1)
        self.zoom_slider.setMaximum(1000)
        self.zoom_slider.valueChanged.connect(self._zoom_changed)
        layout.addWidget(self.zoom_slider)

        self.live_btn = self._create_button("\u25b6 Live")
        self.live_btn.setMinimumWidth(70)
        policy = self.live_btn.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.live_btn.setSizePolicy(policy)
        self.live_btn.hide()
        layout.addWidget(self.live_btn)

        self.calibrate_btn = self._create_button("Calibrate")
        self.calibrate_btn.clicked.connect(self.on_calibrate)
        layout.addWidget(self.calibrate_btn)

        self.reset_btn = self._create_button()
        self.reset_btn.setIcon(QIcon("reset_icon.png"))
        self.reset_btn.clicked.connect(self._reset_calibration)
        layout.addWidget(self.reset_btn)

    def _button_style(self, font_size=None, padding="8px 12px"):
        resolved_font_size = config.FONT_SIZE if font_size is None else font_size
        return f"""
            QPushButton {{
                background-color: rgb(58, 58, 58);
                color: white;
                border: 1px solid rgb(78, 78, 78);
                border-radius: 6px;
                padding: {padding};
                font-size: {resolved_font_size}px;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: rgb(72, 72, 72);
            }}
            QPushButton:pressed {{
                background-color: rgb(92, 92, 92);
            }}
            QPushButton:disabled {{
                background-color: rgb(48, 48, 48);
                color: rgb(150, 150, 150);
            }}
        """

    def _create_button(self, text="", font_size=None, padding="8px 12px", min_width=None):
        button = QPushButton(text)
        button.setMinimumHeight(self.BUTTON_HEIGHT)
        if min_width is not None:
            button.setMinimumWidth(min_width)
        button.setStyleSheet(self._button_style(font_size=font_size, padding=padding))
        return button

    def _create_arrow_button(self, direction):
        button = self._create_button("", padding="0px", min_width=self.BUTTON_HEIGHT)
        button.setIcon(self._create_arrow_icon(direction))
        button.setIconSize(QSize(16, 16))
        return button

    def _create_arrow_icon(self, direction):
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("white"))

        if direction == "left":
            points = [QPoint(10, 3), QPoint(5, 8), QPoint(10, 13)]
        else:
            points = [QPoint(6, 3), QPoint(11, 8), QPoint(6, 13)]

        painter.drawPolygon(QPolygon(points))
        painter.end()
        return QIcon(pixmap)

    # ------------------ New actions ------------------ #
    def _reset_calibration(self):
        new_config = self.user_config.update(psd_db_min=config.PSD_DB_MIN, psd_db_max=config.PSD_DB_MAX)
        self.on_config_change(new_config)

    def _pan(self, direction: int):
        """direction: -1 = left, +1 = right"""
        step = 0.27 # percent of display width
        self.on_pan(direction * step)

    def _zoom_changed(self, value):
        min_minutes, max_minutes = config.DISPLAY_MINUTES_BOUNDS

        t = (value - 1) / 999.0  # 0..1 linear

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

        val_zoom = 1 + 999.0 * t

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
        self.setWindowTitle("DSA Settings")
        self.setMinimumSize(640, 200)

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

        # --- Buttons row ---
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setMinimumHeight(35)
        reset_btn.setStyleSheet(f"font-size: {config.FONT_SIZE}px;")
        reset_btn.clicked.connect(self._reset_to_defaults)

        apply_btn = QPushButton("Apply and Close")
        apply_btn.setMinimumHeight(35)
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


class EEGSettingsDialog(QDialog):
    def __init__(self, user_config, on_config_change, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EEG Settings")
        self.setMinimumSize(360, 220)

        self.user_config = user_config
        self.on_config_change = on_config_change
        self.speed_buttons = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Select EEG sweep speed:")
        title.setStyleSheet(f"font-size: {config.FONT_SIZE + 1}px; font-weight: 600;")
        layout.addWidget(title)

        self.button_group = QButtonGroup(self)
        for value in config.EEG_MM_PER_SECOND_OPTIONS:
            label = f"{value:g} mm/s"
            button = QRadioButton(label)
            button.setStyleSheet(f"padding: 0px 12px;"
                                 f"font-size: {config.FONT_SIZE}px;")
            self.button_group.addButton(button)
            self.speed_buttons[value] = button
            layout.addWidget(button)

        selected_button = self.speed_buttons.get(self.user_config.eeg_mm_per_second)
        if selected_button is not None:
            selected_button.setChecked(True)

        layout.addStretch(1)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)

        reset_btn = QPushButton("Reset to Default")
        reset_btn.setMinimumHeight(35)
        reset_btn.setStyleSheet(f"font-size: {config.FONT_SIZE}px;")
        reset_btn.clicked.connect(self._reset_to_default)
        button_row.addWidget(reset_btn)

        button_row.addStretch(1)

        apply_btn = QPushButton("Apply and Close")
        apply_btn.setMinimumHeight(35)
        apply_btn.setStyleSheet(f"font-size: {config.FONT_SIZE}px;")
        apply_btn.clicked.connect(self._apply)
        button_row.addWidget(apply_btn)

        layout.addLayout(button_row)

    def _reset_to_default(self):
        default_button = self.speed_buttons.get(config.EEG_MM_PER_SECOND)
        if default_button is not None:
            default_button.setChecked(True)

    def _apply(self):
        selected_value = next(
            (value for value, button in self.speed_buttons.items() if button.isChecked()),
            None,
        )

        if selected_value is None:
            msg = QMessageBox(self)
            msg.setWindowTitle("Invalid Configuration")
            msg.setText("Select an EEG sweep speed.")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.setOption(QMessageBox.Option.DontUseNativeDialog, True)
            msg.exec()
            return

        try:
            new_config = self.user_config.update(eeg_mm_per_second=selected_value)
            self.on_config_change(new_config)
            self.accept()
        except ValueError as e:
            msg = QMessageBox(self)
            msg.setWindowTitle("Invalid Configuration")
            msg.setText(str(e))
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.setOption(QMessageBox.Option.DontUseNativeDialog, True)
            msg.exec()
