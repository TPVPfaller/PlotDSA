import datetime
from collections import deque

import numpy as np
import pyqtgraph as pg
from pyqtgraph import ColorBarItem, GridItem
from PySide6.QtCore import QByteArray, QSize, Qt, QEvent, QTimer
from PySide6.QtGui import QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
from scipy.signal import lfilter
import time

import math
from ..core.buffers import DSABuffer
from ..core.calculations import DSACalculator
from .. import config

SETTINGS_GEAR_SVG = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
        <path fill="#FFFFFF" d="M502.325,307.303l-39.006-30.805c-6.215-4.908-9.665-12.429-9.668-20.348c0-0.084,0-0.168,0-0.252
        c-0.014-7.936,3.44-15.478,9.667-20.396l39.007-30.806c8.933-7.055,12.093-19.185,7.737-29.701l-17.134-41.366
        c-4.356-10.516-15.167-16.86-26.472-15.532l-49.366,5.8c-7.881,0.926-15.656-1.966-21.258-7.586
        c-0.059-0.06-0.118-0.119-0.177-0.178c-5.597-5.602-8.476-13.36-7.552-21.225l5.799-49.363
        c1.328-11.305-5.015-22.116-15.531-26.472L337.004,1.939c-10.516-4.356-22.646-1.196-29.701,7.736l-30.805,39.005
        c-4.908,6.215-12.43,9.665-20.349,9.668c-0.084,0-0.168,0-0.252,0c-7.935,0.014-15.477-3.44-20.395-9.667L204.697,9.675
        c-7.055-8.933-19.185-12.092-29.702-7.736L133.63,19.072c-10.516,4.356-16.86,15.167-15.532,26.473l5.799,49.366
        c0.926,7.881-1.964,15.656-7.585,21.257c-0.059,0.059-0.118,0.118-0.178,0.178c-5.602,5.598-13.36,8.477-21.226,7.552
        l-49.363-5.799c-11.305-1.328-22.116,5.015-26.472,15.531L1.939,174.996c-4.356,10.516-1.196,22.646,7.736,29.701l39.006,30.805
        c6.215,4.908,9.665,12.429,9.668,20.348c0,0.084,0,0.167,0,0.251c0.014,7.935-3.44,15.477-9.667,20.395L9.675,307.303
        c-8.933,7.055-12.092,19.185-7.736,29.701l17.134,41.365c4.356,10.516,15.168,16.86,26.472,15.532l49.366-5.799
        c7.882-0.926,15.656,1.965,21.258,7.586c0.059,0.059,0.118,0.119,0.178,0.178c5.597,5.603,8.476,13.36,7.552,21.226l-5.799,49.364
        c-1.328,11.305,5.015,22.116,15.532,26.472l41.366,17.134c10.516,4.356,22.646,1.196,29.701-7.736l30.804-39.005
        c4.908-6.215,12.43-9.665,20.348-9.669c0.084,0,0.168,0,0.251,0c7.936-0.014,15.478,3.44,20.396,9.667l30.806,39.007
        c7.055,8.933,19.185,12.093,29.701,7.736l41.366-17.134c10.516-4.356,16.86-15.168,15.532-26.472l-5.8-49.366
        c-0.926-7.881,1.965-15.656,7.586-21.257c0.059-0.059,0.119-0.119,0.178-0.178c5.602-5.597,13.36-8.476,21.225-7.552l49.364,5.799
        c11.305,1.328,22.117-5.015,26.472-15.531l17.134-41.365C514.418,326.488,511.258,314.358,502.325,307.303z M281.292,329.698
        c-39.68,16.436-85.172-2.407-101.607-42.087c-16.436-39.68,2.407-85.171,42.087-101.608c39.68-16.436,85.172,2.407,101.608,42.088
        C339.815,267.771,320.972,313.262,281.292,329.698z"/>
    </svg>
"""

LEFT_AXIS_WIDTH = 45
AXIS_TEXT_COLOR = config.TEXT_COLOR_STR
AXIS_TICK_FONT_SIZE_PT = config.FONT_SIZE - 7
AXIS_LABEL_STYLE = {
    "color": AXIS_TEXT_COLOR,
    "font-size": f"{config.FONT_SIZE - 6}pt",
}


def create_settings_gear_icon(size=18):
    renderer = QSvgRenderer(QByteArray(SETTINGS_GEAR_SVG.encode("utf-8")))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()

    return QIcon(pixmap)


def create_stepper_icon(direction, size=20):
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    pen = QPen(Qt.white)
    pen.setWidth(2)
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)

    center = size / 2
    inset = max(4, size // 4)
    painter.drawLine(inset, center, size - inset, center)
    if direction == "plus":
        painter.drawLine(center, inset, center, size - inset)

    painter.end()
    return QIcon(pixmap)


def set_uniform_left_axis_width(plot_with_axis):
    plot_with_axis.getAxis("left").setWidth(LEFT_AXIS_WIDTH)


def style_axis(axis):
    axis.setPen(AXIS_TEXT_COLOR)
    axis.setTextPen(AXIS_TEXT_COLOR)
    tick_font = QFont()
    tick_font.setPointSize(AXIS_TICK_FONT_SIZE_PT)
    axis.setTickFont(tick_font)


def set_axis_label(plot_item, axis_name, text, units=None):
    plot_item.setLabel(axis_name, text, units=units, **AXIS_LABEL_STYLE)
    style_axis(plot_item.getAxis(axis_name))


# ------------------ DSA View ------------------ #
class FrequencyAxis(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_freq = None
        style_axis(self)

    def set_max_freq(self, max_f):
        self.max_freq = float(max_f)
        self.update()

    def tickValues(self, minVal, maxVal, size):
        ticks = super().tickValues(minVal, maxVal, size)

        if self.max_freq is None or not ticks:
            return ticks

        spacing = 5.0

        # Stable major ticks
        major_ticks = []

        start = math.ceil(minVal / spacing) * spacing

        v = start
        while v < maxVal:
            major_ticks.append(float(v))
            v += spacing

        # Remove tick near max_freq
        major_ticks = [
            t for t in major_ticks
            if abs(t - self.max_freq) > spacing * 0.7
        ]

        # Always add max_freq
        major_ticks.append(float(self.max_freq))

        # Ensure sorted unique finite values
        major_ticks = sorted(set(
            t for t in major_ticks
            if math.isfinite(t)
        ))

        # Replace ONLY major ticks
        ticks[0] = (spacing, major_ticks)

        return ticks

class DSAView(pg.GraphicsLayoutWidget):
    """Dynamic Spectrum Analysis display with live/pan and pinch-zoom support."""

    def __init__(self, user_config, on_config_change, on_zoom_change):
        super().__init__()

        self.user_config = user_config
        self.on_config_change = on_config_change
        self.on_zoom_change = on_zoom_change
        self.dsa_buffer = DSABuffer()
        self.setMinimumHeight(config.MIN_DSA_HEIGHT)
        self._init_parameters()
        self._init_plot()
        self._init_colormap()
        self._init_colorbar()
        self._init_settings_controls()
        self._init_gestures()
        self.update()

    def _init_parameters(self):
        self.live_mode = True
        self._last_render = time.time()
        self._last_levels = (self.user_config.psd_db_min, self.user_config.psd_db_max)
        self._zoom_factor = 1.0
        self.display_minutes = self.user_config.display_minutes
        self._pan_sec = 0.0
        self._min_zoom = 1.0
        self._max_zoom = 10.0

        self.freq_bins = config.FREQ_BINS[(config.FREQ_BINS <= self.user_config.max_freq_hz)]
        self._dragging = False
        self._last_mouse_pos = None
        self.t0 = datetime.datetime.now().timestamp()

    def _init_plot(self):
        self.time_axis = pg.DateAxisItem("bottom")
        self.freq_axis = FrequencyAxis("left")
        self.plot = self.addPlot(row=0, col=0, axisItems={"bottom": self.time_axis, "left": self.freq_axis})
        set_axis_label(self.plot, "left", "Frequency", units="Hz")
        style_axis(self.time_axis)
        set_uniform_left_axis_width(self.plot)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.showGrid(x=False, y=False)
        self.plot.invertY(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self._update_y_axis()
        self.plot.setContentsMargins(0, 10, 0, 0)
        self.image = pg.ImageItem(axisOrder='col-major', interpolation="linear")
        self.plot.addItem(self.image)

    def _update_y_axis(self):
        max_f = self.user_config.max_freq_hz
        self.plot.setYRange(config.LOWEST_FREQ_HZ, max_f, padding=0)
        self.freq_axis.set_max_freq(max_f)


    def _style_adjust_button(self, button):
        button.setFixedSize(36, 36)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(60, 60, 60, 150);
                color: white;
                border-radius: 4px;
                font-weight: bold;
                font-size: {config.FONT_SIZE + 6}px;
            }}
            QPushButton:hover {{
                background-color: rgba(80, 80, 80, 200);
            }}
            QPushButton:pressed {{
                background-color: rgba(100, 100, 100, 255);
            }}
        """)

    def _adjust_max_freq(self, delta):
        new_max_freq = self.user_config.max_freq_hz + delta
        min_f, max_f = config.MAX_FREQ_HZ_BOUNDS
        if min_f <= new_max_freq <= max_f:
            new_config = self.user_config.update(max_freq_hz=new_max_freq)
            self.on_config_change(new_config)

    def _init_colormap(self):
        colors = [
            (10, 10, 50),
            (20, 40, 120),
            (40, 120, 200),
            (80, 200, 220),
            (80, 220, 140),
            (200, 220, 80),
            (240, 160, 40),
            (240, 80, 40),
            (240, 0, 0),
        ]
        self.cmap = pg.ColorMap(np.linspace(0, 1, len(colors)), colors)
        self.lut = self.cmap.getLookupTable(nPts=256, mode="byte")
        self.image.setLookupTable(self.lut)

    def _init_colorbar(self):
        self.colorbar = ColorBarItem(
            values=(self.user_config.psd_db_min, self.user_config.psd_db_max),
            colorMap=self.cmap,
            interactive=False,
        )
        self.colorbar.setImageItem(self.image)
        self.colorbar.setContentsMargins(0,24,20,24)
        self.addItem(self.colorbar, row=0, col=1)

        cb_axis = self.colorbar.getAxis("left")
        style_axis(cb_axis)
        cb_axis.setLabel("Power", units="dB", **AXIS_LABEL_STYLE)

        number_axis = self.colorbar.getAxis("right")
        style_axis(number_axis)

    def _init_settings_controls(self):
        from PySide6.QtWidgets import QApplication, QFrame, QLabel, QToolButton, QVBoxLayout

        self.dsa_value_labels = {}
        self._app_event_filter_installed = False

        self.settings_button = QToolButton(self.viewport())
        self.settings_button.setCursor(Qt.PointingHandCursor)
        self.settings_button.setFixedSize(32, 32)
        self.settings_button.setIcon(create_settings_gear_icon())
        self.settings_button.setIconSize(QSize(18, 18))
        self.settings_button.setStyleSheet("""
            QToolButton {
                background-color: rgba(60, 60, 60, 170);
                border: 1px solid rgba(255, 255, 255, 45);
                border-radius: 16px;
            }
            QToolButton:hover {
                background-color: rgba(80, 80, 80, 220);
            }
            QToolButton:pressed {
                background-color: rgba(100, 100, 100, 255);
            }
        """)
        self.settings_button.clicked.connect(self._toggle_settings_popup)

        self.settings_popup = QFrame(self.viewport())
        self.settings_popup.setObjectName("dsaSettingsPopup")
        self.settings_popup.hide()
        self.settings_popup.setStyleSheet(f"""
            QFrame#dsaSettingsPopup {{
                background-color: rgba(28, 28, 28, 245);
                border: 1px solid rgba(255, 255, 255, 35);
                border-radius: 8px;
            }}
            QLabel {{
                color: white;
                font-size: {max(config.FONT_SIZE - 2, 8)}px;
                font-weight: 600;
            }}
            QPushButton {{
                background-color: rgba(60, 60, 60, 170);
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: {max(config.FONT_SIZE, 8)}px;
                padding: 0px;
                text-align: center;
            }}
            QPushButton[role="stepper"] {{
                padding: 0px;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: rgba(80, 80, 80, 210);
            }}
            QLabel[role="value"] {{
                border-radius: 4px;
                font-size: {max(config.FONT_SIZE - 1, 8)}px;
                font-weight: bold;
                padding: 6px 8px;
            }}
        """)

        popup_layout = QVBoxLayout(self.settings_popup)
        popup_layout.setContentsMargins(10, 10, 10, 10)
        popup_layout.setSpacing(8)
        popup_layout.addLayout(self._create_step_control_row(
            "Max Freq",
            "max_freq_hz",
            lambda value: f"{value} Hz",
            lambda: self._adjust_max_freq(-1),
            lambda: self._adjust_max_freq(1),
        ))
        popup_layout.addLayout(self._create_step_control_row(
            "Max Power",
            "psd_db_max",
            lambda value: f"{value} dB",
            lambda: self._adjust_psd_level("psd_db_max", -1),
            lambda: self._adjust_psd_level("psd_db_max", 1),
        ))
        popup_layout.addLayout(self._create_step_control_row(
            "Min Power",
            "psd_db_min",
            lambda value: f"{value} dB",
            lambda: self._adjust_psd_level("psd_db_min", -1),
            lambda: self._adjust_psd_level("psd_db_min", 1),
        ))

        self._sync_settings_labels()
        self._update_settings_button_pos()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            self._app_event_filter_installed = True

    def _create_step_control_row(self, title, field_name, formatter, minus_handler, plus_handler):
        from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout

        block = QVBoxLayout()
        block.setContentsMargins(0, 0, 0, 0)
        block.setSpacing(4)
        block.addWidget(QLabel(title))

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        stepper_font = QFont()
        stepper_font.setBold(True)
        stepper_font.setPixelSize(max(config.FONT_SIZE + 4, 12))

        minus_button = QPushButton("−")
        minus_button.setProperty("role", "stepper")
        minus_button.setFixedSize(36, 36)
        minus_button.setFont(stepper_font)
        minus_button.setText("")
        minus_button.setIcon(create_stepper_icon("minus"))
        minus_button.setIconSize(QSize(20, 20))
        minus_button.clicked.connect(minus_handler)
        row.addWidget(minus_button)

        value_label = QLabel()
        value_label.setProperty("role", "value")
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setFixedWidth(70)
        row.addWidget(value_label)
        self.dsa_value_labels[field_name] = (value_label, formatter)

        plus_button = QPushButton("+")
        plus_button.setProperty("role", "stepper")
        plus_button.setFixedSize(36, 36)
        plus_button.setFont(stepper_font)
        plus_button.setText("")
        plus_button.setIcon(create_stepper_icon("plus"))
        plus_button.setIconSize(QSize(20, 20))
        plus_button.clicked.connect(plus_handler)
        row.addWidget(plus_button)

        block.addLayout(row)
        return block

    def _sync_settings_labels(self):
        for field_name, (label, formatter) in self.dsa_value_labels.items():
            label.setText(formatter(getattr(self.user_config, field_name)))

    def _toggle_settings_popup(self):
        if self.settings_popup.isVisible():
            self.settings_popup.hide()
            return

        self._sync_settings_labels()
        self._position_settings_popup()
        self.settings_popup.show()
        self.settings_popup.raise_()

    def _position_settings_popup(self):
        self.settings_popup.adjustSize()
        popup_x = max(0, self.settings_button.x() - self.settings_popup.width() - 6)
        popup_y = max(0, self.settings_button.y())
        self.settings_popup.move(popup_x, popup_y)

    def _update_settings_button_pos(self):
        if not hasattr(self, "settings_button"):
            return

        x = max(0, self.viewport().width() - self.settings_button.width() - 12)
        self.settings_button.move(x, 8)
        if self.settings_popup.isVisible():
            self._position_settings_popup()

    def eventFilter(self, obj, event):
        settings_popup = getattr(self, "settings_popup", None)
        if settings_popup is not None and event.type() == QEvent.MouseButtonPress and settings_popup.isVisible():
            from PySide6.QtWidgets import QApplication

            target = QApplication.widgetAt(event.globalPosition().toPoint())
            if not self._settings_click_is_inside(target):
                settings_popup.hide()

        return super().eventFilter(obj, event)

    def _settings_click_is_inside(self, target):
        if target is None:
            return False

        return (
            target is self.settings_popup
            or target is self.settings_button
            or self.settings_popup.isAncestorOf(target)
            or self.settings_button.isAncestorOf(target)
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_settings_button_pos()

    def closeEvent(self, event):
        if getattr(self, "_app_event_filter_installed", False):
            from PySide6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(self)
            self._app_event_filter_installed = False
        super().closeEvent(event)

    def _adjust_psd_level(self, field_name, delta):
        current_value = getattr(self.user_config, field_name)
        min_bound, max_bound = getattr(config, f"{field_name.upper()}_BOUNDS")
        new_value = current_value + delta

        if not (min_bound <= new_value <= max_bound):
            return

        updated_fields = {field_name: new_value}
        if field_name == "psd_db_min" and new_value >= self.user_config.psd_db_max:
            return
        if field_name == "psd_db_max" and new_value <= self.user_config.psd_db_min:
            return

        self.on_config_change(self.user_config.update(**updated_fields))

    def _init_gestures(self):
        self.grabGesture(Qt.PinchGesture)

    def append(self, ts, psd):
        if psd is not None:
            self.dsa_buffer.append(ts, psd)

    def _visible_width_sec(self):
        return self.display_minutes * 60.0

    def _target_resolution(self, visible_width_sec, divisor):
        return max(config.TIME_RESOLUTION, visible_width_sec / divisor)

    def _render_grid(self, visible_width_sec, target_divisor=2160.0):
        target_resolution = self._target_resolution(visible_width_sec, target_divisor)
        actual_res = min(self.dsa_buffer.RESOLUTIONS, key=lambda x: abs(x - target_resolution))
        n_columns = max(1, int(visible_width_sec / actual_res))
        return actual_res, n_columns

    def _live_window_start(self, visible_width_sec):
        if self.dsa_buffer.t0 is None:
            return self.dsa_buffer.get_oldest_timestamp()

        actual_res, n_columns = self._render_grid(visible_width_sec)
        buf = self.dsa_buffer.buffers[actual_res]
        last_slot = buf["last_slot"]
        if last_slot is None:
            return self.dsa_buffer.get_oldest_timestamp()

        oldest_slot = self.dsa_buffer._get_oldest_slot(actual_res) or 0
        start_slot = max(oldest_slot, last_slot - n_columns + 1)
        return self.dsa_buffer.t0 + start_slot * actual_res

    def _pan_limits(self, visible_width_sec):
        min_offset = self.dsa_buffer.get_oldest_timestamp()
        max_offset = max(min_offset, self._live_window_start(visible_width_sec))
        return min_offset, max_offset

    def _current_pan_start(self, visible_width_sec):
        min_offset, max_offset = self._pan_limits(visible_width_sec)
        if self.live_mode:
            return max_offset
        return float(np.clip(self._pan_sec, min_offset, max_offset))

    def _sync_pan_window(self, visible_width_sec):
        min_offset, max_offset = self._pan_limits(visible_width_sec)

        if self.live_mode:
            self._pan_sec = max_offset
            return

        self._pan_sec = float(np.clip(self._pan_sec, min_offset, max_offset))
        if not self._dragging and self._pan_sec >= max_offset - 0.05:
            self.live_mode = True
            self._pan_sec = max_offset

    def _get_visible_dsa_data(self, target_divisor):
        visible_width_sec = self._visible_width_sec()
        self._sync_pan_window(visible_width_sec)

        self.t0, self.dsa_rect, actual_res = self.dsa_buffer.get_view_at(
            width=visible_width_sec,
            height=len(self.freq_bins),
            pan_sec=self._pan_sec,
            target_resolution=self._target_resolution(visible_width_sec, target_divisor),
        )
        return visible_width_sec, self._psd_to_db(self.dsa_rect), actual_res

    def _psd_to_db(self, data):
        data = np.ascontiguousarray(data)
        np.maximum(data, np.finfo(np.float32).eps, out=data)
        np.log10(data, out=data)
        data *= 10
        return data

    # ------------------ Update & Rendering ------------------ #
    def update(self):
        self._last_render = time.time()
        visible_width_sec, data, actual_res = self._get_visible_dsa_data(2160.0)

        levels = (self.user_config.psd_db_min, self.user_config.psd_db_max)
        if levels != self._last_levels:
            self.image.setLevels(levels, update=False)
            self.colorbar.setLevels(levels)
            self._last_levels = levels

        self.image.setImage(data, nan_policy="omit", autoLevels=False)
        self.image.setRect((self.t0, config.LOWEST_FREQ_HZ, actual_res * data.shape[0], self.user_config.max_freq_hz))
        self.plot.setXRange(self.t0, self.t0 + visible_width_sec, padding=0)

    def calibrate(self):
        _, data, _ = self._get_visible_dsa_data(1000.0)
        data = data[~np.isnan(data)]
        if len(data) > 0:
            data.sort()
            n_1_percent = max(1, int(len(data) * 0.001))
            psd_db_min = np.mean(data[:n_1_percent])
            psd_db_max = np.mean(data[-n_1_percent:])
            psd_db_min = np.clip(int(psd_db_min), config.PSD_DB_MIN_BOUNDS[0], config.PSD_DB_MIN_BOUNDS[1])
            psd_db_max = np.clip(int(psd_db_max), config.PSD_DB_MAX_BOUNDS[0], config.PSD_DB_MAX_BOUNDS[1])
            new_config = self.user_config.update(psd_db_min=psd_db_min, psd_db_max=psd_db_max)
            self.on_config_change(new_config)

    def jump_to_live(self):
        self.live_mode = True
        if self.dsa_buffer.t0 is not None:
            self._pan_sec = self._current_pan_start(self._visible_width_sec())
            self.update()

    def pan(self, delta_percent):
        """External pan control"""
        if self.dsa_buffer.t0 is None:
            return

        if delta_percent == "live":
            self.jump_to_live()
            return

        visible_width_sec = self._visible_width_sec()
        current_start = self._current_pan_start(visible_width_sec)
        self.live_mode = False
        self._pan_sec = current_start + visible_width_sec * delta_percent

        self.update()

    def clear_data(self):
        """Delete all buffered data"""
        self.dsa_buffer = DSABuffer()
        self.live_mode = True
        self._pan_sec = 0.0
        self.update()

    # ------------------ Mouse & Gesture ------------------ #

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)
        if ev.isAccepted():
            return
        if ev.button() == Qt.LeftButton:
            self._dragging = True
            self._last_mouse_pos = ev.pos()
            ev.accept()

    def mouseMoveEvent(self, ev):
        if self.dsa_buffer.t0 is not None and self._dragging and self._last_mouse_pos is not None:
            delta = ev.pos() - self._last_mouse_pos
            self._last_mouse_pos = ev.pos()
            visible_width_sec = self._visible_width_sec()
            width_px = self.plot.width()
            dt = (delta.x() / width_px) * visible_width_sec if width_px else 0
            self._pan_sec = self._current_pan_start(visible_width_sec) - dt
            self.live_mode = False
            self.update()

            ev.accept()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if ev.isAccepted():
            return
        if self._dragging:
            self._dragging = False
            self._last_mouse_pos = None
            if self.dsa_buffer.t0 is not None:
                self.update()
            ev.accept()

    def event(self, ev):
        if ev.type() == QEvent.Gesture:
            return self.gestureEvent(ev)
        return super().event(ev)

    def gestureEvent(self, ev):
        pinch = ev.gesture(Qt.PinchGesture)
        if pinch:
            self.handlePinch(pinch)
            return True
        return False

    def handlePinch(self, pinch):
        if pinch.state() == Qt.GestureUpdated:
            new_minutes = self.display_minutes / pinch.scaleFactor()
            min_m, max_m = config.DISPLAY_MINUTES_BOUNDS
            new_minutes = np.clip(new_minutes, min_m, max_m)
            self.on_zoom_change(new_minutes)
            self.update()
        return True

    def apply_zoom(self, new_minutes):
        if new_minutes is not None:
            old_width = self._visible_width_sec()
            displayed_start = self.t0
            old_res, old_columns = self._render_grid(old_width)
            new_width = new_minutes * 60.0
            new_res, new_columns = self._render_grid(new_width)
            self.display_minutes = new_minutes

            if self.dsa_buffer.t0 is not None:
                min_offset, max_offset = self._pan_limits(new_width)
                if self.live_mode:
                    self._pan_sec = max_offset
                elif old_res == new_res and old_columns == new_columns:
                    self._pan_sec = float(np.clip(displayed_start, min_offset, max_offset))
                else:
                    center = displayed_start + old_width / 2.0
                    self._pan_sec = float(np.clip(center - new_width / 2.0, min_offset, max_offset))
        self.update()

    def apply_config(self, new_config):
        old_config = self.user_config
        self.user_config = new_config
        if new_config.max_freq_hz != old_config.max_freq_hz:
            self.freq_bins = config.FREQ_BINS[(config.FREQ_BINS <= self.user_config.max_freq_hz)]
            self._update_y_axis()

        self._sync_settings_labels()
        self.update()



class PSDView(pg.PlotWidget):
    def __init__(self, user_config, on_config_change=None):
        super().__init__()
        self.user_config = user_config
        self.on_config_change = on_config_change
        self._last_psd = None

        set_axis_label(self.plotItem, "bottom", "Frequency", units="Hz")
        set_axis_label(self.plotItem, "left", "Power", units="dB")
        set_uniform_left_axis_width(self.plotItem)
        self.getPlotItem().setContentsMargins(10, 10, 0, 5)
        self.setMinimumHeight(config.MIN_PSD_HEIGHT)
        self.setMenuEnabled(False)
        self.showGrid(x=True, y=True)
        self.setMouseEnabled(x=False, y=False)

        self.curve = self.plot(pen=pg.mkPen("y", width=2), title="PSD")

        self.setInteractive(False)
        self.apply_config(user_config)

    def update(self, psd):
        self._last_psd = np.asarray(psd, dtype=np.float32)
        psd_db = 10 * np.log10(np.clip(psd, np.finfo(np.float32).eps, None))
        self.curve.setData(config.FREQ_BINS, psd_db)

    def apply_config(self, user_config):
        self.user_config = user_config
        self.setXRange(config.LOWEST_FREQ_HZ, user_config.max_freq_hz, padding=0)
        self.setYRange(user_config.psd_db_min - 5, user_config.psd_db_max + 5, padding=0)
        if self._last_psd is not None:
            self.update(self._last_psd)



# 7.5 mm/sekunde 15 mm/sekunde eeg view skalieren mit application window. 5 microvolt pro millimeter. 27 zoll pc. einstellung in system settings
class EEGView(pg.PlotWidget):
    """Real-time circular EEG display with smooth sweep line + gap (optimized)."""

    RENDER_HZ = 20

    def __init__(self, user_config, on_config_change=None):
        super().__init__()
        self.user_config = user_config
        self.on_config_change = on_config_change

        # --- Plot setup ---
        self.getPlotItem().setContentsMargins(10, 10, 60, 10)
        set_axis_label(self.plotItem, "left", "EEG", units="\N{MICRO SIGN}V")
        set_axis_label(self.plotItem, "bottom", "Time", units="s")
        set_uniform_left_axis_width(self.plotItem)
        self.setMinimumHeight(config.MIN_EEG_HEIGHT)
        self.showGrid(x=False, y=False)
        self.grid = GridItem()
        self.addItem(self.grid)
        self.grid.setZValue(-1)
        self.grid.setTickSpacing(
            x=[1.0],  # 1 second major grid
            y=[50.0]  # optional amplitude grid (50 µV for EEG)
        )
        self.grid.setTextPen(None)
        self.getAxis('bottom').setTickPen(None)
        self.getAxis('left').setTickPen(None)

        self.setMenuEnabled(False)
        self.setClipToView(True)
        self.setDownsampling(auto=False, ds=1, mode="subsample")
        self.setMouseEnabled(False, False)
        self.setInteractive(False)

        self.curve_a = self.plot(pen=pg.mkPen((0, 200, 255), width=1))
        self.missing_curve_a = self.plot(pen=pg.mkPen((255, 80, 80), width=2))
        self.missing_curve_a.setDownsampling(auto=False, ds=1, method="subsample")

        # Sweep line
        self.update_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("w", style=Qt.DashLine))
        self.addItem(self.update_line)
        self._init_settings_controls()

        # --- Buffer setup ---
        self.N = int(config.EEG_VIEW_WINDOW_SEC * config.SAMPLE_RATE_HZ)

        self.display = np.full(self.N, np.nan, dtype=np.float32)
        self.display_head = -1
        self.history = deque(maxlen=5 * 60 * config.SAMPLE_RATE_HZ) # 5 minutes history

        self.x = np.linspace(
            0,
            config.EEG_VIEW_WINDOW_SEC,
            self.N,
            endpoint=False,
            dtype=np.float32,
        )

        # Gap size (in samples)
        self.gap_samples = int(0.05 * config.EEG_VIEW_WINDOW_SEC * config.SAMPLE_RATE_HZ)

        # --- Timing ---
        self._pending = deque()
        self._sample_period = 1.0 / config.SAMPLE_RATE_HZ
        self._last_rendered_head = -1
        self.seconds_visible = config.EEG_VIEW_WINDOW_SEC
        self._init_view_filter()

        # --- View limits ---
        self.setXRange(0, config.EEG_VIEW_WINDOW_SEC, padding=0)
        self._apply_y_range()

        # --- Timer ---
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._render_frame_stable)
        self._timer.start(int(1000 / self.RENDER_HZ))

    def _init_settings_controls(self):
        from PySide6.QtWidgets import QApplication, QFrame, QLabel, QStyle, QToolButton, QVBoxLayout

        self.sweep_buttons = {}
        self.amplitude_buttons = {}
        self.is_paused = False
        self._app_event_filter_installed = False

        self.settings_button = QToolButton(self.viewport())
        self.settings_button.setCursor(Qt.PointingHandCursor)
        self.settings_button.setFixedSize(32, 32)
        self.settings_button.setIcon(create_settings_gear_icon())
        self.settings_button.setIconSize(QSize(18, 18))
        self.settings_button.setStyleSheet("""
            QToolButton {
                background-color: rgba(60, 60, 60, 170);
                border: 1px solid rgba(255, 255, 255, 45);
                border-radius: 16px;
            }
            QToolButton:hover {
                background-color: rgba(80, 80, 80, 220);
            }
            QToolButton:pressed {
                background-color: rgba(100, 100, 100, 255);
            }
        """)
        self.settings_button.clicked.connect(self._toggle_settings_popup)

        self.pause_button = QToolButton(self.viewport())
        self.pause_button.setCursor(Qt.PointingHandCursor)
        self.pause_button.setCheckable(True)
        self.pause_button.setFixedSize(32, 32)
        self.pause_button.setStyleSheet("""
            QToolButton {
                background-color: rgba(60, 60, 60, 170);
                border: 1px solid rgba(255, 255, 255, 45);
                border-radius: 16px;
            }
            QToolButton:hover {
                background-color: rgba(80, 80, 80, 220);
            }
            QToolButton:pressed {
                background-color: rgba(100, 100, 100, 255);
            }
            QToolButton:checked {
                background-color: rgba(0, 150, 220, 220);
            }
        """)
        self.pause_button.clicked.connect(self._toggle_pause)
        self._pause_icon = self.style().standardIcon(QStyle.SP_MediaPause)
        self._play_icon = self.style().standardIcon(QStyle.SP_MediaPlay)
        self._sync_pause_button()

        self.settings_popup = QFrame(self.viewport())
        self.settings_popup.setObjectName("eegSettingsPopup")
        self.settings_popup.hide()
        self.settings_popup.setStyleSheet(f"""
            QFrame#eegSettingsPopup {{
                background-color: rgba(28, 28, 28, 245);
                border: 1px solid rgba(255, 255, 255, 35);
                border-radius: 8px;
            }}
            QLabel {{
                color: white;
                font-size: {max(config.FONT_SIZE - 2, 8)}px;
                font-weight: 600;
            }}
        """)

        popup_layout = QVBoxLayout(self.settings_popup)
        popup_layout.setContentsMargins(10, 10, 10, 10)
        popup_layout.setSpacing(8)

        popup_layout.addWidget(QLabel("Sweep Speed"))
        popup_layout.addLayout(
            self._create_settings_button_row(
                config.EEG_SWEEP_SPEED_OPTIONS,
                self.sweep_buttons,
                self._set_eeg_sweep_speed,
            )
        )

        popup_layout.addWidget(QLabel("Y Range"))
        popup_layout.addLayout(
            self._create_settings_button_row(
                config.EEG_Y_RANGE_OPTIONS,
                self.amplitude_buttons,
                self._set_eeg_y_max,
            )
        )

        self._sync_sweep_buttons()
        self._sync_amplitude_buttons()
        self._update_settings_button_pos()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            self._app_event_filter_installed = True

    def _create_settings_button_row(self, options, button_map, handler):
        from PySide6.QtWidgets import QHBoxLayout, QPushButton

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        for option in options:
            label = f"{option:g}" if isinstance(option, float) else str(option)
            button = QPushButton(label)
            button.setCheckable(True)
            button.setFixedSize(50, 32)
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgba(60, 60, 60, 170);
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: {max(config.FONT_SIZE - 3, 8)}px;
                }}
                QPushButton:hover {{
                    background-color: rgba(80, 80, 80, 210);
                }}
                QPushButton:checked {{
                    background-color: rgba(0, 150, 220, 220);
                }}
            """)
            button.clicked.connect(lambda _, value=option: handler(value))
            layout.addWidget(button)
            button_map[option] = button

        return layout

    def _toggle_settings_popup(self):
        if self.settings_popup.isVisible():
            self.settings_popup.hide()
            return

        self._sync_sweep_buttons()
        self._sync_amplitude_buttons()
        self._position_settings_popup()
        self.settings_popup.show()
        self.settings_popup.raise_()

    def _position_settings_popup(self):
        self.settings_popup.adjustSize()
        popup_x = max(0, self.settings_button.x() - self.settings_popup.width() - 6)
        popup_y = max(0, self.settings_button.y())
        self.settings_popup.move(popup_x, popup_y)

    def _toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self._pending.clear()
        self._sync_pause_button()

    def _init_view_filter(self):
        self._view_filter = DSACalculator(self.user_config.window_sec)
        self._view_filter_b = np.asarray(self._view_filter.bp_b, dtype=np.float32)
        self._view_filter_settle_samples = max(0, (len(self._view_filter_b) - 1) // 2)
        self._reset_view_filter_state()

    def _reset_view_filter_state(self):
        self._view_filter_zi = np.zeros(len(self._view_filter_b) - 1, dtype=np.float32)
        self._view_filter_settle_remaining = self._view_filter_settle_samples

    def _sync_pause_button(self):
        self.pause_button.blockSignals(True)
        self.pause_button.setChecked(self.is_paused)
        self.pause_button.setIcon(self._play_icon if self.is_paused else self._pause_icon)
        self.pause_button.setIconSize(QSize(18, 18))
        self.pause_button.setToolTip("Resume EEG" if self.is_paused else "Pause EEG")
        self.pause_button.blockSignals(False)

    def _update_settings_button_pos(self):
        if not hasattr(self, "settings_button"):
            return

        x = max(0, self.viewport().width() - self.settings_button.width() - 12)
        self.settings_button.move(x, 8)
        if hasattr(self, "pause_button"):
            self.pause_button.move(x, self.settings_button.y() + self.settings_button.height() + 6)
        if self.settings_popup.isVisible():
            self._position_settings_popup()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress and self.settings_popup.isVisible():
            from PySide6.QtWidgets import QApplication

            target = QApplication.widgetAt(event.globalPosition().toPoint())
            if not self._settings_click_is_inside(target):
                self.settings_popup.hide()

        return super().eventFilter(obj, event)

    def _settings_click_is_inside(self, target):
        if target is None:
            return False

        return (
            target is self.settings_popup
            or target is self.settings_button
            or target is self.pause_button
            or self.settings_popup.isAncestorOf(target)
            or self.settings_button.isAncestorOf(target)
            or self.pause_button.isAncestorOf(target)
        )

    def _apply_y_range(self):
        max_uv = self.user_config.eeg_uv_range_max
        self.setYRange(-max_uv, max_uv, padding=0)

    def _sync_amplitude_buttons(self):
        selected = self.user_config.eeg_uv_range_max
        for amplitude, button in self.amplitude_buttons.items():
            button.blockSignals(True)
            button.setChecked(amplitude == selected)
            button.blockSignals(False)

    def _sync_sweep_buttons(self):
        selected = self.user_config.eeg_sweep_speed
        for speed, button in self.sweep_buttons.items():
            button.blockSignals(True)
            button.setChecked(speed == selected)
            button.blockSignals(False)

    def _set_eeg_y_max(self, max_uv):
        if max_uv == self.user_config.eeg_uv_range_max:
            self._sync_amplitude_buttons()
            return

        if self.on_config_change is None:
            self.user_config = self.user_config.update(eeg_uv_range_max=max_uv)
            self._apply_y_range()
            self._sync_amplitude_buttons()
            return

        self.on_config_change(self.user_config.update(eeg_uv_range_max=max_uv))

    def _set_eeg_sweep_speed(self, sweep_speed):
        if sweep_speed == self.user_config.eeg_sweep_speed:
            self._sync_sweep_buttons()
            return

        if self.on_config_change is None:
            self.user_config = self.user_config.update(eeg_sweep_speed=sweep_speed)
            self._sync_sweep_buttons()
            self._update_time_scale()
            return

        self.on_config_change(self.user_config.update(eeg_sweep_speed=sweep_speed))


    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._update_time_scale)

    def _update_time_scale(self):
        if not hasattr(self, "plotItem"):
            return

        win = self.window()
        if win is None:
            return

        handle = win.windowHandle()
        if handle is None:
            return

        screen = handle.screen()
        dpi = screen.logicalDotsPerInch() if screen else 96

        px_per_sec = self.user_config.eeg_sweep_speed * dpi / 25.4

        try:
            view_box = self.getViewBox()
        except AttributeError:
            return
        if view_box is None:
            return

        # Use the actual drawable plot width, not the outer PlotWidget width.
        width_px = view_box.sceneBoundingRect().width()
        if width_px <= 0:
            return

        seconds_visible = width_px / px_per_sec

        new_N = int(seconds_visible * config.SAMPLE_RATE_HZ)

        if new_N <= 10:
            return

        if new_N != self.N:
            # 1. Preserve the relative position of the sweep line
            if self.display_head >= 0:
                old_ratio = (self.display_head + 1) / self.N
            else:
                old_ratio = 0.0

            new_head = int(old_ratio * new_N) - 1
            new_head = np.clip(new_head, -1, new_N - 1)

            # 2. Create fresh buffer
            new_display = np.full(new_N, np.nan, dtype=np.float32)

            # 3. Populate from history
            if self.history:
                h_list = list(self.history)
                h_arr = np.array(h_list, dtype=np.float32)
                M = len(h_arr)

                # We want latest sample (h_arr[-1]) at new_display[new_head]
                # We can fill up to new_N samples.
                count = min(M, new_N)
                to_copy = h_arr[-count:]

                start_idx = (new_head - count + 1) % new_N

                if start_idx + count <= new_N:
                    new_display[start_idx : start_idx + count] = to_copy
                else:
                    first_part = new_N - start_idx
                    new_display[start_idx:] = to_copy[:first_part]
                    new_display[: count - first_part] = to_copy[first_part:]

            self.display = new_display
            self.display_head = new_head
            self.N = new_N
            self._last_rendered_head = -1
            # Gap size (in samples)
            self.gap_samples = int(0.05 * self.N)

        self.seconds_visible = seconds_visible

        self.setXRange(0, seconds_visible, padding=0)

        self.x = np.linspace(
            0,
            seconds_visible,
            self.N,
            endpoint=False,
            dtype=np.float32,
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_time_scale()
        self._update_settings_button_pos()

    def closeEvent(self, event):
        self._timer.stop()
        if getattr(self, "_app_event_filter_installed", False):
            from PySide6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(self)
            self._app_event_filter_installed = False
        super().closeEvent(event)

    def clear_data(self):
        self.history.clear()
        self.display.fill(np.nan)
        self.display_head = -1
        self._pending.clear()
        self._last_rendered_head = -1
        self._reset_view_filter_state()
        self.update_line.setPos(0)
        self.curve_a.clear()
        self.missing_curve_a.clear()

    def apply_config(self, user_config):
        self.user_config = user_config
        self._apply_y_range()
        self._sync_sweep_buttons()
        self._sync_amplitude_buttons()
        self._update_time_scale()

    # ------------------------------------------------------------------
    # Data input
    # ------------------------------------------------------------------
    def append_sample(self, val):
        if val is None:
            return

        if self.is_paused:
            return

        sample = float(val)
        if not np.isfinite(sample):
            sample = np.nan
            self._reset_view_filter_state()
        else:
            filtered_sample, self._view_filter_zi = lfilter(
                self._view_filter_b,
                [1.0],
                np.asarray([sample], dtype=np.float32),
                zi=self._view_filter_zi,
            )
            if self._view_filter_settle_remaining > 0:
                self._view_filter_settle_remaining -= 1
                sample = np.nan
            else:
                sample = float(filtered_sample[0])

        now = time.perf_counter()
        last = self._pending[-1][0] if self._pending else now
        scheduled = max(last + self._sample_period, now)
        self._pending.append((scheduled, sample))

    def _build_visible_curve_data(self, head):
        visible_mask = np.zeros(self.N, dtype=bool)
        gap_end = (head + self.gap_samples) % self.N
        if head < gap_end:
            visible_mask[gap_end:] = True
            visible_mask[: head + 1] = True
        else:
            visible_mask[gap_end: head + 1] = True

        visible_values = np.full(self.N, np.nan, dtype=np.float32)
        visible_values[visible_mask] = self.display[visible_mask]

        missing_mask = visible_mask & np.isnan(self.display)

        valid_values = np.array(visible_values, copy=True)
        valid_values[missing_mask] = np.nan
        return valid_values, missing_mask

    def _build_missing_curve_segments(self, missing_mask):
        sample_width = np.float32(self.seconds_visible / self.N)
        half_sample_width = np.float32(sample_width / 2.0)
        missing_x = []
        missing_y = []
        idx = 0

        while idx < self.N:
            if not missing_mask[idx]:
                idx += 1
                continue

            run_start = idx
            while idx < self.N and missing_mask[idx]:
                idx += 1
            run_end = idx - 1

            missing_x.append(max(0.0, self.x[run_start] - half_sample_width))
            missing_y.append(0.0)
            missing_x.append(min(self.seconds_visible, self.x[run_end] + half_sample_width))
            missing_y.append(0.0)
            missing_x.append(np.nan)
            missing_y.append(np.nan)

        if not missing_x:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        return (
            np.asarray(missing_x, dtype=np.float32),
            np.asarray(missing_y, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render_frame_stable(self):
        if not self._pending and self.display_head < 0:
            return

        now = time.perf_counter()
        changed = False

        while self._pending and self._pending[0][0] <= now:
            _, v = self._pending.popleft()
            self.display_head = (self.display_head + 1) % self.N
            self.display[self.display_head] = v
            self.history.append(v)
            changed = True

        if self._pending:
            next_wall = self._pending[0][0]
            prev_wall = next_wall - self._sample_period
            frac = (now - prev_wall) / self._sample_period
            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0
            interp_head = (self.display_head + frac) % self.N
        else:
            if self.display_head < 0:
                return
            interp_head = float(self.display_head)

        self.update_line.setPos((interp_head / self.N) * self.seconds_visible)

        head = int(interp_head) % self.N
        if not changed and head == self._last_rendered_head:
            return

        self._last_rendered_head = head
        valid_values, missing_mask = self._build_visible_curve_data(head)
        missing_x, missing_y = self._build_missing_curve_segments(missing_mask)
        self.curve_a.setData(self.x, valid_values, connect="finite")
        if len(missing_x) == 0:
            self.missing_curve_a.clear()
        else:
            self.missing_curve_a.setData(missing_x, missing_y, connect="finite")
