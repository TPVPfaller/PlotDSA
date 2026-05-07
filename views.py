import datetime
from collections import deque

import numpy as np
import pyqtgraph as pg
from pyqtgraph import ColorBarItem, GridItem
from PySide6.QtCore import Qt, QEvent, QTimer
import time

import math
from buffers import DSABuffer
import config


# ------------------ DSA View ------------------ #
class FrequencyAxis(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_freq = None

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
        self.plot.setLabel("left", "Frequency", units="Hz")
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.showGrid(x=False, y=False)
        self.plot.invertY(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self._update_y_axis()
        self.plot.setContentsMargins(20, 20, 0, 0)
        self.image = pg.ImageItem(axisOrder='col-major', interpolation="linear")
        self.plot.addItem(self.image)

        self._init_freq_buttons()

    def _update_y_axis(self):
        max_f = self.user_config.max_freq_hz
        self.plot.setYRange(config.LOWEST_FREQ_HZ, max_f, padding=0)
        self.freq_axis.set_max_freq(max_f)

    def _init_freq_buttons(self):
        from PySide6.QtWidgets import QPushButton, QFrame, QVBoxLayout, QGraphicsProxyWidget

        # Create a container widget for the buttons
        self.btn_container = QFrame()
        self.btn_container.setStyleSheet("background: transparent; border: none;")
        layout = QVBoxLayout(self.btn_container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.btn_plus = QPushButton("+")
        self.btn_minus = QPushButton("-")

        # Explicitly set parent to ensure they aren't garbage collected
        # though QGraphicsProxyWidget should handle it.
        self.btn_plus.setParent(self.btn_container)
        self.btn_minus.setParent(self.btn_container)

        for btn in [self.btn_plus, self.btn_minus]:
            self._style_adjust_button(btn)
            layout.addWidget(btn)

        self.btn_plus.clicked.connect(lambda: self._adjust_max_freq(1))
        self.btn_minus.clicked.connect(lambda: self._adjust_max_freq(-1))

        # Add the container to the plot using a proxy
        self.proxy = QGraphicsProxyWidget()
        self.proxy.setWidget(self.btn_container)
        self.proxy.setParentItem(self.plot.vb)
        self.plot.vb.setFlag(pg.GraphicsWidget.ItemClipsChildrenToShape, False)

        # Position it at the top-left of the viewbox
        # We'll update the position when the viewbox is resized
        self.plot.vb.sigResized.connect(self._update_button_pos)
        self._update_button_pos()

    def _update_button_pos(self):
        # Position at the left of the Y-axis
        # The ViewBox (vb) is the plotting area.
        # To move buttons to the left of it (where the Y-axis is), we use a negative X.
        # -45 seems like a good offset to clear the axis labels/ticks.
        self.proxy.setPos(-52, -16)
        self.proxy.setZValue(100)  # Ensure it's above other elements

    def _style_adjust_button(self, button):
        button.setFixedSize(32, 32)
        button.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 60, 60, 150);
                color: white;
                border-radius: 4px;
                font-weight: bold;
                font-size: 22px;
            }
            QPushButton:hover {
                background-color: rgba(80, 80, 80, 200);
            }
            QPushButton:pressed {
                background-color: rgba(100, 100, 100, 255);
            }
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
        from PySide6.QtWidgets import QPushButton, QFrame, QVBoxLayout, QGraphicsProxyWidget

        self.colorbar = ColorBarItem(
            values=(self.user_config.psd_db_min, self.user_config.psd_db_max),
            colorMap=self.cmap,
            label="Power (dB)",
            interactive=False,
        )
        self.colorbar.setImageItem(self.image)
        self.colorbar.setContentsMargins(0,24,0,24)
        self.addItem(self.colorbar, row=0, col=1)
        self.ci.layout.setColumnStretchFactor(0, 10)
        self.ci.layout.setColumnStretchFactor(1, 1)
        self.ci.layout.setContentsMargins(0, 0, 40, 0)

        self.colorbar_max_btn_proxy = self._create_colorbar_button_group(
            plus_handler=lambda: self._adjust_psd_level("psd_db_max", 1),
            minus_handler=lambda: self._adjust_psd_level("psd_db_max", -1),
        )
        self.colorbar_min_btn_proxy = self._create_colorbar_button_group(
            plus_handler=lambda: self._adjust_psd_level("psd_db_min", 1),
            minus_handler=lambda: self._adjust_psd_level("psd_db_min", -1),
        )
        QTimer.singleShot(0, self._position_colorbar_buttons)

    def _create_colorbar_button_group(self, plus_handler, minus_handler):
        from PySide6.QtWidgets import QPushButton, QFrame, QVBoxLayout, QGraphicsProxyWidget

        container = QFrame()
        container.setStyleSheet("background: transparent; border: none;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        plus_button = QPushButton("+")
        minus_button = QPushButton("-")
        self._style_adjust_button(plus_button)
        self._style_adjust_button(minus_button)
        plus_button.clicked.connect(plus_handler)
        minus_button.clicked.connect(minus_handler)
        layout.addWidget(plus_button)
        layout.addWidget(minus_button)

        proxy = QGraphicsProxyWidget(self.colorbar)
        proxy.setWidget(container)
        proxy.setZValue(100)
        return proxy

    def _position_colorbar_buttons(self):
        if not hasattr(self, "colorbar_max_btn_proxy") or not hasattr(self, "colorbar_min_btn_proxy"):
            return

        rect = self.colorbar.boundingRect()
        top_size = self.colorbar_max_btn_proxy.boundingRect()
        bottom_size = self.colorbar_min_btn_proxy.boundingRect()
        x_pos = max(0, rect.width() - max(top_size.width(), bottom_size.width()) + 12)

        self.colorbar_max_btn_proxy.setPos(x_pos, 5)
        self.colorbar_min_btn_proxy.setPos(x_pos, max(0, rect.height() - bottom_size.height())-8)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_colorbar_buttons()

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

    # ------------------ Update & Rendering ------------------ #
    def update(self):
        self._last_render = time.time()
        visible_width_sec = self.display_minutes * 60.0
        n_time_bins = max(1, int(visible_width_sec / config.TIME_RESOLUTION))

        target_res = visible_width_sec / 2160.0 # When displaying more than 6 hours reduce resolution to 10s
        target_res = max(1.0, target_res)

        max_offset = self.dsa_buffer.get_newest_timestamp() - visible_width_sec
        min_offset = self.dsa_buffer.get_oldest_timestamp()

        if self.live_mode:
            self._pan_sec = max_offset
        else:
            self._pan_sec =np.clip(self._pan_sec, min_offset, max_offset)
            if self._pan_sec >= max_offset - 0.05:
                self.live_mode = True
                self._pan_sec = max_offset

        self.t0, self.dsa_rect, actual_res = self.dsa_buffer.get_view_at(
            width=n_time_bins, height=len(self.freq_bins), pan_sec=self._pan_sec,
            target_resolution=target_res
        )
        print(actual_res)
        data = np.ascontiguousarray(self.dsa_rect)
        np.maximum(data, np.finfo(np.float32).eps, out=data)
        np.log10(data, out=data)
        data *= 10

        levels = (self.user_config.psd_db_min, self.user_config.psd_db_max)
        if levels != self._last_levels:
            self.image.setLevels(levels, update=False)
            self.colorbar.setLevels(levels)
            self._last_levels = levels

        self.image.setImage(data, nan_policy="omit", autoLevels=False)
        self.image.setRect((self.t0, config.LOWEST_FREQ_HZ, actual_res * data.shape[0], self.user_config.max_freq_hz))
        self.plot.setXRange(self.t0, self.t0 + visible_width_sec, padding=0)

    def calibrate(self):
        visible_width_sec = self.display_minutes * 60.0
        n_time_bins = max(1, int(visible_width_sec / config.TIME_RESOLUTION))

        # Target resolution for calibration: use Level 0 (1s) if possible for accuracy,
        # but for performance if zoomed out, we can use the same logic as update.
        target_res = visible_width_sec / 1000.0
        target_res = max(1.0, target_res)

        max_offset = self.dsa_buffer.get_newest_timestamp() - visible_width_sec
        min_offset = self.dsa_buffer.get_oldest_timestamp()

        if self.live_mode:
            self._pan_sec = max_offset
        else:
            self._pan_sec = np.clip(self._pan_sec, min_offset, max_offset)
            if self._pan_sec >= max_offset - 0.05:
                self.live_mode = True
                self._pan_sec = max_offset

        _, self.dsa_rect, _ = self.dsa_buffer.get_view_at(
            width=n_time_bins, height=len(self.freq_bins), pan_sec=self._pan_sec,
            target_resolution=target_res
        )
        data = np.ascontiguousarray(self.dsa_rect)

        np.maximum(data, np.finfo(np.float32).eps, out=data)
        np.log10(data, out=data)
        data *= 10
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
            visible_width_sec = self.display_minutes * 60
            self._pan_sec = self.dsa_buffer.get_newest_timestamp() - visible_width_sec
            self.update()

    def pan(self, delta_percent):
        """External pan control"""
        if self.dsa_buffer.t0 is None:
            return

        if delta_percent == "live":
            self.jump_to_live()
            return

        self.live_mode = False
        self._pan_sec += self.display_minutes * delta_percent * 60
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
            visible_width_sec = self.display_minutes * 60
            width_px = self.plot.width()
            dt = (delta.x() / width_px) * visible_width_sec if width_px else 0
            if self._pan_sec == 0.0:
                self._pan_sec = self.dsa_buffer.t0
            self._pan_sec -= dt
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
            old_width = self.display_minutes * 60.0
            new_width = new_minutes * 60.0
            self.display_minutes = new_minutes

            # keep current center
            center = self._pan_sec + old_width / 2.0

            # recompute pan so center stays fixed
            self._pan_sec = center - new_width / 2.0

            if self.dsa_buffer.t0 is not None:
                max_offset = self.dsa_buffer.get_newest_timestamp() - new_width
                min_offset = self.dsa_buffer.get_oldest_timestamp()
                self._pan_sec = max(min_offset, min(self._pan_sec, max_offset))
        self.update()

    def apply_config(self, new_config):
        old_config = self.user_config
        self.user_config = new_config
        if new_config.max_freq_hz != old_config.max_freq_hz:
            self.freq_bins = config.FREQ_BINS[(config.FREQ_BINS <= self.user_config.max_freq_hz)]
            self._update_y_axis()

        self.update()



class PSDView(pg.PlotWidget):
    def __init__(self, user_config, on_config_change=None):
        super().__init__()
        self.user_config = user_config
        self.on_config_change = on_config_change

        self.setLabel("bottom", "Frequency", units="Hz")
        self.setLabel("left", "Power", units="dB")
        self.getPlotItem().setContentsMargins(10, 0, 0, 5)
        self.setMinimumHeight(config.MIN_PSD_HEIGHT)
        self.setMenuEnabled(False)
        self.showGrid(x=True, y=True)
        self.setMouseEnabled(x=False, y=False)

        self.curve = self.plot(pen=pg.mkPen("y", width=2), title="PSD")

        self.setInteractive(False)
        self.setYRange(user_config.psd_db_min - 5, user_config.psd_db_max + 5)

    def update(self, psd):
        psd_db = 10 * np.log10(np.clip(psd, np.finfo(np.float32).eps, None))
        self.curve.setData(config.FREQ_BINS, psd_db)



# 7.5 mm/sekunde 15 mm/sekunde eeg view skalieren mit application window. 5 microvolt pro millimeter. 27 zoll pc. einstellung in system settings
class EEGView(pg.PlotWidget):
    """Real-time circular EEG display with smooth sweep line + gap (optimized)."""

    RENDER_HZ = 20

    def __init__(self, user_config, on_config_change=None):
        super().__init__()
        self.user_config = user_config
        self.on_config_change = on_config_change

        # --- Plot setup ---
        self.setLabel("left", "EEG", units="µV")
        self.getPlotItem().setContentsMargins(10, 10, 60, 3)
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
        self.setDownsampling(mode="peak")
        self.setMouseEnabled(False, False)
        self.setInteractive(False)

        self.curve_a = self.plot(pen=pg.mkPen((0, 200, 255), width=1))
        self.curve_b = self.plot(pen=pg.mkPen((0, 200, 255), width=1))

        # Sweep line
        self.update_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("w", style=Qt.DashLine))
        self.addItem(self.update_line)
        self._init_amplitude_buttons()

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

        # --- View limits ---
        self.setXRange(0, config.EEG_VIEW_WINDOW_SEC, padding=0)
        self._apply_y_range()

        # --- Timer ---
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._render_frame)
        self._timer.start(int(1000 / self.RENDER_HZ))

    def _init_amplitude_buttons(self):
        from PySide6.QtWidgets import QPushButton, QFrame, QVBoxLayout

        self.amplitude_buttons = {}
        self.amplitude_container = QFrame(self.viewport())
        self.amplitude_container.setStyleSheet("background: transparent; border: none;")
        layout = QVBoxLayout(self.amplitude_container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        for amplitude in (50, 75, 125):
            button = QPushButton(str(amplitude))
            button.setCheckable(True)
            button.setFixedSize(40, 30)
            button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(60, 60, 60, 150);
                    color: white;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: rgba(80, 80, 80, 200);
                }
                QPushButton:checked {
                    background-color: rgba(0, 150, 220, 220);
                }
            """)
            button.clicked.connect(lambda _, max_uv=amplitude: self._set_eeg_y_max(max_uv))
            layout.addWidget(button)
            self.amplitude_buttons[amplitude] = button

        self.amplitude_container.raise_()
        self._sync_amplitude_buttons()
        self._update_amplitude_button_pos()

    def _update_amplitude_button_pos(self):
        if not hasattr(self, "amplitude_container"):
            return

        self.amplitude_container.adjustSize()
        x = max(0, self.viewport().width() - self.amplitude_container.width() - 12)
        self.amplitude_container.move(x, 8)

    def _apply_y_range(self):
        max_uv = self.user_config.eeg_uv_range_max
        self.setYRange(-max_uv, max_uv, padding=0)

    def _sync_amplitude_buttons(self):
        selected = self.user_config.eeg_uv_range_max
        for amplitude, button in self.amplitude_buttons.items():
            button.blockSignals(True)
            button.setChecked(amplitude == selected)
            button.blockSignals(False)

    def _set_eeg_y_max(self, max_uv):
        if max_uv == self.user_config.eeg_uv_range_max:
            return

        if self.on_config_change is None:
            self.user_config = self.user_config.update(eeg_uv_range_max=max_uv)
            self._apply_y_range()
            self._sync_amplitude_buttons()
            return

        self.on_config_change(self.user_config.update(eeg_uv_range_max=max_uv))


    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._update_time_scale)

    def _update_time_scale(self):
        win = self.window()
        if win is None:
            return

        handle = win.windowHandle()
        if handle is None:
            return

        screen = handle.screen()
        dpi = screen.logicalDotsPerInch() if screen else 96

        px_per_sec = self.user_config.eeg_mm_per_second * dpi / 25.4

        view_box = self.getViewBox()
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
        self._update_amplitude_button_pos()

    def clear_data(self):
        self.history.clear()
        self.display.fill(np.nan)
        self.display_head = -1
        self._pending.clear()
        self._last_rendered_head = -1
        self.update_line.setPos(0)
        self.curve_a.clear()
        self.curve_b.clear()

    def apply_config(self, user_config):
        self.user_config = user_config
        self._apply_y_range()
        self._sync_amplitude_buttons()
        self._update_time_scale()

    # ------------------------------------------------------------------
    # Data input
    # ------------------------------------------------------------------
    def append_sample(self, val):
        if val is None:
            return

        now = time.perf_counter()
        last = self._pending[-1][0] if self._pending else now
        scheduled = max(last + self._sample_period, now)
        self._pending.append((scheduled, float(val)))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render_frame(self):
        if not self._pending and self.display_head < 0:
            return

        now = time.perf_counter()
        changed = False

        # --- Consume pending samples ---
        while self._pending and self._pending[0][0] <= now:
            _, v = self._pending.popleft()
            self.display_head = (self.display_head + 1) % self.N
            self.display[self.display_head] = v
            self.history.append(v)
            changed = True

        # --- Interpolated head ---
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

        # --- Update sweep line ---
        self.update_line.setPos((interp_head / self.N) * self.seconds_visible)

        head = int(interp_head) % self.N

        if not changed and head == self._last_rendered_head:
            return

        self._last_rendered_head = head

        gap_end = (head + self.gap_samples) % self.N

        if head < gap_end:
            # Case 1: no wrap in gap
            # Draw: gap_end → end  AND  0 → head

            # Segment A: gap_end → end
            if gap_end < self.N:
                self.curve_a.setData(
                    self.x[gap_end:],
                    self.display[gap_end:],
                )
            else:
                self.curve_a.clear()

            # Segment B: 0 → head
            if head >= 0:
                self.curve_b.setData(
                    self.x[: head + 1],
                    self.display[: head + 1],
                )
            else:
                self.curve_b.clear()

        else:
            # Case 2: gap wraps around buffer
            # Draw: gap_end → head (single segment)

            self.curve_a.setData(
                self.x[gap_end: head + 1],
                self.display[gap_end: head + 1],
            )
            self.curve_b.clear()
