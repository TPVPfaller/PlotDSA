import datetime
from collections import deque

import numpy as np
import pyqtgraph as pg
from pyqtgraph import ColorBarItem, GridItem
from PySide6.QtCore import Qt, QEvent, QTimer
import time


from config import EEG_MM_PER_SECOND
from data import DSABuffer
import config


# ------------------ DSA View ------------------ #
class DSAView(pg.GraphicsLayoutWidget):
    """Dynamic Spectrum Analysis display with live/pan and pinch-zoom support."""

    def __init__(self, user_config, on_config_change, on_zoom_change):
        super().__init__()
        self.user_config = user_config
        self.on_config_change = on_config_change
        self.on_zoom_change = on_zoom_change
        self.dsa_buffer = DSABuffer()
        self._init_parameters()
        self._init_plot()
        self._init_colormap()
        self._init_colorbar()
        self._init_gestures()
        self.update(None, force_update=True)

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
        self.plot = self.addPlot(row=0, col=0, axisItems={"bottom": self.time_axis})
        self.plot.setLabel("left", "Frequency", units="Hz")
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.showGrid(x=False, y=False)
        self.plot.invertY(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setYRange(config.LOWEST_FREQ_HZ, self.user_config.max_freq_hz, padding=0)

        self.image = pg.ImageItem(axisOrder='col-major', interpolation="linear")
        self.plot.addItem(self.image)

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
            label="Power (dB)",
            interactive=False,
        )
        self.colorbar.setImageItem(self.image)
        self.addItem(self.colorbar, row=0, col=1)
        self.ci.layout.setColumnStretchFactor(0, 10)
        self.ci.layout.setColumnStretchFactor(1, 1)

    def _init_gestures(self):
        self.grabGesture(Qt.PinchGesture)


    # ------------------ Update & Rendering ------------------ #
    def update(self, dsa_column, force_update=False):
        if dsa_column is not None:
            ts, psd = dsa_column
            self.dsa_buffer.append(ts, psd)
        if time.time() - self._last_render < 0.5 and not force_update:
            return
        self._last_render = time.time()
        visible_width_sec = self.display_minutes * 60.0
        n_time_bins = max(1, int(visible_width_sec / config.TIME_RESOLUTION))

        target_res = visible_width_sec / 1000.0
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
        data = self.dsa_rect.copy()
        data = np.ascontiguousarray(data) # array is stored in a continuous block of memory
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
        data = self.dsa_rect.copy()
        data = np.ascontiguousarray(data)  # array is stored in a continuous block of memory

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
            self.update(None, force_update=True)

    def pan(self, delta_percent):
        """External pan control"""
        if self.dsa_buffer.t0 is None:
            return

        if delta_percent == "live":
            self.jump_to_live()
            return

        self.live_mode = False
        self._pan_sec += self.display_minutes * delta_percent * 60
        self.update(None, force_update=True)

    def clear_data(self):
        """Delete all buffered data"""
        self.dsa_buffer = DSABuffer()
        self.live_mode = True
        self._pan_sec = 0.0
        self.update(None, force_update=True)

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
            self.update(None, force_update=True)

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
            self.update(None, force_update=True)
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
        self.update(None, force_update=True)

    def apply_config(self, new_config):
        old_config = self.user_config
        self.user_config = new_config
        if new_config.max_freq_hz != old_config.max_freq_hz:
            self.plot.setYRange(config.LOWEST_FREQ_HZ, self.user_config.max_freq_hz, padding=0)
            self.freq_bins = config.FREQ_BINS[(config.FREQ_BINS <= self.user_config.max_freq_hz)]

        self.update(None, force_update=True)



class PSDView(pg.PlotWidget):
    def __init__(self, user_config):
        super().__init__()
        self.user_config = user_config

        self.setLabel("bottom", "Frequency", units="Hz")
        self.setLabel("left", "Power", units="dB")
        self.setMenuEnabled(False)
        self.showGrid(x=True, y=True)
        self.setMouseEnabled(x=False, y=False)

        self.curve = self.plot(pen=pg.mkPen("y", width=2), title="PSD")

        self.setInteractive(False)
        self.setYRange(user_config.psd_db_min - 15, user_config.psd_db_max + 15)

    def update(self, psd):
        self.setYRange(self.user_config.psd_db_min - 15, self.user_config.psd_db_max + 15)
        psd_db = 10 * np.log10(np.clip(psd, np.finfo(np.float32).eps, None))
        self.curve.setData(config.FREQ_BINS, psd_db)

    def apply_config(self, user_config):
        self.user_config = user_config


# 7.5 mm/sekunde 15 mm/sekunde eeg view skalieren mit application window. 5 microvolt pro millimeter. 27 zoll pc. einstellung in system settings
class EEGView(pg.PlotWidget):
    """Real-time circular EEG display with smooth sweep line + gap (optimized)."""

    RENDER_HZ = 20

    def __init__(self):
        super().__init__()

        # --- Plot setup ---
        self.setLabel("left", "EEG", units="µV")
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

        self.nan_curve = self.plot(
            pen=None,
            symbol='o',
            symbolBrush='r',  # fill color red
            symbolSize=3,  # size
            symbolPen=None
        )

        self.curve_a = self.plot(pen=pg.mkPen((0, 200, 255), width=1))
        self.curve_b = self.plot(pen=pg.mkPen((0, 200, 255), width=1))

        # Sweep line
        self.update_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("w", style=Qt.DashLine))
        self.addItem(self.update_line)

        # --- Buffer setup ---
        self.N = int(config.EEG_VIEW_WINDOW_SEC * config.SAMPLE_RATE_HZ)

        self.display = np.full(self.N, np.nan, dtype=np.float32)
        self.display_head = -1

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

        # --- View limits ---
        self.setXRange(0, config.EEG_VIEW_WINDOW_SEC, padding=0)
        self.setYRange(-80, 80, padding=0)

        # --- Timer ---
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._render_frame)
        self._timer.start(int(1000 / self.RENDER_HZ))


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

        px_per_sec = EEG_MM_PER_SECOND * dpi / 25.4

        width_px = self.width()
        if width_px <= 0:
            return

        seconds_visible = width_px / px_per_sec

        new_N = int(seconds_visible * config.SAMPLE_RATE_HZ)

        if new_N <= 10:
            return

        if new_N != self.N:
            if self.display_head >= 0:
                # 1. Unroll the existing circular buffer so the most recent sample is at the end.
                unrolled = np.concatenate([
                    self.display[self.display_head + 1:],
                    self.display[:self.display_head + 1]
                ])

                # 2. Maintain temporal sweep line position (e.g., stay at 50% through the window)
                old_ratio = (self.display_head + 1) / self.N
                new_head = int(old_ratio * new_N) - 1
                new_head = np.clip(new_head, -1, new_N - 1)

                # 3. Create a fresh buffer and map samples relative to the new head.
                new_display = np.full(new_N, np.nan, dtype=np.float32)

                # How many samples can we carry over?
                num_to_copy = min(self.N, new_N)
                samples_to_copy = unrolled[-num_to_copy:]

                # We place samples such that samples_to_copy[-1] lands at new_head.
                start_idx = new_head - num_to_copy + 1
                if start_idx >= 0:
                    new_display[start_idx : new_head + 1] = samples_to_copy
                else:
                    # Wraps around the beginning of the circular buffer
                    # Part 1: Fill from 0 to new_head
                    new_display[0 : new_head + 1] = samples_to_copy[-(new_head + 1):]
                    # Part 2: Fill the remainder at the end of the buffer
                    rem = num_to_copy - (new_head + 1)
                    new_display[-rem:] = samples_to_copy[:rem]

                self.display_head = new_head
                self._last_rendered_head = -1
                self.display = new_display
            else:
                self.display = np.full(new_N, np.nan, dtype=np.float32)

            self.N = new_N
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

        # Mask where NaNs originally were
        nan_mask = np.isnan(self.display)

        # --- Consume pending samples ---
        while self._pending and self._pending[0][0] <= now:
            _, v = self._pending.popleft()
            self.display_head = (self.display_head + 1) % self.N
            self.display[self.display_head] = v
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

        nan_indices = np.where(nan_mask)[0]

        if len(nan_indices) > 0:
            self.nan_curve.setData(
                self.x[nan_indices],
                np.zeros_like(nan_indices, dtype=np.float32),
            )
        else:
            self.nan_curve.clear()