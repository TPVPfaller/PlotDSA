import time


import datetime
from collections import deque

import numpy as np
import pyqtgraph as pg
from pyqtgraph import ColorBarItem
from PySide6.QtCore import Qt, QEvent, QTimer

from data import DSABuffer
from config import SystemConfig


# ------------------ DSA View ------------------ #
class DSAView(pg.GraphicsLayoutWidget):
    """Dynamic Spectrum Analysis display with live/pan and pinch-zoom support."""

    def __init__(self, config, on_config_change):
        super().__init__()
        self.config = config
        self.on_config_change = on_config_change
        self.dsa_buffer = DSABuffer(self.config.segment_sec)
        self._init_parameters()
        self._init_plot()
        self._init_colormap()
        self._init_colorbar()
        self._init_gestures()
        self._init_image_buffer()

        pg.setConfigOptions(antialias=False)

    def _init_parameters(self):
        self.live_mode = True
        self._last_render = time.time()
        self._zoom_factor = 1.0
        self.display_minutes = self.config.display_minutes
        self._pan_offset_sec = 0.0
        self._min_zoom = 1.0
        self._max_zoom = 10.0
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

        self.image = pg.ImageItem(interpolation="linear")
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
            values=(self.config.psd_db_min, self.config.psd_db_max),
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

    def _init_image_buffer(self):
        nperseg = int(self.config.segment_sec * SystemConfig.SAMPLE_RATE_HZ)
        freq_bins = np.fft.rfftfreq(nperseg, d=1 / SystemConfig.SAMPLE_RATE_HZ)
        mask = (freq_bins >= SystemConfig.LOWEST_FREQ_HZ) & (freq_bins <= self.config.max_freq_hz)
        self.n_freq_bins = len(freq_bins[mask])
        self.n_time_bins = int(self.config.display_minutes * 60 / SystemConfig.TIME_RESOLUTION)

        self.dsa_rect = np.full((self.n_time_bins, self.n_freq_bins), np.nan, dtype=np.float32)
        self.image.setImage(self.dsa_rect, autoLevels=False)
        self.image.setLevels((self.config.psd_db_min, self.config.psd_db_max))
        self.image.setRect((self.t0, 0, self.config.display_minutes * 60, self.config.max_freq_hz))

    # ------------------ Update & Rendering ------------------ #
    def update(self, dsa_column):
        if time.time() - self._last_render < SystemConfig.TIME_RESOLUTION:
            return
        if dsa_column is not None:
            ts, freqs, psd = dsa_column
            self.dsa_buffer.append(ts, freqs, psd)
        visible_width_sec = self.config.display_minutes * 60.0
        n_visible_bins = max(1, int(visible_width_sec / SystemConfig.TIME_RESOLUTION))

        max_offset = self.dsa_buffer.get_newest_timestamp() - visible_width_sec
        min_offset = self.dsa_buffer.get_oldest_timestamp()

        if self.live_mode:
            self._pan_offset_sec = max_offset
        else:
            self._pan_offset_sec = max(min_offset, min(self._pan_offset_sec, max_offset))
            if self._pan_offset_sec >= max_offset - 0.05:
                self.live_mode = True
                self._pan_offset_sec = max_offset
                self.on_config_change(self.config)

        self.t0, self.dsa_rect = self.dsa_buffer.get_view_at(
            width=n_visible_bins, height=self.n_freq_bins, pan_offset_sec=self._pan_offset_sec
        )
        data = self.dsa_rect
        if self.config.normalize_psd:
            data = self._normalize(data)
            self.colorbar.setLevels((-40, -5))
            self.image.setLevels((-40, -5), update=False)
        else:
            self.colorbar.setLevels((self.config.psd_db_min, self.config.psd_db_max))
            self.image.setLevels((self.config.psd_db_min, self.config.psd_db_max), update=False)

        data_db = 10 * np.log10(np.clip(data, np.finfo(np.float32).eps, None))
        self.image.setImage(data_db, nan_policy="omit", autoLevels=False)
        self.image.setRect((self.t0, SystemConfig.LOWEST_FREQ_HZ, visible_width_sec, self.config.max_freq_hz))

        self.plot.setXRange(self.t0, self.t0 + visible_width_sec, padding=0)
        self.plot.setYRange(SystemConfig.LOWEST_FREQ_HZ, float(self.config.max_freq_hz), padding=0)

    def _normalize(self, data):
        data = data.copy()
        col_sums = np.sum(data, axis=1, keepdims=True)
        col_sums = np.maximum(col_sums, np.finfo(data.dtype).eps)

        np.divide(data, col_sums, out=data)
        return data

    # ------------------ Zoom & Pan ------------------ #
    def set_zoom(self, zoom_factor: float):
        print(zoom_factor)
        self._zoom_factor = np.clip(zoom_factor, self._min_zoom, self._max_zoom)
        self.update(None)

    def jump_to_live(self):
        self.live_mode = True
        if self.dsa_buffer.t0 is not None:
            visible_width_sec = self.config.display_minutes * 60
            self._pan_offset_sec = self.dsa_buffer.get_newest_timestamp() - visible_width_sec
            self.update(None)
        self.on_config_change(self.config)

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
            x_range = self.plot.viewRange()[0][1] - self.plot.viewRange()[0][0]
            width_px = self.plot.width()
            dt = (delta.x() / width_px) * x_range if width_px else 0
            if self._pan_offset_sec == 0.0:
                self._pan_offset_sec = self.dsa_buffer.t0
            self._pan_offset_sec -= dt
            self.live_mode = False
            self.update(None)
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
            new_minutes = self.config.display_minutes / pinch.scaleFactor()
            min_m, max_m = SystemConfig.DISPLAY_MINUTES_BOUNDS
            new_minutes = np.clip(new_minutes, min_m, max_m)
            self.on_config_change(self.config.update(display_minutes=new_minutes))
        return True

    def apply_config(self, config):
        self.config = config
        self.update(None)


class PSDView(pg.PlotWidget):
    def __init__(self, PSD_DB_MIN, PSD_DB_MAX):
        super().__init__()

        self.setLabel("bottom", "Frequency", units="Hz")
        self.setLabel("left", "Power Spectral Density", units="dB/Hz")
        self.setMenuEnabled(False)
        self.showGrid(x=True, y=True)
        self.setMouseEnabled(x=False, y=False)

        self.curve = self.plot(pen=pg.mkPen("y", width=2), title="PSD")

        self.setInteractive(False)

        self.setYRange(
            PSD_DB_MIN - 15,
            PSD_DB_MAX + 15
        )

    def update(self, freqs, psd):
        if freqs is None or psd is None:
            return
        psd_db = 10 * np.log10(np.clip(psd, np.finfo(np.float32).eps, None))
        self.curve.setData(freqs, psd_db)

    def apply_config(self, PSD_DB_MIN, PSD_DB_MAX):
        self.setYRange(
            PSD_DB_MIN - 15,
            PSD_DB_MAX + 15
        )


class EEGView(pg.PlotWidget):
    """Real-time circular EEG display with smooth sweep line."""

    RENDER_HZ = 20

    def __init__(self, window_sec: float):
        super().__init__()

        # --- Plot setup ---
        self.setLabel("left", "EEG", units="µV")
        self.showGrid(x=True, y=True)
        self.setMenuEnabled(False)
        self.setClipToView(True)
        self.setDownsampling(mode="peak")
        self.setMouseEnabled(False, False)
        self.setInteractive(False)

        self.curve = self.plot(pen=pg.mkPen((0, 200, 255), width=2))
        self.fault_curve = self.plot(pen=pg.mkPen((220, 50, 50), width=2))
        self.update_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("w", style=Qt.DashLine))
        self.addItem(self.update_line)

        self.N = int(SystemConfig.EEG_VIEW_WINDOW_SEC * SystemConfig.SAMPLE_RATE_HZ)
        self.display = np.full(self.N, np.nan, np.float32)
        self.display_head = -1

        self.x = np.linspace(0, SystemConfig.EEG_VIEW_WINDOW_SEC, self.N, endpoint=False)
        self.gap_samples = max(1, int(0.1 * SystemConfig.SAMPLE_RATE_HZ))

        self._pending = deque()
        self._sample_period = 1.0 / SystemConfig.SAMPLE_RATE_HZ
        self._last_rendered_head = -1

        self._y_plot = np.empty_like(self.display)
        self._fault_plot = np.empty_like(self.display)

        self.setXRange(0, SystemConfig.EEG_VIEW_WINDOW_SEC, padding=0)
        self.setYRange(-100, 100, padding=0)

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._render_frame)
        self._timer.start(int(1000 / self.RENDER_HZ))

    def append_sample(self, ts: float, val):
        if val is None:
            return

        import time
        now = time.perf_counter()
        last = self._pending[-1][0] if self._pending else now
        scheduled = max(last + self._sample_period, now)
        self._pending.append((scheduled, val))

    def _render_frame(self):
        if not self._pending and self.display_head < 0:
            return

        now = time.perf_counter()

        changed = False
        while self._pending and self._pending[0][0] <= now:
            _, v = self._pending.popleft()
            self.display_head = (self.display_head + 1) % self.N
            self.display[self.display_head] = v
            changed = True

        # --- Interpolated head ---
        if self._pending:
            next_wall = self._pending[0][0]
            prev_wall = next_wall - self._sample_period
            frac = np.clip((now - prev_wall) / self._sample_period, 0, 1)
            interp_head = (self.display_head + frac) % self.N
        else:
            if self.display_head < 0:
                return
            interp_head = float(self.display_head)

        self.update_line.setPos(
            (interp_head / self.N) * SystemConfig.EEG_VIEW_WINDOW_SEC
        )

        head_int = int(interp_head) % self.N
        if not changed and head_int == self._last_rendered_head:
            return

        self._last_rendered_head = head_int

        np.copyto(self._y_plot, self.display)
        np.copyto(self._fault_plot, np.where(np.isnan(self._y_plot), 0.0, np.nan))

        # --- Blank gap ---
        end = head_int + self.gap_samples + 1
        if end < self.N:
            self._y_plot[head_int + 1:end] = np.nan
            self._fault_plot[head_int + 1:end] = np.nan
        else:
            self._y_plot[head_int + 1:] = np.nan
            self._y_plot[:end % self.N] = np.nan
            self._fault_plot[head_int + 1:] = np.nan
            self._fault_plot[:end % self.N] = np.nan

        self.curve.setData(self.x, self._y_plot)
        self.fault_curve.setData(self.x, self._fault_plot)
