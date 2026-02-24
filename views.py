import pyqtgraph as pg
import datetime
from config import SystemConfig
from pyqtgraph import ColorBarItem
from PySide6.QtWidgets import QPinchGesture, QGestureEvent, QPushButton, QGraphicsProxyWidget, QSizePolicy
from PySide6.QtCore import Qt, QEvent, Signal
import numpy as np
from collections import deque


class DSAView(pg.GraphicsLayoutWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.PSD_DB_MIN = config.psd_db_min
        self.PSD_DB_MAX = config.psd_db_max
        self.SEGMENT_SEC = config.segment_sec
        self.DISPLAY_MINUTES = config.display_minutes
        self.MAX_FREQ_HZ = config.max_freq_hz

        self._live_mode = True  # Start in live mode
        self._zoom_factor = 1.0  # 1.0 = full display time
        self._pan_offset_sec = 0.0  # seconds offset from start of buffer
        self._min_zoom = 1.0
        self._max_zoom = 10.0

        # --- Layout ---
        self.time_axis = pg.DateAxisItem("bottom")
        self.plot = self.addPlot(row=0, col=0, axisItems={"bottom": self.time_axis})
        #self.plot.setLabel("bottom", "Time")
        self.plot.setLabel("left", "Frequency", units="Hz")
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.showGrid(x=False, y=False)
        self.plot.invertY(False)

        self.image = pg.ImageItem()
        self.image.setOpts(interpolation='linear')
        self.plot.addItem(self.image)

        # Set interactive to allow events to reach the view
        self.setInteractive(True)
        self.plot.setMouseEnabled(x=False, y=False)  # Disable default pyqtgraph panning/zooming

        # Dragging state
        self._dragging = False
        self._last_mouse_pos = None

        # --- Colormap ---
        self._init_colormap()

        # --- Colorbar ---
        self.colorbar = ColorBarItem(
            values=(self.PSD_DB_MIN, self.PSD_DB_MAX),
            colorMap=self.cmap,
            label="Power (dB)",
            interactive=False,
        )
        self.colorbar.setImageItem(self.image)
        self.addItem(self.colorbar, row=0, col=1)

        self.ci.layout.setColumnStretchFactor(0, 10)
        self.ci.layout.setColumnStretchFactor(1, 1)

        # Enable gestures
        self.grabGesture(Qt.PinchGesture)
        # self.grabGesture(Qt.PanGesture) # Pan is already handled well by mouse events on most touchscreens

        self.t0 = datetime.datetime.now().timestamp()
        nperseg = int(self.SEGMENT_SEC * SystemConfig.SAMPLE_RATE_HZ)
        freq_bins = np.fft.rfftfreq(nperseg, d=1.0 / SystemConfig.SAMPLE_RATE_HZ)
        mask = (freq_bins >= SystemConfig.LOWEST_FREQ_HZ) & (freq_bins <= self.MAX_FREQ_HZ)
        self.n_freq_bins = len(freq_bins[mask])
        self.n_time_bins = int(self.DISPLAY_MINUTES * 60.0 / SystemConfig.UPDATE_STEP_SEC)

        self.dsa_rect = np.full((self.n_time_bins, self.n_freq_bins), np.nan, dtype=np.float32)

        # Image is (freq, time) when displayed
        self.image.setImage(
            self.dsa_rect,
            autoLevels=False
        )

        self.image.setLevels((
            self.PSD_DB_MIN,
            self.PSD_DB_MAX
        ))

        # Set pixel-to-axis mapping
        self.image.setRect(
            (
                self.t0,  # x
                0.0,  # y
                self.DISPLAY_MINUTES * 60,  # width
                self.MAX_FREQ_HZ  # height
            )
        )

    def _init_colormap(self):
        colors = [
            (10, 10, 50),  # very low power (deep blue)
            (20, 40, 120),  # blue
            (40, 120, 200),  # light blue
            (80, 200, 220),  # cyan
            (80, 220, 140),  # green
            (200, 220, 80),  # yellow
            (240, 160, 40),  # orange
            (240, 80, 40),  # red
            (240, 0, 0),
        ]

        pos = np.linspace(0.0, 1.0, len(colors))
        self.cmap = pg.ColorMap(pos, colors)

        self.lut = self.cmap.getLookupTable(nPts=256, mode="byte")

        self.image.setLookupTable(self.lut)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)

    def update(self, dsa_buffer):
        self._buffer = dsa_buffer

        visible_width_sec = float(self.DISPLAY_MINUTES * 60)
        n_visible_bins = max(1, int(visible_width_sec / SystemConfig.UPDATE_STEP_SEC))

        data_span_sec = float(dsa_buffer.get_last_timestamp() - dsa_buffer.t0) if dsa_buffer.t0 else 0.0
        max_offset = max(0.0, data_span_sec - visible_width_sec)

        if self._live_mode:
            self._pan_offset_sec = max_offset
        else:
            self._pan_offset_sec = max(0.0, min(self._pan_offset_sec, max_offset))
            # If manually panned to the end, re-enable live mode (even if max_offset == 0)
            if self._pan_offset_sec >= max_offset - 0.05:  # Small epsilon
                # Only trigger UI refresh if we actually changed state
                self._live_mode = True
                self._pan_offset_sec = max_offset
                if hasattr(self, "on_config_change_callback"):
                    # We don't change config, but we want to trigger a UI refresh (indicator/button)
                    self.on_config_change_callback(self.config)

        # Ask buffer for the slice starting at pan_offset_sec
        self.t0, self.dsa_rect = dsa_buffer.get_view_at(
            width=n_visible_bins,
            height=self.n_freq_bins,
            pan_offset_sec=self._pan_offset_sec
        )

        # Force button visibility check
        self.is_last_dsa_visible()

        x_start = self.t0

        self.image.setImage(
            self.dsa_rect,
            autoLevels=False,
            levels=(self.PSD_DB_MIN, self.PSD_DB_MAX),
            lut=self.lut,
            nan_policy="omit",
        )
        self.image.setRect((
            float(x_start),
            0.0,
            float(visible_width_sec),
            float(self.MAX_FREQ_HZ)
        ))
        self.plot.setXRange(float(x_start), float(x_start + visible_width_sec), padding=0)
        self.plot.setYRange(0, float(self.MAX_FREQ_HZ), padding=0)

    def is_last_dsa_visible(self):
        """Check if the timestamp of the last available DSA column is within display bounds."""
        if not hasattr(self, "_buffer") or self._buffer is None:
            return False
        if self._live_mode:
            return True
        
        last_ts = self._buffer.get_last_timestamp()
        try:
            last_ts_val = float(last_ts)
        except Exception:
            return False
        if not np.isfinite(last_ts_val):
            return False
        
        visible_width_sec = float(self.DISPLAY_MINUTES * 60)
        
        # t0 is the start of the current visible window (x_start in update)
        # Use a more generous epsilon (half an UPDATE_STEP) to handle discrete bin boundaries
        eps = float(SystemConfig.UPDATE_STEP_SEC) * 0.5
        is_visible = (self.t0 <= last_ts_val <= self.t0 + visible_width_sec + eps)
        
        return is_visible

    def jump_to_live(self):
        """Reset view to latest available data."""
        print("Jumping to live mode...")
        self._live_mode = True
        # Explicitly set the pan offset to max_offset immediately 
        if hasattr(self, "_buffer") and self._buffer is not None:
            visible_width_sec = float(self.DISPLAY_MINUTES * 60)
            data_span_sec = float(self._buffer.get_last_timestamp() - self._buffer.t0) if self._buffer.t0 else 0.0
            max_offset = max(0.0, data_span_sec - visible_width_sec)
            self._pan_offset_sec = max_offset
            self.update(self._buffer)
        
        # Inform the main app/topbar so the LIVE indicator updates immediately
        if hasattr(self, "on_config_change_callback"):
            # We don't change config, but we want to trigger a UI refresh.
            self.on_config_change_callback(self.config)

    def apply_config(self, config):
        self.config = config
        # Update color scale limits if needed
        if self.PSD_DB_MIN != config.psd_db_min or self.PSD_DB_MAX != config.psd_db_max:
            self.PSD_DB_MIN = config.psd_db_min
            self.PSD_DB_MAX = config.psd_db_max
            self.colorbar.setLevels((self.PSD_DB_MIN, self.PSD_DB_MAX))
            self.image.setLevels((self.PSD_DB_MIN, self.PSD_DB_MAX))

        # Update frequency axis resolution if segment or max freq changed
        if self.SEGMENT_SEC != config.segment_sec or self.MAX_FREQ_HZ != config.max_freq_hz:
            self.SEGMENT_SEC = config.segment_sec
            self.MAX_FREQ_HZ = config.max_freq_hz
            nperseg = int(self.SEGMENT_SEC * SystemConfig.SAMPLE_RATE_HZ)
            freq_bins = np.fft.rfftfreq(nperseg, d=1.0 / SystemConfig.SAMPLE_RATE_HZ)
            mask = (freq_bins >= SystemConfig.LOWEST_FREQ_HZ) & (freq_bins <= self.MAX_FREQ_HZ)
            self.n_freq_bins = len(freq_bins[mask])

        # Update horizontal time resolution if display window changed
        if self.DISPLAY_MINUTES != config.display_minutes:
            self.DISPLAY_MINUTES = config.display_minutes
            self.n_time_bins = int(self.DISPLAY_MINUTES * 60.0 / SystemConfig.UPDATE_STEP_SEC)

        # Apply immediately by redrawing with current buffer if available
        if hasattr(self, "_buffer") and self._buffer is not None:
            self.update(self._buffer)

    def set_zoom(self, zoom_factor: float):
        """Set zoom factor (1 = full display, >1 = zoom in)."""
        self._zoom_factor = max(self._min_zoom, min(self._max_zoom, zoom_factor))
        if hasattr(self, "_buffer"):
            self.update(self._buffer)

    def set_pan(self, pan: float):
        pass

    def mousePressEvent(self, ev):
        # First let the scene/items (e.g., QGraphicsProxyWidget button) handle the event
        super().mousePressEvent(ev)
        if ev.isAccepted():
            return

        if ev.button() == Qt.LeftButton:
            self._dragging = True
            self._last_mouse_pos = ev.pos()
            ev.accept()

    def mouseMoveEvent(self, ev):
        if self._dragging and self._last_mouse_pos is not None:
            delta = ev.pos() - self._last_mouse_pos
            self._last_mouse_pos = ev.pos()

            # Convert pixel delta to time delta
            # plot.viewRange() returns [[xmin, xmax], [ymin, ymax]]
            view_range = self.plot.viewRange()
            x_range = view_range[0][1] - view_range[0][0]
            width_px = self.plot.width()
            
            if width_px > 0:
                dt = (delta.x() / width_px) * x_range
                # Brushing: dragging left (negative delta.x) moves view forward in time (increases offset)
                # But we want to pan the data, so dragging left should reveal data to the right.
                # In many "brush" implementations, you "pull" the data.
                # If I pull left, the data should move left, meaning I see what's on the right.
                # So _pan_offset_sec should INCREASE.
                self._pan_offset_sec -= dt  # subtracting because dragging right (positive delta) should DECREASE offset (move to past)
                
                # Manual pan disables live mode unless we are at the very end (handled in update)
                self._live_mode = False
                
                if hasattr(self, "_buffer"):
                    self.update(self._buffer)

            ev.accept()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        # Allow scene/items to process release (e.g., complete a button click)
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
            # pinch.scaleFactor() is the factor since the last event
            scale_factor = pinch.scaleFactor()
            if scale_factor > 0 and scale_factor != 1.0:
                min_minutes = SystemConfig.DISPLAY_MINUTES_BOUNDS[0]
                max_minutes = SystemConfig.DISPLAY_MINUTES_BOUNDS[1]
                
                # Zooming in (scale > 1) should decrease DISPLAY_MINUTES
                # Scale factor > 1 means fingers moving apart (zoom in)
                new_minutes = self.config.display_minutes / scale_factor
                new_minutes = max(min_minutes, min(max_minutes, new_minutes))
                
                # In DSAView, we don't have a callback to on_config_change, 
                # but we can trigger it if we store a reference, OR we can just update local and let sync happen.
                # Gestures should ideally inform the main app.
                if hasattr(self, "on_config_change_callback"):
                    new_config = self.config.update(display_minutes=new_minutes)
                    self.on_config_change_callback(new_config)
                else:
                    # Fallback if callback not set
                    self.DISPLAY_MINUTES = new_minutes
                    self.n_time_bins = int(self.DISPLAY_MINUTES * 60.0 / SystemConfig.UPDATE_STEP_SEC)
                    if hasattr(self, "_buffer"):
                        self.update(self._buffer)
        return True


class PSDView(pg.PlotWidget):
    def __init__(self, PSD_DB_MIN, PSD_DB_MAX):
        super().__init__()

        self.setLabel("bottom", "Frequency", units="Hz")
        self.setLabel("left", "Power Spectral Density", units="dB/Hz")
        self.setMenuEnabled(False)
        self.showGrid(x=True, y=True)

        self.curve = self.plot(pen=pg.mkPen("y", width=2), title="PSD")

        self.setInteractive(False)

        self.setYRange(
            PSD_DB_MIN - 15,
            PSD_DB_MAX + 15
        )

    def update(self, freqs, psd_db):
        if freqs is None or psd_db is None:
            return

        self.curve.setData(freqs, psd_db)

    def apply_config(self, PSD_DB_MIN, PSD_DB_MAX):
        self.setYRange(
            PSD_DB_MIN - 15,
            PSD_DB_MAX + 15
        )

class EEGView(pg.PlotWidget):
    """
    Displays the raw EEG time-series in a fixed 2-second circular buffer.
    - Updated on every incoming sample.
    - Left-to-right timeline with wrap-around.
    - A small blank gap at the right edge (ahead of the newest sample).
    """

    def __init__(self, window_sec: float):
        # Numeric axis (seconds)
        super().__init__()
        # Enforce 2 seconds maximum display
        self.window_sec = SystemConfig.EEG_VIEW_WINDOW_SEC
        #self.setLabel("bottom", "Time", units="s")
        self.setLabel("left", "EEG", units="µV")
        self.setMenuEnabled(False)
        self.showGrid(x=True, y=True)
        self.setClipToView(True)
        self.setDownsampling(mode='peak')
        self.setInteractive(False)
        self.setMouseEnabled(x=False, y=False)

        # Fixed-size circular buffer
        self.N = int(round(self.window_sec * SystemConfig.SAMPLE_RATE_HZ))
        self.values = np.full(self.N, np.nan, dtype=float)
        self.head = -1  # index of newest sample
        self.x = np.linspace(0.0, self.window_sec, self.N, endpoint=False)
        # Gap ahead of head (blank area at right edge)
        self.gap_sec = 0.1
        self.gap_samples = max(1, int(round(self.gap_sec * SystemConfig.SAMPLE_RATE_HZ)))

        self.curve = self.plot(pen=pg.mkPen((0, 200, 255), width=2))
        
        # Add a vertical update line
        self.update_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('w', width=1, style=Qt.DashLine))
        self.addItem(self.update_line)
        
        self.setXRange(0.0, self.window_sec, padding=0)
        self.setYRange(-100, 100, padding=0)
        self._last_plot_t = None  # last timestamp plotted (epoch seconds)

    def append_samples(self, samples):
        """Backward-compatible batch append: calls `append_sample` for each tuple."""
        if not samples:
            return
        for ts, val in samples:
            self.append_sample(ts, val)

    def append_sample(self, ts, val):
        """Append a single sample and redraw immediately."""
        if ts is None or val is None:
            return
        try:
            v = float(val)
        except Exception:
            return

        # Advance head and store value
        self.head = (self.head + 1) % self.N
        self.values[self.head] = v
        
        # Current position in seconds
        x_head = self.x[self.head]
        self.update_line.setPos(x_head)

        # Build y for plotting: stationary values with a gap ahead of the head
        y_plot = self.values.copy()
        
        # Clear a gap ahead of the head
        if self.gap_samples > 0:
            for i in range(1, self.gap_samples + 1):
                idx = (self.head + i) % self.N
                y_plot[idx] = np.nan

        self.curve.setData(self.x, y_plot)

    def redraw_current(self):
        """Force an immediate redraw from the current circular buffer."""
        if self.head < 0:
            return
            
        x_head = self.x[self.head]
        self.update_line.setPos(x_head)

        y_plot = self.values.copy()
        if self.gap_samples > 0:
            for i in range(1, self.gap_samples + 1):
                idx = (self.head + i) % self.N
                y_plot[idx] = np.nan
        
        self.curve.setData(self.x, y_plot)