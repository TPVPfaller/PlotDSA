import pyqtgraph as pg
import datetime
from config import SystemConfig
from pyqtgraph import ColorBarItem
import numpy as np
from collections import deque


class DSAView(pg.GraphicsLayoutWidget):
    def __init__(self, config):
        super().__init__()
        self.PSD_DB_MIN = config.PSD_DB_MIN
        self.PSD_DB_MAX = config.PSD_DB_MAX
        self.SEGMENT_SEC = config.SEGMENT_SEC
        self.DISPLAY_MINUTES = config.DISPLAY_MINUTES
        self.MAX_FREQ_HZ = config.MAX_FREQ_HZ

        # --- Layout ---
        self.time_axis = pg.DateAxisItem("bottom")
        self.plot = self.addPlot(row=0, col=0, axisItems={"bottom": self.time_axis})
        #self.plot.setLabel("bottom", "Time")
        self.plot.setLabel("left", "Frequency", units="Hz")
        self.plot.setMenuEnabled(False)
        self.plot.showGrid(x=False, y=False)
        self.plot.invertY(False)

        self.setInteractive(False)
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)

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
            (255, 0, 0),  # high power (burst)
        ]

        pos = np.linspace(0.0, 1.0, len(colors))
        self.cmap = pg.ColorMap(pos, colors)

        self.lut = self.cmap.getLookupTable(nPts=256, mode="byte")

        self.image.setLookupTable(self.lut)

    def update(self, dsa_buffer):
        self.t0, self.dsa_rect = dsa_buffer.get_view(width=self.n_time_bins, height=self.n_freq_bins)

        self.image.setImage(
            self.dsa_rect,
            autoLevels=False,
            levels=(self.PSD_DB_MIN, self.PSD_DB_MAX),
            lut=self.cmap.getLookupTable(),
            nan_policy="omit",
        )

        self.image.setRect(
            (
                self.t0,  # x
                0.0,  # y
                self.DISPLAY_MINUTES*60,  # width
                self.MAX_FREQ_HZ  # height
            )
        )

    def update_config(self, config):
        if self.PSD_DB_MIN != config.PSD_DB_MIN or self.PSD_DB_MAX != config.PSD_DB_MAX:
            self.PSD_DB_MIN = config.PSD_DB_MIN
            self.PSD_DB_MAX = config.PSD_DB_MAX
            self.colorbar.setLevels((self.PSD_DB_MIN, self.PSD_DB_MAX))
            self.image.setLevels((
                self.PSD_DB_MIN,
                self.PSD_DB_MAX
            ))

        if self.SEGMENT_SEC != config.SEGMENT_SEC or self.MAX_FREQ_HZ != config.MAX_FREQ_HZ:
            self.SEGMENT_SEC = config.SEGMENT_SEC
            self.MAX_FREQ_HZ = config.MAX_FREQ_HZ

            nperseg = int(self.SEGMENT_SEC * SystemConfig.SAMPLE_RATE_HZ)
            freq_bins = np.fft.rfftfreq(nperseg, d=1.0 / SystemConfig.SAMPLE_RATE_HZ)
            mask = (freq_bins >= SystemConfig.LOWEST_FREQ_HZ) & (freq_bins <= self.MAX_FREQ_HZ)
            self.n_freq_bins = len(freq_bins[mask])

        if self.DISPLAY_MINUTES != config.DISPLAY_MINUTES:
            self.DISPLAY_MINUTES = config.DISPLAY_MINUTES
            self.n_time_bins = int(self.DISPLAY_MINUTES * 60.0 / SystemConfig.UPDATE_STEP_SEC)

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

    def update_config(self, PSD_DB_MIN, PSD_DB_MAX):
        self.setYRange(
            PSD_DB_MIN - 15,
            PSD_DB_MAX + 15
        )

class EEGView(pg.PlotWidget):
    """
    Displays the raw EEG time-series. Uses a DateAxis on the bottom and keeps a sliding window
    of the last WINDOW_SEC seconds based on SystemConfig.SAMPLE_RATE_HZ.
    UI redraw is throttled to once per WINDOW_SEC (samples are still buffered continuously).
    Also provides a forced redraw method to synchronize with DSA/PSD updates.
    """

    def __init__(self, window_sec: float):
        # Use DateAxisItem for time axis
        super().__init__(axisItems={"bottom": pg.DateAxisItem(orientation="bottom")})
        self.window_sec = float(window_sec)
        self.setTitle("Raw EEG")
        #self.setLabel("bottom", "Time")
        self.setLabel("left", "EEG", units="µV")
        self.setMenuEnabled(False)
        self.showGrid(x=True, y=True)
        self.setClipToView(True)
        self.setDownsampling(mode='peak')
        self.setMouseEnabled(x=False, y=False)

        maxlen = int(self.window_sec * SystemConfig.SAMPLE_RATE_HZ)
        self._times = deque(maxlen=maxlen)  # epoch seconds (float)
        self._values = deque(maxlen=maxlen)

        self.curve = self.plot(pen=pg.mkPen((0, 200, 255), width=2))
        self._last_update_count = 0
        self._last_plot_t = None  # epoch seconds of last redraw

    def append_samples(self, samples):
        """Append a list of (datetime, value) tuples and refresh the plot at most once per window_sec."""
        if not samples:
            return
        count_added = 0
        for ts, val in samples:
            if ts is None or val is None or not np.isfinite(val):
                continue
            # Convert to epoch seconds (float)
            try:
                t = ts.timestamp()
            except Exception:
                # If ts is already a float
                t = float(ts)
            self._times.append(t)
            self._values.append(float(val))
            count_added += 1

        if count_added == 0:
            return

        # Trim data outside the window based on time to be robust to jitter
        t_now = self._times[-1]
        window_start = t_now - self.window_sec
        # While oldest is older than window_start, pop left
        while self._times and self._times[0] < window_start:
            self._times.popleft()
            self._values.popleft()

        # Throttle UI redraws to once per window_sec
        should_redraw = False
        if self._last_plot_t is None:
            should_redraw = True
        else:
            if (t_now - self._last_plot_t) >= self.window_sec:
                should_redraw = True

        if should_redraw:
            self.curve.setData(list(self._times), list(self._values))
            self._auto_range_y()
            self._last_plot_t = t_now

    def redraw_current(self):
        """Force an immediate redraw using the current buffer (used to sync with DSA/PSD updates)."""
        if not self._times:
            return
        self.curve.setData(list(self._times), list(self._values))
        self._auto_range_y()
        # Keep last plot time coherent with latest timestamp
        self._last_plot_t = self._times[-1]

    def _auto_range_y(self):
        if len(self._values) < 2:
            return
        v = np.asarray(self._values, dtype=float)
        finite = np.isfinite(v)
        if not finite.any():
            return
        v = v[finite]
        vmin, vmax = float(np.min(v)), float(np.max(v))
        if vmin == vmax:
            pad = 1.0
        else:
            pad = 0.1 * (vmax - vmin)
        self.setYRange(vmin - pad, vmax + pad, padding=0)

    def update_config(self, window_sec: float):
        window_sec = float(window_sec)
        if window_sec <= 0:
            return
        if window_sec == self.window_sec:
            return
        self.window_sec = window_sec
        maxlen = int(self.window_sec * SystemConfig.SAMPLE_RATE_HZ)
        # Rebuild deques with new maxlen, keeping most recent samples
        times = list(self._times)
        values = list(self._values)
        if len(times) > maxlen:
            times = times[-maxlen:]
            values = values[-maxlen:]
        self._times = deque(times, maxlen=maxlen)
        self._values = deque(values, maxlen=maxlen)
        # Force a redraw after config change
        if self._times:
            self.curve.setData(list(self._times), list(self._values))
            self._auto_range_y()
            self._last_plot_t = self._times[-1]