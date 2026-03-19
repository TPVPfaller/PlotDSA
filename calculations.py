from scipy.signal import sosfilt, welch, butter, filtfilt, iirnotch, tf2sos

import numpy as np
from config import SystemConfig


class DSACalculator:
    """Computes PSD columns for DSA using Welch's method, optimized."""

    def __init__(self, window_sec, segment_sec, segment_overlap):
        self.window_sec = window_sec
        self.segment_sec = segment_sec
        self.segment_overlap = segment_overlap

        self.notch_freq = 50.0
        self.notch_quality = 100.0

        self._precompute_filters()

        # Precompute Welch parameters
        self._update_welch_params()

    def _precompute_filters(self):
        low = SystemConfig.LOWEST_FREQ_HZ / (0.5 * SystemConfig.SAMPLE_RATE_HZ)
        high = SystemConfig.MAX_FREQ_HZ_BOUNDS[1] / (0.5 * SystemConfig.SAMPLE_RATE_HZ)

        bp_b, bp_a = butter(4, [low, high], btype="band")
        notch_b, notch_a = iirnotch(self.notch_freq / (0.5 * SystemConfig.SAMPLE_RATE_HZ), self.notch_quality)

        # convert to SOS and stack
        sos_bp = tf2sos(bp_b, bp_a)
        sos_notch = tf2sos(notch_b, notch_a)

        self.sos = np.vstack([sos_bp, sos_notch])

    def _update_welch_params(self):
        self.nperseg = int(SystemConfig.SAMPLE_RATE_HZ * self.segment_sec)
        self.noverlap = int(self.segment_overlap * self.nperseg)

        # Precompute frequency mask once
        f_dummy = np.fft.rfftfreq(self.nperseg, 1 / SystemConfig.SAMPLE_RATE_HZ)
        self.freq_mask = ((f_dummy >= SystemConfig.LOWEST_FREQ_HZ)& (f_dummy <= SystemConfig.MAX_FREQ_HZ_BOUNDS[1]))

    def _bandpass_filter(self, data):
        return filtfilt(self.bp_b, self.bp_a, data)

    def _notch_filter(self, data):
        return filtfilt(self.notch_b, self.notch_a, data)

    def compute_psd_column(self, eeg_values):
        min_samples = int(self.window_sec * SystemConfig.SAMPLE_RATE_HZ)
        if len(eeg_values) < min_samples:
            return None, None

        #filtered = sosfilt(self.sos, eeg_values)
        filtered = np.ascontiguousarray(eeg_values, dtype=np.float32)
        f, psd = welch(
            filtered,
            fs=SystemConfig.SAMPLE_RATE_HZ,
            window="hann",
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            scaling="density",
            detrend=False,  # faster than "constant"
            average="mean",
            return_onesided=True,
        )

        f = f[self.freq_mask]
        psd = psd[self.freq_mask]

        np.any(psd)
        if not np.all(psd):
            return f, np.full(len(psd), np.nan, np.float32)

        return f, psd

    def update_config(self, window_sec, segment_sec, segment_overlap):
        self.window_sec = window_sec
        self.segment_sec = segment_sec
        self.segment_overlap = segment_overlap
        self._update_welch_params()
