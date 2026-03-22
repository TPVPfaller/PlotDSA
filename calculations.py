import numpy as np
from scipy.signal import welch, firwin, lfilter
from config import SystemConfig

class DSACalculator:
    """Computes PSD columns for DSA using Welch's method (Defender-safe)."""

    def __init__(self, window_sec, segment_sec, segment_overlap):
        self.window_sec = window_sec
        self.segment_sec = segment_sec
        self.segment_overlap = segment_overlap

        self.notch_freq = 50.0  # line noise
        self.notch_bw_hz = 1.0  # bandwidth for simple notch FIR

        self.freq_mask = None

        self._precompute_filters()
        self._update_welch_params()

    def _precompute_filters(self):
        # FIR bandpass design
        numtaps = 101  # filter length
        nyq = 0.5 * SystemConfig.SAMPLE_RATE_HZ
        low = SystemConfig.LOWEST_FREQ_HZ / nyq
        high = SystemConfig.MAX_FREQ_HZ_BOUNDS[1] / nyq
        self.bp_b = firwin(numtaps, [low, high], pass_zero=False)
        self.bp_a = [1.0]  # FIR denominator

        # Simple FIR notch filter approximation (high attenuation at 50Hz)
        notch_low = (self.notch_freq - self.notch_bw_hz/2) / nyq
        notch_high = (self.notch_freq + self.notch_bw_hz/2) / nyq
        self.notch_b = firwin(numtaps, [notch_low, notch_high], pass_zero=True)
        self.notch_a = [1.0]

    def _apply_filters(self, data):
        # Apply notch first, then bandpass
        filtered = lfilter(self.notch_b, self.notch_a, data)
        filtered = lfilter(self.bp_b, self.bp_a, filtered)
        return filtered

    def _update_welch_params(self):
        self.nperseg = int(SystemConfig.SAMPLE_RATE_HZ * self.segment_sec)
        self.noverlap = int(self.segment_overlap * self.nperseg)
        f_dummy = np.fft.rfftfreq(self.nperseg, 1 / SystemConfig.SAMPLE_RATE_HZ)
        self.freq_mask = (f_dummy >= SystemConfig.LOWEST_FREQ_HZ) & (
            f_dummy <= SystemConfig.MAX_FREQ_HZ_BOUNDS[1]
        )

    def compute_psd_column(self, eeg_values):
        min_samples = int(self.window_sec * SystemConfig.SAMPLE_RATE_HZ)
        if len(eeg_values) < min_samples:
            return None, None

        # Apply FIR filters
        filtered = self._apply_filters(np.asarray(eeg_values, dtype=np.float32))

        # 5 seconds window, max 10%-20% overlap

        f, psd = welch(
            filtered,
            fs=SystemConfig.SAMPLE_RATE_HZ,
            nperseg=2 * SystemConfig.SAMPLE_RATE_HZ,
            noverlap=None,  # matches MATLAB [] default (50% overlap internally)
            #nfft=2 * SystemConfig.SAMPLE_RATE_HZ,
            return_onesided=True
        )
        self.freq_mask = (f >= SystemConfig.LOWEST_FREQ_HZ) & (f <= SystemConfig.MAX_FREQ_HZ_BOUNDS[1])
        f = f[self.freq_mask]
        psd = psd[self.freq_mask]

        if not np.all(psd):
            return f, np.full(len(psd), np.nan, np.float32)

        return f, psd

    def update_config(self, window_sec, segment_sec, segment_overlap):
        self.window_sec = window_sec
        self.segment_sec = segment_sec
        self.segment_overlap = segment_overlap
        self._update_welch_params()