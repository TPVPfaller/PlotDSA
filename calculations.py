import numpy as np
from scipy.signal import welch, firwin, lfilter
import config

class DSACalculator:
    """Computes PSD columns for DSA using Welch's method (Defender-safe)."""

    def __init__(self, window_sec):
        self.window_sec = window_sec

        self.notch_freq = 50.0  # line noise
        self.notch_bw_hz = 1.0  # bandwidth for simple notch FIR

        self._precompute_filters()

    def _precompute_filters(self):
        # FIR bandpass design
        numtaps = 101  # filter length
        nyq = 0.5 * config.SAMPLE_RATE_HZ
        low = config.LOWEST_FREQ_HZ / nyq
        high = config.MAX_FREQ_HZ_BOUNDS[1] / nyq
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

    def compute_psd_column(self, eeg_values):
        min_samples = int(self.window_sec * config.SAMPLE_RATE_HZ)
        if len(eeg_values) < min_samples:
            return None, None

        # Apply FIR filters
        filtered = self._apply_filters(np.asarray(eeg_values, dtype=np.float32))

        # 5 seconds window, max 10%-20% overlap

        _, psd = welch(
            filtered,
            fs=config.SAMPLE_RATE_HZ,
            nperseg=config.N_PER_SEGMENT,
            noverlap=None,
            return_onesided=True
        )

        psd = psd[config.FREQ_MASK]

        if not np.all(psd):
            return np.full(len(psd), np.nan, np.float32)

        return psd

    def update_config(self, window_sec):
        self.window_sec = window_sec
