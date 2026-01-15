from scipy.signal import welch
import numpy as np
from config import SystemConfig

class DSACalculator:
    """Computes PSD columns for DSA using Welch's method."""

    def __init__(self, window_sec, segment_sec, overlap_psd, sample_rate=SystemConfig.SAMPLE_RATE_HZ):
        self.window_sec = window_sec
        self.segment_sec = segment_sec
        self.overlap_psd = overlap_psd
        self.sample_rate = sample_rate

    def compute_psd_column(self, eeg_values):
        if len(eeg_values) < self.window_sec * self.sample_rate:
            return None, None
        
        nperseg = int(self.sample_rate * self.segment_sec)
        f, psd = welch(
            np.array(eeg_values),
            fs=self.sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=int(self.overlap_psd * nperseg),
            scaling="density",
            detrend="constant",
            average="mean",
            return_onesided=True,
        )

        mask = (f >= SystemConfig.LOWEST_FREQ_HZ) & (f <= SystemConfig.MAX_FREQ_HZ_BOUNDS[1])
        f = f[mask]
        psd = psd[mask]

        # Convert to dB with a small epsilon to avoid log(0)
        psd_db = 10.0 * np.log10(psd + 1e-12)

        return f, psd_db

    def update_config(self, WINDOW_SEC, SEGMENT_SEC, OVERLAP_PSD):
        self.window_sec = WINDOW_SEC
        self.segment_sec = SEGMENT_SEC
        self.overlap_psd = OVERLAP_PSD