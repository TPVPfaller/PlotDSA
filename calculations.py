from scipy.signal import welch, butter, filtfilt, iirnotch
import numpy as np
from config import SystemConfig


class DSACalculator:
    """Computes PSD columns for DSA using Welch's method, with optional filtering."""

    def __init__(self, window_sec, segment_sec, segment_overlap, sample_rate=SystemConfig.SAMPLE_RATE_HZ):
        self.window_sec = window_sec
        self.segment_sec = segment_sec
        self.segment_overlap = segment_overlap
        self.sample_rate = sample_rate

        # Default filter parameters
        self.lowcut = SystemConfig.LOWEST_FREQ_HZ
        self.highcut = SystemConfig.MAX_FREQ_HZ_BOUNDS[1]
        self.notch_freq = 50.0  # Change to 60.0 if using 60 Hz mains
        self.notch_quality = 30.0  # Q factor for notch filter

    def _bandpass_filter(self, data):
        nyq = 0.5 * self.sample_rate
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(N=4, Wn=[low, high], btype="band")
        return filtfilt(b, a, data)

    def _notch_filter(self, data):
        nyq = 0.5 * self.sample_rate
        freq = self.notch_freq / nyq
        b, a = iirnotch(freq, self.notch_quality)
        return filtfilt(b, a, data)

    def compute_psd_column(self, eeg_values):
        if len(eeg_values) < self.window_sec * self.sample_rate:
            return None, None

        # Apply filters
        filtered = self._bandpass_filter(eeg_values)
        filtered = self._notch_filter(filtered)

        nperseg = int(self.sample_rate * self.segment_sec)
        f, psd = welch(
            filtered,
            fs=self.sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=int(self.segment_overlap * nperseg),
            scaling="density",
            detrend="constant",
            average="mean",
            return_onesided=True,
        )

        mask = (f >= SystemConfig.LOWEST_FREQ_HZ) & (f <= SystemConfig.MAX_FREQ_HZ_BOUNDS[1])
        f = f[mask]
        psd = psd[mask]

        if np.count_nonzero(psd) < len(psd):
            return f, np.full((len(psd)), np.nan, np.float32)

        psd_db = 10.0 * np.log10(psd)
        return f, psd_db

    def update_config(self, window_sec, segment_sec, segment_overlap):
        self.window_sec = window_sec
        self.segment_sec = segment_sec
        self.segment_overlap = segment_overlap
