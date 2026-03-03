from scipy.signal import welch, butter, filtfilt, iirnotch
import numpy as np
from config import SystemConfig


class DSACalculator:
    """Computes PSD columns for DSA using Welch's method, with optional filtering."""

    def __init__(self, window_sec, segment_sec, segment_overlap):
        self.window_sec = window_sec
        self.segment_sec = segment_sec
        self.segment_overlap = segment_overlap

        # Default filter parameters
        self.lowcut = SystemConfig.LOWEST_FREQ_HZ
        self.highcut = SystemConfig.MAX_FREQ_HZ_BOUNDS[1]
        self.notch_freq = 50.0  # Change to 60.0 if using 60 Hz mains
        self.notch_quality = 30.0  # Q factor for notch filter

    def _bandpass_filter(self, data):
        nyq = 0.5 * SystemConfig.SAMPLE_RATE_HZ
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(N=4, Wn=[low, high], btype="band")
        return filtfilt(b, a, data)

    def _notch_filter(self, data):
        nyq = 0.5 * SystemConfig.SAMPLE_RATE_HZ
        freq = self.notch_freq / nyq
        b, a = iirnotch(freq, self.notch_quality)
        return filtfilt(b, a, data)

    def compute_psd_column(self, eeg_values):
        if len(eeg_values) < self.window_sec * SystemConfig.SAMPLE_RATE_HZ:
            print("Not enough data for PSD")
            return None, None

        # Apply filters
        filtered = self._bandpass_filter(eeg_values)
        filtered = self._notch_filter(filtered)

        nperseg = int(SystemConfig.SAMPLE_RATE_HZ * self.segment_sec)
        f, psd = welch(
            filtered,
            fs=SystemConfig.SAMPLE_RATE_HZ,
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
            print("PSD has a zero entry.")
            return f, np.full((len(psd)), np.nan, np.float32)

        return f, psd


    def update_config(self, window_sec, segment_sec, segment_overlap):
        self.window_sec = window_sec
        self.segment_sec = segment_sec
        self.segment_overlap = segment_overlap