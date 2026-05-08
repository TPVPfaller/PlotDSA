import numpy as np
from scipy.signal import welch, firwin, lfilter
from scipy.signal.windows import dpss
import config


class DSACalculator:
    """Computes PSD columns for DSA using Welch's method (Defender-safe)."""

    def __init__(self, window_sec):
        self.window_sec = window_sec

        self.notch_freq = 50.0  # line noise

        self._precompute_filters()

    def _precompute_filters(self):
        fs = config.SAMPLE_RATE_HZ

        self.bp_b = firwin(
            numtaps=101,
            cutoff=[config.LOWEST_FREQ_HZ, config.MAX_FREQ_HZ_BOUNDS[1]],
            fs=fs,
            pass_zero=False,
            window='hamming'
        )

        self.notch_b = firwin(
            numtaps=81,
            cutoff=[48.5, 51.5],
            fs=fs,
            pass_zero=True,
            window='hamming'
        )

        self.bp_a = [1.0]
        self.notch_a = [1.0]

    def _apply_filters(self, data):
        # Apply notch first, then bandpass
        filtered = lfilter(self.notch_b, self.notch_a, data)
        filtered = lfilter(self.bp_b, self.bp_a, filtered)
        return filtered

    def multitaper_method(self, eeg_values):
        """
        Multitaper PSD estimation.
        """
        TW = 1
        K = 3  # number of tapers
        nw = config.N_PER_SEGMENT # number of samples per window (800 for 400hz)
        N = len(eeg_values)//config.N_PER_SEGMENT # number of windows

        eeg_values = eeg_values[:(N * nw)] # cut off extra samples
        eeg_values = np.reshape(eeg_values, (N, nw)) # reshape to (N, nw)

        fs = config.SAMPLE_RATE_HZ

        tapers, _ = dpss(nw, NW=TW, Kmax=K, return_ratios=True) # (K, nw)
        spect = np.zeros((N, nw)) # N//2+1 because fft output is Hermitian-symmetric

        for i in range(N):
            # tapers is (K, nw), eeg_values[i] is (nw,)
            window_data = eeg_values[i]
            tapered_data = tapers * window_data # Broadcasting (K, nw) * (nw,) -> (K, nw)
            
            fourier = np.fft.fft(tapered_data, n=nw, axis=1) # shape: (K, nw//2+1)

            # Power per taper
            power = (np.abs(fourier) ** 2) / fs

            # For rfft, we need to double the power for all bins except DC and Nyquist if we want one-sided PSD
            power[:, 1:-1] *= 2
            
            # Average across tapers
            spect[i] = np.mean(power, axis=0)

        # Average across windows
        return np.mean(spect, axis=0)


    def compute_psd_column(self, eeg_values, method='multitaper'):
        min_samples = int(self.window_sec * config.SAMPLE_RATE_HZ)
        if len(eeg_values) < min_samples:
            return None, None

        # Apply FIR filters
        filtered = self._apply_filters(np.asarray(eeg_values, dtype=np.float32))

        if method == 'multitaper':
            psd = self.multitaper_method(filtered)
            if psd is None:
                return np.full(len(config.FREQ_BINS), np.nan, np.float32)
        else:
            _, psd = welch(
                filtered,
                fs=config.SAMPLE_RATE_HZ,
                nperseg=config.N_PER_SEGMENT,
                noverlap=None, # If None then overlap is 50%
                return_onesided=True
            )
        psd = psd[:251]
        psd = psd[config.FREQ_MASK]

        if not np.all(psd):
            return np.full(len(psd), np.nan, np.float32)

        return psd

    def update_config(self, window_sec):
        self.window_sec = window_sec
