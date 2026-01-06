from config import SystemConfig
from scipy.signal import welch
import numpy as np

class DSACalculator:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.window_samples = int(
            config.WINDOW_SEC * config.SAMPLE_RATE_HZ
        )

        # Standard Welch choice for EEG
        self.nperseg = self.window_samples

    def compute_psd_column(self, eeg_buffer):
        if len(eeg_buffer) < self.window_samples:
            return None, None

        f, psd = welch(
            eeg_buffer,
            fs=self.config.SAMPLE_RATE_HZ,
            window="hann",
            nperseg=int(self.config.SAMPLE_RATE_HZ * self.config.WINDOW_SEC),
            noverlap=int(self.config.OVERLAP * self.nperseg),
            scaling="density",
            detrend="constant",
            average="mean"
        )

        mask = f <= self.config.MAX_FREQ_HZ
        f = f[mask]
        psd = psd[mask]

        # Romagnoli et al. (2024). Non-invasive technology for brain monitoring: definition and meaning of the principal
        # parameters for the International PRactice On TEChnology neuro-moniToring group (I-PROTECT).
        # Journal of Clinical Monitoring and Computing. 38. 1-19. 10.1007/s10877-024-01146-1.
        psd_db = 10.0 * np.log10(psd + 1e-12)

        return f, psd_db