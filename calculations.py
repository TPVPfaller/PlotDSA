from scipy.signal import welch
import numpy as np
from config import SystemConfig as config

class DSACalculator:

    def __init__(self):
        self.window_sec = config.WINDOW_SEC
        self.sample_rate_hz = config.SAMPLE_RATE_HZ
        self.segment_sec = config.SEGMENT_SEC
        self.overlap = config.OVERLAP

    def compute_psd_column(self, eeg_values):
        if len(eeg_values) < self.window_sec * self.sample_rate_hz:
            return None, None
        nperseg = self.sample_rate_hz * self.segment_sec
        f, psd = welch(
            np.array(eeg_values),
            fs=self.sample_rate_hz,
            window="hann",
            nperseg=int(nperseg),
            noverlap=int(self.overlap * nperseg),
            scaling="density",
            detrend="constant",
            average="mean",
            return_onesided=True,
        )

        mask = (f >= config.LOWEST_FREQ_HZ) & (f <= config.MAX_FREQ_HZ_BOUNDS[1])
        f = f[mask]
        psd = psd[mask]

        # convert to db
        # Romagnoli et al. (2024). Non-invasive technology for brain monitoring: definition and meaning of the principal
        # parameters for the International PRactice On TEChnology neuro-moniToring group (I-PROTECT).
        # Journal of Clinical Monitoring and Computing. 38. 1-19. 10.1007/s10877-024-01146-1.
        psd_db = 10.0 * np.log10(psd + 1e-12)

        return f, psd_db

    def update_config(self):
        self.window_sec = config.WINDOW_SEC
        self.sample_rate_hz = config.SAMPLE_RATE_HZ
        self.segment_sec = config.SEGMENT_SEC
        self.overlap = config.OVERLAP