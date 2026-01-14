from scipy.signal import welch
import numpy as np
from config import SystemConfig

class DSACalculator:

    def __init__(self, WINDOW_SEC, SEGMENT_SEC, OVERLAP_PSD):
        self.window_sec = WINDOW_SEC
        self.segment_sec = SEGMENT_SEC
        self.overlap_psd = OVERLAP_PSD

    def compute_psd_column(self, eeg_values):
        if len(eeg_values) < self.window_sec * SystemConfig.SAMPLE_RATE_HZ:
            return None, None
        nperseg = SystemConfig.SAMPLE_RATE_HZ * self.segment_sec
        f, psd = welch(
            np.array(eeg_values),
            fs=SystemConfig.SAMPLE_RATE_HZ,
            window="hann",
            nperseg=int(nperseg),
            noverlap=int(self.overlap_psd * nperseg),
            scaling="density",
            detrend="constant",
            average="mean",
            return_onesided=True,
        )

        mask = (f >= SystemConfig.LOWEST_FREQ_HZ) & (f <= SystemConfig.MAX_FREQ_HZ_BOUNDS[1])
        f = f[mask]
        psd = psd[mask]

        # convert to db
        # Romagnoli et al. (2024). Non-invasive technology for brain monitoring: definition and meaning of the principal
        # parameters for the International PRactice On TEChnology neuro-moniToring group (I-PROTECT).
        # Journal of Clinical Monitoring and Computing. 38. 1-19. 10.1007/s10877-024-01146-1.
        psd_db = 10.0 * np.log10(psd + 1e-12)

        return f, psd_db

    def update_config(self, WINDOW_SEC, SEGMENT_SEC, OVERLAP_PSD):
        self.window_sec = WINDOW_SEC
        self.segment_sec = SEGMENT_SEC
        self.overlap_psd = OVERLAP_PSD