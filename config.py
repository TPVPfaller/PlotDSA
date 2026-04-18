"""
IEC 62304 – Class B

System Configuration Module
Centralized configuration definition and validation.
"""
from dataclasses import replace, dataclass
from typing import Tuple
import math
import numpy as np


"""
Immutable system constants.
These should never change during runtime.
"""
SAMPLE_RATE_HZ: int = 250
TIME_RESOLUTION: float = 1.0
DSA_FPS: float = 0.5
INTERVAL: float = 1.1
LOWEST_FREQ_HZ: float = 0.1
DSA_TIME_DIFF_TOLERANCE: float = 2.0 / SAMPLE_RATE_HZ
EEG_TIME_DIFF_TOLERANCE: float = 0.5 / SAMPLE_RATE_HZ
BASE_DIR: str = "C:\\temp\\VSCaptureWave"
LSL_STREAM_NAME: str = "EEG_DATA"
EEG_VIEW_WINDOW_SEC: float = 4.0
EEG_MM_PER_SECOND: int = 30
N_PER_SEGMENT: int = SAMPLE_RATE_HZ * 2 # 2 second segments for Welch's method
FONT_SIZE = 15


# Default values
WINDOW_SEC: int = 10
WINDOW_OVERLAP: float = 0.10
DISPLAY_MINUTES: float = 30.0
MAX_FREQ_HZ: int = 30
PSD_DB_MIN: int = -25
PSD_DB_MAX: int = 20

# Bounds (class-level, not instance attributes)
WINDOW_SEC_BOUNDS: Tuple[int, int] = (max(1, math.ceil(TIME_RESOLUTION)), 30)
WINDOW_OVERLAP_BOUNDS: Tuple[float, float] = (0.0, 0.99)
DISPLAY_MINUTES_BOUNDS: Tuple[float, float] = (0.5, 60.0*24.0*7) # 1 week limit to avoid 4GB allocation but still plenty
MAX_FREQ_HZ_BOUNDS: Tuple[int, int] = (20, 50)
PSD_DB_MIN_BOUNDS: Tuple[int, int] = (-50, 0)
PSD_DB_MAX_BOUNDS: Tuple[int, int] = (0, 50)
EEG_BOUNDS: Tuple[int, int] = (-250, 250)
USE_MULTITAPER: bool = True

_all_freq_bins = np.fft.rfftfreq(N_PER_SEGMENT, d=1 / SAMPLE_RATE_HZ)
FREQ_MASK = ((_all_freq_bins >= LOWEST_FREQ_HZ) & (_all_freq_bins <= MAX_FREQ_HZ_BOUNDS[1]))
FREQ_BINS = _all_freq_bins[FREQ_MASK]


@dataclass(frozen=True)
class UserConfig:
    """
    User-configurable settings with validation.
    Immutable at runtime; updates return a new instance.
    """

    window_sec: int = WINDOW_SEC
    window_overlap: float = WINDOW_OVERLAP
    display_minutes: float = DISPLAY_MINUTES
    max_freq_hz: int = MAX_FREQ_HZ
    psd_db_min: int = PSD_DB_MIN
    psd_db_max: int = PSD_DB_MAX
    use_multitaper: bool = USE_MULTITAPER

    def __post_init__(self):
        """Validate on creation."""
        self.validate()

    def validate(self) -> None:
        """
        Validates the entire configuration.
        Raises ValueError if invalid.
        """
        self._check_bounds("window_sec", self.window_sec, WINDOW_SEC_BOUNDS)
        self._check_bounds("window_overlap", self.window_overlap, WINDOW_OVERLAP_BOUNDS)
        self._check_bounds("display_minutes", self.display_minutes, DISPLAY_MINUTES_BOUNDS)
        self._check_bounds("max_freq_hz", self.max_freq_hz, MAX_FREQ_HZ_BOUNDS)
        self._check_bounds("psd_db_min", self.psd_db_min, PSD_DB_MIN_BOUNDS)
        self._check_bounds("psd_db_max", self.psd_db_max, PSD_DB_MAX_BOUNDS)

        if self.psd_db_min >= self.psd_db_max:
            raise ValueError(
                f"psd_db_min ({self.psd_db_min}) must be less than psd_db_max ({self.psd_db_max})"
            )

    def update(self, **kwargs) -> 'UserConfig':
        """
        Returns a new UserConfig instance with updated values.
        """
        new_config = replace(self, **kwargs)
        new_config.validate()
        return new_config

    @staticmethod
    def _check_bounds(name: str, value: float, bounds: Tuple[float, float]) -> None:
        """Check if value is within bounds."""
        if not (bounds[0] <= value <= bounds[1]):
            raise ValueError(
                f"{name} must be within [{bounds[0]}, {bounds[1]}], got {value}"
            )
