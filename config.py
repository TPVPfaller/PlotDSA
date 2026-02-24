"""
IEC 62304 – Class B

System Configuration Module
Centralized configuration definition and validation.
"""

from dataclasses import dataclass, field
from typing import Tuple


class SystemConfig:
    """
    Immutable system constants.
    These should never change during runtime.
    """
    SAMPLE_RATE_HZ: int = 400
    UPDATE_STEP_SEC: float = 0.25
    INTERVAL: float = 1.1
    NO_DATA_VALUE: float = -10000.0
    LOWEST_FREQ_HZ: float = 0.1
    TIME_DIFF_TOLERANCE: float = 0.5 / SAMPLE_RATE_HZ
    BASE_DIR: str = "C:\\temp\\VSCaptureWave"
    EEG_VIEW_WINDOW_SEC: float = 4.0

    # Bounds (class-level, not instance attributes)
    WINDOW_SEC_BOUNDS: Tuple[float, float] = (UPDATE_STEP_SEC, 60.0)
    SEGMENT_SEC_BOUNDS: Tuple[float, float] = (1.0, 4.0)
    WINDOW_OVERLAP_BOUNDS: Tuple[float, float] = (0.0, 1.0) # exclusive
    SEGMENT_OVERLAP_BOUNDS: Tuple[float, float] = (0.0, 1.0) # exclusive
    DISPLAY_MINUTES_BOUNDS: Tuple[float, float] = (0.5, 300.0)
    MAX_FREQ_HZ_BOUNDS: Tuple[int, int] = (20, 50)
    PSD_DB_MIN_BOUNDS: Tuple[int, int] = (-50, 0)
    PSD_DB_MAX_BOUNDS: Tuple[int, int] = (0, 50)
    EEG_BOUNDS: Tuple[int, int] = (-300, 300)



@dataclass(frozen=True)
class UserConfig:
    """
    User-configurable settings with validation.
    Immutable at runtime; updates return a new instance.
    """

    # Current values
    window_sec: float = 4.0
    segment_sec: float = 2.0
    segment_overlap: float = 0.5
    window_overlap: float = 0.85
    display_minutes: float = 10.0
    max_freq_hz: int = 30
    psd_db_min: int = -20
    psd_db_max: int = 20

    def __post_init__(self):
        """Validate on creation."""
        self.validate()

    def validate(self) -> None:
        """
        Validates the entire configuration.
        Raises ValueError if invalid.
        """
        self._check_bounds("window_sec", self.window_sec, SystemConfig.WINDOW_SEC_BOUNDS)
        self._check_bounds("segment_sec", self.segment_sec, SystemConfig.SEGMENT_SEC_BOUNDS)
        self._check_bounds("display_minutes", self.display_minutes, SystemConfig.DISPLAY_MINUTES_BOUNDS)
        self._check_bounds("max_freq_hz", self.max_freq_hz, SystemConfig.MAX_FREQ_HZ_BOUNDS)
        self._check_bounds("psd_db_min", self.psd_db_min, SystemConfig.PSD_DB_MIN_BOUNDS)
        self._check_bounds("psd_db_max", self.psd_db_max, SystemConfig.PSD_DB_MAX_BOUNDS)

        # Exclusive bounds for overlaps
        if not (SystemConfig.WINDOW_OVERLAP_BOUNDS[0] < self.window_overlap < SystemConfig.WINDOW_OVERLAP_BOUNDS[1]):
            raise ValueError(
                f"overlap must be between {SystemConfig.WINDOW_OVERLAP_BOUNDS[0]} and {SystemConfig.WINDOW_OVERLAP_BOUNDS[1]} (exclusive), "
                f"got {self.window_overlap}"
            )

        if not (SystemConfig.SEGMENT_OVERLAP_BOUNDS[0] < self.segment_overlap < SystemConfig.SEGMENT_OVERLAP_BOUNDS[1]):
            raise ValueError(
                f"segment_overlap must be between {SystemConfig.SEGMENT_OVERLAP_BOUNDS[0]} and "
                f"{SystemConfig.SEGMENT_OVERLAP_BOUNDS[1]} (exclusive), got {self.segment_overlap}"
            )

        # Cross-field validation
        if self.segment_sec > self.window_sec:
            raise ValueError(
                f"segment_sec ({self.segment_sec}) must not exceed window_sec ({self.window_sec})"
            )

        if self.psd_db_min >= self.psd_db_max:
            raise ValueError(
                f"psd_db_min ({self.psd_db_min}) must be less than psd_db_max ({self.psd_db_max})"
            )

    def update(self, **kwargs) -> 'UserConfig':
        """
        Returns a new UserConfig instance with updated values.
        """
        from dataclasses import replace
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