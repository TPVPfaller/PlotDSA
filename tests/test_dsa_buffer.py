import numpy as np
from data import DSABuffer
from config import SystemConfig, UserConfig


def get_freq_bins(segment_sec=SystemConfig.SEGMENT_SEC):
    nperseg = int(segment_sec * SystemConfig.SAMPLE_RATE_HZ)
    freq_bins = np.fft.rfftfreq(nperseg, d=1.0 / SystemConfig.SAMPLE_RATE_HZ)
    mask = (freq_bins >= SystemConfig.LOWEST_FREQ_HZ) & (freq_bins <= SystemConfig.MAX_FREQ_HZ_BOUNDS[1])
    return freq_bins[mask]


# ------------------------------------------------------------
# DSA time alignment
# ------------------------------------------------------------

def test_get_view_at_basic_pan():
    buf = DSABuffer(1.0)
    f = get_freq_bins(1.0)
    psd = np.ones(len(f))

    t0 = 1000.0
    for i in range(5):
        buf.append(t0 + i * SystemConfig.TIME_RESOLUTION, f, psd * i)

    # Pan 2 time slots forward
    pan_sec = t0 + 2.0
    t_start, view = buf.get_view_at(width=3, height=10, pan_sec=pan_sec)

    assert abs(t_start - pan_sec) < 1e-6
    assert not np.isnan(view[0]).all()


def test_get_view_at_clamps_to_last_slot():
    buf = DSABuffer(1.0)
    f = get_freq_bins(1.0)
    psd = np.ones(len(f))

    t0 = 1000.0
    buf.append(t0, f, psd)

    # Request far beyond available data
    t_start, view = buf.get_view_at(width=5, height=10, pan_sec=t0 + 1000)

    # Should not crash and should contain valid first frame
    assert not np.isnan(view[0]).all()


def test_frequency_mismatch_resets_buffer():
    buf = DSABuffer(1.0)
    f = get_freq_bins(1.0)
    psd = np.ones(len(f))

    buf.append(1000.0, f, psd)

    # Use wrong frequency size
    wrong_f = np.arange(10)
    buf.append(1001.0, wrong_f, np.ones(10))

    assert buf.t0 is None
    assert buf.last_slot is None


def test_apply_config_resets_when_changed():
    buf = DSABuffer(1.0)
    f = get_freq_bins(1.0)
    psd = np.ones(len(f))

    buf.append(1000.0, f, psd)
    assert buf.t0 is not None

    buf.apply_config(2.0)

    assert buf.t0 is None
    assert buf.last_slot is None
