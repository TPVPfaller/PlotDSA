import numpy as np
from data import DSABuffer
import config


def get_freq_bins():
    return config.FREQ_BINS


# ------------------------------------------------------------
# DSA time alignment
# ------------------------------------------------------------

def test_get_view_at_basic_pan():
    buf = DSABuffer()
    f = get_freq_bins()
    psd = np.ones(len(f))

    t0 = 1000.0
    for i in range(5):
        buf.append(t0 + i * config.TIME_RESOLUTION, psd * i)

    # Pan 2 time slots forward
    pan_sec = t0 + 2.0
    t_start, view, res = buf.get_view_at(width=3, height=10, pan_sec=pan_sec)

    assert abs(t_start - pan_sec) < 1e-6
    assert not np.isnan(view[0]).all()


def test_get_view_at_clamps_to_last_slot():
    buf = DSABuffer()
    f = get_freq_bins()
    psd = np.ones(len(f))

    t0 = 1000.0
    buf.append(t0, psd)

    # Request far beyond available data
    t_start, view, res = buf.get_view_at(width=5, height=10, pan_sec=t0 + 1000)

    # Should not crash and should contain valid first frame
    assert not np.isnan(view[0]).all()


def test_apply_config_resets_when_changed():
    buf = DSABuffer()
    f = get_freq_bins()
    psd = np.ones(len(f))

    buf.append(1000.0, psd)
    assert buf.t0 is not None

    buf.apply_config(2.0)

    assert buf.t0 is None
