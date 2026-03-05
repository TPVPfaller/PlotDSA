import numpy as np
from data import DSABuffer
from config import SystemConfig


def get_freq_bins(segment_sec=SystemConfig.SEGMENT_SEC):
    nperseg = int(segment_sec * SystemConfig.SAMPLE_RATE_HZ)
    freq_bins = np.fft.rfftfreq(nperseg, d=1.0 / SystemConfig.SAMPLE_RATE_HZ)
    mask = (freq_bins >= SystemConfig.LOWEST_FREQ_HZ) & (freq_bins <= SystemConfig.MAX_FREQ_HZ_BOUNDS[1])
    return freq_bins[mask]

def test_append_and_retrieve_single_frame():
    buffer = DSABuffer(SEGMENT_SEC=1)

    f = get_freq_bins(1)
    psd = np.ones(50)

    ts = 1000.0
    buffer.append(ts, f, psd)

    t0, frame = buffer.get_frame(width=1, height=50)

    assert frame.shape == (1, 50)
    assert np.all(frame[0] == 1)


def test_gap_filling_inserts_nan():
    buffer = DSABuffer(4)

    f = get_freq_bins(4)
    psd = np.ones(50)

    buffer.append(1000.0, f, psd)
    buffer.append(1002.0, f, psd)  # intentional gap

    t0, frame = buffer.get_frame(width=3, height=50)

    # There must be at least one NaN row
    assert np.any(np.isnan(frame))

def test_dsabuffer_gap_filling():
    buf = DSABuffer(2.0)

    f = get_freq_bins(2.0)
    psd = np.ones(10)

    t0 = 1000.0
    buf.append(t0, f, psd)
    buf.append(t0 + 2 * SystemConfig.TIME_RESOLUTION, f, psd)

    t, view = buf.get_view(5, 10)

    # middle column must be NaN
    assert np.isnan(view[1]).all()


def test_dsabuffer_wraparound():
    buf = DSABuffer(1.0)
    f = get_freq_bins(1.0)
    psd = np.ones(5)

    n = buf.max_frames + 10

    for i in range(n):
        buf.append(i * SystemConfig.TIME_RESOLUTION, f, psd)

    t0, view = buf.get_view(buf.max_frames, 5)

    assert not np.isnan(view[-1]).all(), "Newest frame must exist after wrap"


# ------------------------------------------------------------
# DSA time alignment
# ------------------------------------------------------------

def test_dsa_timestamp_monotonic():
    buf = DSABuffer(2.0)
    f = get_freq_bins(2.0)
    psd = np.ones(10)

    times = [1000.0 + i * SystemConfig.TIME_RESOLUTION for i in range(10)]

    for t in times:
        buf.append(t, f, psd)

    t0, _ = buf.get_view(10, 10)

    assert abs(t0 - times[0]) < 1e-6