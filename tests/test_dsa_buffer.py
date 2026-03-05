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
    psd = np.ones(len(f))

    buffer.append(1000.0, f, psd)
    buffer.append(1002.0, f, psd)  # intentional gap

    t0, frame = buffer.get_frame(width=3, height=50)

    # There must be at least one NaN row
    assert np.any(np.isnan(frame))


def test_dsabuffer_gap_filling():
    buf = DSABuffer(2.0)

    f = get_freq_bins(2.0)
    psd = np.ones(len(f))

    t0 = 1000.0
    buf.append(t0, f, psd)
    buf.append(t0 + 2 * SystemConfig.TIME_RESOLUTION, f, psd)

    t, view = buf.get_view(5, 10)

    # middle column must be NaN
    assert np.isnan(view[1]).all()


def test_dsabuffer_wraparound():
    buf = DSABuffer(1.0)
    f = get_freq_bins(1.0)
    psd = np.ones(len(f))

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
    psd = np.ones(len(f))

    times = [1000.0 + i * SystemConfig.TIME_RESOLUTION for i in range(10)]

    for t in times:
        buf.append(t, f, psd)

    t0, _ = buf.get_view(10, 10)

    assert abs(t0 - times[0]) < 1e-6


def test_get_view_at_basic_pan():
    buf = DSABuffer(1.0)
    f = get_freq_bins(1.0)
    psd = np.ones(len(f))

    t0 = 1000.0
    for i in range(5):
        buf.append(t0 + i * SystemConfig.TIME_RESOLUTION, f, psd * i)

    # Pan 2 time slots forward
    pan_sec = 2 * SystemConfig.TIME_RESOLUTION
    t_start, view = buf.get_view_at(width=3, height=10, pan_offset_sec=pan_sec)

    assert abs(t_start - (t0 + pan_sec)) < 1e-6
    assert not np.isnan(view[0]).all()


def test_get_view_at_clamps_to_last_slot():
    buf = DSABuffer(1.0)
    f = get_freq_bins(1.0)
    psd = np.ones(len(f))

    t0 = 1000.0
    buf.append(t0, f, psd)

    # Request far beyond available data
    t_start, view = buf.get_view_at(width=5, height=10, pan_offset_sec=1000)

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
