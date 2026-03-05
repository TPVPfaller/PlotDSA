import numpy as np
import datetime
from data import EEGBuffer
from config import SystemConfig


def make_timestamp_series(start, n, sr):
    dt = datetime.timedelta(seconds=1 / sr)
    return [start + i * dt for i in range(n)]


def generate_samples(n, start_time):
    samples = []
    delta = datetime.timedelta(milliseconds=1000 / SystemConfig.SAMPLE_RATE_HZ)

    ts = start_time
    for i in range(n):
        samples.append((ts, 10.0))
        ts += delta

    return samples


def test_window_produces_dsa_column():
    buffer = EEGBuffer(
        window_sec=4,
        segment_sec=1,
        segment_overlap=0.5,
        overlap=0.5
    )

    start = datetime.datetime.now()
    samples = generate_samples(
        int(4 * SystemConfig.SAMPLE_RATE_HZ),
        start
    )

    dsa_cols, raw = buffer.get_dsa_columns(samples)

    assert len(dsa_cols) > 0


def test_timestamp_fault_resets_buffer():
    buffer = EEGBuffer(4, 1, 0.5, 0.5)

    start = datetime.datetime.now()
    samples = generate_samples(100, start)

    # inject fault
    samples.append((start + datetime.timedelta(seconds=10), 10.0))

    dsa_cols, raw = buffer.get_dsa_columns(samples)

    assert len(buffer.eeg_values) == 0


def test_eegbuffer_sliding_window_count():
    sr = SystemConfig.SAMPLE_RATE_HZ
    window = 2.0
    overlap = 0.5

    buf = EEGBuffer(window, 1.0, 0.5, overlap)

    n = int(5 * sr)
    ts = make_timestamp_series(datetime.datetime.now(), n, sr)
    sig = np.random.randn(n)

    samples = list(zip(ts, sig))

    out, _ = buf.get_dsa_columns(samples)

    expected = int((5 - window) / (window * (1 - overlap))) + 1

    assert abs(len(out) - expected) <= 1


def test_eegbuffer_resets_on_gap():
    sr = SystemConfig.SAMPLE_RATE_HZ
    buf = EEGBuffer(2.0, 1.0, 0.5, 0.5)

    t0 = datetime.datetime.now()

    good = [
        (t0 + datetime.timedelta(seconds=i / sr), 1.0)
        for i in range(200)
    ]

    gap = [
        (t0 + datetime.timedelta(seconds=10), 1.0)
    ]

    out1, _ = buf.get_dsa_columns(good)
    out2, _ = buf.get_dsa_columns(gap)

    assert len(out2) == 0, "Gap must reset EEGBuffer"


def test_eegbuffer_resets_on_timestamp_gap():
    window_sec = 1.0
    buf = EEGBuffer(window_sec, 1.0, 0.5, 0.5)

    base = datetime.datetime.now()

    good_sample = (base, 10.0)
    bad_sample = (base + datetime.timedelta(seconds=1), 10.0)  # big gap

    out, samples = buf.get_dsa_columns([good_sample, bad_sample])

    # Gap should cause reset → no DSA output
    assert len(out) == 0


def test_eegbuffer_produces_dsa_column():
    window_sec = 1.0
    sample_rate = SystemConfig.SAMPLE_RATE_HZ
    buf = EEGBuffer(window_sec, 1.0, 0.5, 0.5)

    base = datetime.datetime.now()

    data = []
    for i in range(int(window_sec * sample_rate)):
        ts = base + datetime.timedelta(milliseconds=i * (1000 / sample_rate))
        data.append((ts, 1.0))

    out, samples = buf.get_dsa_columns(data)

    assert len(out) >= 1
