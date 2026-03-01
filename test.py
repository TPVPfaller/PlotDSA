import numpy as np
import datetime
import pytest

from calculations import DSACalculator
from config import SystemConfig
from data import EEGBuffer, DSABuffer   # adjust import if main file named differently


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def generate_sine(freq, seconds, sr, noise=0.0):
    t = np.arange(0, seconds, 1 / sr)
    sig = np.sin(2 * np.pi * freq * t)
    if noise > 0:
        sig += np.random.normal(0, noise, size=len(sig))
    return sig


def make_timestamp_series(start, n, sr):
    dt = datetime.timedelta(seconds=1 / sr)
    return [start + i * dt for i in range(n)]


# ------------------------------------------------------------
# DSACalculator
# ------------------------------------------------------------

def test_psd_detects_correct_frequency():
    sr = SystemConfig.SAMPLE_RATE_HZ
    window_sec = 4.0
    segment_sec = 2.0

    eeg = generate_sine(freq=10, seconds=window_sec, sr=sr)

    calc = DSACalculator(window_sec, segment_sec, 0.5)
    f, psd = calc.compute_psd_column(eeg)

    peak_freq = f[np.argmax(psd)]

    assert abs(peak_freq - 10.0) < 0.5, "PSD peak should be near 10 Hz"


def test_psd_is_nan_safe():
    calc = DSACalculator(4.0, 2.0, 0.5)

    eeg = np.zeros(int(4.0 * SystemConfig.SAMPLE_RATE_HZ))
    f, psd = calc.compute_psd_column(eeg)

    assert np.all(np.isfinite(psd)), "PSD must not contain NaNs or infs"


# ------------------------------------------------------------
# EEGBuffer
# ------------------------------------------------------------

def test_eegbuffer_sliding_window_count():
    sr = SystemConfig.SAMPLE_RATE_HZ
    window = 2.0
    overlap = 0.5

    buf = EEGBuffer(window, 1.0, 0.5, overlap)

    n = int(5 * sr)
    ts = make_timestamp_series(datetime.datetime.now(), n, sr)
    sig = np.random.randn(n)

    samples = list(zip(ts, sig))

    out = buf.extend_and_process(samples)

    expected = int((5 - window) / (window * (1 - overlap))) + 1

    assert abs(len(out) - expected) <= 1


def test_eegbuffer_resets_on_gap():
    sr = SystemConfig.SAMPLE_RATE_HZ
    buf = EEGBuffer(2.0, 1.0, 0.5, 0.5)

    t0 = datetime.datetime.now()

    good = [
        (t0 + datetime.timedelta(seconds=i/sr), 1.0)
        for i in range(200)
    ]

    gap = [
        (t0 + datetime.timedelta(seconds=10), 1.0)
    ]

    out1 = buf.extend_and_process(good)
    out2 = buf.extend_and_process(gap)

    assert len(out2) == 0, "Gap must reset EEGBuffer"


# ------------------------------------------------------------
# DSABuffer
# ------------------------------------------------------------

def test_dsabuffer_gap_filling():
    buf = DSABuffer(2.0)

    f = np.linspace(0, 40, 10)
    psd = np.ones(10)

    t0 = 1000.0
    buf.append(t0, f, psd)
    buf.append(t0 + 2 * SystemConfig.UPDATE_STEP_SEC, f, psd)

    t, view = buf.get_view(5, 10)

    # middle column must be NaN
    assert np.isnan(view[1]).all()


def test_dsabuffer_wraparound():
    buf = DSABuffer(1.0)
    f = np.linspace(0, 40, 5)
    psd = np.ones(5)

    n = buf.max_frames + 10

    for i in range(n):
        buf.append(i * SystemConfig.UPDATE_STEP_SEC, f, psd)

    t0, view = buf.get_view(buf.max_frames, 5)

    assert not np.isnan(view[-1]).all(), "Newest frame must exist after wrap"


# ------------------------------------------------------------
# DSA time alignment
# ------------------------------------------------------------

def test_dsa_timestamp_monotonic():
    buf = DSABuffer(2.0)
    f = np.linspace(0, 40, 10)
    psd = np.ones(10)

    times = [1000.0 + i * SystemConfig.UPDATE_STEP_SEC for i in range(10)]

    for t in times:
        buf.append(t, f, psd)

    t0, _ = buf.get_view(10, 10)

    assert abs(t0 - times[0]) < 1e-6