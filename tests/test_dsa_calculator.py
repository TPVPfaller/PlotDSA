import numpy as np
from calculations import DSACalculator
from config import SystemConfig


def generate_sine(freq, seconds, sr, noise=0.0):
    t = np.arange(0, seconds, 1 / sr)
    sig = np.sin(2 * np.pi * freq * t)
    if noise > 0:
        sig += np.random.normal(0, noise, size=len(sig))
    return sig


def test_psd_returns_correct_frequency_range():
    calc = DSACalculator(window_sec=4, segment_sec=1, segment_overlap=0.5)

    # Generate synthetic sine at 10 Hz
    fs = SystemConfig.SAMPLE_RATE_HZ
    t = np.arange(0, 4, 1/fs)
    signal = np.sin(2 * np.pi * 10 * t)

    f, psd = calc.compute_psd_column(signal)

    assert f is not None
    assert psd is not None
    assert len(f) == len(psd)
    assert np.all(f >= SystemConfig.LOWEST_FREQ_HZ)
    assert np.all(f <= SystemConfig.MAX_FREQ_HZ_BOUNDS[1])


def test_psd_detects_10hz_peak():
    calc = DSACalculator(4, 1, 0.5)

    fs = SystemConfig.SAMPLE_RATE_HZ
    t = np.arange(0, 4, 1/fs)
    signal = np.sin(2 * np.pi * 10 * t)

    f, psd = calc.compute_psd_column(signal)

    peak_freq = f[np.argmax(psd)]
    assert abs(peak_freq - 10) < 0.5


def test_psd_returns_nan_on_invalid_signal():
    calc = DSACalculator(4, 1, 0.5)
    signal = np.zeros(int(4 * SystemConfig.SAMPLE_RATE_HZ))

    f, psd = calc.compute_psd_column(signal)

    assert np.all(np.isnan(psd))

def test_psd_detects_correct_frequency():
    sr = SystemConfig.SAMPLE_RATE_HZ
    window_sec = 4.0
    segment_sec = 2.0

    eeg = generate_sine(freq=10, seconds=window_sec, sr=sr)

    calc = DSACalculator(window_sec, segment_sec, 0.5)
    f, psd = calc.compute_psd_column(eeg)

    peak_freq = f[np.argmax(psd)]

    assert abs(peak_freq - 10.0) < 0.5, "PSD peak should be near 10 Hz"