import numpy as np
import os
from data import Output
from config import SystemConfig


def test_csv_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(SystemConfig, "BASE_DIR", tmp_path)

    freqs = np.array([1, 2, 3])
    psd = np.array([10, 20, 30])

    Output.save_psd_to_csv(1000.0, freqs, psd)

    files = list(tmp_path.glob("*.csv"))
    assert len(files) == 1

    with open(files[0]) as f:
        lines = f.readlines()

    assert "timestamp" in lines[0]
    assert len(lines) == 2