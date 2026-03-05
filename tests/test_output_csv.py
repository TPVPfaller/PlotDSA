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


def test_output_build_filename(tmp_path):
    freqs = np.array([1.0, 2.0, 3.0])
    base_dir = tmp_path

    filename = Output._build_filename(str(base_dir), freqs)

    assert "df1.0Hz" in filename
    assert filename.endswith(".csv")


def test_save_psd_to_csv_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(SystemConfig, "BASE_DIR", str(tmp_path))

    freqs = np.array([1.0, 2.0])
    psd = np.array([10.0, 20.0])

    Output.save_psd_to_csv(1000.0, freqs, psd)

    files = list(tmp_path.glob("*.csv"))
    assert len(files) == 1

    content = files[0].read_text()
    assert "timestamp" in content
    assert "f_1.00_Hz" in content
