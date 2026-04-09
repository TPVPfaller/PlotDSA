import numpy as np
from data import Output
import config


def test_csv_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)

    psd = np.array([10, 20, 30])

    Output.save_psd_to_csv(1000.0, psd)

    files = list(tmp_path.glob("*.csv"))
    assert len(files) == 1

    with open(files[0]) as f:
        lines = f.readlines()

    assert "timestamp" in lines[0]
    assert len(lines) == 2


def test_save_psd_to_csv_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BASE_DIR", str(tmp_path))

    psd = np.array([10.0, 20.0])

    Output.save_psd_to_csv(1000.0, psd)

    files = list(tmp_path.glob("*.csv"))
    assert len(files) == 1

    content = files[0].read_text()
    assert "timestamp" in content
    assert "f_1.00_Hz" in content
