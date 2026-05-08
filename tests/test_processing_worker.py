import pytest
import numpy as np
from worker import ProcessingWorker
from config import UserConfig
import config


class DummyStream:
    def __init__(self):
        self.receiving = False

    def connect(self):
        self.receiving = False


def test_apply_config_stashes_new_config(monkeypatch):
    monkeypatch.setattr("worker.EEGStream", DummyStream)

    worker = ProcessingWorker(UserConfig())
    new_config = worker.user_config.update(window_sec=4)

    worker.apply_config(new_config)
    worker.stop()

    assert worker._new_config == new_config


def test_run_emits_samples_and_saves_generated_columns(monkeypatch):
    saved_calls = []
    emitted_samples = []
    emitted_columns = []

    class ConnectedStream:
        def __init__(self):
            self.receiving = True

        def connect(self):
            self.receiving = True

        def read_lsl_samples(self):
            return [(0, 1.0)]

    class FakeExecutor:
        def submit(self, fn, *args):
            saved_calls.append((fn, args))

        def shutdown(self, wait=True):
            pass

    monkeypatch.setattr("worker.EEGStream", ConnectedStream)
    monkeypatch.setattr("worker.Output.save_psd_to_csv", lambda *args: None)

    worker = ProcessingWorker(UserConfig(window_sec=2, window_overlap=0.5))
    worker._io_executor = FakeExecutor()

    def fake_get_dsa_columns(samples, method="multitaper"):
        worker.running = False
        return [(123.0, np.array([1.0, 2.0], dtype=np.float32))], [0.1, 0.2]

    worker.eeg_buffer.get_dsa_columns = fake_get_dsa_columns
    worker.new_samples.connect(emitted_samples.append)
    worker.new_dsa_column.connect(lambda ts, psd, steps: emitted_columns.append((ts, psd, steps)))

    worker.run()

    assert emitted_samples == [[0.1, 0.2]]
    assert len(emitted_columns) == 1
    assert emitted_columns[0][0] == 123.0
    assert emitted_columns[0][2] == 2
    assert saved_calls[0][1][0] == 123.0
    assert saved_calls[0][1][1] == 2 * config.TIME_RESOLUTION
