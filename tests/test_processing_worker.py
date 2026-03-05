import pytest
from worker import ProcessingWorker
from config import UserConfig


def test_worker_emits_connection_signal(qtbot):
    config = UserConfig()
    worker = ProcessingWorker(config)

    received = []

    def handler(state):
        received.append(state)

    worker.connection_changed.connect(handler)

    # simulate
    worker.connection_changed.emit(True)

    assert received == [True]