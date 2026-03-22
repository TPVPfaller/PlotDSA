import time
import math
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import QObject, Signal, Slot
from data import EEGStream, EEGBuffer, Output
from config import SystemConfig


class ProcessingWorker(QObject):
    new_dsa_column = Signal(float, object, object)  # ts, freqs, psd
    new_samples = Signal(object)  # list of eeg values

    def __init__(self, config):
        super().__init__()
        self.running = True
        self.config = config
        self._new_config = None

        self.stream = EEGStream()
        self._last_connection_state = self.stream.receiving
        self.eeg_buffer = EEGBuffer(
            self.config.window_sec,
            self.config.segment_sec,
            self.config.segment_overlap,
            self.config.window_overlap
        )
        self._io_executor = ThreadPoolExecutor(max_workers=1)

    @Slot(object)
    def apply_config(self, config):
        self._new_config = config

    @Slot()
    def run(self):
        next_time = time.time()
        while self.running:
            try:
                if self._new_config:
                    self.config = self._new_config
                    self._new_config = None
                    self.eeg_buffer.apply_config(
                        self.config.window_sec,
                        self.config.segment_sec,
                        self.config.segment_overlap,
                        self.config.window_overlap
                    )
                if not self.stream.receiving:
                    self.stream.connect()
                    # Emit on change
                    if self.stream.receiving != self._last_connection_state:
                        self._last_connection_state = self.stream.receiving
                    time.sleep(0.5)
                    continue

                samples = self.stream.read_samples()
                dsa_columns, checked_samples = self.eeg_buffer.get_dsa_columns(samples)

                if checked_samples:
                    self.new_samples.emit(checked_samples)

                for ts, freqs, psd in dsa_columns:
                    if psd is None:
                        continue

                    # Calculate how many update steps this window covers.
                    # We use ceil to ensure we bridge the gap to the next column's expected timestamp.
                    hop_sec = self.eeg_buffer.hop_len / SystemConfig.SAMPLE_RATE_HZ
                    steps = math.ceil(hop_sec / SystemConfig.TIME_RESOLUTION)

                    # Ensure at least one step is filled
                    steps = max(1, steps)

                    for i in range(steps):
                        self.new_dsa_column.emit(ts + i * SystemConfig.TIME_RESOLUTION, freqs, psd)

                    self._io_executor.submit(Output.save_psd_to_csv, ts, freqs, psd)

                next_time += SystemConfig.TIME_RESOLUTION
                sleep = max(0.0, next_time - time.time())
                time.sleep(sleep)
            except Exception as e:
                print("Worker error:", e)


    def stop(self):
        self.running = False
        self._io_executor.shutdown(wait=True)
