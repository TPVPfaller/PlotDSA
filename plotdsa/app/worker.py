import time
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import QObject, Signal, Slot
from ..core.buffers import EEGBuffer
from ..io.output import Output
from ..io.input import EEGStream
from .. import config


class ProcessingWorker(QObject):
    new_dsa_column = Signal(float, object, int)  # timestamp, power spectral density array (psd), steps
    new_samples = Signal(object)  # list of eeg values

    def __init__(self, user_config):
        super().__init__()
        self.running = True
        self.user_config = user_config
        self._new_config = None

        self.stream = EEGStream()
        self._last_connection_state = self.stream.receiving
        self._dsa_time_origin = None
        self.eeg_buffer = EEGBuffer(
            self.user_config.window_sec,
            self.user_config.window_overlap
        )
        self._io_executor = ThreadPoolExecutor(max_workers=1)

    @Slot(object)
    def apply_config(self, user_config):
        self._new_config = user_config

    @Slot()
    def run(self):
        next_time = time.time()
        try:
            while self.running:

                if self._new_config:
                    self.user_config = self._new_config
                    self._new_config = None
                    self.eeg_buffer.apply_config(self.user_config.window_sec, self.user_config.window_overlap)
                if not self.stream.receiving:
                    self.stream.connect()
                    # Emit on change
                    if self.stream.receiving != self._last_connection_state:
                        self._last_connection_state = self.stream.receiving
                    time.sleep(0.5)
                    continue

                samples = self.stream.read_lsl_samples()
                self.new_samples.emit([sample[1] for sample in samples])

                method = 'multitaper' if self.user_config.use_multitaper else 'welch'
                dsa_columns = self.eeg_buffer.get_dsa_columns(samples, method=method)

                for ts, psd in dsa_columns:
                    if psd is None:
                        continue

                    steps = self._get_dsa_steps(ts)

                    duration = steps * config.TIME_RESOLUTION

                    self.new_dsa_column.emit(ts, psd, steps)

                    self._io_executor.submit(Output.save_psd_to_csv, ts, duration, psd)

                next_time += config.TIME_RESOLUTION
                sleep = max(0.0, next_time - time.time())
                time.sleep(sleep)
        finally:
            self._io_executor.shutdown(wait=True)



    def stop(self):
        self.running = False

    def _get_dsa_steps(self, ts):
        current_slot = self._get_dsa_slot(ts)
        hop_sec = self.eeg_buffer.hop_len / config.SAMPLE_RATE_HZ
        next_slot = self._get_dsa_slot(ts + hop_sec)
        return max(1, next_slot - current_slot)

    def _get_dsa_slot(self, ts):
        if self._dsa_time_origin is None:
            self._dsa_time_origin = ts
        offset = (ts - self._dsa_time_origin) / config.TIME_RESOLUTION
        return int(offset + 0.5)
