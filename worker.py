import time
import math
from concurrent.futures import ThreadPoolExecutor
from PySide6.QtCore import QObject, Signal, Slot
from data import EEGStream, EEGBuffer, DSABuffer, Output
from config import SystemConfig


class ProcessingWorker(QObject):
    new_data = Signal(object, object, object)  # dsa_buffer, freqs, psd
    new_sample = Signal(float, float)  # epoch_seconds, eeg_value
    connection_changed = Signal(bool)  # LSL connection state

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
        self.dsa_buffer = DSABuffer(self.config.segment_sec)
        self._io_executor = ThreadPoolExecutor(max_workers=1)

    @Slot(object)
    def apply_config(self, config):
        self._new_config = config

    @Slot()
    def run(self):
        # Emit initial connection state
        try:
            self.connection_changed.emit(bool(self.stream.receiving))
        except Exception:
            pass

        while self.running:
            # Apply new config if available
            if self._new_config:
                self.config = self._new_config
                self._new_config = None
                self.eeg_buffer.apply_config(
                    self.config.window_sec,
                    self.config.segment_sec,
                    self.config.segment_overlap,
                    self.config.window_overlap
                )
                self.dsa_buffer.apply_config(self.config.segment_sec)

            if not self.stream.receiving:
                self.stream.connect()
                # Emit on change
                if self.stream.receiving != self._last_connection_state:
                    self._last_connection_state = self.stream.receiving
                    try:
                        self.connection_changed.emit(bool(self.stream.receiving))
                    except Exception:
                        pass
                time.sleep(0.5)
                continue

            samples = self.stream.read_samples()
            dsa_columns, checked_samples = self.eeg_buffer.get_dsa_columns(samples)

            # Emit each individual sample for the EEG view
            for ts_val, eeg_val in checked_samples:
                self.new_sample.emit(ts_val, eeg_val)

            for ts, freqs, psd in dsa_columns:
                if psd is None:
                    continue

                # Calculate how many update steps this window covers.
                # We use ceil to ensure we bridge the gap to the next column's expected timestamp.
                hop_duration = self.eeg_buffer.hop_len / SystemConfig.SAMPLE_RATE_HZ
                steps = math.ceil(hop_duration / SystemConfig.TIME_RESOLUTION)

                # Ensure at least one step is filled
                steps = max(1, steps)

                for i in range(steps):
                    self.dsa_buffer.append(
                        ts + i * SystemConfig.TIME_RESOLUTION,
                        freqs,
                        psd
                    )

                self.new_data.emit(self.dsa_buffer, freqs, psd)
                self._io_executor.submit(Output.save_psd_to_csv, ts, freqs, psd)

            time.sleep(SystemConfig.TIME_RESOLUTION)

    def stop(self):
        self.running = False
        self._io_executor.shutdown(wait=True)
