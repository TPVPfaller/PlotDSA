import os, sys

# fix lsl for single file .exe
if getattr(sys, "frozen", False):
    base = sys._MEIPASS if hasattr(sys, "_MEIPASS") else os.path.dirname(sys.executable)
    os.environ["PYLSL_LIB"] = os.path.join(base, "pylsl", "lib", "lsl.dll")

from pylsl import StreamInlet, resolve_byprop
from datetime import datetime
import numpy as np


class EEGStream:
    def __init__(self):
        streams = resolve_byprop("name", "EEG_DATA")
        if not streams:
            raise RuntimeError("EEG stream not found")

        self._inlet = StreamInlet(streams[0])

    def read_samples(self):
        samples = []
        while True:
            sample, _ = self._inlet.pull_sample(timeout=0)
            if sample is None:
                break

            try:
                timestamp, eeg_str = sample[0].split(",")
                value = float(eeg_str)
                timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
                timestamp_sec = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second + timestamp.microsecond / 1e6
                if np.isfinite(value):
                    samples.append((timestamp, value))
            except Exception as e:
                # Invalid sample discarded
                print("Invalid sample: ",  e)
                continue

        return samples