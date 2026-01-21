import os, sys

# fix lsl for single file .exe
if getattr(sys, "frozen", False):
    base = sys._MEIPASS if hasattr(sys, "_MEIPASS") else os.path.dirname(sys.executable)
    os.environ["PYLSL_LIB"] = os.path.join(base, "pylsl", "lib", "lsl.dll")

from pylsl import StreamInlet, resolve_byprop
import csv
from datetime import datetime as dt
import numpy as np
from config import SystemConfig


class EEGStream:
    def __init__(self):
        self.receiving = False
        self._inlet = None
        self.connect()

    def connect(self):
        try:
            streams = resolve_byprop("name", "EEG_DATA", timeout=2)
            if streams:
                self._inlet = StreamInlet(streams[0])
                print(f"Connected to: {streams[0].name()} (uid: {streams[0].uid()})")
                self.receiving = True
            else:
                self.receiving = False
        except Exception as e:
            print(f"LSL Connection error: {e}")
            self.receiving = False

    def read_samples(self):
        if not self.receiving or self._inlet is None:
            return []

        samples = []
        while True:
            sample, _ = self._inlet.pull_sample(timeout=0)
            if sample is None:
                break

            try:
                timestamp, eeg_str = sample[0].split(",")
                value = float(eeg_str)
                timestamp = dt.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
                if not np.isfinite(value):
                    value = np.nan
                elif value < SystemConfig.EEG_BOUNDS[0] or value > SystemConfig.EEG_BOUNDS[1]:
                    print(f"Out of bounds: {value}")
                    value = np.nan

                samples.append((timestamp, value))

            except Exception as e:
                # Invalid sample discarded
                print("Invalid sample: ",  e)
                continue

        return samples


def build_filename(base_dir, freqs, start_time=None):
    freqs = np.asarray(freqs)
    if freqs.size < 2:
        raise ValueError("At least 2 frequency bins required")

    df = round(freqs[1] - freqs[0], 6)

    if start_time is None:
        start_time = dt.now()
    else:
        start_time = dt.fromtimestamp(start_time)
    ts = start_time.strftime("%Y-%m-%d")

    filename = f"dsa_{ts}_df{df:.1f}Hz.csv"
    return os.path.join(base_dir, filename)

def save_psd_to_csv(freqs, psd_db, base_dir, timestamp=None):
    """
    Save a single PSD column to a CSV file.

    File format:
    - Header written on first creation: 'timestamp' followed by one column per frequency bin labeled
      like 'f_{freq:.2f}_Hz'.
    - Each call appends a new row with the provided timestamp and PSD values (dB/Hz).

    Args:
        freqs: 1D array-like of frequency bin centers (Hz), length N.
        psd_db: 1D array-like of PSD values in dB/Hz, length N.
        filepath: Destination CSV file path (will be created if it does not exist).
        timestamp: Optional timestamp for the row. If a datetime is provided, it will be formatted
                   in ISO 8601 with milliseconds. If None, the current time is used.

    Raises:
        ValueError: If shapes of inputs mismatch or if appending to an existing file with an
                    incompatible number of columns (i.e., different frequency bins).
    """

    # Normalize inputs
    freqs = np.asarray(freqs).ravel()
    psd_db = np.asarray(psd_db).ravel()
    if freqs.shape[0] != psd_db.shape[0]:
        raise ValueError("freqs and psd_db must have the same length")

    # Prepare timestamp string
    if timestamp is None:
        ts_str = dt.now().isoformat(timespec="milliseconds")
    else:
        if isinstance(timestamp, dt):
            ts_str = timestamp.isoformat(timespec="milliseconds")
        else:
            ts_str = str(timestamp)
    # Ensure directory exists
    filepath = build_filename(base_dir, freqs, timestamp)
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0

    # If appending to an existing file, validate column count
    if not write_header:
        try:
            with open(filepath, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    write_header = True
                else:
                    expected_cols = 1 + len(freqs)  # timestamp + N frequencies
                    if len(header) != expected_cols:
                        raise ValueError(
                            f"Existing CSV has {len(header)} columns but expected {expected_cols}. "
                            "Frequency bins must remain identical across saves."
                        )
        except FileNotFoundError:
            write_header = True
    # Write
    # TODO: Save in a predefined resolution of 0.5 Hz
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            freq_headers = [f"f_{freq:.2f}_Hz" for freq in freqs]
            writer.writerow(["timestamp"] + freq_headers)

        row_values = [int(np.round(x, 0)) if np.isfinite(x) else "" for x in psd_db]
        writer.writerow([ts_str] + row_values)
