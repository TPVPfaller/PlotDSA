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
        self.receiving = False
        streams = resolve_byprop("name", "EEG_DATA", timeout=5)

        if len(streams) > 0:
            self._inlet = StreamInlet(streams[0])
            print(f"Connected to: {streams[0].name()} (uid: {streams[0].uid()})")
            self.receiving = True

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
                if np.isfinite(value):
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

    # Frequency resolution (Δf)
    df = round(freqs[1] - freqs[0], 6)

    # Timestamp string
    if start_time is None:
        start_time = datetime.now()
    ts = start_time.strftime("%Y-%m-%d")

    # Build filename
    filename = f"dsa_{ts}_df{df:.1f}Hz.csv"

    return base_dir + "\\" + filename

def save_psd_to_csv(freqs, psd_db, base_dir, timestamp=None):
    """
    Save a single PSD column to a CSV file.

    File format:
    - Header written on first creation: 'timestamp' followed by one column per frequency bin labelled
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
    import csv
    from datetime import datetime as _dt
    import numpy as _np
    import os as _os

    # Normalize inputs
    freqs = _np.asarray(freqs).ravel()
    psd_db = _np.asarray(psd_db).ravel()

    if freqs.shape[0] != psd_db.shape[0]:
        raise ValueError("freqs and psd_db must have the same length")

    # Prepare timestamp string
    if timestamp is None:
        ts_str = _dt.now().isoformat(timespec="milliseconds")
    else:
        if isinstance(timestamp, _dt):
            ts_str = timestamp.isoformat(timespec="milliseconds")
        else:
            ts_str = str(timestamp)

    # Ensure directory exists
    filepath = build_filename(base_dir, freqs, timestamp)
    directory = _os.path.dirname(filepath)
    if directory and not _os.path.exists(directory):
        _os.makedirs(directory, exist_ok=True)

    write_header = not _os.path.exists(filepath) or _os.path.getsize(filepath) == 0

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
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            freq_headers = [f"f_{freq:.2f}_Hz" for freq in freqs]
            writer.writerow(["timestamp"] + freq_headers)

        row_values = [np.round(x, 1) if _np.isfinite(x) else "" for x in psd_db]
        writer.writerow([ts_str] + row_values)
