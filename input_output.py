import csv
import datetime
import glob
import os
import sys
from datetime import datetime as dt

import numpy as np
from pylsl import StreamInlet, resolve_byprop

import config


# fix lsl for single file .exe
if getattr(sys, "frozen", False):
    base = sys._MEIPASS if hasattr(sys, "_MEIPASS") else os.path.dirname(sys.executable)
    os.environ["PYLSL_LIB"] = os.path.join(base, "pylsl", "lib", "lsl.dll")


class EEGStream:
    def __init__(self):
        self.receiving = False
        self._inlet = None
        self.connect()

    def connect(self):
        try:
            streams = resolve_byprop("name", config.LSL_STREAM_NAME, timeout=2)
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
                try:
                    timestamp = dt.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
                except ValueError:
                    time_part = dt.strptime(timestamp, "%H:%M:%S.%f").time()
                    timestamp = dt.combine(dt.now().date(), time_part)

                if not np.isfinite(value):
                    value = np.nan
                elif value < config.EEG_BOUNDS[0] or value > config.EEG_BOUNDS[1]:
                    print(f"Out of bounds: {value}")
                    value = np.nan

                samples.append((timestamp, value))
            except Exception as e:
                print("Invalid sample: ", e)
                continue

        return samples


class Output:
    @staticmethod
    def _build_filename(base_dir, start_time=None):
        if start_time is None:
            start_time = dt.now()
        else:
            start_time = dt.fromtimestamp(start_time)
        ts = start_time.strftime("%Y-%m-%d")
        filename = f"dsa_{ts}.csv"
        return os.path.join(base_dir, filename)

    @staticmethod
    def load_psd_from_time(start_time_dt):
        """
        Loads PSD data starting from a specific datetime.
        Returns a list of (timestamp, duration, psd) tuples.
        """
        threshold = start_time_dt
        loaded_data = []
        pattern = os.path.join(config.BASE_DIR, "dsa_*.csv")
        filepaths = sorted(glob.glob(pattern))

        for filepath in filepaths:
            if not os.path.exists(filepath):
                continue

            filename = os.path.basename(filepath)
            file_date = None
            if filename.startswith("dsa_") and filename.endswith(".csv"):
                date_part = filename[4:-4]
                try:
                    file_date = dt.strptime(date_part, "%Y-%m-%d").date()
                except ValueError:
                    file_date = None

            try:
                with open(filepath, "r", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if not header:
                        continue

                    has_duration = "duration" in header
                    expected_cols = (2 if has_duration else 1) + len(config.FREQ_BINS)
                    if len(header) != expected_cols:
                        continue

                    for row in reader:
                        if not row:
                            continue
                        try:
                            ts_str = row[0]
                            try:
                                row_dt = dt.fromisoformat(ts_str)
                            except ValueError:
                                time_part = dt.strptime(ts_str, "%H:%M:%S.%f").time()
                                if file_date is None:
                                    continue
                                row_dt = dt.combine(file_date, time_part)

                            if row_dt < threshold:
                                continue

                            if has_duration:
                                duration = float(row[1])
                                psd_start_idx = 2
                            else:
                                duration = config.TIME_RESOLUTION
                                psd_start_idx = 1

                            psd = []
                            for val_str in row[psd_start_idx:]:
                                if val_str == "" or val_str.lower() == "nan":
                                    psd.append(np.nan)
                                else:
                                    psd.append(float(val_str))

                            loaded_data.append((row_dt.timestamp(), duration, np.array(psd, dtype=np.float32)))
                        except (ValueError, IndexError):
                            continue
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        loaded_data.sort(key=lambda x: x[0])
        return loaded_data

    @staticmethod
    def save_psd_to_csv(timestamp, duration, psd):
        psd = np.asarray(psd).ravel()

        if timestamp is None:
            ts_dt = dt.now()
        elif isinstance(timestamp, dt):
            ts_dt = timestamp
        elif isinstance(timestamp, (int, float)):
            ts_dt = dt.fromtimestamp(timestamp)
        else:
            try:
                ts_dt = dt.fromisoformat(str(timestamp))
            except ValueError:
                ts_dt = dt.now()

        ts_str = ts_dt.strftime("%H:%M:%S.%f")[:-3]
        filepath = Output._build_filename(config.BASE_DIR, ts_dt.timestamp())
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
        existing_rows = []
        timestamp_exists = False
        truncated = False

        if not write_header:
            try:
                with open(filepath, "r", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header is None:
                        write_header = True
                    else:
                        expected_cols = 2 + len(config.FREQ_BINS)
                        if len(header) != expected_cols:
                            raise ValueError(
                                f"Existing CSV has {len(header)} columns but expected {expected_cols}. "
                                "Frequency bins must remain identical across saves."
                            )

                        for row in reader:
                            if not row:
                                continue
                            row_ts = row[0]
                            if row_ts > ts_str:
                                truncated = True
                                continue
                            existing_rows.append(row)
                            if row_ts == ts_str:
                                timestamp_exists = True
            except FileNotFoundError:
                print(f"File not found: {filepath} creating new file")
                write_header = True

        if truncated:
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                freq_headers = [f"f_{freq:.1f}_Hz" for freq in config.FREQ_BINS]
                writer.writerow(["timestamp", "duration"] + freq_headers)
                writer.writerows(existing_rows)
            write_header = False

        if timestamp_exists:
            return

        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                freq_headers = [f"f_{freq:.1f}_Hz" for freq in config.FREQ_BINS]
                writer.writerow(["timestamp", "duration"] + freq_headers)

            row_values = [np.round(x, 4) if np.isfinite(x) else "nan" for x in psd]
            writer.writerow([ts_str, f"{duration:.3f}"] + row_values)
