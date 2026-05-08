import csv
import glob
import os
from datetime import datetime as dt

import numpy as np

from .. import config


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
        threshold = start_time_dt
        loaded_data = []

        pattern = os.path.join(config.BASE_DIR, "dsa_*.csv")
        for filepath in sorted(glob.glob(pattern)):
            try:
                filename = os.path.basename(filepath)
                file_date = None
                try:
                    if filename.startswith("dsa_"):
                        file_date = dt.strptime(filename[4:-4], "%Y-%m-%d").date()
                except ValueError:
                    pass

                with open(filepath, "r", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if not header:
                        continue

                    has_duration = "duration" in header
                    psd_offset = 2 if has_duration else 1
                    if len(header) != psd_offset + len(config.FREQ_BINS):
                        continue

                    for row in reader:
                        if not row:
                            continue
                        try:
                            ts = row[0]
                            try:
                                row_dt = dt.fromisoformat(ts)
                            except ValueError:
                                if file_date is None:
                                    continue
                                row_dt = dt.combine(
                                    file_date,
                                    dt.strptime(ts, "%H:%M:%S.%f").time()
                                )

                            if row_dt < threshold:
                                continue

                            duration = float(row[1]) if has_duration else config.TIME_RESOLUTION

                            psd = np.fromiter(
                                (float(v) if v and v.lower() != "nan" else np.nan for v in row[psd_offset:]),
                                dtype=np.float32
                            )

                            loaded_data.append((row_dt.timestamp(), duration, psd))

                        except (ValueError, IndexError):
                            continue

            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        loaded_data.sort(key=lambda x: x[0])
        return loaded_data

    @staticmethod
    def save_psd_to_csv(timestamp, duration, psd):
        psd = np.asarray(psd).ravel()

        try:
            ts_dt = (
                dt.now() if timestamp is None else
                timestamp if isinstance(timestamp, dt) else
                dt.fromtimestamp(timestamp) if isinstance(timestamp, (int, float)) else
                dt.fromisoformat(str(timestamp))
            )
        except ValueError:
            ts_dt = dt.now()

        ts_str = ts_dt.strftime("%H:%M:%S.%f")[:-3]
        filepath = Output._build_filename(config.BASE_DIR, ts_dt.timestamp())
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
        existing_rows, timestamp_exists, truncated = [], False, False

        if not write_header:
            try:
                with open(filepath, "r", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)

                    if not header:
                        write_header = True
                    else:
                        if len(header) != 2 + len(config.FREQ_BINS):
                            raise ValueError("Frequency bins mismatch.")

                        for row in reader:
                            if not row:
                                continue
                            if row[0] > ts_str:
                                truncated = True
                                continue
                            existing_rows.append(row)
                            if row[0] == ts_str:
                                timestamp_exists = True

            except FileNotFoundError:
                write_header = True

        if truncated:
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "duration"] +
                                [f"f_{f:.1f}_Hz" for f in config.FREQ_BINS])
                writer.writerows(existing_rows)
            write_header = False

        if timestamp_exists:
            return

        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow(["timestamp", "duration"] +
                                [f"f_{f:.1f}_Hz" for f in config.FREQ_BINS])

            writer.writerow(
                [ts_str, f"{int(duration)}"] +
                [np.round(x, 4) if np.isfinite(x) else "nan" for x in psd]
            )
