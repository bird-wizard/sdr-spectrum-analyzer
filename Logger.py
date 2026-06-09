from PyQt6.QtCore import pyqtSignal, QObject
import sqlite3
import csv
from datetime import datetime
from Config import classify_band

DB_PATH = "sdr_activity.db"

class ActivityLogger(QObject):
    # for the GUI activity table.
    # (band, count, last_seen_timestamp, peak_power_dbm)
    stats_ready = pyqtSignal(list)

    def __init__(self):
        # run QObject
        super().__init__()
        self._init_db()

    def _init_db(self):
        con = sqlite3.connect(DB_PATH)
        con.execute("""
            CREATE TABLE IF NOT EXISTS freq_activity (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                freq_mhz        REAL    NOT NULL,
                power_dbm       REAL    NOT NULL,
                band            TEXT,
                matched_station TEXT
            )
        """)
        con.commit()
        con.close()

    def log_peaks(self, peaks: list):
        # called by Worker
        if not peaks:
            return
        now = datetime.utcnow().isoformat(timespec="seconds")
        rows = [
            (now, fm, pm, classify_band(fm), str(station) if station else None)
            for fm, pm, station in peaks
        ]
        con = sqlite3.connect(DB_PATH)
        con.executemany(
            """INSERT INTO freq_activity
               (timestamp, freq_mhz, power_dbm, band, matched_station)
               VALUES (?, ?, ?, ?, ?)""",
            rows,
        )
        con.commit()
        con.close()

    def fetch_band_stats(self):
        # SQL query for GUI table
        con = sqlite3.connect(DB_PATH)
        rows = con.execute("""
            SELECT   ROUND(freq_mhz, 1)  AS freq,
                     band,
                     COUNT(*)            AS detections,
                     MAX(timestamp)      AS last_seen,
                     MAX(power_dbm)      AS peak_power
            FROM     freq_activity
            GROUP BY ROUND(freq_mhz, 1)
            ORDER BY detections DESC
        """).fetchall()
        con.close()
        self.stats_ready.emit(rows)

    def export_csv(self, path: str):
        # SQL query for CSV export
        con = sqlite3.connect(DB_PATH)
        rows = con.execute(
            """SELECT timestamp, freq_mhz, power_dbm, band, matched_station
               FROM freq_activity ORDER BY timestamp"""
        ).fetchall()
        con.close()
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "freq_mhz", "power_dbm", "band", "matched_station"])
            writer.writerows(rows)

    def clear_history(self):
        # Clear option
        con = sqlite3.connect(DB_PATH)
        con.execute("DELETE FROM freq_activity")
        con.commit()
        con.close()
