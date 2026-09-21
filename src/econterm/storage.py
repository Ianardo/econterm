import sqlite3
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "econterm.db"

class Repository:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
    def init_db(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS series (
                series_id TEXT PRIMARY KEY,
                title TEXT,
                units TEXT,
                frequency TEXT,
                last_updated TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                series_id TEXT,
                date TEXT,
                value REAL,
                PRIMARY KEY (series_id, date)
            )
        """)
        
    def save_series(self, series_id, title, units, frequency, last_updated):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO series (series_id, title, units, frequency, last_updated)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(series_id) DO UPDATE SET
                title = excluded.title,
                units = excluded.units,
                frequency = excluded.frequency,
                last_updated = excluded.last_updated
        """, (series_id, title, units, frequency, last_updated))

    def save_observations(self, series_id, observations):
        cur = self.conn.cursor()
        rows = [(series_id, date, value) for date, value in observations]
        cur.executemany("""
            INSERT INTO observations (series_id, date, value)
            VALUES (?, ?, ?)
            ON CONFLICT(series_id, date) DO UPDATE SET
                value = excluded.value
        """, rows)
    def get_observations(self, series_id):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT date, value
            FROM observations
            WHERE series_id = ?
            ORDER BY date
        """, (series_id,))
        return cur.fetchall()

    def list_series(self):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT series_id, title, units, frequency, last_updated
            FROM series
            ORDER BY series_id
        """)
        return cur.fetchall()
    
    @contextmanager
    def transaction(self):
        conn = self.conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    def close(self):
        self.conn.close()
            
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        

if __name__ == "__main__":
    with Repository(DB_PATH) as repo:
        repo.init_db()
        with repo.transaction():
            repo.save_series("GDPC1", "Real Gross Domestic Product",
                            "Billions of Chained 2017 Dollars", "Quarterly", "2026-09-18")
            repo.save_observations("GDPC1", [("2026-01-01", 23000.0)])
        print(repo.list_series())