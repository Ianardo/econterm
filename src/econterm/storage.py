"""SQLite storage for FRED series metadata and observations."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime

from econterm.models import Observation, SeriesInfo


class Repository:
    """Read and write series data in a local SQLite database.

    Use as a context manager so the connection is always closed:

        with Repository(path) as repo:
            obs = repo.get_observations("GDPC1")
    """

    def __init__(self, db_path):
        # sqlite3.connect creates the database file if it's missing,
        # but not the folder it lives in.
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        # Runs on every open so a fresh install needs no setup step.
        # IF NOT EXISTS makes this a no-op once the tables exist.
        self._create_tables()

    def _row_to_info(self, row):
        """Convert a series row into a SeriesInfo, parsing fetched_at back to a datetime."""
        return SeriesInfo(
            series_id=row["series_id"],
            title=row["title"],
            units=row["units"],
            frequency=row["frequency"],
            last_updated=row["last_updated"],
            fetched_at=datetime.fromisoformat(row["fetched_at"]),
        )

    def _create_tables(self):
        """Create the series and observations tables if they don't exist yet."""
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS series (
                series_id TEXT PRIMARY KEY,
                title TEXT,
                units TEXT,
                frequency TEXT,
                last_updated TEXT,
                fetched_at TEXT
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
        self.conn.commit()

    def save_series(self, series_id, title, units, frequency, last_updated, fetched_at):
        """Insert or update one series' metadata.

        Does not commit: call inside repo.transaction().
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO series (series_id, title, units, frequency, last_updated, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(series_id) DO UPDATE SET
                title = excluded.title,
                units = excluded.units,
                frequency = excluded.frequency,
                last_updated = excluded.last_updated,
                fetched_at = excluded.fetched_at
        """,
            (series_id, title, units, frequency, last_updated, fetched_at.isoformat()),
        )

    def save_observations(self, series_id, observations):
        """Insert or update observations for one series.

        observations is an iterable of (date, value) pairs.

        Does not commit: call inside repo.transaction().
        """
        cur = self.conn.cursor()
        rows = [(series_id, date, value) for date, value in observations]
        cur.executemany(
            """
            INSERT INTO observations (series_id, date, value)
            VALUES (?, ?, ?)
            ON CONFLICT(series_id, date) DO UPDATE SET
                value = excluded.value
        """,
            rows,
        )

    def get_observations(self, series_id):
        """Return all observations for a series in date order, or [] if none are stored."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT date, value
            FROM observations
            WHERE series_id = ?
            ORDER BY date
        """,
            (series_id,),
        )
        rows = cur.fetchall()
        return [Observation(date=row["date"], value=row["value"]) for row in rows]

    def list_series(self):
        """Return metadata for every stored series, sorted by series ID."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT series_id, title, units, frequency, last_updated, fetched_at
            FROM series
            ORDER BY series_id
        """)
        return [self._row_to_info(row) for row in cur.fetchall()]

    def get_series_metadata(self, series_id):
        """Return metadata for one series, or None if it isn't stored."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT series_id, title, units, frequency, last_updated, fetched_at
            FROM series
            WHERE series_id = ?
        """,
            (series_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_info(row)

    @contextmanager
    def transaction(self):
        """Commit everything inside the block together, or roll all of it back on error."""
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