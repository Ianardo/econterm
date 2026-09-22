from econterm.storage import Repository
import sqlite3
import pytest
from pathlib import Path
from contextlib import contextmanager

def test_aligning(tmp_path):
    repo = Repository(tmp_path / "test.db")
    repo.init_db()
    
    y_obs = [
        ("2020-01-01", 100.0),
        ("2020-04-01", 101.5),
        ("2020-07-01", 99.8),
        ("2020-10-01", 102.3),   # only in y
    ]
    x_obs = [
        ("2019-10-01", 3.9),     # only in x
        ("2020-01-01", 3.6),
        ("2020-04-01", 13.0),
        ("2020-07-01", 10.2),
    ]

    with repo.transaction():
        repo.save_observations("y_obs", y_obs)
        repo.save_observations("x_obs", x_obs)

    rows = repo.get_aligned("y_obs", "x_obs")
    assert [tuple(row) for row in rows] == [
        ("2020-01-01", 100.0, 3.6),
        ("2020-04-01", 101.5, 13.0),
        ("2020-07-01", 99.8, 10.2),
    ]