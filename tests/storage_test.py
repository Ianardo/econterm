from econterm.storage import Repository
import sqlite3
import pytest
from pathlib import Path
from contextlib import contextmanager

def test_save_and_list_series(tmp_path):
    repo = Repository(tmp_path / "test.db")
    repo.init_db()

    with pytest.raises(ValueError):
        with repo.transaction():
            repo.save_series("TEST1", "Test", "Units", "Monthly", "2026-01-01")
            raise ValueError("boom")

    rows = repo.list_series()
    assert rows == []