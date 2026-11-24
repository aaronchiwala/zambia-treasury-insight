import os
import tempfile
from pathlib import Path

# must be set before the app modules are imported
_tmp = Path(tempfile.mkdtemp()) / "test.sqlite"
os.environ["ZTI_DB"] = str(_tmp)
os.environ["ZTI_AUTO_INGEST"] = "0"
os.environ.pop("ZTI_PASSWORD", None)

import pytest  # noqa: E402

from app import db, demo_data  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def demo_db():
    db.init()
    demo_data.load(seed=7)
    yield


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """An isolated, empty database for tests that write (leaves the shared demo DB untouched)."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "isolated.sqlite")
    db.init()
    return db.DB_PATH
