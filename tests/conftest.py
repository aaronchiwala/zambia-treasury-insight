import os
import tempfile
from pathlib import Path

# must be set before the app modules are imported
_tmp = Path(tempfile.mkdtemp()) / "test.sqlite"
os.environ["ZTI_DB"] = str(_tmp)

import pytest  # noqa: E402

from app import db, demo_data  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def demo_db():
    db.init()
    demo_data.load(seed=7)
    yield
