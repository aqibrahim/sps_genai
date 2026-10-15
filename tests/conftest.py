"""Shared test setup: make the project root importable and start one API client."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def client():
    # Imported here so the spaCy and CNN models load once for the whole test run
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app)


@pytest.fixture(scope="session")
def sample_dir():
    return ROOT / "sample_images"
