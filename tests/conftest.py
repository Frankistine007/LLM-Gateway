import pathlib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app import models
from app.config import settings
from app.database import create_sqlite_engine
from app.dependencies import get_db
from app.main import app

TEST_DB_PATH = pathlib.Path(__file__).parent / "test_gateway.db"
TEST_DATABASE_URL = f"sqlite:///{TEST_DB_PATH}"

# Same engine config as the app, so tests exercise the real locking behaviour.
engine = create_sqlite_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db


@pytest.fixture(scope="session", autouse=True)
def _test_db():
    """Create a throwaway test_gateway.db for the whole test session, then delete it."""
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
    models.Base.metadata.create_all(bind=engine)
    yield
    engine.dispose()
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def db_session():
    """A fresh SQLAlchemy session per test, so reads see other sessions' commits."""
    session = TestSessionLocal()
    yield session
    session.close()


@pytest.fixture()
def session_factory():
    """The raw session factory, for tests that need several sessions at once
    (e.g. driving concurrent requests from multiple threads)."""
    return TestSessionLocal


@pytest.fixture()
def api_key(client):
    """Creates a fresh throwaway client via POST /clients and returns its API key."""
    resp = client.post(
        "/clients",
        json={"name": "pytest-client"},
        headers={"X-Admin-Key": settings.admin_api_key},
    )
    assert resp.status_code == 200
    return resp.json()["api_key"]
