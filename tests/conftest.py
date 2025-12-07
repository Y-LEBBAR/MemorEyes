# tests/conftest.py
"""
Shared pytest fixtures for MemorEyes backend testing.

All tests use an in-memory SQLite database so that:
- No state persists between tests
- Tests are fast and deterministic
- The real database file is never touched

Fixtures:
    engine          → Creates all tables in a fresh in-memory DB
    db_session      → Yields a SQLAlchemy session per test
    sample_embedding → Returns a normalized 512-D vector for testing
    slightly_different_embedding → Returns another controlled vector
"""

import pytest
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from core.db import Base  # SQLAlchemy declarative base with all models
from collections.abc import Generator

# ---------------------------------------------------------------------------
# Engine fixture: new, clean in-memory DB for each test session
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def engine():
    # Use SQLite's in-memory mode. Perfect for tests.
    engine = create_engine("sqlite:///:memory:", echo=False)

    # Create all tables defined in core.db.Base.metadata
    Base.metadata.create_all(engine)

    return engine


# ---------------------------------------------------------------------------
# db_session fixture: yields a fresh session per test
# ---------------------------------------------------------------------------
@pytest.fixture
def db_session(engine) -> Generator[Session, None, None]:
    """
    Provides a clean SQLAlchemy session for each test.
    Rolls back changes after every test automatically.
    """
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        yield session
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Embedding fixtures: deterministic vectors for testing identity logic
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_embedding():
    """
    Returns a fixed normalized 512-D vector.
    Useful for consistent matching tests.
    """
    vec = np.ones(512, dtype="float32")
    vec /= np.linalg.norm(vec)
    return vec


@pytest.fixture
def slightly_different_embedding():
    """
    Returns another 512-D unit vector, slightly different from sample_embedding.
    Creates predictable cosine distance for match ranking tests.
    """
    vec = np.zeros(512, dtype="float32")
    vec[:256] = 1.0  # first half ones, second half zeros
    vec /= np.linalg.norm(vec)
    return vec
