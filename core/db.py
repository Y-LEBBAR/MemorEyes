# core/db.py
"""
Database configuration for MemorEyes.

This module:
- Creates the SQLAlchemy engine (currently SQLite, easy to swap later).
- Defines the Base class all ORM models inherit from.
- Exposes a get_session() context manager for safe DB access.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------

# For now we use a local SQLite file. Later we can replace this with:
#   postgresql+psycopg2://user:password@host:port/dbname
DATABASE_URL = "sqlite:///./memoreyes.db"

# The engine manages the actual DB connection.
# `check_same_thread=False` is required when SQLite is used in a multi-threaded
# FastAPI app.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

# Session factory. Each request should use its own Session instance.
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Base class that all ORM models must inherit from.
Base = declarative_base()


# ---------------------------------------------------------------------------
# Session helper
# ---------------------------------------------------------------------------

@contextmanager
def get_session() -> Iterator[Session]:
    """
    Context manager that yields a database session and handles commit/rollback.

    Usage:
        from core.db import get_session

        with get_session() as db:
            db.add(obj)
            ...
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        # If anything goes wrong, rollback to keep DB consistent.
        session.rollback()
        raise
    finally:
        # Always close the connection.
        session.close()
