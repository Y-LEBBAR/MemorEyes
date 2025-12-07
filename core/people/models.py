# core/people/models.py
"""
ORM models for people and their stored face embeddings.

These are *only* about identity:
- Person: "who is this human?"
- FaceSample: "one embedding (and optional bbox/image) we captured for them"

Higher-level memory/thought models will attach to Person later.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from sqlalchemy import DateTime

from sqlalchemy import (
    Column,
    String,
    DateTime,
    Integer,
    ForeignKey,
    Text,
    JSON,
)
from sqlalchemy.dialects.sqlite import BLOB
from sqlalchemy.orm import relationship

from core.db import Base


def gen_uuid() -> str:
    """Generate a random UUID as a string. Used for primary keys."""
    return str(uuid.uuid4())


class Person(Base):
    """
    A person the system knows about.

    This is the "profile" node that other things (face samples, thoughts,
    episodes, relationship stats) will attach to.
    """

    __tablename__ = "person"

    # Primary key. We use a stringified UUID to stay DB-agnostic.
    id = Column(String, primary_key=True, default=gen_uuid)

    # Display name to show in UIs. Can be edited by the user.
    display_name = Column(String, nullable=True)

    # Free-form notes written by the wearer ("met at conference", etc.).
    notes = Column(Text, nullable=True)

    # Simple JSON array of tags ("friend", "client", "family", ...).
    tags = Column(JSON, nullable=True)

    # One "canonical" face embedding used for quick matching.
    # We also store all samples in FaceSample; this is a running average.
    avg_embedding = Column(BLOB, nullable=True)

    # How many FaceSample embeddings we used to build the avg_embedding.
    num_samples = Column(Integer, nullable=False, default=0)

    # Timestamp of the last time we saw this person (for recency sorting).
    last_seen_at = Column(DateTime(timezone=True), nullable=True)

    # Timestamps.
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationship: list of all stored face samples for this person.
    face_samples = relationship(
        "FaceSample",
        back_populates="person",
        cascade="all, delete-orphan",
    )


class FaceSample(Base):
    """
    A single stored face embedding.

    This might be:
    - linked to a known Person (person_id not null)
    - or temporarily "unassigned" before the user labels it.
    """

    __tablename__ = "face_sample"

    # Primary key.
    id = Column(String, primary_key=True, default=gen_uuid)

    # Which person this sample belongs to (can be null until labeled).
    person_id = Column(
        String,
        ForeignKey("person.id"),
        nullable=True,
    )
    person = relationship("Person", back_populates="face_samples")

    # Embedding bytes (we pack a float32 numpy array into this).
    embedding = Column(BLOB, nullable=False)

    # Optional: where this sample came from, for debugging.
    # e.g. "data/faces/jane_001.jpg"
    image_path = Column(String, nullable=True)

    # Optional: bounding box within the original image.
    # Example:
    #   {"x1": 120, "y1": 80, "x2": 220, "y2": 200}
    bbox = Column(JSON, nullable=True)

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
