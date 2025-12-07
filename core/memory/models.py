# core/memory/models.py
"""
Memory-layer ORM models: Episode and Thought.

- Episode: one interaction (meeting, call, etc.) with a primary person.
- Thought: one atomic memory/fact tied to a person (and optionally an episode).

Both link back to core.people.models.Person via person_id.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    DateTime,
    ForeignKey,
    Text,
    Float,
    JSON,
)
from sqlalchemy.orm import relationship

from ..db import Base   # relative import: core/memory → core/db


def gen_uuid() -> str:
    """Generate a random UUID string (for primary keys)."""
    return str(uuid.uuid4())


class Episode(Base):
    """
    One interaction event.

    Examples:
    - A visit to a pharmacy.
    - A coffee with a friend.
    - A Zoom call.

    For now this is minimal but extended to support transcripts with speaker info.
    """

    __tablename__ = "episode"

    id = Column(String, primary_key=True, default=gen_uuid)

    # Main person in this interaction (optional; could be group).
    # This is the *other* person, not the wearer.
    primary_person_id = Column(
        String,
        ForeignKey("person.id"),
        nullable=True,
    )

    started_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )
    ended_at = Column(
        DateTime,
        nullable=True,
    )

    # Optional context
    location = Column(String, nullable=True)
    context_notes = Column(Text, nullable=True)

    # ------------------------------------------------------------------
    # Transcript / audio fields
    # ------------------------------------------------------------------

    # Path to the raw audio file for this interaction (if you store it).
    # Example: "data/audio/episode_123.wav"
    audio_path = Column(String, nullable=True)          # NEW

    # Full transcript collapsed into a single text blob for quick search / debug.
    # Example: "You: Hi / Them: Hello, nice to see you / You: How have you been? ..."
    transcript_text = Column(Text, nullable=True)       # NEW

    # Language code of the transcript (e.g. "en", "fr"), if known.
    transcript_language = Column(String, nullable=True) # NEW

    # Diarized transcript segments with speaker info.
    #
    # Stored as JSON list of segments. Each segment is expected to look like:
    # {
    #   "start": 0.0,                   # float, seconds from episode start
    #   "end": 3.2,                     # float, seconds from episode start
    #   "speaker_role": "wearer" | "other",  # who is speaking at a high level
    #   "speaker_person_id": "uuid" | null,  # if we can map "other" to a Person
    #   "text": "Hi, good to see you"
    # }
    #
    # You can also add extra keys later (e.g. "emotion", "sentiment").
    transcript_segments = Column(JSON, nullable=True)   # NEW

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    # Relationship to Thought
    thoughts = relationship("Thought", back_populates="episode")


class Thought(Base):
    """
    One atomic memory / fact.

    Key ideas:
    - A Thought can be about *anything* (people, products, places, ideas).
    - It is primarily tied to:
        • who expressed it (speaker_person_id), if we know that
        • optional people it is about (main_about_person_id + about_person_ids)
        • the episode where it arose (episode_id)

    Examples:
    - Speaker (pharmacist) says:
        "I'm interested in Product A but the price is too high."
      → speaker_person_id = pharmacist.id
      → main_about_person_id = pharmacist.id (their attitude)
      → entities["products"] = ["Product A"]

    - Wearer thinks:
        "She seemed nervous when we talked about her mother."
      → speaker_person_id = wearer.id (internal note)
      → main_about_person_id = her.id
    """

    __tablename__ = "thought"

    id = Column(String, primary_key=True, default=gen_uuid)

    # Optional link to the Episode in which this thought arose.
    episode_id = Column(
        String,
        ForeignKey("episode.id"),
        nullable=True,
    )

    # Who expressed this thought/content (if known).
    # For spoken content, this is the speaker.
    # For internal notes, it is typically the wearer.
    speaker_person_id = Column(
        String,
        ForeignKey("person.id"),
        nullable=True,
    )

    # Person this thought is primarily ABOUT (if any).
    # E.g. "she loves pistachio ice cream" → main_about_person_id = her.id
    main_about_person_id = Column(
        String,
        ForeignKey("person.id"),
        nullable=True,
    )

    # Additional people this thought is about (0..N).
    # Stored as a JSON list of person_id strings.
    # E.g. a group memory about several people.
    about_person_ids = Column(JSON, nullable=True)

    # Generic entities beyond people: products, places, orgs, etc.
    # Example structure:
    # {
    #   "products": ["Product A"],
    #   "places":   ["Jules' Gelato"],
    #   "topics":   ["pricing","ice_cream"]
    # }
    entities = Column(JSON, nullable=True)

    # Core content
    short_title = Column(String, nullable=False)  # 1-line summary for glasses
    text = Column(Text, nullable=False)           # full descriptive text

    # Simple generic metadata (can be extended later)
    tags = Column(JSON, nullable=True)           # ["preference","pricing","ice_cream"]

    # Rough “size” or importance of this memory (0..1).
    importance_score = Column(
        Float,
        nullable=False,
        default=0.5,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    # Relationship back to Episode; we keep this, it’s still valid.
    episode = relationship("Episode", back_populates="thoughts")
