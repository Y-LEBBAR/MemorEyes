# app/routes/episodes.py
"""
Episode-related API endpoints.

An Episode represents one interaction event, e.g.:
- a visit to a pharmacy,
- a coffee with a friend,
- a meeting with a client.

This router provides:
- POST /episodes
    → create/start an Episode
- PATCH /episodes/{episode_id}
    → update/close an Episode (ended_at, notes, etc.)
- GET /episodes/{episode_id}
    → fetch a single Episode
- GET /episodes/by-person/{person_id}
    → list Episodes where this person was the primary counterpart
"""

from __future__ import annotations

from typing import Optional, List, Generator
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.db import get_session
from core.memory.models import Episode
from core.people.models import Person


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/episodes", tags=["episodes"])


# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session.
    """
    with get_session() as session:
        yield session


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class EpisodeCreate(BaseModel):
    """
    Payload to create/start an Episode.

    Fields:
    - primary_person_id: main person you’re interacting with (optional)
    - location: free-text location string (optional)
    - context_notes: arbitrary notes (optional)
    """
    primary_person_id: Optional[str] = None
    location: Optional[str] = None
    context_notes: Optional[str] = None


class EpisodeUpdate(BaseModel):
    """
    Payload to update an Episode.

    Typical use:
    - when the interaction ends: set ended_at
    - update location / notes
    """
    ended_at: Optional[datetime] = None
    location: Optional[str] = None
    context_notes: Optional[str] = None


class EpisodeOut(BaseModel):
    """
    Public representation of an Episode.

    Only includes fields we actually need right now.
    """
    id: str
    primary_person_id: Optional[str]
    started_at: datetime
    ended_at: Optional[datetime]
    location: Optional[str]
    context_notes: Optional[str]

    created_at: datetime

    class Config:
        # Pydantic v2: replace old orm_mode=True
        from_attributes = True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("", response_model=EpisodeOut)
def create_episode(payload: EpisodeCreate, db: Session = Depends(get_db)):
    """
    Create/start a new Episode.

    For now:
    - Optionally validate primary_person_id
    - started_at is set by the ORM default (datetime.utcnow)
    """
    if payload.primary_person_id is not None:
        person = db.query(Person).get(payload.primary_person_id)
        if not person:
            raise HTTPException(status_code=400, detail="Unknown primary_person_id")

    episode = Episode(
        primary_person_id=payload.primary_person_id,
        location=payload.location,
        context_notes=payload.context_notes,
    )
    db.add(episode)
    db.flush()  # assign episode.id

    return episode


@router.patch("/{episode_id}", response_model=EpisodeOut)
def update_episode(
    episode_id: str,
    payload: EpisodeUpdate,
    db: Session = Depends(get_db),
):
    """
    Update an Episode.

    Examples:
    - When the interaction ends, set ended_at.
    - Update location or context_notes.
    """
    episode = db.query(Episode).get(episode_id)
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")

    if payload.ended_at is not None:
        episode.ended_at = payload.ended_at

    if payload.location is not None:
        episode.location = payload.location

    if payload.context_notes is not None:
        episode.context_notes = payload.context_notes

    return episode


@router.get("/{episode_id}", response_model=EpisodeOut)
def get_episode(episode_id: str, db: Session = Depends(get_db)):
    """
    Fetch a single Episode by id.
    """
    episode = db.query(Episode).get(episode_id)
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    return episode


@router.get("/by-person/{person_id}", response_model=List[EpisodeOut])
def list_episodes_for_person(person_id: str, db: Session = Depends(get_db)):
    """
    List Episodes where this person is the primary counterpart.

    Ordered newest-first by started_at.
    """
    episodes = (
        db.query(Episode)
        .filter(Episode.primary_person_id == person_id)
        .order_by(Episode.started_at.desc())
        .all()
    )
    return episodes
