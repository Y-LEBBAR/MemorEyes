# app/routes/thoughts.py
"""
Thought-related API endpoints.

This module exposes:
- POST /thoughts
    → Create a new Thought (memory/fact) tied to:
        - who expressed it (speaker_person_id)
        - who/what it is about (main_about_person_id, about_person_ids, entities)
        - optionally the Episode where it arose.

- GET /thoughts/by-person/{person_id}
    → List thoughts where this person is the main subject (main_about_person_id).
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.db import get_session
from core.memory.models import Thought, Episode
from core.people.models import Person

router = APIRouter(prefix="/thoughts", tags=["thoughts"])


# -----------------------------------------------------------------------------
# DB dependency (shared pattern with other routers)
# -----------------------------------------------------------------------------

def get_db():
    """
    FastAPI dependency that yields a DB Session.

    Wraps core.db.get_session() (context manager).
    """
    with get_session() as session:
        yield session


# -----------------------------------------------------------------------------
# Pydantic schemas
# -----------------------------------------------------------------------------

class ThoughtCreate(BaseModel):
    """
    Payload to create a new Thought.

    Notes:
    - speaker_person_id: who expressed this content (wearer, pharmacist, etc.)
    - main_about_person_id: who this thought is primarily ABOUT (optional)
    - about_person_ids: other people this thought is about (optional list)
    - entities: generic non-person subjects (products, places, topics, etc.)
    """
    episode_id: Optional[str] = None

    speaker_person_id: Optional[str] = None
    main_about_person_id: Optional[str] = None
    about_person_ids: Optional[List[str]] = None

    entities: Optional[Dict[str, Any]] = None

    short_title: str
    text: str

    importance_score: float = 0.5
    tags: Optional[List[str]] = None


class ThoughtOut(BaseModel):
    """
    What the API returns for a Thought.
    """
    id: str

    episode_id: Optional[str]
    speaker_person_id: Optional[str]
    main_about_person_id: Optional[str]
    about_person_ids: Optional[List[str]]
    entities: Optional[Dict[str, Any]]

    short_title: str
    text: str

    importance_score: float
    tags: Optional[List[str]]

    created_at: str

    class Config:
        orm_mode = True  # allow returning SQLAlchemy model instances directly


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@router.post("", response_model=ThoughtOut)
def create_thought_endpoint(payload: ThoughtCreate, db: Session = Depends(get_db)):
    """
    Create a new Thought.

    Typical usage:
    - The mobile app / glasses companion app calls this when the user
      manually adds a note about someone.
    - Later, an automatic pipeline (from transcripts) can also call this.

    Validation:
    - If speaker_person_id / main_about_person_id / about_person_ids are given,
      verify that those Person rows exist.
    - If episode_id is given, verify that Episode exists.
    """

    # Validate speaker_person_id if provided
    if payload.speaker_person_id is not None:
        speaker = db.query(Person).get(payload.speaker_person_id)
        if not speaker:
            raise HTTPException(status_code=400, detail="Unknown speaker_person_id")

    # Validate main_about_person_id if provided
    if payload.main_about_person_id is not None:
        about_person = db.query(Person).get(payload.main_about_person_id)
        if not about_person:
            raise HTTPException(status_code=400, detail="Unknown main_about_person_id")

    # Validate each about_person_ids entry if provided
    if payload.about_person_ids:
        for pid in payload.about_person_ids:
            person = db.query(Person).get(pid)
            if not person:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown person in about_person_ids: {pid}",
                )

    # Validate episode_id if provided
    if payload.episode_id is not None:
        episode = db.query(Episode).get(payload.episode_id)
        if not episode:
            raise HTTPException(status_code=400, detail="Unknown episode_id")

    thought = Thought(
        episode_id=payload.episode_id,
        speaker_person_id=payload.speaker_person_id,
        main_about_person_id=payload.main_about_person_id,
        about_person_ids=payload.about_person_ids,
        entities=payload.entities,
        short_title=payload.short_title,
        text=payload.text,
        importance_score=payload.importance_score,
        tags=payload.tags,
    )

    db.add(thought)
    db.flush()  # assign id

    return thought


@router.get("/by-person/{person_id}", response_model=List[ThoughtOut])
def list_thoughts_for_person(person_id: str, db: Session = Depends(get_db)):
    """
    List Thoughts where this person is the main subject.

    For now we keep it simple:
    - filter on Thought.main_about_person_id == person_id
    - order newest first

    Later, we can extend this to also include:
    - Thoughts where person_id is in about_person_ids
    - Thoughts where person_id == speaker_person_id (e.g. self-disclosures)
    """
    thoughts = (
        db.query(Thought)
        .filter(Thought.main_about_person_id == person_id)
        .order_by(Thought.created_at.desc())
        .all()
    )

    return thoughts
