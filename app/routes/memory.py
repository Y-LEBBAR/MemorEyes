# app/routes/memory.py
"""
Memory retrieval endpoints.

For now we keep it simple:
- POST /memory/retrieve
    → Given a person_id, return top-N Thought snippets for HUD display.

Ranking:
- sort by importance_score (descending), then created_at (newest first).
- Later we can replace this with a more advanced scoring function that
  uses context embeddings, episode info, etc.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.db import get_session
from core.memory.models import Thought
from core.people.models import Person

router = APIRouter(prefix="/memory", tags=["memory"])


# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------

def get_db():
    with get_session() as session:
        yield session


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class MemoryRetrieveRequest(BaseModel):
    """
    Request body for /memory/retrieve.

    person_id:
        The person we are currently looking at / interacting with.
    limit:
        Maximum number of thought snippets to return (HUD is limited).
    """
    person_id: str
    limit: int = 3


class MemorySnippet(BaseModel):
    """
    Minimal information about a Thought suitable for HUD display.
    """
    thought_id: str
    short_title: str
    importance_score: float
    created_at: str
    tags: Optional[list[str]] = None


class MemoryRetrieveResponse(BaseModel):
    """
    Response for /memory/retrieve.
    """
    person_id: str
    thoughts: List[MemorySnippet]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/retrieve", response_model=MemoryRetrieveResponse)
def retrieve_memories(payload: MemoryRetrieveRequest, db: Session = Depends(get_db)):
    """
    Return the top-N thoughts for a given person, ordered by:
        1) importance_score (descending)
        2) created_at (newest first)

    This is the first version of the "what should I show on the glasses
    right now?" brain.
    """
    # Ensure the person exists (optional but cleaner)
    person = db.query(Person).get(payload.person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    # Select thoughts where this person is the main subject.
    # Later we can also include:
    #   - thoughts where person_id in about_person_ids
    #   - self-thoughts where speaker_person_id == person_id
    thoughts_q = (
        db.query(Thought)
        .filter(Thought.main_about_person_id == payload.person_id)
        .order_by(
            Thought.importance_score.desc(),
            Thought.created_at.desc(),
        )
        .limit(payload.limit)
    )

    thoughts = thoughts_q.all()

    snippets: list[MemorySnippet] = []
    for t in thoughts:
        snippets.append(
            MemorySnippet(
                thought_id=t.id,
                short_title=t.short_title,
                importance_score=t.importance_score,
                created_at=t.created_at.isoformat(),
                tags=t.tags,
            )
        )

    return MemoryRetrieveResponse(
        person_id=payload.person_id,
        thoughts=snippets,
    )
