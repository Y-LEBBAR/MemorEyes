# core/people/service.py
"""
Service layer for Person and FaceSample.

This module contains *logic* (no FastAPI, no HTTP), so it can be unit-tested
directly and reused by different front-ends.

Responsibilities:
- Create a new Person.
- Add face samples to a Person and maintain the running-average embedding.
- Find the best matching Person for a given query embedding (brute-force).
"""

from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from core.people.models import Person, FaceSample

from typing import Any, Dict, Union
from core.db import get_session


# ---------------------------------------------------------------------------
# Embedding serialization helpers
# ---------------------------------------------------------------------------

def _pack_embedding(emb: np.ndarray) -> bytes:
    """
    Convert a float32 numpy vector into raw bytes suitable for storing in BLOB.

    We always force float32 to keep size consistent.
    """
    emb = emb.astype("float32")
    return emb.tobytes()


def _unpack_embedding(buf: bytes) -> np.ndarray:
    """
    Convert raw bytes from the DB back into a float32 numpy vector.
    """
    arr = np.frombuffer(buf, dtype="float32")
    return arr


# ---------------------------------------------------------------------------
# Person creation / enrollment
# ---------------------------------------------------------------------------

def create_person(
    db: Session,
    display_name: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    first_embedding: np.ndarray | None = None,
) -> Person:
    """
    Create a new Person record.

    Optionally:
    - attach an initial face embedding
    - create a first FaceSample for that embedding
    """
    person = Person(
        display_name=display_name,
        tags=tags or [],
        notes=notes,
    )

    if first_embedding is not None:
        # Store the first embedding as both the primary and a sample.
        packed = _pack_embedding(first_embedding)
        person.avg_embedding = packed
        person.num_samples = 1

        sample = FaceSample(
            embedding=packed,
        )
        person.face_samples.append(sample)

    db.add(person)
    # No commit here; the caller (route or higher layer) controls transaction.
    db.flush()  # assign person.id and sample.id

    return person


def add_face_sample(
    db: Session,
    person: Person,
    embedding: np.ndarray,
) -> FaceSample:
    """
    Add a new face embedding for an existing Person and update their primary
    embedding as a running average.

    Running average formula:
        new_avg = (old_avg * n + new_sample) / (n + 1)
    """
    # Create and attach the sample row.
    sample = FaceSample(
        person_id=person.id,
        embedding=_pack_embedding(embedding),
    )
    db.add(sample)

    # If we already have a primary embedding, update it as a running average.
    if person.avg_embedding:
        old = _unpack_embedding(person.avg_embedding)
        n = person.num_samples
        new_avg = (old * n + embedding) / (n + 1)
    else:
        # First ever embedding for this person.
        new_avg = embedding

    person.avg_embedding = _pack_embedding(new_avg.astype("float32"))
    person.num_samples = person.num_samples + 1

    # We do not commit here; caller controls commit.
    return sample


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def find_best_match(
    db: Session,
    query_embedding: np.ndarray,
    threshold: float = 0.4,
) -> tuple[Person | None, float]:
    """
    Find the closest Person for the given embedding using cosine distance.

    Returns:
        (best_person, distance)

        - best_person: Person instance or None if no one passes the threshold
        - distance: 1 - cosine_similarity, lower is better

    threshold:
        Maximum allowed distance to consider this a match.
        You can tune this empirically using your models.
    """
    # Fetch all people that actually have a avg_embedding.
    people = db.query(Person).filter(Person.avg_embedding.isnot(None)).all()
    if not people:
        # No one enrolled yet.
        return None, float("inf")

    q = query_embedding.astype("float32")
    q_norm = np.linalg.norm(q) or 1e-8

    best_person = None
    best_dist = float("inf")

    for p in people:
        # Unpack the stored embedding and compute cosine similarity.
        emb = _unpack_embedding(p.avg_embedding)
        denom = (np.linalg.norm(emb) * q_norm) or 1e-8
        sim = float(np.dot(emb, q) / denom)

        # Cosine distance = 1 - cosine similarity
        dist = 1.0 - sim

        if dist < best_dist:
            best_dist = dist
            best_person = p

    # If best distance is above threshold, treat as "no match".
    if best_dist > threshold:
        return None, best_dist

    return best_person, best_dist

def get_person_facts(person_key: Union[int, str]) -> Dict[str, Any]:
    """
    Return lightweight facts about a person for UI display.

    person_key:
        - usually a Person.id (int / str)

    Returns a dict such as:
        {
            "exists": True/False,
            "id": ...,
            "name": ...,
            "created_at": ...,
            "last_seen_at": ...,
            "episode_count": 0,   # TODO: wire to Episode when ready
            "notes": "",
        }
    """
    with get_session() as db:
        # Try by primary key
        person = db.get(Person, person_key)  # type: ignore[arg-type]

        if person is None:
            return {
                "exists": False,
                "name": str(person_key),
                "episode_count": 0,
                "notes": "",
            }

        facts: Dict[str, Any] = {
            "exists": True,
            "id": person.id,
            "name": getattr(person, "display_name", None) or str(person.id),
            "created_at": getattr(person, "created_at", None),
            "last_seen_at": getattr(person, "last_seen_at", None),
            "episode_count": 0,   # TODO: compute once Episodes are wired
            "notes": getattr(person, "notes", "") or "",
        }
        return facts