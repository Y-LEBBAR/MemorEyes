# app/routes/persons.py
"""
FastAPI routes for Person management and face-based identity operations.

This module exposes:
- POST /persons                  → create empty person profile
- GET  /persons/{id}             → retrieve a person
- POST /persons/enroll           → attach face embedding to a person
- POST /persons/search           → match an embedding to known persons
- POST /persons/create-with-image → full pipeline: detect → embed → dedupe → create
- GET  /persons                  → list all persons

These routes sit above the face pipeline and the people_service layer.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.db import get_session
from core.people.models import Person
from core.people.service import (
    create_person,
    add_face_sample,
    find_best_match,
)

# Face pipeline
from core.utils.image import load_image_rgb
from core.face.detector import detect_faces
from core.face.embeddings import get_face_embedding

import core.people.service as people_service

router = APIRouter()

DUPLICATE_THRESHOLD = 0.45


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class PersonCreate(BaseModel):
    display_name: Optional[str] = None
    tags: list[str] | None = None
    notes: Optional[str] = None


class PersonOut(BaseModel):
    id: str
    display_name: Optional[str]
    tags: list[str] | None
    notes: Optional[str]
    num_samples: int

    class Config:
        orm_mode = True


class FaceEnrollRequest(BaseModel):
    person_id: str
    embedding: list[float]


class FaceSearchRequest(BaseModel):
    embedding: list[float]
    distance_threshold: float = 0.4


class FaceSearchResult(BaseModel):
    person: Optional[PersonOut]
    distance: float


# DB dependency
def get_db():
    with get_session() as session:
        yield session


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("", response_model=PersonOut)
def create_person_endpoint(payload: PersonCreate, db: Session = Depends(get_db)):
    """
    Create an empty person profile (no face samples yet).
    """
    person = create_person(
        db,
        display_name=payload.display_name,
        tags=payload.tags or [],
        notes=payload.notes,
    )
    return person


@router.get("/{person_id}", response_model=PersonOut)
def get_person(person_id: str, db: Session = Depends(get_db)):
    """
    Retrieve a person profile by ID.
    """
    person = db.query(Person).get(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    return person


@router.post("/enroll", response_model=PersonOut)
def enroll_face(payload: FaceEnrollRequest, db: Session = Depends(get_db)):
    """
    Add a face embedding to an existing person.
    Updates the running-average embedding and sample count.
    """
    person = db.query(Person).get(payload.person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    emb = np.array(payload.embedding, dtype="float32")
    add_face_sample(db, person, emb)

    return person


@router.post("/search", response_model=FaceSearchResult)
def search_by_face(payload: FaceSearchRequest, db: Session = Depends(get_db)):
    """
    Find the best matching person given an embedding.
    Returns person=None if no match is above threshold.
    """
    emb = np.array(payload.embedding, dtype="float32")

    person, dist = find_best_match(
        db,
        emb,
        threshold=payload.distance_threshold,
    )

    if person is None:
        return FaceSearchResult(person=None, distance=dist)

    return FaceSearchResult(person=person, distance=dist)


@router.get("", summary="List all persons")
def list_persons(db: Session = Depends(get_db)):
    """
    Lightweight list of all persons.
    """
    persons = db.query(Person).all()

    return [
        {
            "id": p.id,
            "display_name": p.display_name,
            "num_samples": p.num_samples,
        }
        for p in persons
    ]


# ---------------------------------------------------------------------------
# CREATE PERSON WITH IMAGE (FULL PIPELINE)
# ---------------------------------------------------------------------------

@router.post("/create-with-image", summary="Create a person directly from a photo")
def create_person_with_image(
    file: UploadFile = File(...),
    display_name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    db: Session = Depends(get_db),
):
    """
    Full identity creation pipeline:

    Steps:
      1. Load uploaded image.
      2. Detect the most confident face.
      3. Compute embedding.
      4. Check for duplicates (via find_best_match).
      5. If duplicate → return duplicate info.
      6. Else create a new Person and attach a FaceSample.

    This bypasses the need to call /face/detect and /face/embedding separately.
    """

    # 1 — Read image
    img = load_image_rgb(file.file)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # 2 — Detect faces
    detections = detect_faces(img)
    if not detections:
        raise HTTPException(status_code=400, detail="No face detected")

    # Choose the best-scoring face
    best = max(detections, key=lambda d: d["score"])
    bbox = best["bbox"]

    # 3 — Compute embedding
    embedding = get_face_embedding(img, bbox)
    if embedding is None:
        raise HTTPException(status_code=500, detail="Embedding failed")

    # 4 — Check for duplicates
    matched_person, dist = people_service.find_best_match(db, embedding)

    if matched_person and dist < DUPLICATE_THRESHOLD:
        return {
            "duplicate": True,
            "person": {
                "id": matched_person.id,
                "display_name": matched_person.display_name,
                "num_samples": matched_person.num_samples,
            },
            "distance": dist,
        }

    # 5 — Create new person
    person = people_service.create_person(
        db,
        display_name=display_name or "Unnamed",
        tags=tags or [],
    )

    # Attach new sample
    add_face_sample(
        db=db,
        person=person,
        embedding=embedding,
        image_bbox=bbox,
    )

    db.commit()
    db.refresh(person)

    return {
        "duplicate": False,
        "person": {
            "id": person.id,
            "display_name": person.display_name,
            "num_samples": person.num_samples,
        },
        "distance_to_existing": dist,
    }

# ---------------------------------------------------------------------------
# IDENTIFY PERSON FROM IMAGE (FULL PIPELINE)
# ---------------------------------------------------------------------------

@router.post("/identify", summary="Identify the person in an uploaded image")
def identify_person(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Full identification pipeline.

    Steps:
      1. Load uploaded image (RGB ndarray)
      2. Detect the most confident face
      3. Compute embedding
      4. Compare embedding using people_service.find_best_match
      5. If above threshold → return matched person
      6. Else return 'unknown'

    Returns example:
    {
        "match": {
            "id": "...",
            "display_name": "Yannis",
            "num_samples": 4
        },
        "distance": 0.31,
        "is_unknown": false
    }
    """

    # 1 — Read image
    img = load_image_rgb(file.file)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # 2 — Detect faces
    detections = detect_faces(img)
    if not detections:
        return {
            "match": None,
            "distance": None,
            "is_unknown": True,
            "reason": "no_face_detected",
        }

    # Choose the best-scoring face
    best = max(detections, key=lambda d: d["score"])
    bbox = best["bbox"]

    # 3 — Compute embedding
    embedding = get_face_embedding(img, bbox)
    if embedding is None:
        raise HTTPException(status_code=500, detail="Embedding failed")

    # 4 — Compare against existing persons
    matched_person, dist = people_service.find_best_match(db, embedding)

    # No match above threshold
    if matched_person is None:
        return {
            "match": None,
            "distance": dist,
            "is_unknown": True,
            "reason": "no_match_above_threshold",
        }

    # 5 — Success
    return {
        "match": {
            "id": matched_person.id,
            "display_name": matched_person.display_name,
            "num_samples": matched_person.num_samples,
        },
        "distance": dist,
        "is_unknown": False,
    }