"""
face.py

Purpose:
    FastAPI routes related to face processing and representation.
    This file provides:
        - POST /face/detect
            → detect face bounding boxes in an uploaded image
        - POST /face/embedding
            → compute a 512-D ArcFace embedding for the first detected face
        - POST /face/identify
            → detect all faces and try to match each to a known Person

Used by:
    - AR headset client for real-time face detection & identity lookup
    - Profile creation pipeline (capturing embeddings)
    - Identity matching & memory retrieval flow
    - Debugging and model evaluation workflows

Notes:
    - MVP behavior for /face/embedding uses the FIRST detected face only.
    - All embeddings are L2-normalized and ready for cosine similarity.
"""

from __future__ import annotations
from datetime import datetime, timezone

from typing import List, Optional, Generator

import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.utils.image import load_image_from_bytes, to_rgb
from core.face.detector import detect_faces
from core.face.embeddings import get_face_embedding
from core.db import get_session
from core.people.service import find_best_match

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

# Prefix is defined here; main.py includes this router WITHOUT an extra prefix.
router = APIRouter(prefix="/face", tags=["face"])


# ---------------------------------------------------------------------------
# DB dependency helper
# ---------------------------------------------------------------------------

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session.

    Wraps core.db.get_session() so each request gets its own Session with
    automatic commit/rollback.
    """
    with get_session() as session:
        yield session


# ---------------------------------------------------------------------------
# Pydantic models for /face/identify
# ---------------------------------------------------------------------------

class FaceMatchPerson(BaseModel):
    """
    Minimal information about a recognized person, suitable for HUD display.
    """
    id: str
    display_name: Optional[str] = None
    distance: float  # cosine distance; lower = more similar


class FaceIdentifyResult(BaseModel):
    """
    One detected face in the image.
    """
    bbox: List[float]           # [x1, y1, x2, y2]
    confidence: float
    person: Optional[FaceMatchPerson] = None


class IdentifyResponse(BaseModel):
    """
    Response wrapper for /face/identify.
    """
    faces: List[FaceIdentifyResult]


# ---------------------------------------------------------------------------
# /face/detect
# ---------------------------------------------------------------------------

@router.post("/detect")
async def detect_face_endpoint(file: UploadFile = File(...)):
    """
    Detect faces in an uploaded image.

    Returns:
        {
            "faces": [
                {"bbox": [x1, y1, x2, y2], "confidence": 0.98},
                ...
            ]
        }
    """
    # Validate upload type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type")

    # Load and convert image to numpy RGB
    file_bytes = await file.read()
    image = load_image_from_bytes(file_bytes)
    image = to_rgb(image)

    # Run detection
    detections = detect_faces(image)  # expected: list[(bbox, score)]

    # Format results
    response = []
    for box, score in detections:
        response.append(
            {
                "bbox": box.tolist(),
                "confidence": float(score),
            }
        )

    return {"faces": response}


# ---------------------------------------------------------------------------
# /face/embedding
# ---------------------------------------------------------------------------

@router.post("/embedding")
async def embedding_endpoint(file: UploadFile = File(...)):
    """
    Upload an image → detect face → compute 512-D embedding
    for the FIRST detected face.

    Returns:
        {
            "embedding": [...512 floats...]
        }
    """
    # Validate input type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type")

    # Read image bytes → numpy array
    file_bytes = await file.read()
    image = load_image_from_bytes(file_bytes)
    image = to_rgb(image)

    # Detect face(s)
    detections = detect_faces(image)
    if not detections:
        raise HTTPException(status_code=404, detail="No face detected")

    # Use first detected face (MVP behavior)
    bbox, score = detections[0]

    # Compute 512-D embedding
    embedding = get_face_embedding(image, bbox)

    return {"embedding": embedding.tolist()}


# ---------------------------------------------------------------------------
# /face/identify
# ---------------------------------------------------------------------------

@router.post("/identify", response_model=IdentifyResponse)
async def identify_faces_endpoint(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    High-level endpoint:

    - Detect ALL faces in an uploaded image
    - For each face:
        • compute a 512-D embedding
        • match against known Persons using cosine distance

    Returns:
        {
          "faces": [
            {
              "bbox": [...],
              "confidence": 0.98,
              "person": {
                  "id": "...",
                  "display_name": "Alice",
                  "distance": 0.12
              }
            },
            {
              "bbox": [...],
              "confidence": 0.95,
              "person": null
            }
          ]
        }
    """
    # Validate upload type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type")

    # Load and convert image
    file_bytes = await file.read()
    image = load_image_from_bytes(file_bytes)
    image = to_rgb(image)

    # 1) Detect faces
    detections = detect_faces(image)  # [(bbox, score), ...]
    if not detections:
        # No faces: return empty list (not an error)
        return IdentifyResponse(faces=[])

    results: list[FaceIdentifyResult] = []

    # 2) For each detection: compute embedding and try to match a person
    for bbox, score in detections:
        # Compute 512-D embedding for this face
        emb = get_face_embedding(image, bbox)

        # Ensure numpy array (in case underlying function changes later)
        if not isinstance(emb, np.ndarray):
            emb = np.asarray(emb, dtype="float32")

        # Match against known persons in DB
        person, dist = find_best_match(db, emb)

        if person is not None:
            person.last_seen_at = datetime.now(timezone.utc)
            db.commit()

            person_out = FaceMatchPerson(
                id=person.id,
                display_name=person.display_name,
                distance=dist,
            )
        else:
            person_out = None

        # Convert bbox to plain list[float] for JSON
        bbox_list = [float(v) for v in bbox.tolist()]

        results.append(
            FaceIdentifyResult(
                bbox=bbox_list,
                confidence=float(score),
                person=person_out,
            )
        )

    return IdentifyResponse(faces=results)
