# tests/test_create_with_image_route.py
"""
Tests for POST /persons/create-with-image using FastAPI's TestClient.

We DO NOT run the real face detector / embedding models here.
Instead, we patch:
    - detect_faces
    - get_face_embedding

This isolates endpoint logic and makes tests deterministic and fast.

What we test:
    1. Creating a NEW person from an image
    2. Detecting a DUPLICATE person when embeddings match closely
"""

import io
import numpy as np
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from core.people.service import create_person, add_face_sample


client = TestClient(app)


# ---------------------------------------------------------------------------
# Helper: create a dummy image file in memory
# ---------------------------------------------------------------------------
def dummy_image_bytes():
    """
    Produces a tiny valid PNG image (1x1 pixel black).
    FastAPI UploadFile accepts any bytes-like object as file content.
    """
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
        b"\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDAT"
        b"\x08\xd77`\x00\x00\x00\x02\x00\x01\xe2!"
        b"\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    )


# ---------------------------------------------------------------------------
# TEST 1 — New person creation
# ---------------------------------------------------------------------------
@patch("app.routes.persons.detect_faces")
@patch("app.routes.persons.get_face_embedding")
def test_create_new_person(mock_embed, mock_detect, db_session):
    """
    Endpoint should create a new person when:
        - face detection finds a bbox
        - embedding doesn't match existing persons
    """

    # Fake face detection: one face, high confidence
    mock_detect.return_value = [{"bbox": [0, 0, 10, 10], "score": 0.99}]

    # Fake embedding: stable deterministic 512-D vector
    embedding = np.ones(512, dtype="float32")
    embedding /= np.linalg.norm(embedding)
    mock_embed.return_value = embedding

    file_bytes = dummy_image_bytes()
    files = {"file": ("test.png", file_bytes, "image/png")}

    response = client.post(
        "/persons/create-with-image",
        files=files,
        data={"display_name": "Alice"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["duplicate"] is False
    assert data["person"]["display_name"] == "Alice"
    assert data["person"]["num_samples"] == 1


# ---------------------------------------------------------------------------
# TEST 2 — Duplicate detection when embedding matches existing person
# ---------------------------------------------------------------------------
@patch("app.routes.persons.detect_faces")
@patch("app.routes.persons.get_face_embedding")
def test_duplicate_detection(mock_embed, mock_detect, db_session):
    """
    If the embedded vector matches an existing person closely (distance < threshold),
    endpoint should return duplicate=True and not create a new person.
    """

    # Setup: Create an existing person with an embedding
    person = create_person(db_session, display_name="Bob", tags=[], notes=None)

    embedding = np.ones(512, dtype="float32")
    embedding /= np.linalg.norm(embedding)

    add_face_sample(
        db_session,
        person_id=person.id,
        embedding=embedding,
        image_bbox=[0, 0, 10, 10],
    )

    # Fake detection + embedding for this request (same embedding)
    mock_detect.return_value = [{"bbox": [0, 0, 10, 10], "score": 0.99}]
    mock_embed.return_value = embedding

    file_bytes = dummy_image_bytes()
    files = {"file": ("test.png", file_bytes, "image/png")}

    response = client.post(
        "/persons/create-with-image",
        files=files,
        data={"display_name": "Someone"},
    )

    assert response.status_code == 200
    data = response.json()

    # Must identify as duplicate
    assert data["duplicate"] is True
    assert data["person"]["id"] == person.id
    assert "distance" in data
