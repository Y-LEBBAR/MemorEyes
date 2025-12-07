# tests/test_last_seen_at_update.py
"""
Tests that /face/identify updates person.last_seen_at correctly
when a recognized person is matched.
"""

import io
import numpy as np
from unittest.mock import patch
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from app.main import app
from core.people.service import create_person, add_face_sample


client = TestClient(app)


def dummy_image_bytes():
    """1x1 PNG for test uploads."""
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
        b"\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDAT"
        b"\x08\xd77`\x00\x00\x00\x02\x00\x01\xe2!"
        b"\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    )


@patch("app.routes.face.detect_faces")
@patch("app.routes.face.get_face_embedding")
def test_last_seen_at_updates(mock_embed, mock_detect, db_session):
    """
    When a person is identified via /face/identify,
    their last_seen_at field must update.
    """

    # Setup: Create person + embedding sample
    person = create_person(db_session, display_name="Alice", tags=[], notes=None)

    emb = np.ones(512, dtype="float32")
    emb /= np.linalg.norm(emb)

    add_face_sample(db_session, person.id, emb, [0, 0, 10, 10])

    db_session.commit()
    db_session.refresh(person)

    # Record current timestamp BEFORE identification
    before = datetime.now(timezone.utc)

    # Mock detector + embedding to return the same embedding
    mock_detect.return_value = [([0, 0, 10, 10], 0.99)]
    mock_embed.return_value = emb

    # POST request
    files = {"file": ("test.png", dummy_image_bytes(), "image/png")}

    response = client.post("/face/identify", files=files)
    assert response.status_code == 200

    # Refresh the person record
    db_session.refresh(person)

    # last_seen_at MUST be updated and later than 'before'
    assert person.last_seen_at is not None
    assert person.last_seen_at > before


@patch("app.routes.face.detect_faces")
@patch("app.routes.face.get_face_embedding")
def test_last_seen_at_not_updated_when_no_match(mock_embed, mock_detect, db_session):
    """
    If no existing person matches the embedding, last_seen_at must NOT update.
    """

    # Create sample person with some embedding
    person = create_person(db_session, "Bob", tags=[], notes=None)
    emb_ref = np.ones(512, dtype="float32") / np.sqrt(512)
    add_face_sample(db_session, person.id, emb_ref, [0, 0, 10, 10])
    db_session.commit()

    # Nothing matches this vector → very different embedding
    emb_query = np.zeros(512, dtype="float32")
    emb_query[0] = 1.0

    before = datetime.now(timezone.utc)

    mock_detect.return_value = [([0, 0, 10, 10], 0.99)]
    mock_embed.return_value = emb_query

    files = {"file": ("test.png", dummy_image_bytes(), "image/png")}

    response = client.post("/face/identify", files=files)
    assert response.status_code == 200

    db_session.refresh(person)

    # last_seen_at must remain None — no match occurred
    assert person.last_seen_at is None or person.last_seen_at <= before
