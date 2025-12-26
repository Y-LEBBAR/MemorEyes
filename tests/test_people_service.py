# tests/test_people_service.py
"""
Unit tests for core.people.service.

We test:
    - Creating a person
    - Adding face samples
    - Running-average embedding updates
    - Best-match selection logic
"""

import numpy as np
from core.people import service as people_service


# ---------------------------------------------------------------------------
# Helper: create a simple person for testing
# ---------------------------------------------------------------------------
def create_basic_person(db, name="TestUser"):
    """
    Creates a basic person using the real create_person() service
    so that tests reflect real system behavior.
    """
    return people_service.create_person(
        db=db,
        display_name=name,
        tags=["test"],
        notes="unit test",
    )


# ---------------------------------------------------------------------------
# TEST 1 — Creating a person
# ---------------------------------------------------------------------------
def test_create_person(db_session):
    person = create_basic_person(db_session, name="Alice")

    assert person.id is not None
    assert person.display_name == "Alice"
    assert person.tags == ["test"]
    assert person.notes == "unit test"
    assert person.num_samples == 0
    assert person.primary_embedding is None  # No samples enrolled yet


# ---------------------------------------------------------------------------
# TEST 2 — Adding a face sample updates num_samples + primary_embedding
# ---------------------------------------------------------------------------
def test_add_face_sample(db_session, sample_embedding):
    person = create_basic_person(db_session)

    people_service.add_face_sample(
        db=db_session,
        person_id=person.id,
        embedding=sample_embedding,
        image_bbox=[0, 0, 1, 1],  # dummy bbox
    )

    db_session.commit()
    db_session.refresh(person)

    assert person.num_samples == 1
    assert person.primary_embedding is not None
    assert np.allclose(person.primary_embedding, sample_embedding)


# ---------------------------------------------------------------------------
# TEST 3 — Running-average embedding logic
# ---------------------------------------------------------------------------
def test_running_average(db_session, sample_embedding, slightly_different_embedding):
    person = create_basic_person(db_session)

    # Add first embedding
    people_service.add_face_sample(db_session, person.id, sample_embedding, [0, 0, 10, 10])

    # Add second embedding
    people_service.add_face_sample(
        db_session, person.id, slightly_different_embedding, [0, 0, 10, 10]
    )

    db_session.commit()
    db_session.refresh(person)

    # Expected running average:
    avg = (sample_embedding + slightly_different_embedding) / 2
    avg /= np.linalg.norm(avg)

    assert person.num_samples == 2
    assert np.allclose(person.primary_embedding, avg, atol=1e-5)


# ---------------------------------------------------------------------------
# TEST 4 — find_best_match returns correct person for exact embedding
# ---------------------------------------------------------------------------
def test_find_best_match_exact(db_session, sample_embedding):
    person = create_basic_person(db_session)

    # Enroll embedding so the system can match it
    people_service.add_face_sample(db_session, person.id, sample_embedding, [0, 0, 10, 10])

    match = people_service.find_best_match(db_session, sample_embedding)

    assert match is not None
    assert match.person.id == person.id
    assert match.distance == 0.0


# ---------------------------------------------------------------------------
# TEST 5 — find_best_match returns None when above threshold
# ---------------------------------------------------------------------------
def test_find_best_match_none(db_session, sample_embedding):
    person = create_basic_person(db_session)

    people_service.add_face_sample(db_session, person.id, sample_embedding, [0, 0, 10, 10])

    # A vector pointing in another direction → force distance > threshold
    distant_vector = np.zeros(512, dtype="float32")
    distant_vector[0] = 1.0

    match = people_service.find_best_match(
        db_session,
        distant_vector,
        threshold=0.2,  # strict threshold to guarantee no match
    )

    assert match is None


# ---------------------------------------------------------------------------
# TEST 6 — Closest match among multiple persons
# ---------------------------------------------------------------------------
def test_best_of_two_people(db_session, sample_embedding, slightly_different_embedding):
    # Person A
    person_a = create_basic_person(db_session, "A")
    people_service.add_face_sample(db_session, person_a.id, sample_embedding, [0, 0, 10, 10])

    # Person B
    person_b = create_basic_person(db_session, "B")
    people_service.add_face_sample(db_session, person_b.id, slightly_different_embedding, [0, 0, 10, 10])

    # Query should match B, since input vector equals B's embedding
    match = people_service.find_best_match(db_session, slightly_different_embedding)

    assert match is not None
    assert match.person.id == person_b.id
