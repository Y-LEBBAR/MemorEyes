# core/nlp/thoughts_service.py
# ----------------------------
# Stubs for creating Thoughts from text segments.
#
# For the MVP demo we are not yet wiring real ASR; these functions define
# the contract that the live client and the future NLP pipeline will use.

from __future__ import annotations

from typing import Optional

import numpy as np

from core.db import get_session
from core.memory.models import Thought  # adjust import to real path
from core.memory.search import add_thought_embedding


def embed_text(text: str) -> np.ndarray:
    """
    Placeholder sentence embedding function.

    TODO: replace with a real sentence-transformer model.
    For now, this should just raise or return a dummy vector.
    """
    # Example dummy vector (not suitable for production):
    return np.random.normal(size=(128,)).astype(np.float32)


def create_thought_from_segment(
    episode_id: str,
    text: str,
    t_start: float,
    t_end: float,
    primary_person_id: Optional[str] = None,
) -> Thought:
    """
    Create and persist a Thought from a text segment.

    This is the main entry point for the future ASR/segmentation pipeline:
      - clean text
      - infer labels (MVP: leave empty)
      - embed text
      - insert Thought row into DB
      - register embedding in the memory index
    """
    emb = embed_text(text)
    labels: list[str] = []  # TODO: implement label inference

    with get_session() as session:
        thought = Thought(
            episode_id=episode_id,
            timestamp_start=t_start,
            timestamp_end=t_end,
            text=text,
            labels=labels,
            importance_score=0.0,
            embedding=emb.tolist(),
        )
        session.add(thought)
        session.commit()
        session.refresh(thought)

    # Register in in-memory index for retrieval
    add_thought_embedding(thought.id, emb)

    return thought
