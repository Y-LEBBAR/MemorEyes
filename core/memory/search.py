# core/memory/search.py
# ----------------------
# Memory search and ranking for MemorEyes.
#
# This module provides:
#   - An in-memory vector index for Thought embeddings
#   - cosine_similarity(): numerical similarity between two vectors
#   - RetrievalContext: query context (who is present, context embedding, etc.)
#   - get_top_insight(): given a context, return the best matching Thought(s)
#
# It replaces the earlier "profile matching" logic. Face/person identity
# matching is handled via the people/face pipeline; this module is focused on
# retrieving *thoughts/memories* given a semantic context.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from core.db import get_session
from core.memory.models import Thought  # adapt import if your model lives elsewhere


# ---------------------------------------------------------------------------
# Basic similarity
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.

    Embeddings must be 1D numpy arrays. They do not need to be pre-normalized;
    we normalize inside this function.
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Embeddings must be 1D vectors")

    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0

    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Simple in-memory index for Thought embeddings
# (You can swap this for FAISS/HNSW later while keeping the same interface.)
# ---------------------------------------------------------------------------


class InMemoryIndex:
    """
    Minimal in-process vector index.

    Stores {thought_id -> embedding} in memory and supports cosine-similarity
    search. This is sufficient for the MVP and for unit/integration tests.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors: Dict[str, np.ndarray] = {}

    def add(self, key: str, vec: np.ndarray) -> None:
        if vec.ndim != 1:
            raise ValueError("Embedding must be a 1D vector")
        if vec.shape[0] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {vec.shape[0]}")
        self._vectors[key] = vec.astype(np.float32)

    def search(self, query: np.ndarray, top_k: int = 50) -> List[Tuple[str, float]]:
        if query.ndim != 1:
            raise ValueError("Query embedding must be a 1D vector")
        if not self._vectors:
            return []

        q = query.astype(np.float32)
        keys = list(self._vectors.keys())
        mat = np.stack([self._vectors[k] for k in keys], axis=0)  # (N, D)

        # Cosine similarity for all stored vectors vs query
        q_norm = np.linalg.norm(q) + 1e-8
        mat_norms = np.linalg.norm(mat, axis=1) + 1e-8
        sims = (mat @ q) / (mat_norms * q_norm)

        order = np.argsort(-sims)[:top_k]
        return [(keys[i], float(sims[i])) for i in order]


_index: InMemoryIndex | None = None


def _get_index(dim: int) -> InMemoryIndex:
    global _index
    if _index is None:
        _index = InMemoryIndex(dim)
    return _index


def add_thought_embedding(thought_id: str, emb: np.ndarray) -> None:
    """
    Register (or update) a Thought's embedding in the in-memory index.

    Call this whenever a Thought is created or its embedding changes.
    """
    idx = _get_index(emb.shape[0])
    idx.add(thought_id, emb)


def search_similar(emb: np.ndarray, top_k: int = 50) -> List[Tuple[str, float]]:
    """
    Return up to top_k similar Thought IDs and scores for a given embedding.

    Returns:
        List of (thought_id, similarity_score) sorted by score descending.
    """
    if _index is None:
        return []
    return _index.search(emb, top_k=top_k)


# ---------------------------------------------------------------------------
# Retrieval context and scoring
# ---------------------------------------------------------------------------


@dataclass
class RetrievalContext:
    """
    Query context for memory retrieval.
    """
    person_ids_present: List[str]
    context_embedding: np.ndarray
    inferred_labels: List[str]
    now: datetime


# Weight configuration for the scoring function.
W_SIM = 0.50
W_PERSON = 0.20
W_LABEL = 0.10
W_TIME = 0.10
W_IMP = 0.05
W_RARITY = 0.05


def person_overlap_score(
    thought_person_ids: List[str], context_person_ids: List[str]
) -> float:
    """
    Score in [0, 1] based on overlap between persons tied to this Thought
    and the persons currently present.

    MVP: 1.0 if there is any intersection, else 0.0.
    Later we can make this proportional to the size of the overlap.
    """
    if not thought_person_ids or not context_person_ids:
        return 0.0
    return 1.0 if set(thought_person_ids) & set(context_person_ids) else 0.0


def label_match_score(thought_labels: List[str], context_labels: List[str]) -> float:
    """
    Score in [0, 1] based on overlap between Thought labels and context labels.
    """
    if not thought_labels or not context_labels:
        return 0.0
    return 1.0 if set(thought_labels) & set(context_labels) else 0.0


def recency_score(thought_ts: datetime | None, now: datetime) -> float:
    """
    Exponential decay based on how long ago the Thought occurred.

    Returns value in (0, 1]. If timestamp is missing, we return 0.5 as a neutral default.
    """
    if thought_ts is None:
        return 0.5

    # Difference in days
    delta_days = abs((now - thought_ts).total_seconds()) / 86400.0
    tau = 90.0  # time-scale in days; tweak as needed
    return float(np.exp(-delta_days / tau))


def rarity_score(_: Thought) -> float:
    """
    Placeholder rarity score.

    TODO: Implement real rarity based on how many similar thoughts exist.
    For now, always return 1.0 so it doesn't affect ranking.
    """
    return 1.0


def compute_score(
    thought: Thought,
    base_sim: float,
    ctx: RetrievalContext,
    thought_person_ids: List[str],
) -> float:
    """
    Combine semantic similarity + person overlap + label match + recency +
    importance + rarity into a single scalar score.
    """
    p_over = person_overlap_score(thought_person_ids, ctx.person_ids_present)
    l_match = label_match_score(getattr(thought, "labels", []) or [], ctx.inferred_labels)
    r_time = recency_score(getattr(thought, "timestamp_end", None), ctx.now)
    imp = getattr(thought, "importance_score", 0.0) or 0.0
    rarity = rarity_score(thought)

    return (
        W_SIM * base_sim
        + W_PERSON * p_over
        + W_LABEL * l_match
        + W_TIME * r_time
        + W_IMP * imp
        + W_RARITY * rarity
    )


# ---------------------------------------------------------------------------
# Top insight retrieval
# ---------------------------------------------------------------------------


def _get_person_ids_for_thought(thought: Thought) -> List[str]:
    """
    Return the list of person IDs associated with a Thought.

    This depends on how you model relationships in core.memory.models.
    MVP assumption: Thought.episode has a `primary_person_id`, and later
    we may add a many-to-many between Episode and Person.

    Adjust this helper once your models are finalized.
    """
    person_ids: List[str] = []

    # Example: use episode.primary_person_id if available.
    try:
        episode = getattr(thought, "episode", None)
        if episode is not None:
            primary_id = getattr(episode, "primary_person_id", None)
            if primary_id is not None:
                person_ids.append(primary_id)
    except Exception:
        # Keep it robust even if relationships are not wired yet.
        pass

    return person_ids


def get_top_insight(
    ctx: RetrievalContext,
    max_results: int = 1,
    search_k: int = 100,
) -> List[Thought]:
    """
    Core retrieval function.

    Given a RetrievalContext, query the vector index for similar Thoughts,
    load them from the DB, compute scores, and return the top-N Thoughts.

    Args:
        ctx: RetrievalContext with person_ids_present, context_embedding, etc.
        max_results: how many Thoughts to return (usually 1 for HUD).
        search_k: how many nearest neighbors to request from the index.

    Returns:
        A list of Thought instances sorted by descending relevance score.
    """
    if ctx.context_embedding.ndim != 1:
        raise ValueError("context_embedding must be a 1D vector")

    id_sim_list = search_similar(ctx.context_embedding, top_k=search_k)
    if not id_sim_list:
        return []

    thought_ids = [tid for tid, _ in id_sim_list]

    with get_session() as session:
        # Fetch Thoughts corresponding to candidate IDs
        thoughts: List[Thought] = (
            session.query(Thought)
            .filter(Thought.id.in_(thought_ids))
            .all()
        )

    thoughts_by_id: Dict[str, Thought] = {t.id: t for t in thoughts}

    scored: List[Tuple[Thought, float]] = []
    for tid, base_sim in id_sim_list:
        thought = thoughts_by_id.get(tid)
        if thought is None:
            continue
        person_ids = _get_person_ids_for_thought(thought)
        score = compute_score(thought, base_sim, ctx, person_ids)
        scored.append((thought, score))

    if not scored:
        return []

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [t for (t, _) in scored[:max_results]]
