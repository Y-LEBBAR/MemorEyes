# core/memory/search.py
# -----------
# This module handles identity matching by comparing a query face embedding
# against all stored profile embeddings in the SQLite database. It provides:
#   - cosine_similarity(): numerical similarity between two L2-normalized vectors
#   - find_best_match(): retrieves stored profiles, compares embeddings, and
#     returns the highest-scoring profile if it exceeds a confidence threshold.
# This file is part of the MemorEyes MVP pipeline for face recognition.

import numpy as np
from typing import List, Tuple, Optional
from core.db import fetch_all_profiles


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    Embeddings must be 1D numpy arrays after L2 normalization.
    """
    # Avoid division by zero
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Embeddings must be 1D vectors")

    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)

    if norm == 0:
        return 0.0

    return float(dot / norm)


def find_best_match(
    query_embedding: np.ndarray,
    threshold: float = 0.45  # adjust after testing
) -> Tuple[Optional[dict], float]:
    """
    Compare a query embedding against all stored profile embeddings.

    Returns:
        (best_profile_dict_or_None, best_similarity_score)

    If no match exceeds the threshold → returns (None, best_score).
    """

    profiles = fetch_all_profiles()  # returns rows as dicts
    if not profiles:
        return None, 0.0

    best_profile = None
    best_score = -1.0

    for p in profiles:
        stored_vec = np.array(p["vector"], dtype=np.float32)  # already decoded from DB
        score = cosine_similarity(query_embedding, stored_vec)

        if score > best_score:
            best_score = score
            best_profile = p

    # No confident match
    if best_score < threshold:
        return None, best_score

    return best_profile, best_score
