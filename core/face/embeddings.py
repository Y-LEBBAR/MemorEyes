"""
embeddings.py

Purpose:
    Provide all face embedding functionality for MemorEyes.
    This module loads the ArcFace ONNX model (512-D) and exposes:
        - get_face_embedding(image, bbox)  → 512-D embedding for one face
        - get_face_embeddings(image)       → list of {bbox, embedding, score}
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import onnxruntime as ort

from core.face.detector import detect_faces  # make sure this exists and is imported correctly

# ONNX embedding model file path
MODEL_PATH = "models/arcfaceresnet100-8.onnx"

# Embedding dimension for ArcFace R100 (always 512)
EMBED_DIM = 512

# Singleton instance
_embedding_session: Optional[ort.InferenceSession] = None


def load_embedding_model() -> ort.InferenceSession:
    """
    Load the ONNX ArcFace model only once (singleton).
    """
    global _embedding_session

    if _embedding_session is None:
        providers = ["CPUExecutionProvider"]
        _embedding_session = ort.InferenceSession(MODEL_PATH, providers=providers)

    return _embedding_session


def preprocess_face(image: np.ndarray, bbox) -> np.ndarray:
    """
    Crop and preprocess the face for embedding extraction.

    Args:
        image (np.ndarray): Original RGB image.
        bbox (list or np.ndarray): [x1, y1, x2, y2].

    Returns:
        np.ndarray: Preprocessed face (1, 3, 112, 112)
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Crop face
    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        raise ValueError("Invalid crop: face area is empty.")

    # Resize to 112x112
    face = cv2.resize(cropped, (112, 112))

    # Convert RGB -> BGR (InsightFace convention)
    face = face[:, :, ::-1]

    # HWC -> CHW
    face = np.transpose(face, (2, 0, 1))  # (3, 112, 112)

    # Normalize to [-1, 1]
    face = (face - 127.5) / 127.5

    # Add batch dimension → (1, 3, 112, 112)
    face = np.expand_dims(face, axis=0).astype(np.float32)

    return face


def get_face_embedding(image: np.ndarray, bbox) -> np.ndarray:
    """
    Extract a 512-D embedding from a detected face.

    Args:
        image (np.ndarray): RGB image.
        bbox: Bounding box of face.

    Returns:
        np.ndarray: Normalized 512-D embedding vector.
    """
    session = load_embedding_model()

    # Preprocess
    face_input = preprocess_face(image, bbox)

    # Run inference
    outputs = session.run(None, {"input": face_input})
    embedding = outputs[0][0]  # (512,)

    # L2 normalization (critical)
    norm = np.linalg.norm(embedding) or 1e-8
    embedding = embedding / norm

    return embedding.astype(np.float32)


# ---------------------------------------------------------------------------
# New high-level helper for full-frame processing
# ---------------------------------------------------------------------------

def get_face_embeddings(image_rgb: np.ndarray) -> List[Dict[str, Any]]:
    """
    High-level helper used by the live demo:

    Given a full RGB image:
      - detect all faces
      - compute ArcFace embeddings for each
      - return list of dicts with bbox, embedding, score.

    Args:
        image_rgb: H x W x 3 RGB frame.

    Returns:
        List of dicts:
            {
                "bbox": [x1, y1, x2, y2],
                "embedding": <1D np.ndarray>,
                "score": float
            }
    """
    # detect_faces should return something like:
    #   [{"bbox": [x1, y1, x2, y2], "score": float, ...}, ...]
    # If your detector returns a different shape, adjust this loop accordingly.
    faces = detect_faces(image_rgb)

    results: List[Dict[str, Any]] = []

    for f in faces:
        if isinstance(f, dict):
            bbox = f.get("bbox")
            score = float(f.get("score", 1.0))
        else:
            # If detector returns plain bbox arrays, treat f as bbox
            bbox = f
            score = 1.0

        if bbox is None:
            continue

        emb = get_face_embedding(image_rgb, bbox)

        results.append(
            {
                "bbox": bbox,
                "embedding": emb,  # demo will np.array(...) this
                "score": score,
            }
        )

    return results
