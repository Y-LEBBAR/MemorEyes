"""
embeddings.py

Purpose:
    Provide all face embedding functionality for MemorEyes.
    This module loads the ArcFace ONNX model (512-D) and exposes a simple
    function `get_face_embedding(image, bbox)` that returns a 512-D vector.

Description:
    - Loads the embedding model ONCE using a singleton pattern.
    - Takes a cropped face region, aligns and resizes it to 112x112.
    - Converts to RGB, normalizes, and runs the ONNX session.
    - Returns a high-quality 512-D unit-normalized embedding.

Used by:
    - /face/embedding API route
    - /profile/create (store embeddings)
    - /face/match (compare embeddings for identification)

Dependencies:
    - core.utils.image     → resizing, cropping, preprocessing
    - onnxruntime          → model inference
"""

import numpy as np
import onnxruntime as ort
import cv2
from typing import Optional

# ONNX embedding model file path
MODEL_PATH = "models/arcfaceresnet100-8.onnx"

# Embedding dimension for ArcFace R100 (always 512)
EMBED_DIM = 512

# Singleton instance
_embedding_session: Optional[ort.InferenceSession] = None


def load_embedding_model() -> ort.InferenceSession:
    """
    Load the ONNX ArcFace model only once (singleton).

    Returns:
        onnxruntime.InferenceSession
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

    # crop face
    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        raise ValueError("Invalid crop: face area is empty.")

    # resize
    face = cv2.resize(cropped, (112, 112))

    # convert RGB -> BGR if needed (InsightFace expects BGR)
    face = face[:, :, ::-1]

    # transpose to CHW (3, 112, 112)
    face = np.transpose(face, (0, 1, 2))
    face = np.transpose(face, (2, 0, 1))

    # normalize to [-1, 1]
    face = (face - 127.5) / 127.5

    # add batch dimension → (1, 3, 112, 112)
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

    # preprocess
    face_input = preprocess_face(image, bbox)

    # run inference
    outputs = session.run(None, {"input": face_input})
    embedding = outputs[0][0]  # (512,)

    # L2 normalization (critical)
    embedding = embedding / np.linalg.norm(embedding)

    return embedding.astype(np.float32)
