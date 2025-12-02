"""
image.py

Purpose:
    Utility functions for handling image data across the MemorEyes system.
    This module converts uploaded image bytes into numpy arrays, performs
    RGB conversion, handles face cropping using bounding boxes, and provides
    resizing helpers for embedding models (e.g., ArcFace).

Used by:
    - core.face.detector      → to preprocess frames before detection
    - core.face.embeddings    → to prepare cropped faces for embedding
    - FastAPI routes          → to parse incoming image uploads

Key Functions:
    load_image_from_bytes(bytes)  → numpy array (RGB)
    to_rgb(image)                 → ensures RGB format
    crop(image, box)              → crops face region
    resize(image, size=112)       → resizes for embedding models
"""

import numpy as np
import cv2
from PIL import Image
from io import BytesIO

def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Convert raw uploaded bytes into a numpy RGB image.
    """
    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    return np.array(image)

def to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is RGB format.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def crop(image: np.ndarray, box: tuple) -> np.ndarray:
    """
    Crop a face using a bounding box: (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = map(int, box)
    return image[y1:y2, x1:x2]

def resize(image: np.ndarray, size: int = 112) -> np.ndarray:
    """
    Resize image for embedding models (ArcFace uses 112x112).
    """
    return cv2.resize(image, (size, size))
