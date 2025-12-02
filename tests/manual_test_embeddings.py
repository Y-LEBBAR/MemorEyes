"""
manual_test_embeddings.py

Manual test script for the MemorEyes embedding pipeline.
Loads a test image (face.jpg), runs face detection + embeddings,
and prints basic information about each detected face.
"""

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image

from core.face.embeddings import get_face_embeddings


def main() -> None:
    """
    Entry point for manual testing of the embedding pipeline.
    """
    # Resolve repo root and face.jpg relative to this file:
    img_path = r"C:\Users\yanni\MemorEyes\Data\face.jpg" #Path(__file__).resolve().parents[2] (alternative)
    print(f"Using image: {img_path}")

    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    results: List[Dict[str, Any]] = get_face_embeddings(img_np)

        # Print how many faces were detected in the image
    print(f"Num faces: {len(results)}")

    # Iterate over each detected face and report basic info
    for i, r in enumerate(results):
        # Extract the embedding vector for this face
        emb = r["embedding"]

        # Log face index, detector confidence, bounding box, and embedding dimension
        print(
            f"Face {i}: "
            f"score={r['score']:.3f}, "
            f"bbox={r['bbox']}, " #bounding box in format box = np.array([x1, y1, x2, y2], dtype=float)
            f"emb_dim={len(emb)}"
        )



if __name__ == "__main__":
    main()
