# MVP_MemorEyes/manual_test_detector.py

from pathlib import Path

import numpy as np
from PIL import Image

from core.face.detector import detect_faces


def main() -> None:
    # Adjust path if your test image is elsewhere
    img_path = Path(r"C:\Users\yanni\MemorEyes\Data\face.jpg")  # from MVP_MemorEyes/ to repo root
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    results = detect_faces(img_np)

    print(f"Num faces: {len(results)}")
    for i, (bbox, score) in enumerate(results):
        print(f"Face {i}: bbox={bbox.tolist()}, score={score:.3f}")


if __name__ == "__main__":
    main()
