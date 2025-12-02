"""
detector.py

Face detection for MemorEyes using the SCRFD model from InsightFace (ONNX).
"""

from typing import List, Tuple

import numpy as np
import insightface

DETECTION_MODEL_PATH = "models/scrfd_10g_bnkps.onnx"
INPUT_SIZE = (640, 640)  # SCRFD default input size

class FaceDetector:
    """
    Wrapper class for the InsightFace SCRFD detector.
    Loads the model once and exposes a detect() method.
    """

    def __init__(self) -> None:
        """Initialize and load the detector."""
        self.detector = insightface.model_zoo.get_model(DETECTION_MODEL_PATH)
        # ctx_id=0 → CPU; change to GPU index if needed later
        self.detector.prepare(ctx_id=0)

    def detect(self, image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Perform face detection using SCRFD (ONNX) on an RGB image.

        Parameters:
            image (np.ndarray): RGB image of shape (H, W, 3).

        Returns:
            List[Tuple[np.ndarray, float]]: list of (bbox, score) where
                bbox is np.array([x1, y1, x2, y2]) and score is float.
            Empty list if no faces detected.
        """
        # SCRFD requires BGR input (OpenCV format)
        img_bgr = image[:, :, ::-1]

        # SCRFD detection call (ONLY valid signature)
        bboxes, kpss = self.detector.detect(
            img_bgr,
            input_size=INPUT_SIZE,
            max_num=0,        # 0 = return all faces
            metric="default", # default NMS metric
        )

        # Handle no detections
        if bboxes is None or len(bboxes) == 0:
            return []

        results: List[Tuple[np.ndarray, float]] = []
        for bbox in bboxes:
            # SCRFD bbox format is [x1, y1, x2, y2, score]
            x1, y1, x2, y2, score = bbox
            box = np.array([x1, y1, x2, y2], dtype=float)
            results.append((box, float(score)))

        return results


# Singleton pattern for efficiency (load model once)
_detector_instance: FaceDetector | None = None


def load_detector() -> FaceDetector:
    """Return a singleton detector instance, loading the model only once."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FaceDetector()
    return _detector_instance


def detect_faces(image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
    """
    Convenience function for one-line face detection.

    Equivalent to:
        detector = load_detector()
        return detector.detect(image)
    """
    detector = load_detector()
    return detector.detect(image)
