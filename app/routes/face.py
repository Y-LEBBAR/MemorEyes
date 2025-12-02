"""
face.py

Purpose:
    FastAPI routes related to face processing and representation.
    This file provides:
        - `/face/detect`   → detect face bounding boxes in an uploaded image
        - `/face/embedding` → compute a 512-D ArcFace embedding for the first detected face

Used by:
    - AR headset client for real-time face detection & identity lookup
    - Profile creation pipeline (capturing embeddings)
    - Identity matching & memory retrieval flow
    - Debugging and model evaluation workflows

Dependencies:
    - core.utils.image
        → image loading, conversion, preprocessing utilities

    - core.face.detector
        → ONNX face detection model (SCRFD / RetinaFace)

    - core.face.embeddings
        → ArcFace 512-D embedding model (ResNet100)

Notes:
    - MVP behavior uses the FIRST detected face only.
    - All embeddings are L2-normalized and ready for cosine similarity.
"""


from fastapi import APIRouter, UploadFile, File, HTTPException
from core.utils.image import load_image_from_bytes, to_rgb
from core.face.detector import detect_faces
from core.face.embeddings import get_face_embedding

router = APIRouter(prefix="/face", tags=["face"])


@router.post("/detect")
async def detect_face_endpoint(file: UploadFile = File(...)):
    """
    Detect faces in an uploaded image.

    Returns:
        - bounding boxes in format [x1, y1, x2, y2]
        - detection confidence
    """
    # Validate upload type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type")

    # Load and convert image to numpy RGB
    file_bytes = await file.read()
    image = load_image_from_bytes(file_bytes)
    image = to_rgb(image)

    # Run detection
    detections = detect_faces(image)

    # Format results
    response = []
    for box, score in detections:
        response.append({
            "bbox": box.tolist(),
            "confidence": float(score)
        })

    return {"faces": response}

@router.post("/embedding")
async def embedding_endpoint(file: UploadFile = File(...)):
    """
    Upload an image → detect face → compute 512-D embedding.

    Returns:
        {
            "embedding": [...512 floats...]
        }
    """

    # Validate input type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type")

    # Read image bytes → numpy array
    file_bytes = await file.read()
    image = load_image_from_bytes(file_bytes)
    image = to_rgb(image)

    # Detect face(s)
    detections = detect_faces(image)
    if not detections:
        raise HTTPException(status_code=404, detail="No face detected")

    # Use first detected face (MVP behavior)
    bbox, score = detections[0]

    # Compute 512-D embedding
    embedding = get_face_embedding(image, bbox)

    return {"embedding": embedding.tolist()}