# demo/live_mvp.py
# ----------------
# Local desktop demo for MemorEyes.
#
# - Uses laptop webcam for live video.
# - Draws green 3-px bounding boxes around detected faces and shows identity labels.
# - Left panel: person facts (for the "active" face) + LIVE transcript lines.
# - Right panel: fake Thought JSON objects (for now).
#
# Transcript comes from core.nlp.asr_whisper using Whisper on the microphone input.

from __future__ import annotations

import json
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.db import get_session
from core.face.embeddings import get_face_embeddings
from core.nlp.asr_whisper import start_asr_background
from core.people.service import find_best_match, get_person_facts  # you added get_person_facts earlier


# -----------------------------
# Config
# -----------------------------

LEFT_PANEL_WIDTH = 320
RIGHT_PANEL_WIDTH = 360
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
FONT_THICKNESS = 1
LINE_HEIGHT = 18
BOX_COLOR = (0, 255, 0)  # green BGR
BOX_THICKNESS = 3        # 3 pixels


# -----------------------------
# Face pipeline wrapper
# -----------------------------


def identify_faces_bboxes(
    frame_bgr: np.ndarray,
) -> List[Tuple[Tuple[int, int, int, int], str, Optional[int]]]:
    """
    Detect faces in the frame, embed them, and identify each person.

    Returns:
        List of (bbox, label, person_id_or_None) where:
          bbox      = (x1, y1, x2, y2)
          label     = person display name or "Unknown"
          person_id = Person.id or None if unknown
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # We assume get_face_embeddings(frame_rgb) returns:
    #   [{"bbox": [x1, y1, x2, y2], "embedding": [..], "score": ..}, ...]
    results = get_face_embeddings(frame_rgb)

    identified: List[Tuple[Tuple[int, int, int, int], str, Optional[int]]] = []

    with get_session() as db:
        for r in results:
            bbox = r["bbox"]
            emb = np.array(r["embedding"], dtype=np.float32)

            person, dist = find_best_match(db, emb)  # uses core.people.service

            if person is None:
                label = "Unknown"
                person_id = None
            else:
                name = getattr(person, "display_name", None)
                label = name or f"Person {person.id}"
                person_id = person.id

            x1, y1, x2, y2 = map(int, bbox)
            identified.append(((x1, y1, x2, y2), label, person_id))

    return identified


def choose_active_person(
    identified: List[Tuple[Tuple[int, int, int, int], str, Optional[int]]],
) -> Optional[int]:
    """
    Choose 'active' person_id based on largest bounding box area.
    """
    if not identified:
        return None

    max_area = -1
    active_person_id: Optional[int] = None

    for (x1, y1, x2, y2), _, person_id in identified:
        if person_id is None:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            active_person_id = person_id

    return active_person_id


# -----------------------------
# Panels rendering
# -----------------------------


def render_left_panel(
    height: int,
    active_person_id: Optional[int],
    transcript_lines: List[str],
) -> np.ndarray:
    """
    Build the left panel image: person facts + transcript.
    """
    panel = np.zeros((height, LEFT_PANEL_WIDTH, 3), dtype=np.uint8)

    y = 20
    cv2.putText(panel, "PERSON FACTS", (10, y), FONT, FONT_SCALE, (255, 255, 255), 1)
    y += LINE_HEIGHT

    if active_person_id is None:
        cv2.putText(
            panel,
            "No known person in focus.",
            (10, y),
            FONT,
            FONT_SCALE,
            (200, 200, 200),
            1,
        )
        y += LINE_HEIGHT
    else:
        facts_lines = get_person_facts_lines(active_person_id)
        for line in facts_lines:
            if y > height - LINE_HEIGHT * 8:
                break
            cv2.putText(panel, line, (10, y), FONT, FONT_SCALE, (200, 255, 200), 1)
            y += LINE_HEIGHT

    # Separator
    cv2.line(panel, (0, y + 4), (LEFT_PANEL_WIDTH, y + 4), (80, 80, 80), 1)
    y += LINE_HEIGHT

    cv2.putText(panel, "TRANSCRIPT (live)", (10, y), FONT, FONT_SCALE, (255, 255, 255), 1)
    y += LINE_HEIGHT

    max_lines = (height - y) // LINE_HEIGHT
    for line in transcript_lines[-max_lines:]:
        cv2.putText(panel, line, (10, y), FONT, FONT_SCALE, (200, 200, 200), 1)
        y += LINE_HEIGHT
        if y >= height - 5:
            break

    return panel


def render_right_panel(height: int, thoughts: List[dict]) -> np.ndarray:
    """
    Build the right panel image: recent Thoughts as JSON (still fake).
    """
    panel = np.zeros((height, RIGHT_PANEL_WIDTH, 3), dtype=np.uint8)

    y = 20
    cv2.putText(panel, "THOUGHTS (demo)", (10, y), FONT, FONT_SCALE, (255, 255, 255), 1)
    y += LINE_HEIGHT

    max_lines = (height - y) // LINE_HEIGHT

    json_lines: List[str] = []
    for t in thoughts[-20:]:
        s = json.dumps(t, ensure_ascii=False)
        if len(s) > 60:
            s = s[:57] + "..."
        json_lines.append(s)

    for line in json_lines[-max_lines:]:
        cv2.putText(panel, line, (10, y), FONT, FONT_SCALE, (200, 200, 255), 1)
        y += LINE_HEIGHT
        if y >= height - 5:
            break

    return panel


def get_person_facts_lines(person_id: int) -> List[str]:
    """
    Use get_person_facts(person_id) to build display lines.
    """
    facts = get_person_facts(person_id)

    if not facts.get("exists", True):
        return [f"Unknown person id={person_id}"]

    lines: List[str] = []
    name = facts.get("name", f"id={person_id}")
    lines.append(f"Name: {name}")

    last_seen = facts.get("last_seen_at")
    if last_seen:
        lines.append(f"Last seen: {last_seen}")

    episode_count = facts.get("episode_count")
    if episode_count is not None:
        lines.append(f"Episodes: {episode_count}")

    notes = facts.get("notes")
    if notes:
        lines.append(f"Notes: {notes}")

    return lines


# -----------------------------
# Fake thoughts (for now)
# -----------------------------


def update_fake_thoughts(frame_index: int, thoughts: List[dict]) -> None:
    """
    For now, generate fake Thought entries every N frames.
    Later this will be replaced by real Thought creation from ASR segments.
    """
    if frame_index % 90 == 0:  # ~every 3 seconds at 30fps
        t = {
            "id": f"fake-{len(thoughts)+1}",
            "timestamp": time.strftime("%H:%M:%S"),
            "text": "This is a fake Thought – replace with real NLP output.",
            "labels": ["demo"],
        }
        thoughts.append(t)


# -----------------------------
# Main loop
# -----------------------------


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: could not read initial frame from webcam.")
        cap.release()
        return

    height, width = frame.shape[:2]

    transcript_lines: List[str] = []
    thoughts: List[dict] = []

    # Start ASR background thread to fill transcript_lines
    stop_asr_event = start_asr_background(transcript_lines)

    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1

            # Identify faces
            try:
                identified = identify_faces_bboxes(frame)
            except Exception:
                identified = []

            # Draw bounding boxes + labels
            for (x1, y1, x2, y2), label, _person_id in identified:
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 5, 0)),
                    FONT,
                    0.5,
                    BOX_COLOR,
                    1,
                    lineType=cv2.LINE_AA,
                )

            active_person_id = choose_active_person(identified)

            # Update fake thoughts (real ones will come from NLP pipeline later)
            update_fake_thoughts(frame_index, thoughts)

            # Build panels
            left_panel = render_left_panel(height, active_person_id, transcript_lines)
            right_panel = render_right_panel(height, thoughts)

            combined = np.hstack([left_panel, frame, right_panel])

            cv2.imshow("MemorEyes Live MVP", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        stop_asr_event.set()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
