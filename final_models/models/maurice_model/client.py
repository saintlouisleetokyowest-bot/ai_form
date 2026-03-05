"""
Form Analyzer – Webcam Client
==============================
Captures webcam frames, forwards them to the inference server, and renders
the pose skeleton + HUD overlay locally.  No ML dependencies required.

Usage
-----
    python client.py                        # server on localhost:8000
    python client.py --server http://1.2.3.4:8000
"""

import argparse
import queue
import threading
from collections import deque

import cv2
import numpy as np
import requests

# ── Server ────────────────────────────────────────────────────────────────────
DEFAULT_SERVER = "http://localhost:8000"

# ── MediaPipe landmark integer indices (no mediapipe import needed) ────────────
NOSE            = 0
LEFT_SHOULDER   = 11;  RIGHT_SHOULDER  = 12
LEFT_ELBOW      = 13;  RIGHT_ELBOW     = 14
LEFT_WRIST      = 15;  RIGHT_WRIST     = 16
LEFT_HIP        = 23;  RIGHT_HIP       = 24
LEFT_KNEE       = 25;  RIGHT_KNEE      = 26
LEFT_ANKLE      = 27;  RIGHT_ANKLE     = 28
LEFT_HEEL       = 29;  RIGHT_HEEL      = 30
LEFT_FOOT_INDEX = 31;  RIGHT_FOOT_INDEX = 32

CONNECTIONS = [
    (LEFT_SHOULDER,  RIGHT_SHOULDER),
    (LEFT_SHOULDER,  LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP,       RIGHT_HIP),
    (LEFT_SHOULDER,  LEFT_ELBOW),
    (LEFT_ELBOW,     LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW,    RIGHT_WRIST),
    (LEFT_HIP,       LEFT_KNEE),
    (LEFT_KNEE,      LEFT_ANKLE),
    (LEFT_ANKLE,     LEFT_HEEL),
    (LEFT_HEEL,      LEFT_FOOT_INDEX),
    (RIGHT_HIP,      RIGHT_KNEE),
    (RIGHT_KNEE,     RIGHT_ANKLE),
    (RIGHT_ANKLE,    RIGHT_HEEL),
    (RIGHT_HEEL,     RIGHT_FOOT_INDEX),
]

GHOST_CONNECTIONS = [
    (LEFT_SHOULDER,  RIGHT_SHOULDER),
    (LEFT_SHOULDER,  LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP,       RIGHT_HIP),
    (LEFT_SHOULDER,  LEFT_ELBOW),
    (LEFT_ELBOW,     LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW,    RIGHT_WRIST),
]

# ── Visual config (matches 5_form_analyzer.py) ────────────────────────────────
COLOR_GHOST = (0, 165, 255)   # orange BGR
COLOR_TEXT  = (255, 255, 255)
SKELETON_THICKNESS = 2
GHOST_THICKNESS    = 2
LANDMARK_RADIUS    = 5


def score_to_color(score):
    if score >= 70:
        return (0, 255, int(255 * (100 - score) / 30.0))   # green → yellow
    return (0, int(255 * score / 70.0), 255)                # yellow → red


def draw_skeleton(frame, landmarks, connections, color, thickness, h, w):
    for a, b in connections:
        if a in landmarks and b in landmarks:
            pt_a = (int(landmarks[a][0] * w), int(landmarks[a][1] * h))
            pt_b = (int(landmarks[b][0] * w), int(landmarks[b][1] * h))
            cv2.line(frame, pt_a, pt_b, color, thickness)
    for idx, pos in landmarks.items():
        cv2.circle(frame, (int(pos[0] * w), int(pos[1] * h)), LANDMARK_RADIUS, color, -1)
    if NOSE in landmarks:
        nose = (int(landmarks[NOSE][0] * w), int(landmarks[NOSE][1] * h))
        cv2.circle(frame, nose, 18, color, 2)


def draw_hud(frame, score, is_bad, fault_text, h, w):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "LATERAL RAISE", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
    cv2.putText(frame, f"Form: {score:.0f}/100", (15, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_to_color(score), 2)

    bar_x, bar_y, bar_w, bar_h = w - 220, 20, 200, 20
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * score / 100), bar_y + bar_h),
                  score_to_color(score), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_TEXT, 1)

    status = "GOOD FORM" if not is_bad else (fault_text or "ADJUST FORM")
    cv2.putText(frame, status, (w - 220, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_to_color(score), 2)

    cv2.putText(frame, "Q = Quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    cv2.circle(frame, (w - 160, h - 20), 6, COLOR_GHOST, -1)
    cv2.putText(frame, "= Ideal form", (w - 148, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GHOST, 1)


def parse_lm(lm_json):
    """JSON {str: [x,y]} → {int: [x,y]}"""
    if not lm_json:
        return {}
    return {int(k): v for k, v in lm_json.items()}


# Resolution sent to the server — smaller = faster inference, landmarks are
# normalised so they scale back to the display frame perfectly.
INFERENCE_WIDTH = 480


def _sender_thread(url, frame_q, result, score_buf, lm_buf):
    """Background thread: pulls frames from the queue, POSTs to server."""
    session = requests.Session()
    while True:
        small = frame_q.get()
        if small is None:       # poison pill — shut down
            session.close()
            return
        _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        try:
            resp = session.post(url,
                                files={"file": ("f.jpg", buf.tobytes(), "image/jpeg")},
                                timeout=3.0)
            if resp.ok:
                data      = resp.json()
                raw_lms   = parse_lm(data.get("actual_landmarks", {}))
                raw_score = data.get("score", 100.0)
                if raw_lms:
                    score_buf.append(raw_score)
                    lm_buf.append(raw_lms)
                result["score"]      = float(np.mean(score_buf)) if score_buf else 100.0
                result["is_bad"]     = data.get("is_bad", False)
                result["fault_text"] = data.get("fault_text", "")
                result["ghost_lms"]  = parse_lm(data.get("ghost_landmarks") or {})
                if lm_buf:
                    result["actual_lms"] = {
                        idx: np.mean([f[idx] for f in lm_buf if idx in f], axis=0).tolist()
                        for idx in lm_buf[0]
                    }
                result["server_ok"] = True
        except requests.exceptions.ConnectionError:
            result["server_ok"] = False
        except requests.exceptions.Timeout:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help="Base URL of the inference server")
    args = parser.parse_args()
    url  = args.server.rstrip("/") + "/analyze"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    # Shared result state (written by sender thread, read by main thread)
    result = {
        "score": 100.0, "is_bad": False, "fault_text": "",
        "actual_lms": {}, "ghost_lms": {}, "server_ok": True,
    }
    score_buf = deque(maxlen=3)
    lm_buf    = deque(maxlen=2)

    # Queue holds at most 1 frame — if server is busy the frame is dropped,
    # keeping the display loop always responsive.
    frame_q = queue.Queue(maxsize=1)
    sender  = threading.Thread(target=_sender_thread,
                               args=(url, frame_q, result, score_buf, lm_buf),
                               daemon=True)
    sender.start()
    print(f"Sending frames to {url} — press Q to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Downscale only for inference; draw on the full-res frame
        scale = INFERENCE_WIDTH / w
        small = cv2.resize(frame, (INFERENCE_WIDTH, int(h * scale)))
        try:
            frame_q.put_nowait(small)   # drop frame if sender is still busy
        except queue.Full:
            pass

        # Read last known result
        actual_lms = result["actual_lms"]
        ghost_lms  = result["ghost_lms"]
        score      = result["score"]
        is_bad     = result["is_bad"]
        fault_text = result["fault_text"]

        if not result["server_ok"]:
            cv2.putText(frame, "SERVER OFFLINE", (w // 2 - 130, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        if actual_lms:
            draw_skeleton(frame, actual_lms, CONNECTIONS,
                          score_to_color(score), SKELETON_THICKNESS, h, w)
        if ghost_lms:
            draw_skeleton(frame, ghost_lms, GHOST_CONNECTIONS,
                          COLOR_GHOST, GHOST_THICKNESS, h, w)

        draw_hud(frame, score, is_bad, fault_text, h, w)
        cv2.imshow("Workout Form Analyzer", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    frame_q.put(None)   # stop sender thread
    cap.release()
    cv2.destroyAllWindows()
    print("Session ended.")


if __name__ == "__main__":
    main()
