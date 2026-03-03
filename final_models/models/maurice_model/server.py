"""
Form Analyzer – Inference Server
=================================
FastAPI server that accepts a JPEG frame, runs MediaPipe pose detection and
the supervised model, then returns the form score + landmark positions as JSON.
The heavy ML stack (TensorFlow, MediaPipe) lives here inside Docker.

Endpoints
---------
GET  /health   — liveness check
POST /analyze  — multipart JPEG → JSON result
"""

import json
import pickle
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
LM      = mp_pose.PoseLandmark

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "saved_models"

# ── Global model state (loaded once at startup) ───────────────────────────────
_model    = None
_X_scaler = None
_y_scaler = None
_meta     = None
_pose     = None

app = FastAPI()


@app.on_event("startup")
async def startup():
    global _model, _X_scaler, _y_scaler, _meta, _pose
    safe = "lateral_raise"
    _model    = tf.keras.models.load_model(MODELS_DIR / f"{safe}_supervised.keras", compile=False)
    with open(MODELS_DIR / f"{safe}_X_scaler.pkl", "rb") as f:
        _X_scaler = pickle.load(f)
    with open(MODELS_DIR / f"{safe}_y_scaler.pkl", "rb") as f:
        _y_scaler = pickle.load(f)
    with open(MODELS_DIR / f"{safe}_meta.json") as f:
        _meta = json.load(f)
    _pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    print("[server] Models loaded and ready.")


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS  (mirrored from 5_form_analyzer.py)
# ══════════════════════════════════════════════════════════════════════════════

def _coords(landmarks, lm_enum):
    lm = landmarks[lm_enum.value]
    return np.array([lm.x, lm.y, lm.z])

def _angle(a, b, c):
    ba = a - b; bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

def _mid(a, b):
    return (a + b) / 2.0


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_lateral_raise_features(landmarks):
    l_shoulder = _coords(landmarks, LM.LEFT_SHOULDER)
    r_shoulder = _coords(landmarks, LM.RIGHT_SHOULDER)
    l_elbow    = _coords(landmarks, LM.LEFT_ELBOW)
    r_elbow    = _coords(landmarks, LM.RIGHT_ELBOW)
    l_wrist    = _coords(landmarks, LM.LEFT_WRIST)
    r_wrist    = _coords(landmarks, LM.RIGHT_WRIST)
    l_hip      = _coords(landmarks, LM.LEFT_HIP)
    r_hip      = _coords(landmarks, LM.RIGHT_HIP)

    spine_vec  = _mid(l_shoulder, r_shoulder) - _mid(l_hip, r_hip)
    vertical   = np.array([0, -1, 0])
    cosine     = np.dot(spine_vec, vertical) / (np.linalg.norm(spine_vec) + 1e-6)
    left_raise  = _angle(l_hip, l_shoulder, l_elbow)
    right_raise = _angle(r_hip, r_shoulder, r_elbow)

    return {
        "left_arm_raise":             left_raise,
        "right_arm_raise":            right_raise,
        "left_elbow_angle":           _angle(l_shoulder, l_elbow, l_wrist),
        "right_elbow_angle":          _angle(r_shoulder, r_elbow, r_wrist),
        "torso_lean":                 np.degrees(np.arccos(np.clip(cosine, -1, 1))),
        "arm_symmetry":               abs(left_raise - right_raise),
        "left_wrist_above_shoulder":  float(l_shoulder[1] - l_wrist[1]),
        "right_wrist_above_shoulder": float(r_shoulder[1] - r_wrist[1]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════════

def score_form(features):
    feature_cols      = [f.replace("_bad", "") for f in _meta["input_features"]]
    vec               = np.array([[features[f] for f in feature_cols]], dtype=np.float32)
    prediction_scaled = _model.predict(_X_scaler.transform(vec), verbose=0)
    prediction        = _y_scaler.inverse_transform(prediction_scaled)[0]
    errors            = np.abs(np.array([features[f] for f in feature_cols]) - prediction)
    score             = max(0.0, 100.0 * (1.0 - float(np.max(errors)) / 60.0))
    return score, prediction, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# FAULT TEXT
# ══════════════════════════════════════════════════════════════════════════════

def get_fault_text(features, prediction, feature_cols):
    ideal       = dict(zip(feature_cols, prediction))
    faults      = []
    avg_raise   = (features["left_arm_raise"]   + features["right_arm_raise"])   / 2
    ideal_raise = (ideal["left_arm_raise"]       + ideal["right_arm_raise"])       / 2
    avg_elbow   = (features["left_elbow_angle"]  + features["right_elbow_angle"]) / 2
    ideal_elbow = (ideal["left_elbow_angle"]     + ideal["right_elbow_angle"])    / 2

    if ideal_raise - avg_raise > 15:
        faults.append((ideal_raise - avg_raise,                   "RAISE ARMS HIGHER"))
    if avg_raise - ideal_raise > 8:
        faults.append((avg_raise - ideal_raise,                   "ARMS TOO HIGH"))
    if ideal_elbow - avg_elbow > 20:
        faults.append((ideal_elbow - avg_elbow,                   "BEND ELBOWS LESS"))
    if features["torso_lean"] - ideal["torso_lean"] > 10:
        faults.append((features["torso_lean"] - ideal["torso_lean"], "STOP SWINGING"))
    if features["arm_symmetry"] - ideal["arm_symmetry"] > 15:
        faults.append((features["arm_symmetry"],                   "EVEN OUT YOUR ARMS"))

    return faults[0][1] if faults else ""


# ══════════════════════════════════════════════════════════════════════════════
# GHOST SKELETON
# ══════════════════════════════════════════════════════════════════════════════

def reconstruction_to_landmarks(prediction, feature_cols, raw_landmarks):
    """Returns {LM enum: np.array([x, y])} for ghost skeleton joints."""
    ideal  = dict(zip(feature_cols, prediction))

    def lm_xy(lm_enum):
        lm = raw_landmarks[lm_enum.value]
        return np.array([lm.x, lm.y])

    ghost = {lm_enum: lm_xy(lm_enum) for lm_enum in LM}

    for side, shoulder_lm, elbow_lm, wrist_lm, hip_lm in [
        ("left",  LM.LEFT_SHOULDER,  LM.LEFT_ELBOW,  LM.LEFT_WRIST,  LM.LEFT_HIP),
        ("right", LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST, LM.RIGHT_HIP),
    ]:
        shoulder_pos  = lm_xy(shoulder_lm)
        hip_pos       = lm_xy(hip_lm)
        upper_arm_len = np.linalg.norm(lm_xy(elbow_lm) - shoulder_pos)
        forearm_len   = np.linalg.norm(lm_xy(wrist_lm)  - lm_xy(elbow_lm))

        ideal_raise       = float(np.clip(ideal.get(f"{side}_arm_raise",   90.0), 30.0, 150.0))
        ideal_elbow_angle = float(np.clip(ideal.get(f"{side}_elbow_angle", 160.0), 90.0, 180.0))

        torso_dir     = (hip_pos - shoulder_pos) / (np.linalg.norm(hip_pos - shoulder_pos) + 1e-6)
        sign          = 1 if side == "left" else -1
        cos_r, sin_r  = np.cos(np.radians(ideal_raise)), np.sin(np.radians(ideal_raise))
        rot           = np.array([[cos_r, sign * -sin_r], [sin_r, sign * cos_r]])
        upper_arm_dir = rot @ (-torso_dir)
        ideal_elbow   = shoulder_pos + upper_arm_len * upper_arm_dir

        bend_rad      = np.radians(180.0 - ideal_elbow_angle)
        cos_e, sin_e  = np.cos(bend_rad), np.sin(bend_rad)
        forearm_dir   = np.array([[cos_e, -sin_e], [sin_e, cos_e]]) @ upper_arm_dir
        ideal_wrist   = ideal_elbow + forearm_len * forearm_dir

        ghost[shoulder_lm] = shoulder_pos
        ghost[hip_lm]      = hip_pos
        ghost[elbow_lm]    = np.clip(ideal_elbow, 0.0, 1.0)
        ghost[wrist_lm]    = np.clip(ideal_wrist,  0.0, 1.0)

    return ghost


def _serialize(lm_dict):
    """Convert {LM enum | int: np.array([x,y])} to JSON-safe {str: [float, float]}."""
    out = {}
    for k, v in lm_dict.items():
        key = k.value if hasattr(k, "value") else int(k)
        out[str(key)] = [float(v[0]), float(v[1])]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    img_bytes = await file.read()
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "invalid image"}, status_code=400)

    result = _pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not result.pose_landmarks:
        return {
            "score": 100.0, "is_bad": False, "fault_text": "",
            "actual_landmarks": {}, "ghost_landmarks": None,
        }

    raw = result.pose_landmarks.landmark
    features = extract_lateral_raise_features(raw)

    actual_lm_dict = {
        lm_enum: np.array([raw[lm_enum.value].x, raw[lm_enum.value].y])
        for lm_enum in LM
    }

    # Skip scoring while arms are at rest (matches 5_form_analyzer.py)
    avg_raise = (features["left_arm_raise"] + features["right_arm_raise"]) / 2
    if avg_raise < 30:
        return {
            "score": 100.0, "is_bad": False, "fault_text": "",
            "actual_landmarks": _serialize(actual_lm_dict),
            "ghost_landmarks":  None,
        }

    score, prediction, feature_cols = score_form(features)
    is_bad = score < 70

    ghost_lm_dict = None
    fault_text    = ""
    if is_bad:
        ghost_lm_dict = reconstruction_to_landmarks(prediction, feature_cols, raw)
        fault_text    = get_fault_text(features, prediction, feature_cols)

    return {
        "score":            float(score),
        "is_bad":           is_bad,
        "fault_text":       fault_text,
        "actual_landmarks": _serialize(actual_lm_dict),
        "ghost_landmarks":  _serialize(ghost_lm_dict) if ghost_lm_dict else None,
    }
