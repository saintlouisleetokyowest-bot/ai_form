"""
Real-Time Form Analyzer
================================
Opens webcam, detects pose with MediaPipe, runs the autoencoder,
displays a form score with specific fault text, and renders a
corrective ghost skeleton showing ideal form.

Controls:
    Q         — quit

"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import json
from pathlib import Path
from collections import deque

# ── MediaPipe setup ────────────────────────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
LM         = mp_pose.PoseLandmark

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "saved_models"

# ── Visual config ──────────────────────────────────────────────────────────────
COLOR_USER_GOOD    = (0,   255, 0)    # Green  — good form
COLOR_USER_BAD     = (0,   0,   255)  # Red    — bad form
COLOR_GHOST        = (0,   165, 255)  # Orange — ghost skeleton (BGR)
COLOR_TEXT         = (255, 255, 255)  # White
SKELETON_THICKNESS = 2
GHOST_THICKNESS    = 2
LANDMARK_RADIUS    = 5

# ── Pose connections ───────────────────────────────────────────────────────────
CONNECTIONS = [
    (LM.LEFT_SHOULDER,  LM.RIGHT_SHOULDER),
    (LM.LEFT_SHOULDER,  LM.LEFT_HIP),
    (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    (LM.LEFT_HIP,       LM.RIGHT_HIP),
    (LM.LEFT_SHOULDER,  LM.LEFT_ELBOW),
    (LM.LEFT_ELBOW,     LM.LEFT_WRIST),
    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW),
    (LM.RIGHT_ELBOW,    LM.RIGHT_WRIST),
    (LM.LEFT_HIP,       LM.LEFT_KNEE),
    (LM.LEFT_KNEE,      LM.LEFT_ANKLE),
    (LM.LEFT_ANKLE,     LM.LEFT_HEEL),
    (LM.LEFT_HEEL,      LM.LEFT_FOOT_INDEX),
    (LM.RIGHT_HIP,      LM.RIGHT_KNEE),
    (LM.RIGHT_KNEE,     LM.RIGHT_ANKLE),
    (LM.RIGHT_ANKLE,    LM.RIGHT_HEEL),
    (LM.RIGHT_HEEL,     LM.RIGHT_FOOT_INDEX),
]

LATERAL_RAISE_GHOST_CONNECTIONS = [
    (LM.LEFT_SHOULDER,  LM.RIGHT_SHOULDER),
    (LM.LEFT_SHOULDER,  LM.LEFT_HIP),
    (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    (LM.LEFT_HIP,       LM.RIGHT_HIP),
    (LM.LEFT_SHOULDER,  LM.LEFT_ELBOW),
    (LM.LEFT_ELBOW,     LM.LEFT_WRIST),
    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW),
    (LM.RIGHT_ELBOW,    LM.RIGHT_WRIST),
]


# ══════════════════════════════════════════════════════════════════════════════
# SMOOTHING
# ══════════════════════════════════════════════════════════════════════════════

class Smoother:
    """Rolling average smoother for scores and landmark positions."""
    def __init__(self, window=5): #Averages score over 5 frames to make results less jittery
        self.window = window
        self.scores = deque(maxlen=window)
        self.lm_buf = deque(maxlen=window)

    def update(self, score, actual_lms):
        self.scores.append(score)
        self.lm_buf.append(actual_lms)

    def smooth_score(self):
        if not self.scores:
            return 100.0
        return float(np.mean(self.scores))

    def smooth_landmarks(self):
        if not self.lm_buf:
            return {}
        smoothed = {}
        for lm_enum in self.lm_buf[0]:
            positions = [frame[lm_enum] for frame in self.lm_buf if lm_enum in frame]
            smoothed[lm_enum] = np.mean(positions, axis=0)
        return smoothed

    def reset(self):
        self.scores.clear()
        self.lm_buf.clear()


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_model_artifacts(exercise_name):
    """Load supervised model, scalers, and metadata for a given exercise."""
    safe = exercise_name.replace(" ", "_")

    model = tf.keras.models.load_model(MODELS_DIR / f"{safe}_supervised.keras", compile=False)

    with open(MODELS_DIR / f"{safe}_X_scaler.pkl", "rb") as f:
        X_scaler = pickle.load(f)

    with open(MODELS_DIR / f"{safe}_y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)

    with open(MODELS_DIR / f"{safe}_meta.json") as f:
        meta = json.load(f)

    print(f"[{exercise_name}] Loaded supervised model")
    return model, X_scaler, y_scaler, meta


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_coords(landmarks, landmark_enum):
    lm = landmarks[landmark_enum.value]
    return np.array([lm.x, lm.y, lm.z])

def angle_between(a, b, c):
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def midpoint(a, b):
    return (a + b) / 2.0


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_lateral_raise_features(landmarks):
    l_shoulder = get_coords(landmarks, LM.LEFT_SHOULDER)
    r_shoulder = get_coords(landmarks, LM.RIGHT_SHOULDER)
    l_elbow    = get_coords(landmarks, LM.LEFT_ELBOW)
    r_elbow    = get_coords(landmarks, LM.RIGHT_ELBOW)
    l_wrist    = get_coords(landmarks, LM.LEFT_WRIST)
    r_wrist    = get_coords(landmarks, LM.RIGHT_WRIST)
    l_hip      = get_coords(landmarks, LM.LEFT_HIP)
    r_hip      = get_coords(landmarks, LM.RIGHT_HIP)
    mid_shoulder = midpoint(l_shoulder, r_shoulder)
    mid_hip      = midpoint(l_hip, r_hip)
    spine_vec    = mid_shoulder - mid_hip
    vertical     = np.array([0, -1, 0])
    cosine       = np.dot(spine_vec, vertical) / (np.linalg.norm(spine_vec) + 1e-6)
    left_raise   = angle_between(l_hip, l_shoulder, l_elbow)
    right_raise  = angle_between(r_hip, r_shoulder, r_elbow)

    return {
        "left_arm_raise":             left_raise,
        "right_arm_raise":            right_raise,
        "left_elbow_angle":           angle_between(l_shoulder, l_elbow, l_wrist),
        "right_elbow_angle":          angle_between(r_shoulder, r_elbow, r_wrist),
        "torso_lean":                 np.degrees(np.arccos(np.clip(cosine, -1, 1))),
        "arm_symmetry":               abs(left_raise - right_raise),
        "left_wrist_above_shoulder":  float(l_shoulder[1] - l_wrist[1]),
        "right_wrist_above_shoulder": float(r_shoulder[1] - r_wrist[1]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# FORM SCORING
# ══════════════════════════════════════════════════════════════════════════════

def score_form(features, meta, X_scaler, y_scaler, model):
    feature_cols = [f.replace("_bad", "") for f in meta["input_features"]]
    vec = np.array([[features[f] for f in feature_cols]], dtype=np.float32)
    vec_scaled = X_scaler.transform(vec)

    prediction_scaled = model.predict(vec_scaled, verbose=0)
    prediction = y_scaler.inverse_transform(prediction_scaled)[0]

    input_vals = np.array([features[f] for f in feature_cols])
    errors    = np.abs(input_vals - prediction)
    max_error = float(np.max(errors))
    score     = max(0.0, 100.0 * (1.0 - max_error / 60.0))

    return score, max_error, prediction, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# FAULT TEXT
# ══════════════════════════════════════════════════════════════════════════════

def get_fault_text(features, prediction, feature_cols):
    ideal = dict(zip(feature_cols, prediction))
    faults = []

    avg_raise   = (features["left_arm_raise"] + features["right_arm_raise"]) / 2
    ideal_raise = (ideal["left_arm_raise"] + ideal["right_arm_raise"]) / 2
    avg_elbow   = (features["left_elbow_angle"] + features["right_elbow_angle"]) / 2
    ideal_elbow = (ideal["left_elbow_angle"] + ideal["right_elbow_angle"]) / 2

    if ideal_raise - avg_raise > 15:
        faults.append((ideal_raise - avg_raise, "RAISE ARMS HIGHER"))
    if avg_raise - ideal_raise > 8:
        faults.append((avg_raise - ideal_raise, "ARMS TOO HIGH"))
    if ideal_elbow - avg_elbow > 20:
        faults.append((ideal_elbow - avg_elbow, "BEND ELBOWS LESS"))
    if features["torso_lean"] - ideal["torso_lean"] > 10:
        faults.append((features["torso_lean"] - ideal["torso_lean"], "STOP SWINGING"))
    if features["arm_symmetry"] - ideal["arm_symmetry"] > 15:
        faults.append((features["arm_symmetry"], "EVEN OUT YOUR ARMS"))

    if not faults:
        return ""
    faults.sort(reverse=True)
    return faults[0][1]

# ══════════════════════════════════════════════════════════════════════════════
# GHOST SKELETON
# ══════════════════════════════════════════════════════════════════════════════

def reconstruction_to_landmarks(prediction, feature_cols, actual_landmarks):
    ideal = dict(zip(feature_cols, prediction))

    def lm_xy(lm_enum):
        lm = actual_landmarks[lm_enum.value]
        return np.array([lm.x, lm.y])

    ghost = {lm_enum: lm_xy(lm_enum) for lm_enum in LM}

    for side, shoulder_lm, elbow_lm, wrist_lm, hip_lm in [
        ("left",  LM.LEFT_SHOULDER,  LM.LEFT_ELBOW,  LM.LEFT_WRIST,  LM.LEFT_HIP),
        ("right", LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST, LM.RIGHT_HIP),
    ]:
        shoulder_pos  = lm_xy(shoulder_lm)
        hip_pos       = lm_xy(hip_lm)
        actual_elbow  = lm_xy(elbow_lm)
        actual_wrist  = lm_xy(wrist_lm)
        upper_arm_len = np.linalg.norm(actual_elbow - shoulder_pos)
        forearm_len   = np.linalg.norm(actual_wrist - actual_elbow)
        ideal_raise       = float(np.clip(ideal.get(f"{side}_arm_raise", 90.0), 30.0, 150.0))
        ideal_elbow_angle = float(np.clip(ideal.get(f"{side}_elbow_angle", 160.0), 90.0, 180.0))
        torso_vec = hip_pos - shoulder_pos
        torso_dir = torso_vec / (np.linalg.norm(torso_vec) + 1e-6)
        raise_rad = np.radians(ideal_raise)
        sign      = 1 if side == "left" else -1
        cos_r, sin_r = np.cos(raise_rad), np.sin(raise_rad)
        rot  = np.array([[cos_r, sign * -sin_r], [sin_r, sign * cos_r]])
        upper_arm_dir   = rot @ (-torso_dir)
        ideal_elbow_pos = shoulder_pos + upper_arm_len * upper_arm_dir
        elbow_bend_rad  = np.radians(180.0 - ideal_elbow_angle)
        cos_e, sin_e    = np.cos(elbow_bend_rad), np.sin(elbow_bend_rad)
        rot2            = np.array([[cos_e, -sin_e], [sin_e, cos_e]])
        forearm_dir     = rot2 @ upper_arm_dir
        ideal_wrist_pos = ideal_elbow_pos + forearm_len * forearm_dir
        ghost[shoulder_lm] = shoulder_pos
        ghost[hip_lm]      = hip_pos
        ghost[elbow_lm]    = np.clip(ideal_elbow_pos, 0.0, 1.0)
        ghost[wrist_lm]    = np.clip(ideal_wrist_pos, 0.0, 1.0)

    return ghost

# ══════════════════════════════════════════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def draw_skeleton(frame, landmarks, connections, color, thickness, h, w):
    for lm_a, lm_b in connections:
        if lm_a in landmarks and lm_b in landmarks:
            pt_a = (int(landmarks[lm_a][0] * w), int(landmarks[lm_a][1] * h))
            pt_b = (int(landmarks[lm_b][0] * w), int(landmarks[lm_b][1] * h))
            cv2.line(frame, pt_a, pt_b, color, thickness)
    for lm_enum, pos in landmarks.items():
        pt = (int(pos[0] * w), int(pos[1] * h))
        cv2.circle(frame, pt, LANDMARK_RADIUS, color, -1)

def actual_landmarks_to_dict(landmarks):
    return {lm_enum: np.array([landmarks[lm_enum.value].x, landmarks[lm_enum.value].y])
            for lm_enum in LM}


# ══════════════════════════════════════════════════════════════════════════════
# HUD
# ══════════════════════════════════════════════════════════════════════════════
def score_to_color(score):
        if score >= 70:
            # Green to yellow
            t = (100 - score) / 30.0
            r = int(255 * t)
            g = 255
        else:
            # Yellow to red
            t = score / 70.0
            r = 255
            g = int(255 * t)
        return (0, g, r)  # BGR


def draw_hud(frame, exercise, score, is_bad, h, w, fault_text=""):
    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Exercise name
    cv2.putText(frame, exercise.upper(), (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)

    # Form score


    cv2.putText(frame, f"Form: {score:.0f}/100", (15, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_to_color(score), 2)

    # Score bar
    bar_x, bar_y, bar_w, bar_h = w - 220, 20, 200, 20
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    filled = int(bar_w * score / 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), score_to_color(score), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_TEXT, 1)

    # Status / fault text
    status = "GOOD FORM" if not is_bad else (fault_text if fault_text else "ADJUST FORM")
    cv2.putText(frame, status, (w - 220, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_to_color(score), 2)

    # Controls hint
    cv2.putText(frame, "Q=Quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # Ghost legend
    cv2.circle(frame, (w - 160, h - 20), 6, COLOR_GHOST, -1)
    cv2.putText(frame, "= Ideal form", (w - 148, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GHOST, 1)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run(start_exercise="lateral raise"):
    print("Loading models...")
    model, X_scaler, y_scaler, meta = load_model_artifacts("lateral raise")
    print("Model loaded. Opening webcam...")

    smoother    = Smoother(window=5)
    cap         = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as pose:

        score      = 100.0
        is_bad     = False
        ghost_lms  = {}
        fault_text = ""

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            h, w   = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                landmarks  = result.pose_landmarks.landmark
                features   = extract_lateral_raise_features(landmarks)
                actual_lms = actual_landmarks_to_dict(landmarks)

                avg_raise = (features["left_arm_raise"] + features["right_arm_raise"]) / 2

                if avg_raise < 30:
                    score      = 100.0
                    is_bad     = False
                    ghost_lms  = {}
                    fault_text = ""
                    smoother.reset()
                else:
                    raw_score, max_error, prediction, feature_cols = score_form(
                        features, meta, X_scaler, y_scaler, model
                    )

                    print(f"Raise: L={features['left_arm_raise']:.0f} R={features['right_arm_raise']:.0f} | Predicted: L={prediction[0]:.0f} R={prediction[1]:.0f} | Score: {raw_score:.0f}")

                    smoother.update(raw_score, actual_lms)
                    score      = smoother.smooth_score()
                    actual_lms = smoother.smooth_landmarks()
                    is_bad     = score < 70

                    if is_bad:
                        ghost_lms = reconstruction_to_landmarks(prediction, feature_cols, landmarks)
                        fault_text = get_fault_text(features, prediction, feature_cols)
                    else:
                        ghost_lms  = {}
                        fault_text = ""

                # Draw actual skeleton
                draw_skeleton(frame, actual_lms, CONNECTIONS, score_to_color(score),
                              SKELETON_THICKNESS, h, w)

                # Draw ghost skeleton
                if ghost_lms:
                    draw_skeleton(frame, ghost_lms, LATERAL_RAISE_GHOST_CONNECTIONS,
                                  COLOR_GHOST, GHOST_THICKNESS, h, w)

            draw_hud(frame, "lateral raise", score, is_bad, h, w, fault_text)
            cv2.imshow("Workout Form Analyzer", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Session ended.")


run(start_exercise="lateral raise")
