"""
Step 3: Real-Time Form Analyzer
================================
Opens webcam, detects pose with MediaPipe, runs the autoencoder,
displays a form score with specific fault text, and renders a
corrective ghost skeleton showing ideal form.

Controls:
    Q         — quit
    S         — switch to squats mode
    L         — switch to lateral raise mode

Author: You (guided by Claude)
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
MODELS_DIR = Path("models")

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

SQUAT_GHOST_CONNECTIONS = [
    (LM.LEFT_SHOULDER,  LM.RIGHT_SHOULDER),
    (LM.LEFT_SHOULDER,  LM.LEFT_HIP),
    (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    (LM.LEFT_HIP,       LM.RIGHT_HIP),
    (LM.LEFT_HIP,       LM.LEFT_KNEE),
    (LM.LEFT_KNEE,      LM.LEFT_ANKLE),
    (LM.RIGHT_HIP,      LM.RIGHT_KNEE),
    (LM.RIGHT_KNEE,     LM.RIGHT_ANKLE),
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
    def __init__(self, window=5):
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
    """Load autoencoder, scaler, and metadata for a given exercise."""
    safe = exercise_name.replace(" ", "_")

    autoencoder = tf.keras.models.load_model(MODELS_DIR / f"{safe}_autoencoder.keras", compile=False)
    decoder     = tf.keras.models.load_model(MODELS_DIR / f"{safe}_decoder.keras",     compile=False)
    encoder     = tf.keras.models.load_model(MODELS_DIR / f"{safe}_encoder.keras",     compile=False)

    with open(MODELS_DIR / f"{safe}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open(MODELS_DIR / f"{safe}_meta.json") as f:
        meta = json.load(f)

    print(f"[{exercise_name}] Loaded — threshold: {meta['threshold']:.4f}")
    return autoencoder, encoder, decoder, scaler, meta


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

def extract_squat_features(landmarks):
    l_hip      = get_coords(landmarks, LM.LEFT_HIP)
    r_hip      = get_coords(landmarks, LM.RIGHT_HIP)
    l_knee     = get_coords(landmarks, LM.LEFT_KNEE)
    r_knee     = get_coords(landmarks, LM.RIGHT_KNEE)
    l_ankle    = get_coords(landmarks, LM.LEFT_ANKLE)
    r_ankle    = get_coords(landmarks, LM.RIGHT_ANKLE)
    l_shoulder = get_coords(landmarks, LM.LEFT_SHOULDER)
    r_shoulder = get_coords(landmarks, LM.RIGHT_SHOULDER)
    l_heel     = get_coords(landmarks, LM.LEFT_HEEL)
    r_heel     = get_coords(landmarks, LM.RIGHT_HEEL)
    l_foot     = get_coords(landmarks, LM.LEFT_FOOT_INDEX)
    r_foot     = get_coords(landmarks, LM.RIGHT_FOOT_INDEX)
    mid_hip      = midpoint(l_hip, r_hip)
    mid_shoulder = midpoint(l_shoulder, r_shoulder)
    spine_vec    = mid_shoulder - mid_hip
    vertical     = np.array([0, -1, 0])
    cosine       = np.dot(spine_vec, vertical) / (np.linalg.norm(spine_vec) + 1e-6)

    return {
        "left_knee_angle":   angle_between(l_hip, l_knee, l_ankle),
        "right_knee_angle":  angle_between(r_hip, r_knee, r_ankle),
        "left_hip_angle":    angle_between(l_shoulder, l_hip, l_knee),
        "right_hip_angle":   angle_between(r_shoulder, r_hip, r_knee),
        "torso_lean":        np.degrees(np.arccos(np.clip(cosine, -1, 1))),
        "left_knee_valgus":  float(l_knee[0] - l_foot[0]),
        "right_knee_valgus": float(r_knee[0] - r_foot[0]),
        "hip_depth_left":    float(l_hip[1] - l_knee[1]),
        "hip_depth_right":   float(r_hip[1] - r_knee[1]),
        "left_ankle_angle":  angle_between(l_knee, l_ankle, l_heel),
        "right_ankle_angle": angle_between(r_knee, r_ankle, r_heel),
    }


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


FEATURE_EXTRACTORS = {
    "squat":         extract_squat_features,
    "lateral raise": extract_lateral_raise_features,
}


# ══════════════════════════════════════════════════════════════════════════════
# FORM SCORING
# ══════════════════════════════════════════════════════════════════════════════

def score_form(features, meta, scaler, autoencoder):
    feature_cols = meta["features"]
    threshold    = meta["threshold"]
    vec          = np.array([[features[f] for f in feature_cols]], dtype=np.float32)
    vec_scaled   = scaler.transform(vec)
    reconstruction = autoencoder.predict(vec_scaled, verbose=0)
    error          = float(np.mean(np.square(vec_scaled - reconstruction)))
    score          = max(0.0, 100.0 * np.exp(-error / (threshold * 5)))
    is_bad         = error > threshold
    return score, error, reconstruction[0], is_bad


# ══════════════════════════════════════════════════════════════════════════════
# FAULT TEXT
# ══════════════════════════════════════════════════════════════════════════════

def get_fault_text(features, meta, scaler, reconstruction, exercise):
    """
    Compare actual features against ideal reconstruction to identify
    the most important fault. Returns a short instruction string.
    """
    ideal_vec    = scaler.inverse_transform(reconstruction.reshape(1, -1))[0]
    feature_cols = meta["features"]
    ideal        = dict(zip(feature_cols, ideal_vec))
    faults       = []

    if exercise == "squat":
        avg_knee        = (features["left_knee_angle"] + features["right_knee_angle"]) / 2
        ideal_knee      = (ideal["left_knee_angle"] + ideal["right_knee_angle"]) / 2
        avg_hip_depth   = (features["hip_depth_left"] + features["hip_depth_right"]) / 2
        ideal_hip_depth = (ideal["hip_depth_left"] + ideal["hip_depth_right"]) / 2

        if avg_knee - ideal_knee > 15:
            faults.append((avg_knee - ideal_knee, "GO DEEPER"))
        if features["torso_lean"] - ideal["torso_lean"] > 10:
            faults.append((features["torso_lean"] - ideal["torso_lean"], "KEEP BACK STRAIGHT"))
        if features["left_knee_valgus"] - ideal["left_knee_valgus"] > 0.04:
            faults.append((0.5, "LEFT KNEE CAVING IN"))
        if features["right_knee_valgus"] - ideal["right_knee_valgus"] < -0.04:
            faults.append((0.5, "RIGHT KNEE CAVING IN"))
        if avg_hip_depth - ideal_hip_depth > 0.03:
            faults.append((avg_hip_depth - ideal_hip_depth, "HIPS TOO HIGH"))

    else:
        avg_raise   = (features["left_arm_raise"] + features["right_arm_raise"]) / 2

        # Hard rules for obvious extremes
        if avg_raise > 110:
            return "ARMS TOO HIGH"
        if avg_raise < 35:
            return "RAISE ARMS HIGHER"


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

def place_joint_at_angle(anchor, far_anchor, bone_len, ideal_angle_deg, side="left"):
    anchor_to_far = far_anchor - anchor
    total_len     = np.linalg.norm(anchor_to_far)
    if total_len < 1e-6 or bone_len < 1e-6:
        return anchor + np.array([0, bone_len])
    axis_dir  = anchor_to_far / total_len
    perp      = np.array([-axis_dir[1], axis_dir[0]])
    if side == "right":
        perp = -perp
    angle_rad = np.radians(ideal_angle_deg)
    along     = bone_len * np.cos(np.pi - angle_rad)
    across    = bone_len * np.sin(np.pi - angle_rad)
    joint_pos = anchor + along * axis_dir + across * perp
    return np.clip(joint_pos, 0.0, 1.0)


def reconstruction_to_landmarks(reconstruction, scaler, meta, actual_landmarks):
    ideal_vec    = scaler.inverse_transform(reconstruction.reshape(1, -1))[0]
    feature_cols = meta["features"]
    ideal        = dict(zip(feature_cols, ideal_vec))

    def lm_xy(lm_enum):
        lm = actual_landmarks[lm_enum.value]
        return np.array([lm.x, lm.y])

    # Start with all actual landmarks as base
    ghost = {lm_enum: lm_xy(lm_enum) for lm_enum in LM}

    if "left_knee_angle" in ideal:
        # SQUAT — reposition knees and shoulders
        for side, hip_lm, knee_lm, ankle_lm in [
            ("left",  LM.LEFT_HIP,  LM.LEFT_KNEE,  LM.LEFT_ANKLE),
            ("right", LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
        ]:
            hip_pos     = lm_xy(hip_lm)
            ankle_pos   = lm_xy(ankle_lm)
            actual_knee = lm_xy(knee_lm)
            femur_len   = np.linalg.norm(actual_knee - hip_pos)
            ideal_angle = float(np.clip(ideal.get(f"{side}_knee_angle", 90.0), 30.0, 170.0))
            ghost[knee_lm]  = place_joint_at_angle(hip_pos, ankle_pos, femur_len, ideal_angle, side)
            ghost[hip_lm]   = hip_pos
            ghost[ankle_lm] = ankle_pos

        ideal_lean = float(np.clip(ideal.get("torso_lean", 15.0), 0.0, 60.0))
        lean_rad   = np.radians(ideal_lean)
        for hip_lm, shoulder_lm in [
            (LM.LEFT_HIP, LM.LEFT_SHOULDER),
            (LM.RIGHT_HIP, LM.RIGHT_SHOULDER),
        ]:
            hip_pos   = lm_xy(hip_lm)
            actual_sh = lm_xy(shoulder_lm)
            torso_len = np.linalg.norm(actual_sh - hip_pos)
            direction = np.array([np.sin(lean_rad), -np.cos(lean_rad)])
            ghost[shoulder_lm] = np.clip(hip_pos + torso_len * direction, 0.0, 1.0)

    else:
        # LATERAL RAISE — reposition elbows and wrists
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

    # Head circle at nose
    if LM.NOSE in landmarks:
        nose = (int(landmarks[LM.NOSE][0] * w), int(landmarks[LM.NOSE][1] * h))
        cv2.circle(frame, nose, 18, color, 2)

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
    cv2.putText(frame, "S=Squat  L=Lateral Raise  Q=Quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # Ghost legend
    cv2.circle(frame, (w - 160, h - 20), 6, COLOR_GHOST, -1)
    cv2.putText(frame, "= Ideal form", (w - 148, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GHOST, 1)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run(start_exercise="squat"):
    print("Loading models...")
    models = {}
    for ex in ["squat", "lateral raise"]:
        autoencoder, encoder, decoder, scaler, meta = load_model_artifacts(ex)
        models[ex] = {
            "autoencoder": autoencoder,
            "encoder":     encoder,
            "decoder":     decoder,
            "scaler":      scaler,
            "meta":        meta,
        }
    print("Models loaded. Opening webcam...")

    current_exercise = start_exercise
    smoother         = Smoother(window=5)
    cap              = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
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

            m = models[current_exercise]
            ghost_connections = (SQUAT_GHOST_CONNECTIONS
                                 if current_exercise == "squat"
                                 else LATERAL_RAISE_GHOST_CONNECTIONS)

            if result.pose_landmarks:
                landmarks  = result.pose_landmarks.landmark
                extract_fn = FEATURE_EXTRACTORS[current_exercise]
                features   = extract_fn(landmarks)

                if current_exercise == "lateral raise":
                    avg_raise = (features["left_arm_raise"] + features["right_arm_raise"]) / 2
                    if avg_raise > 110 or avg_raise < 35:
                        is_bad = True

                raw_score, error, reconstruction, raw_is_bad = score_form(
                    features, m["meta"], m["scaler"], m["autoencoder"]
                )

                actual_lms = actual_landmarks_to_dict(landmarks)
                smoother.update(raw_score, actual_lms)
                score      = smoother.smooth_score()
                actual_lms = smoother.smooth_landmarks()
                is_bad     = score < 70

                if is_bad:
                    ghost_lms  = reconstruction_to_landmarks(
                        reconstruction, m["scaler"], m["meta"], landmarks
                    )
                    fault_text = get_fault_text(
                        features, m["meta"], m["scaler"], reconstruction, current_exercise
                    )
                else:
                    ghost_lms  = {}
                    fault_text = ""

                # Draw actual skeleton
                score_color = score_to_color(score)
                cv2.putText(frame, f"Form: {score:.0f}/100", (15, 62),

            cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
                draw_skeleton(frame, actual_lms, CONNECTIONS, score_to_color(score),
                              SKELETON_THICKNESS, h, w)

                # Draw ghost skeleton
                if ghost_lms:
                    draw_skeleton(frame, ghost_lms, ghost_connections, COLOR_GHOST,
                                  GHOST_THICKNESS, h, w)

            draw_hud(frame, current_exercise, score, is_bad, h, w, fault_text)
            cv2.imshow("Workout Form Analyzer", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                current_exercise = "squat"
                smoother.reset()
                ghost_lms  = {}
                fault_text = ""
                print("Switched to: squat")
            elif key == ord("l"):
                current_exercise = "lateral raise"
                smoother.reset()
                ghost_lms  = {}
                fault_text = ""
                print("Switched to: lateral raise")

    cap.release()
    cv2.destroyAllWindows()
    print("Session ended.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

run(start_exercise="squat")
