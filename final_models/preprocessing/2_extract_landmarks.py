"""
Landmark & Angle Extractor (Autoencoder Version)
=========================================================
Processes workout videos with MediaPipe Pose and extracts joint angles
for Lateral Raises.

Orignal good-form with paired bad-form repetitions by Jinnosuke
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from pathlib import Path

# ── MediaPipe setup ────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose

SUPPORTED_EXERCISES = {"lateral raise"}

# ── Landmark indices ───────────────────────────────────────────────────────────
LM = mp_pose.PoseLandmark


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_coords(landmarks, landmark_enum):
    """Extract (x, y, z) from a landmark."""
    lm = landmarks[landmark_enum.value]
    return np.array([lm.x, lm.y, lm.z])


def angle_between(a, b, c):
    """
    Compute the angle at point B formed by vectors BA and BC.
    Returns angle in degrees (0-180).
    """
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def midpoint(a, b):
    return (a + b) / 2.0

# ══════════════════════════════════════════════════════════════════════════════
# ANGLE EXTRACTION — LATERAL RAISES
# ══════════════════════════════════════════════════════════════════════════════

def extract_lateral_raise_angles(landmarks):
    """
    Extract biomechanically relevant angles for lateral raise form analysis.
    Returns a dict of named angles/features.
    """
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

    features = {}

    # Arm raise height
    features["left_arm_raise"]  = angle_between(l_hip, l_shoulder, l_elbow)
    features["right_arm_raise"] = angle_between(r_hip, r_shoulder, r_elbow)

    # Elbow bend
    features["left_elbow_angle"]  = angle_between(l_shoulder, l_elbow, l_wrist)
    features["right_elbow_angle"] = angle_between(r_shoulder, r_elbow, r_wrist)

    # Torso stability
    spine_vec = mid_shoulder - mid_hip
    vertical  = np.array([0, -1, 0])
    cosine    = np.dot(spine_vec, vertical) / (np.linalg.norm(spine_vec) + 1e-6)
    features["torso_lean"] = np.degrees(np.arccos(np.clip(cosine, -1, 1)))

    # Arm symmetry
    features["arm_symmetry"] = abs(features["left_arm_raise"] - features["right_arm_raise"])

    # Wrist height relative to shoulder (positive = wrist above shoulder)
    features["left_wrist_above_shoulder"]  = float(l_shoulder[1] - l_wrist[1])
    features["right_wrist_above_shoulder"] = float(r_shoulder[1] - r_wrist[1])

    return features


def is_lateral_raise_active(features):
    """
    Returns True if the person is actively raising (not arms at rest).
    """
    avg_raise = (features["left_arm_raise"] + features["right_arm_raise"]) / 2
    return avg_raise > 30  # Arms raised = active movement


# ══════════════════════════════════════════════════════════════════════════════
# EXERCISE REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

EXERCISE_CONFIG = {
    "lateral raise": {
        "extract_fn":   extract_lateral_raise_angles,
        "is_active_fn": is_lateral_raise_active,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

def process_video(video_path, exercise, frame_skip=3):
    """
    Run MediaPipe on a single video. Extract angles from active frames only.

    Returns:
        List of row dicts ready for a DataFrame.
    """
    config     = EXERCISE_CONFIG[exercise]
    extract_fn = config["extract_fn"]
    active_fn  = config["is_active_fn"]
    rows       = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Could not open {video_path.name}")
        return rows

    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                features  = extract_fn(landmarks)

                # Skip resting/transition frames
                if not active_fn(features):
                    frame_idx += 1
                    continue

                row = {
                    "video":    video_path.name,
                    "exercise": exercise,
                    "frame":    frame_idx,
                }
                row.update(features)
                rows.append(row)

            frame_idx += 1

    cap.release()
    return rows


def process_directory(video_dir, output_csv="angles.csv", frame_skip=3):
    """
    Walk the video directory, process all supported exercises, save CSV.
    """
    video_dir = Path(video_dir)
    all_rows  = []

    for exercise in SUPPORTED_EXERCISES:
        exercise_dir = video_dir / exercise
        if not exercise_dir.exists():
            print(f"[INFO] No folder found for '{exercise}', skipping.")
            continue

        video_files = (
            list(exercise_dir.glob("*.mp4")) +
            list(exercise_dir.glob("*.avi")) +
            list(exercise_dir.glob("*.mov"))
        )

        print(f"\n[{exercise.upper()}] Found {len(video_files)} videos")

        for i, vf in enumerate(video_files):
            print(f"  ({i+1}/{len(video_files)}) Processing {vf.name}...")
            rows = process_video(vf, exercise, frame_skip=frame_skip)
            all_rows.extend(rows)
            print(f"    → {len(rows)} active frames extracted")

    if not all_rows:
        print("\n[ERROR] No data extracted. Check your folder structure and video paths.")
        return None

    df = pd.DataFrame(all_rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("EXTRACTION COMPLETE")
    print("="*50)
    for ex in df["exercise"].unique():
        sub = df[df["exercise"] == ex]
        print(f"\n  {ex}:")
        print(f"    Total active frames: {len(sub)}")
        print(f"    Videos processed:    {sub['video'].nunique()}")
        feature_cols = [c for c in sub.columns if c not in ["video", "exercise", "frame"]]
        print(f"    Feature means:")
        for col in feature_cols:
            print(f"      {col}: {sub[col].mean():.2f}")

    df.to_csv(output_csv, index=False)
    print(f"\nSaved to: {output_csv}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — edit video_dir path and run this cell in Jupyter
# ══════════════════════════════════════════════════════════════════════════════

process_directory(
    video_dir="/Users/mauriceengel/code/saintlouisleetokyowest-bot/ai_form/final_models_more_videos",
    output_csv=str(Path(__file__).parent / "new_angles.csv"),
    frame_skip=3
)
