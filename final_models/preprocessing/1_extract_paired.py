"""
Extract Paired Landmarks for Supervised Training
==================================================
Processes paired bad-form and good-form lateral raise videos.
Extracts MediaPipe joint angles from each pair and outputs a single CSV
with _bad and _good columns, ready for supervised model training.

Folder structure expected:
    videos/
    ├── bad/
    │   ├── badform_1.MOV
    │   ├── badform_2.MOV
    │   └── ...
    └── good/
        ├── goodform_1.MOV
        ├── goodform_2.MOV
        └── ...

Pairing: badform_N matched with goodform_N by number.

Usage:
    python extract_paired.py

Output:
    paired_data.csv
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path

# ── MediaPipe setup ────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
LM = mp_pose.PoseLandmark


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR   = Path(__file__).parent.parent  # final_models/
VIDEO_DIR  = BASE_DIR / "videos"
OUTPUT_CSV = Path(__file__).parent / "paired_data.csv"
FRAME_SKIP  = 3

LATERAL_RAISE_FEATURES = [
    "left_arm_raise",
    "right_arm_raise",
    "left_elbow_angle",
    "right_elbow_angle",
    "torso_lean",
    "arm_symmetry",
    "left_wrist_above_shoulder",
    "right_wrist_above_shoulder",
]


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_coords(landmarks, landmark_enum):
    """Extract (x, y, z) from a landmark."""
    lm = landmarks[landmark_enum.value]
    return np.array([lm.x, lm.y, lm.z])


def angle_between(a, b, c):
    """Angle at point B formed by vectors BA and BC, in degrees."""
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def midpoint(a, b):
    return (a + b) / 2.0


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — LATERAL RAISES
# ══════════════════════════════════════════════════════════════════════════════

def extract_lateral_raise_angles(landmarks):
    """Extract biomechanically relevant angles for lateral raise form."""
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

    spine_vec = mid_shoulder - mid_hip
    vertical  = np.array([0, -1, 0])
    cosine    = np.dot(spine_vec, vertical) / (np.linalg.norm(spine_vec) + 1e-6)

    left_raise  = angle_between(l_hip, l_shoulder, l_elbow)
    right_raise = angle_between(r_hip, r_shoulder, r_elbow)

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


def is_active(features):
    """Returns True if the person is actively raising (not arms at rest)."""
    avg_raise = (features["left_arm_raise"] + features["right_arm_raise"]) / 2
    return avg_raise > 30


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

def extract_frames(video_path, frame_skip=3):
    """
    Run MediaPipe on a single video and extract angles from active frames.

    Returns:
        List of feature dicts (one per active frame)
    """
    frames = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Could not open {video_path.name}")
        return frames

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
                features = extract_lateral_raise_angles(result.pose_landmarks.landmark)

                if is_active(features):
                    frames.append(features)

            frame_idx += 1

    cap.release()
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# PAIR DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def find_pairs(video_dir):
    """
    Scan bad/ and good/ folders, match by number.

    Returns:
        List of (pair_number, bad_path, good_path) tuples
    """
    video_dir = Path(video_dir)
    bad_dir   = video_dir / "bad"
    good_dir  = video_dir / "good"

    if not bad_dir.exists():
        raise FileNotFoundError(f"Bad form folder not found: {bad_dir}")
    if not good_dir.exists():
        raise FileNotFoundError(f"Good form folder not found: {good_dir}")

    # Build lookup: number → file path
    def index_folder(folder, prefix):
        lookup = {}
        for f in sorted(folder.glob("*.*")):
            name = f.stem.lower()  # e.g. "badform_1"
            parts = name.split("_")
            if len(parts) >= 2 and parts[-1].isdigit():
                num = int(parts[-1])
                lookup[num] = f
        return lookup

    bad_files  = index_folder(bad_dir, "badform")
    good_files = index_folder(good_dir, "goodform")

    # Match pairs
    bad_nums  = set(bad_files.keys())
    good_nums = set(good_files.keys())
    matched   = sorted(bad_nums & good_nums)
    unmatched_bad  = sorted(bad_nums - good_nums)
    unmatched_good = sorted(good_nums - bad_nums)

    if unmatched_bad:
        print(f"  [WARN] Bad clips with no good match: {unmatched_bad}")
    if unmatched_good:
        print(f"  [WARN] Good clips with no bad match: {unmatched_good}")

    pairs = [(n, bad_files[n], good_files[n]) for n in matched]
    print(f"  Found {len(pairs)} matched pairs")

    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# PAIR PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_pair(pair_num, bad_path, good_path, frame_skip=3):
    """
    Extract frames from one bad/good pair.

    Since bad and good clips may have different frame counts, we align them
    by sampling the shorter list to match the longer, or taking the minimum
    of both. Each bad frame is paired with the closest good frame by index
    proportion (i.e. frame position within the rep).

    Returns:
        List of row dicts with _bad and _good columns
    """
    bad_frames  = extract_frames(bad_path, frame_skip)
    good_frames = extract_frames(good_path, frame_skip)

    if not bad_frames or not good_frames:
        print(f"    [SKIP] Pair {pair_num}: bad={len(bad_frames)}, good={len(good_frames)} active frames")
        return []

    rows = []
    n_bad  = len(bad_frames)
    n_good = len(good_frames)

    for i in range(n_bad):
        # Map bad frame index proportionally to good frame index
        # So frame 0 of bad maps to frame 0 of good,
        # and the last bad frame maps to the last good frame
        j = int(round(i * (n_good - 1) / max(n_bad - 1, 1)))

        bad_feat  = bad_frames[i]
        good_feat = good_frames[j]

        row = {
            "pair":  pair_num,
            "frame": i,
        }

        for feat in LATERAL_RAISE_FEATURES:
            row[f"{feat}_bad"]  = bad_feat[feat]
            row[f"{feat}_good"] = good_feat[feat]

        rows.append(row)

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("EXTRACTING PAIRED LATERAL RAISE DATA")
    print("=" * 60)

    pairs = find_pairs(VIDEO_DIR)

    all_rows = []

    for pair_num, bad_path, good_path in pairs:
        print(f"\n  Pair {pair_num}: {bad_path.name} ↔ {good_path.name}")
        rows = process_pair(pair_num, bad_path, good_path, frame_skip=FRAME_SKIP)
        all_rows.extend(rows)
        print(f"    → {len(rows)} paired frames")

    if not all_rows:
        print("\n[ERROR] No data extracted. Check your video paths and folder structure.")
        return None

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Total paired frames: {len(df)}")
    print(f"  Pairs processed:     {df['pair'].nunique()}")
    print(f"  Frames per pair:")
    for p in sorted(df["pair"].unique()):
        count = len(df[df["pair"] == p])
        print(f"    Pair {p}: {count} frames")

    print(f"\n  Output columns:")
    for feat in LATERAL_RAISE_FEATURES:
        print(f"    {feat}_bad  /  {feat}_good")

    print(f"\nSaved to: {OUTPUT_CSV}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
