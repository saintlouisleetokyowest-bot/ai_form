"""
Generate Synthetic Bad-Form Data
=================================
Takes your existing good-form angles.csv and creates paired (bad, good) data
for supervised training: F(A) → A*

Each good-form frame gets multiple "bad" variants with realistic, patterned
perturbations that mimic common exercise mistakes — not just random noise.

Usage:
    python generate_bad_form.py

Output:
    paired_data.csv — columns: exercise, variant, [features]_bad, [features]_good

Author: You (guided by Claude)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_CSV  = "new_angles.csv"
OUTPUT_CSV = "synthetic_paired_data.csv"
VARIANTS_PER_FRAME = 3       # How many bad variants to generate per good frame
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# ══════════════════════════════════════════════════════════════════════════════
# PERTURBATION TEMPLATES — common real-world mistakes
# ══════════════════════════════════════════════════════════════════════════════
#
# Each template is a dict of feature_name → (bias, noise_std)
#   bias:      systematic shift (mimics a consistent mistake)
#   noise_std: random variation on top (mimics inconsistency)
#
# Positive bias = increase the value, negative = decrease.
# For angles in degrees; for positional features in normalised coords.

SQUAT_MISTAKES = [
    {
        "name": "knees_caving_in",
        "description": "Knee valgus — knees collapse inward during descent",
        "perturbations": {
            "left_knee_valgus":  (0.06, 0.02),    # knee drifts inward
            "right_knee_valgus": (-0.06, 0.02),   # knee drifts inward (opposite sign)
            "left_knee_angle":   (5.0, 3.0),      # slightly less bent (compensating)
            "right_knee_angle":  (5.0, 3.0),
        }
    },
    {
        "name": "not_deep_enough",
        "description": "Shallow squat — hips stay too high",
        "perturbations": {
            "left_knee_angle":   (25.0, 10.0),    # knees much straighter
            "right_knee_angle":  (25.0, 10.0),
            "hip_depth_left":    (0.06, 0.02),    # hips higher relative to knees
            "hip_depth_right":   (0.06, 0.02),
            "left_hip_angle":    (15.0, 5.0),     # hips less flexed
            "right_hip_angle":   (15.0, 5.0),
        }
    },
    {
        "name": "excessive_forward_lean",
        "description": "Torso tips too far forward — back not upright",
        "perturbations": {
            "torso_lean":        (15.0, 5.0),     # big forward lean
            "left_hip_angle":    (-10.0, 4.0),    # hips close more
            "right_hip_angle":   (-10.0, 4.0),
        }
    },
    {
        "name": "asymmetric_descent",
        "description": "One side drops more than the other — weight shift",
        "perturbations": {
            "left_knee_angle":   (-15.0, 5.0),    # left goes deeper
            "right_knee_angle":  (10.0, 5.0),     # right stays higher
            "hip_depth_left":    (-0.04, 0.01),
            "hip_depth_right":   (0.03, 0.01),
        }
    },
    {
        "name": "heels_rising",
        "description": "Ankles lose dorsiflexion — heels come off ground",
        "perturbations": {
            "left_ankle_angle":  (-20.0, 8.0),    # ankle angle decreases
            "right_ankle_angle": (-20.0, 8.0),
            "torso_lean":        (8.0, 3.0),      # compensatory lean
        }
    },
]

LATERAL_RAISE_MISTAKES = [
    {
        "name": "arms_too_high",
        "description": "Raising arms past shoulder height — traps take over",
        "perturbations": {
            "left_arm_raise":    (25.0, 8.0),     # arms go way above 90°
            "right_arm_raise":   (25.0, 8.0),
            "torso_lean":        (5.0, 2.0),      # slight lean from momentum
        }
    },
    {
        "name": "arms_too_low",
        "description": "Not raising arms high enough — half reps",
        "perturbations": {
            "left_arm_raise":    (-25.0, 8.0),
            "right_arm_raise":   (-25.0, 8.0),
        }
    },
    {
        "name": "elbows_too_bent",
        "description": "Bending elbows excessively — turns into a curl",
        "perturbations": {
            "left_elbow_angle":  (-30.0, 10.0),   # elbows bend a lot
            "right_elbow_angle": (-30.0, 10.0),
        }
    },
    {
        "name": "torso_swinging",
        "description": "Using momentum — body swings to cheat the weight up",
        "perturbations": {
            "torso_lean":        (12.0, 5.0),     # big torso sway
            "left_arm_raise":    (10.0, 5.0),     # arms higher from momentum
            "right_arm_raise":   (10.0, 5.0),
        }
    },
    {
        "name": "asymmetric_raise",
        "description": "One arm higher than the other — uneven strength",
        "perturbations": {
            "left_arm_raise":    (15.0, 5.0),     # left goes higher
            "right_arm_raise":   (-10.0, 5.0),    # right stays lower
            "arm_symmetry":      (20.0, 5.0),     # symmetry breaks
        }
    },
]

EXERCISE_MISTAKES = {
    "squat":         SQUAT_MISTAKES,
    "lateral raise": LATERAL_RAISE_MISTAKES,
}

EXERCISE_FEATURES = {
    "squat": [
        "left_knee_angle", "right_knee_angle",
        "left_hip_angle", "right_hip_angle",
        "torso_lean",
        "left_knee_valgus", "right_knee_valgus",
        "hip_depth_left", "hip_depth_right",
        "left_ankle_angle", "right_ankle_angle",
    ],
    "lateral raise": [
        "left_arm_raise", "right_arm_raise",
        "left_elbow_angle", "right_elbow_angle",
        "torso_lean", "arm_symmetry",
        "left_wrist_above_shoulder", "right_wrist_above_shoulder",
    ],
}

# Reasonable physical bounds for clipping
FEATURE_BOUNDS = {
    "left_knee_angle":   (30, 180),
    "right_knee_angle":  (30, 180),
    "left_hip_angle":    (30, 180),
    "right_hip_angle":   (30, 180),
    "torso_lean":        (0, 60),
    "left_knee_valgus":  (-0.3, 0.3),
    "right_knee_valgus": (-0.3, 0.3),
    "hip_depth_left":    (-0.3, 0.3),
    "hip_depth_right":   (-0.3, 0.3),
    "left_ankle_angle":  (60, 180),
    "right_ankle_angle": (60, 180),
    "left_arm_raise":    (5, 170),
    "right_arm_raise":   (5, 170),
    "left_elbow_angle":  (30, 180),
    "right_elbow_angle": (30, 180),
    "arm_symmetry":      (0, 80),
    "left_wrist_above_shoulder":  (-0.4, 0.4),
    "right_wrist_above_shoulder": (-0.4, 0.4),
}


# ══════════════════════════════════════════════════════════════════════════════
# PERTURBATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def apply_perturbation(good_row, mistake_template, feature_cols):
    """
    Apply a single mistake template to a good-form row.

    Returns a dict of perturbed (bad) feature values.
    """
    bad = {}
    for feat in feature_cols:
        good_val = good_row[feat]

        if feat in mistake_template["perturbations"]:
            bias, noise_std = mistake_template["perturbations"][feat]
            perturbation = bias + np.random.normal(0, noise_std)
            bad_val = good_val + perturbation
        else:
            # Features not in the template get light random noise
            # (a person with bad form doesn't hold everything else perfectly)
            noise = np.random.normal(0, 2.0) if "angle" in feat else np.random.normal(0, 0.005)
            bad_val = good_val + noise

        # Clip to physical bounds
        if feat in FEATURE_BOUNDS:
            lo, hi = FEATURE_BOUNDS[feat]
            bad_val = np.clip(bad_val, lo, hi)

        bad[feat] = bad_val

    return bad


def generate_pairs(df, exercise, variants_per_frame):
    """
    Generate (bad, good) pairs for a single exercise.

    Returns a list of row dicts with columns:
        exercise, variant, feature_bad, feature_good for each feature
    """
    feature_cols = EXERCISE_FEATURES[exercise]
    mistakes = EXERCISE_MISTAKES[exercise]
    ex_df = df[df["exercise"] == exercise].copy()

    if len(ex_df) == 0:
        print(f"  [SKIP] No data for {exercise}")
        return []

    rows = []

    for _, good_row in ex_df.iterrows():
        # Pick random mistake templates for this frame
        chosen = np.random.choice(len(mistakes), size=variants_per_frame, replace=True)

        for i, mistake_idx in enumerate(chosen):
            template = mistakes[mistake_idx]
            bad_features = apply_perturbation(good_row, template, feature_cols)

            row = {
                "exercise": exercise,
                "variant":  template["name"],
                "source_video": good_row.get("video", "unknown"),
                "source_frame": good_row.get("frame", -1),
            }

            # Bad (input) features
            for feat in feature_cols:
                row[f"{feat}_bad"] = bad_features[feat]

            # Good (target) features
            for feat in feature_cols:
                row[f"{feat}_good"] = good_row[feat]

            rows.append(row)

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("GENERATING SYNTHETIC BAD-FORM DATA")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV)
    print(f"\nLoaded {len(df)} good-form frames from {INPUT_CSV}")

    all_rows = []

    for exercise in EXERCISE_FEATURES:
        ex_count = len(df[df["exercise"] == exercise])
        print(f"\n[{exercise.upper()}]")
        print(f"  Good frames: {ex_count}")
        print(f"  Variants per frame: {VARIANTS_PER_FRAME}")
        print(f"  Mistake templates: {len(EXERCISE_MISTAKES[exercise])}")

        for m in EXERCISE_MISTAKES[exercise]:
            print(f"    • {m['name']}: {m['description']}")

        rows = generate_pairs(df, exercise, VARIANTS_PER_FRAME)
        all_rows.extend(rows)
        print(f"  Generated: {len(rows)} (bad, good) pairs")

    paired_df = pd.DataFrame(all_rows)
    paired_df.to_csv(OUTPUT_CSV, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total pairs: {len(paired_df)}")
    for ex in paired_df["exercise"].unique():
        sub = paired_df[paired_df["exercise"] == ex]
        print(f"\n  {ex}:")
        print(f"    Total pairs: {len(sub)}")
        print(f"    Variant distribution:")
        for variant, count in sub["variant"].value_counts().items():
            print(f"      {variant}: {count}")

    print(f"\nSaved to: {OUTPUT_CSV}")
    print(f"\nColumn format:")
    print(f"  *_bad  = synthetic bad-form input  (X)")
    print(f"  *_good = original good-form target  (y)")
    print(f"\nNext step: update train_autoencoder.py to train a supervised")
    print(f"model on this paired data: F(X_bad) → y_good")

    return paired_df


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
