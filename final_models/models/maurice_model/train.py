
import pandas as pd
import numpy as np


EXERCISE_FEATURES = {
    "lateral raise": [
        "left_arm_raise",
        "right_arm_raise",
        "left_elbow_angle",
        "right_elbow_angle",
        "torso_lean",
        "arm_symmetry",
        "left_wrist_above_shoulder",
        "right_wrist_above_shoulder",
    ],
}


file_path = "/Users/mauriceengel/code/saintlouisleetokyowest-bot/ai_form/final_models/preprocessing/paired_data.csv"


def load_paired_data(csv_path):
    df = pd.read_csv(csv_path)

    bad_cols  = [f"{feat}_bad" for feat in EXERCISE_FEATURES["lateral raise"]]
    good_cols = [f"{feat}_good" for feat in EXERCISE_FEATURES["lateral raise"]]

    X = df[bad_cols]
    y = df[good_cols]

    return X, y


X, y = load_paired_data(file_path)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
