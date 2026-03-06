import pandas as pd
import numpy as np
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
from keras import layers, Sequential, Input, optimizers, callbacks
from keras.callbacks import EarlyStopping

import pickle
import json

PREPROCESS_DIR = Path(__file__).parent.parent.parent / "preprocessing"

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

# ── Load & filter data ────────────────────────────────────────────────────────
real      = pd.read_csv(str(PREPROCESS_DIR / "paired_data.csv"))
synthetic = pd.read_csv(str(PREPROCESS_DIR / "synthetic_paired_data.csv"))

# Filter extreme pairs from BOTH real and synthetic data
for feat in EXERCISE_FEATURES["lateral raise"]:
    real      = real     [abs(real     [f"{feat}_bad"] - real     [f"{feat}_good"]) < 40]
    synthetic = synthetic[abs(synthetic[f"{feat}_bad"] - synthetic[f"{feat}_good"]) < 40]

# Identity pairs: good form → good form (teaches model not to over-correct)
good_only = pd.read_csv(str(PREPROCESS_DIR / "new_angles.csv"))
feature_list = EXERCISE_FEATURES["lateral raise"]
identity_rows = []
for _, row in good_only.iterrows():
    r = {}
    for feat in feature_list:
        r[f"{feat}_bad"] = row[feat]
        r[f"{feat}_good"] = row[feat]
    identity_rows.append(r)
identity_df = pd.DataFrame(identity_rows)

feature_cols = [c for c in real.columns if c.endswith("_bad") or c.endswith("_good")]
combined = pd.concat([real[feature_cols], synthetic[feature_cols], identity_df], ignore_index=True)
combined.to_csv(str(PREPROCESS_DIR / "combined_paired_data.csv"), index=False)

print(f"Real:      {len(real)}")
print(f"Synthetic: {len(synthetic)}")
print(f"Identity:  {len(identity_df)}")
print(f"Combined:  {len(combined)}")

# ── Load paired data ──────────────────────────────────────────────────────────
file_path = str(Path(__file__).parent.parent.parent / "preprocessing" / "combined_paired_data.csv")

def load_paired_data(csv_path):
    df = pd.read_csv(csv_path)
    bad_cols  = [f"{feat}_bad"  for feat in EXERCISE_FEATURES["lateral raise"]]
    good_cols = [f"{feat}_good" for feat in EXERCISE_FEATURES["lateral raise"]]
    X = df[bad_cols]
    y = df[good_cols]
    return X, y

X, y = load_paired_data(file_path)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Scale ─────────────────────────────────────────────────────────────────────
X_bad_scaler  = StandardScaler()
y_good_scaler = StandardScaler()

X_train_scaled = X_bad_scaler.fit_transform(X_train)
X_test_scaled  = X_bad_scaler.transform(X_test)
y_train_scaled = y_good_scaler.fit_transform(y_train)
y_test_scaled  = y_good_scaler.transform(y_test)

input_dim  = X_train_scaled.shape[1]
output_dim = y_train_scaled.shape[1]

# ── Model ─────────────────────────────────────────────────────────────────────
model = Sequential([
    Input(shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.15),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.15),
    layers.Dense(32, activation='relu'),
    layers.Dense(output_dim, activation='linear')
])

model.compile(
    loss='mse',
    optimizer=optimizers.Adam(learning_rate=1e-3),
    metrics=['mae']
)

es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_test_scaled, y_test_scaled),
    epochs=300,
    callbacks=[es]
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred_scaled = model.predict(X_test_scaled)

abs_errors_scaled = np.abs(y_test_scaled - y_pred_scaled)
max_errors_scaled = np.max(abs_errors_scaled, axis=1)
print(f"\nMax Error — mean across samples: {np.mean(max_errors_scaled):.4f}")
print(f"Max Error — worst sample:        {np.max(max_errors_scaled):.4f}")
print(f"MAE (for comparison):             {np.mean(abs_errors_scaled):.4f}")

y_pred_real = y_good_scaler.inverse_transform(y_pred_scaled)
y_test_real = y_good_scaler.inverse_transform(y_test_scaled)

abs_errors_real = np.abs(y_test_real - y_pred_real)
max_errors_real = np.max(abs_errors_real, axis=1)

print(f"\nMax Error — mean across samples: {np.mean(max_errors_real):.2f} degrees")
print(f"Max Error — worst sample:        {np.max(max_errors_real):.2f} degrees")
print(f"MAE (for comparison):             {np.mean(abs_errors_real):.2f} degrees")

# ── Worst sample ──────────────────────────────────────────────────────────────
worst_idx     = np.argmax(max_errors_real)
worst_feature = np.argmax(abs_errors_real[worst_idx])
feature_names = EXERCISE_FEATURES["lateral raise"]

print(f"\nWorst sample index: {worst_idx}")
print(f"Worst feature:      {feature_names[worst_feature]}")
print(f"Predicted good:     {y_pred_real[worst_idx][worst_feature]:.2f}°")
print(f"Actual good:        {y_test_real[worst_idx][worst_feature]:.2f}°")
print(f"Error:              {abs_errors_real[worst_idx][worst_feature]:.2f}°")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("saved_models", exist_ok=True)

def save_model_artifacts(model, X_scaler, y_scaler, exercise_name, output_dir):
    safe_name = exercise_name.replace(" ", "_")

    model.save(os.path.join(output_dir, f"{safe_name}_supervised.keras"))

    with open(os.path.join(output_dir, f"{safe_name}_X_scaler.pkl"), "wb") as f:
        pickle.dump(X_scaler, f)

    with open(os.path.join(output_dir, f"{safe_name}_y_scaler.pkl"), "wb") as f:
        pickle.dump(y_scaler, f)

    meta = {
        "exercise":        exercise_name,
        "prediction_type": "absolute",
        "input_features":  [f"{feat}_bad"  for feat in EXERCISE_FEATURES[exercise_name]],
        "output_features": [f"{feat}_good" for feat in EXERCISE_FEATURES[exercise_name]],
        "input_dim":       len(EXERCISE_FEATURES[exercise_name]),
        "output_dim":      len(EXERCISE_FEATURES[exercise_name]),
    }
    with open(os.path.join(output_dir, f"{safe_name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to: {output_dir}/")
    print(f"  {safe_name}_supervised.keras")
    print(f"  {safe_name}_X_scaler.pkl")
    print(f"  {safe_name}_y_scaler.pkl")
    print(f"  {safe_name}_meta.json")

    print(f"Real:      {len(real)}")
    print(f"Synthetic: {len(synthetic)}")
    print(f"Combined:  {len(combined)}")

    good_cols = ["left_arm_raise_good", "right_arm_raise_good"]
    print(combined[good_cols].describe())

save_model_artifacts(model, X_bad_scaler, y_good_scaler, "lateral raise", "saved_models")

#======================================================================
# Metric            First model (451 samples)     Best model (current)
#----------------------------------------------------------------------
# MAE               5.90°                         3.44°
# Mean Max Error    17.27°                        9.59°
# Worst Max Error   45.16°                        40.67°
#======================================================================
