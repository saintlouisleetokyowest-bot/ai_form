import pandas as pd
import numpy as np
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
from keras import layers, Sequential, Input, layers, optimizers, callbacks
from tensorflow.keras.callbacks import EarlyStopping

import pickle
import json

PREPROCESS_DIR = Path(__file__).parent.parent.parent / "preprocessing"

real = pd.read_csv(str(PREPROCESS_DIR / "paired_data.csv"))
synthetic = pd.read_csv(str(PREPROCESS_DIR / "synthetic_paired_data.csv"))

# Keep only the feature columns both share
feature_cols = [c for c in real.columns if c.endswith("_bad") or c.endswith("_good")]

combined = pd.concat([real[feature_cols], synthetic[feature_cols]], ignore_index=True)
combined.to_csv(str(PREPROCESS_DIR / "combined_paired_data.csv"), index=False)

print(f"Real:      {len(real)}")
print(f"Synthetic: {len(synthetic)}")
print(f"Combined:  {len(combined)}")
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
file_path = str(Path(__file__).parent.parent.parent / "preprocessing" / "combined_paired_data.csv")
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_bad_scaler = StandardScaler()
y_good_scaler = StandardScaler()
X_train_scaled = X_bad_scaler.fit_transform(X_train)
X_test_scaled = X_bad_scaler.transform(X_test)
y_train_scaled = y_good_scaler.fit_transform(y_train)
y_test_scaled = y_good_scaler.transform(y_test)
input_dim = X_train_scaled.shape[1]
output_dim = y_train_scaled.shape[1]


model = Sequential()
model.add(Input(shape=(input_dim,)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(output_dim, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=["mae"])
es = EarlyStopping(monitor='val_loss', patience=20)

model.fit(X_train_scaled, y_train_scaled, validation_data= (X_test_scaled, y_test_scaled), epochs=300, callbacks=[es])
# Get predictions
y_pred_scaled = model.predict(X_test_scaled)

# Per-feature absolute error for each sample
abs_errors = np.abs(y_test_scaled - y_pred_scaled)

# Max error across features for each sample
max_errors = np.max(abs_errors, axis=1)

# Summary
print(f"Max Error — mean across samples: {np.mean(max_errors):.4f}")
print(f"Max Error — worst sample:        {np.max(max_errors):.4f}")
print(f"MAE (for comparison):             {np.mean(abs_errors):.4f}")
y_pred_real = y_good_scaler.inverse_transform(y_pred_scaled)
y_test_real = y_good_scaler.inverse_transform(y_test_scaled)

abs_errors_real = np.abs(y_test_real - y_pred_real)
max_errors_real = np.max(abs_errors_real, axis=1)

print(f"Max Error — mean across samples: {np.mean(max_errors_real):.2f} degrees")
print(f"Max Error — worst sample:        {np.max(max_errors_real):.2f} degrees")
print(f"MAE (for comparison):             {np.mean(abs_errors_real):.2f} degrees")
np.argmax(abs_errors_real, axis=1)
os.makedirs("saved_models", exist_ok=True)
def save_model_artifacts(model, X_scaler, y_scaler,
                          exercise_name, output_dir):
    safe_name = exercise_name.replace(" ", "_")

    # Save model
    model.save(os.path.join(output_dir, f"{safe_name}_supervised.keras"))

    # Save both scalers
    with open(os.path.join(output_dir, f"{safe_name}_X_scaler.pkl"), "wb") as f:
        pickle.dump(X_scaler, f)

    with open(os.path.join(output_dir, f"{safe_name}_y_scaler.pkl"), "wb") as f:
        pickle.dump(y_scaler, f)

    # Save metadata
    meta = {
        "exercise":      exercise_name,
        "input_features":  [f"{feat}_bad" for feat in EXERCISE_FEATURES[exercise_name]],
        "output_features": [f"{feat}_good" for feat in EXERCISE_FEATURES[exercise_name]],
        "input_dim":     len(EXERCISE_FEATURES[exercise_name]),
        "output_dim":    len(EXERCISE_FEATURES[exercise_name]),
    }
    with open(os.path.join(output_dir, f"{safe_name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to: {output_dir}/")
    print(f"  {safe_name}_supervised.keras")
    print(f"  {safe_name}_X_scaler.pkl")
    print(f"  {safe_name}_y_scaler.pkl")
    print(f"  {safe_name}_meta.json")
save_model_artifacts(model, X_bad_scaler, y_good_scaler, "lateral raise", "saved_models")
worst_idx = np.argmax(max_errors_real)
worst_feature = np.argmax(abs_errors_real[worst_idx])
feature_names = [f.replace("_good", "") for f in EXERCISE_FEATURES["lateral raise"]]

print(f"Worst sample index: {worst_idx}")
print(f"Worst feature: {feature_names[worst_feature]}")
print(f"Predicted: {y_pred_real[worst_idx][worst_feature]:.2f}")
print(f"Actual:    {y_test_real[worst_idx][worst_feature]:.2f}")
print(f"Error:     {abs_errors_real[worst_idx][worst_feature]:.2f} degrees")
