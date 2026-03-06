Denoiser training stub:

- Expected input: clean good-form reps (body-frame, scale-normalized), shape `(T, 33, 3)` with visibility mask.
- Corruptions to simulate NG: shrug (shoulder y+), swing (torso tilt + hip drift), elbow over-bend, asymmetry, timing jitter.
- Target: recover clean coordinates while preserving individual proportions.

Implement `train.py` here with your preferred stack (PyTorch/TF), then export to ONNX/TFJS for `/packages/models`.
