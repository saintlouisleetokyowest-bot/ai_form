Offline template builder (good reps → mean/std):

1) Export MediaPipe Pose sequences to `.npy` with shape `(frames, 33, 3)` in world coordinates plus visibility mask `(frames, 33)`.
2) Run `python build_template.py --input data/good_reps/*.npy --out ../assets/templates/lateral_raise`.
3) The script will:
   - body-frame align each rep,
   - scale-normalize,
   - resample to 100 frames,
   - compute mean/std for xyz and features.

Add your `build_template.py` here when ready; this folder is a stub to keep repository structure aligned with the blueprint.
