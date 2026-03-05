# AI Form Coach

Real-time form feedback for lateral raises: MediaPipe pose, body-frame alignment, and two overlays.

## How It Works

The app gets 33 body landmarks from the camera and puts them in a body-frame (hip-centered, scale-normalized) for comparison.

- **Left panel (Ideal Form)** — Reference skeleton driven by rep phase (arm angles). Shows target pose as teal dashed lines. After each rep, scoring compares your motion to the reference and highlights faults with hints.
- **Right panel (ML Correction)** — ONNX model takes raw landmarks and phase and outputs corrected pose. The model is trained on good reps; gold dashed lines show the predicted “correct” pose. Adding more training data (e.g. more reps or more users) improves correction quality.

## Quick Start

```bash
npm install
npm run dev:web
```
Open http://localhost:5173/ and allow camera. Face the camera at ~45° and do a lateral raise.

## Build

```bash
npm run build:web
```

## Structure

- `apps/web` — Vite + React
- `packages/core` — pose processing, scoring, overlays
- `packages/models` — ONNX corrector
- `tools/train_denoiser` — training notebooks & scripts
