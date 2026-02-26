# Workout Form Analyzer

A real-time workout form analyzer that uses MediaPipe for pose detection and an autoencoder to learn correct exercise form. When your form deviates from ideal, a corrective ghost skeleton appears showing where your joints should be.

## How it works

1. **MediaPipe** detects 33 body landmarks from your webcam in real time
2. **Joint angles** are computed from the landmarks (knee angle, torso lean, arm raise etc.)
3. An **autoencoder** trained on good form data reconstructs what ideal angles should look like
4. The **reconstruction error** drives a 0-100 form score
5. The **decoder output** is converted back into joint positions and rendered as a corrective ghost skeleton

## Supported exercises

- Squats (`S` key)
- Lateral raises (`L` key)

## Project structure

```
claude_guide/
├── form_analyzer.py        # Main real-time app
├── extract_landmarks.py    # Extracts joint angles from video files
├── train_autoencoder.py    # Trains the autoencoder models
├── angles.csv              # Extracted training data
├── models/                 # Trained model artifacts
│   ├── squat_autoencoder.keras
│   ├── squat_encoder.keras
│   ├── squat_decoder.keras
│   ├── squat_scaler.pkl
│   ├── squat_meta.json
│   ├── lateral_raise_autoencoder.keras
│   ├── lateral_raise_encoder.keras
│   ├── lateral_raise_decoder.keras
│   ├── lateral_raise_scaler.pkl
│   └── lateral_raise_meta.json
└── README.md
```

## Setup

### 1. Create a Python 3.10 virtual environment

Using pyenv:
```bash
pyenv virtualenv 3.10.6 ai_form_v2
pyenv activate ai_form_v2
```

Or using venv:
```bash
python3.10 -m venv ai_form_v2
source ai_form_v2/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python form_analyzer.py
```

Make sure your webcam is connected and the `models/` folder is in the same directory.

## Controls

| Key | Action |
|-----|--------|
| `S` | Switch to squat mode |
| `L` | Switch to lateral raise mode |
| `Q` | Quit |

## Training your own models

### Step 1 — Collect video data

Record or download videos of correct exercise form. Organise them like this:

```
videos/
├── squat/
│   ├── video1.mp4
│   └── video2.mp4
└── lateral raise/
    ├── video1.mp4
    └── video2.mp4
```

### Step 2 — Extract landmarks

```bash
python extract_landmarks.py
```

This processes all videos and saves joint angles to `angles.csv`.

### Step 3 — Train the autoencoder

```bash
python train_autoencoder.py
```

Trained models are saved to the `models/` folder, ready for inference.

## Dependencies

See `requirements.txt`. Key packages:

- `mediapipe==0.10.9` — pose detection
- `tensorflow==2.13.0` — autoencoder training and inference
- `opencv-python==4.8.0.76` — webcam capture and rendering
- `scikit-learn` — feature scaling
- `numpy`, `pandas` — data processing

> **Note:** MediaPipe and TensorFlow have conflicting protobuf requirements in newer versions. The pinned versions in `requirements.txt` are known to work together on Python 3.10.

## Improving accuracy

The model accuracy depends heavily on training data. To improve results:

1. Record yourself doing good form reps with your own webcam
2. Download additional YouTube workout videos using `yt-dlp`
3. Run them through `extract_landmarks.py` and append to `angles.csv`
4. Retrain with `train_autoencoder.py`

The more data from your specific camera angle and setup, the more accurate the model will be.
