FROM python:3.11-slim

# System deps for OpenCV GUI, MediaPipe, and webcam
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY 5_form_analyzer.py .
COPY saved_models/ ./saved_models/

CMD ["python", "5_form_analyzer.py"]
