#!/usr/bin/env bash
# Build and run the form analyzer Docker container.
#
# Linux:  works out of the box (webcam + X11 display)
# macOS:  Docker Desktop cannot pass the webcam through to the container.
#         Run natively instead:
#           pip install -r requirements.txt
#           python 5_form_analyzer.py

set -e

IMAGE="form-analyzer"

echo "==> Building Docker image..."
docker build -t "$IMAGE" .

OS="$(uname -s)"

if [ "$OS" = "Linux" ]; then
    # Allow the container to connect to the local X server
    xhost +local:docker 2>/dev/null || true

    echo "==> Running on Linux (webcam=/dev/video0, display=$DISPLAY)..."
    docker run --rm \
        --device=/dev/video0 \
        --env DISPLAY="$DISPLAY" \
        --volume /tmp/.X11-unix:/tmp/.X11-unix \
        "$IMAGE"
else
    echo ""
    echo "NOTE: macOS / Docker Desktop does not support webcam passthrough."
    echo "Run the app natively instead:"
    echo ""
    echo "  pip install -r requirements.txt"
    echo "  python 5_form_analyzer.py"
    echo ""
fi
