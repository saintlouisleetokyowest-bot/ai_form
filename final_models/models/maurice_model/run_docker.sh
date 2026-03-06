#!/usr/bin/env bash
# Builds and starts the inference server in Docker, then guides you to run
# the lightweight client natively (works on macOS, Linux, and Windows).

set -e

cd "$(dirname "$0")"

echo "==> Building and starting inference server..."
docker compose up --build -d

echo ""
echo "==> Waiting for server at http://localhost:8000 ..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "    Server is up!"
        break
    fi
    sleep 2
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Server running at  http://localhost:8000"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo " Install client dependencies (once):"
echo "   pip install -r requirements_client.txt"
echo ""
echo " Run the client:"
echo "   python client.py"
echo ""
echo " To stop the server:"
echo "   docker compose down"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
