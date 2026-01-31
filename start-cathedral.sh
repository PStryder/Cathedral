#!/bin/bash
# Cathedral Startup Script for Unix/Linux/macOS
# Requires Docker to be installed and running

set -e

echo ""
echo "========================================"
echo " Cathedral - Memory-Augmented Chat"
echo "========================================"
echo ""

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "[ERROR] Docker is not running."
    echo "Please start Docker and try again."
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "[INFO] No .env file found. Creating from template..."
    cp .env.docker .env
    echo ""
    echo "[IMPORTANT] Please edit .env and add your API keys:"
    echo "  - OPENROUTER_API_KEY"
    echo "  - OPENAI_API_KEY"
    echo ""
    echo "Then run this script again."

    # Try to open in default editor
    if command -v nano >/dev/null 2>&1; then
        nano .env
    elif command -v vim >/dev/null 2>&1; then
        vim .env
    else
        echo "Edit .env with your preferred text editor."
    fi
    exit 0
fi

# Check if API keys are set
if grep -q "your-key-here" .env 2>/dev/null; then
    echo "[WARNING] It looks like you haven't set your API keys in .env"
    echo "Please edit .env and add your actual API keys."
    echo ""
    read -p "Continue anyway? (y/N): " CONTINUE
    if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
        exit 0
    fi
fi

echo "[INFO] Starting Cathedral..."
echo ""

# Pull latest images and start
docker-compose pull
docker-compose up -d

echo ""
echo "[SUCCESS] Cathedral is starting up!"
echo ""
echo "  Web Interface: http://localhost:8000"
echo "  Health Check:  http://localhost:8000/api/health"
echo ""
echo "  View logs:     docker-compose logs -f"
echo "  Stop:          docker-compose down"
echo ""

# Wait for health check
echo "[INFO] Waiting for services to be ready..."
sleep 5

# Try to open browser (works on macOS and some Linux)
if command -v open >/dev/null 2>&1; then
    open http://localhost:8000
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open http://localhost:8000
fi

echo ""
echo "Press Ctrl+C to stop viewing logs..."
docker-compose logs -f
