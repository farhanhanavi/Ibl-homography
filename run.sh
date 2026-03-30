#!/usr/bin/env bash
# Court Vision — start the development server
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install dependencies if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "Installing Python dependencies..."
  pip install -r "$SCRIPT_DIR/backend/requirements.txt"
fi

echo "Starting Court Vision at http://localhost:8000"
cd "$SCRIPT_DIR/backend"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
