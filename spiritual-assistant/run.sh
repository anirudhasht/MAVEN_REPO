#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/venv/bin/activate"
cd "$SCRIPT_DIR"
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload