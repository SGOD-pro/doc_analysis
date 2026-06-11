#!/usr/bin/env bash
# Start the Document Analytics Backend
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting LLM Document Analytics Engine Backend..."
echo "   Working directory: $SCRIPT_DIR"

# Check if uv is available
if ! command -v uv &>/dev/null; then
  echo "❌ uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

# Install/sync dependencies
echo "📦 Installing dependencies..."
uv sync

# Download spaCy model if not present
echo "🧠 Checking spaCy model..."
uv run python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null || \
  uv run python -m spacy download en_core_web_sm

echo "✅ Starting FastAPI server on http://localhost:8000"
echo "   API Docs: http://localhost:8001/docs"
echo ""

uv run uvicorn backend.api.main:app --host 0.0.0.0 --port 8001 --reload
