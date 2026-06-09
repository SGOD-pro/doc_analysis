#!/usr/bin/env bash
# Start the React frontend dev server
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/frontend"

echo "🎨 Starting DocAnalytics Frontend..."
echo "   URL: http://localhost:5173"
echo ""

npm run dev
