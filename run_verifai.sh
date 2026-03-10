#!/usr/bin/env bash
set -euo pipefail

# run_verifai.sh -- One-command quickstart for VerifAI
#
# Prerequisites:
#   - Python 3.10+
#   - OPENAI_API_KEY set in environment
#
# Usage:
#   chmod +x run_verifai.sh
#   ./run_verifai.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PRINCIPLES="${PRINCIPLES_FILE:-principles.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-./my-verifier}"
PORT="${PORT:-8000}"

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Error: OPENAI_API_KEY is not set."
    echo "  export OPENAI_API_KEY=sk-..."
    exit 1
fi

if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found. Please install Python 3.10+."
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: Install dependencies
# ---------------------------------------------------------------------------
echo "==> Installing dependencies..."
pip install -q -r requirements.txt

# ---------------------------------------------------------------------------
# Step 2: Train the verifier
# ---------------------------------------------------------------------------
echo "==> Training verifier from $PRINCIPLES..."
python3 train_verifier.py \
    --principles "$PRINCIPLES" \
    --output-dir "$OUTPUT_DIR"

echo "==> Training complete. Model saved to $OUTPUT_DIR"
echo "    Metrics: $OUTPUT_DIR/eval_metrics.json"
cat "$OUTPUT_DIR/eval_metrics.json"

# ---------------------------------------------------------------------------
# Step 3: Serve the verifier
# ---------------------------------------------------------------------------
echo ""
echo "==> Starting verifier API on port $PORT..."
echo "    POST http://localhost:$PORT/verify"
echo "    GET  http://localhost:$PORT/metrics"
echo "    GET  http://localhost:$PORT/health"
echo ""
echo "    Press Ctrl+C to stop."

export VERIFIER_MODEL_PATH="$OUTPUT_DIR"
uvicorn serve_verifier:app --host 0.0.0.0 --port "$PORT"
