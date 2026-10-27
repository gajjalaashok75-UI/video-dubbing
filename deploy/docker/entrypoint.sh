#!/usr/bin/env bash
#
# Multiva container entrypoint.
#
# Two jobs: make sure the runtime tree and the model weights exist on the
# mounted volume, then start the API (which serves the built studio interface
# from /app/ too). Everything is idempotent, so `docker compose up` twice is
# fast the second time.
#
#   serve        fetch missing models, then run the server (default)
#   models       fetch missing models and exit
#   shell        drop into a shell in /app
#   <anything>   run it as-is
#
set -euo pipefail

cd /app

RUNTIME_ROOT="${RUNTIME_ROOT:-/data}"
UPLOAD_DIR="${MULTIVA_UPLOAD_DIR:-$RUNTIME_ROOT/uploads}"
OUTPUT_DIR="${MULTIVA_OUTPUT_DIR:-$RUNTIME_ROOT/output}"
HF_HOME="${HF_HOME:-$RUNTIME_ROOT/hf}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
SKIP_MODELS="${SKIP_MODEL_DOWNLOAD:-0}"

export MULTIVA_UPLOAD_DIR="$UPLOAD_DIR"
export MULTIVA_OUTPUT_DIR="$OUTPUT_DIR"
export HF_HOME

mkdir -p "$UPLOAD_DIR" "$OUTPUT_DIR" "$HF_HOME"

banner() { printf '\n  Multiva — %s\n  %s\n\n' "$1" "------------------------------------------------"; }

banner "container starting"

# --- what the GPU looks like -------------------------------------------------
# Not fatal if there is none: the engine falls back to CPU on its own. Printed
# here because "why is this slow" is answered by this line, not by the code.
python3 - <<'PY' || true
import os
try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU:    {p.name} ({p.total_memory / 1024**3:.1f} GB VRAM)")
        print(f"  CUDA:   {torch.version.cuda}  |  torch {torch.__version__}")
    else:
        print("  GPU:    none visible — renders will run on CPU and be slow.")
        print("          Pass --gpus all (docker) or a GPU reservation (compose).")
except Exception as exc:  # noqa: BLE001
    print(f"  GPU:    torch not importable ({exc})")
PY

# --- models ------------------------------------------------------------------
# Wav2Lip's two checkpoints are excluded from the build context (they are
# hundreds of megabytes) and fetched by the downloader below, alongside the
# Hub models — faster-whisper, NLLB, and the gated IndicF5 voice model. All of
# them land on $HF_HOME or under engine/vendor. Fetching them here means the
# first render does not stall for ten minutes wondering what is happening.
if [ "$SKIP_MODELS" != "1" ]; then
    banner "checking models"
    python3 scripts/download_models.py --check || true
    if ! python3 scripts/download_models.py --check 2>/dev/null | grep -q "0 item"; then
        echo "  Fetching the missing models (resumable; ~7.5 GB first time)..."
        python3 scripts/download_models.py || {
            echo "  Download did not finish. Continuing; the first render will retry."
        }
    else
        echo "  All models present."
    fi
fi

case "${1:-serve}" in
    serve)
        banner "starting the API on ${HOST}:${PORT}"
        echo "  Interface: http://localhost:${PORT}/app/"
        echo "  Health:    http://localhost:${PORT}/api/health"
        echo "  Renders:   ${OUTPUT_DIR}"
        echo
        exec python3 -m uvicorn app:app \
            --app-dir /app/engine \
            --host "$HOST" --port "$PORT" --workers 1
        ;;
    models)
        banner "models"
        # "${@:2}" so the subcommand name is not forwarded as a flag to the script.
        exec python3 scripts/download_models.py "${@:2}"
        ;;
    shell)
        exec /bin/bash
        ;;
    *)
        exec "$@"
        ;;
esac
