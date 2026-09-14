#!/usr/bin/env bash
# Container entrypoint: make sure the directories and the checkpoint exist,
# then hand the process over to uvicorn (exec, so SIGTERM reaches the server).
set -euo pipefail

cd /app

# CUDA allocator: expandable segments reduce fragmentation for the steady
# per-request churn of a long-lived server (overridable from compose).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/data/models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

OUTPUT_DIR="${H3D_OUTPUT_DIR:-/data/outputs}"
UPLOAD_DIR="${H3D_UPLOAD_DIR:-/data/uploads}"
mkdir -p "$OUTPUT_DIR" "$UPLOAD_DIR" "$HF_HOME"

# Pre-download the weights when the cache volume is empty. A failure here is
# not fatal: the API still starts and reports the problem through /ready
# instead of crash-looping the container.
if [ "${H3D_DOWNLOAD_MODELS:-1}" = "1" ]; then
    if ! python scripts/download_models.py; then
        echo "[entrypoint] checkpoint pre-download failed - the API will retry on the first request" >&2
    fi
fi

# One worker on purpose: the ~7.5 GB paint model must live in exactly one
# process, and inference is serialised behind a lock anyway. Long jobs need a
# long keep-alive.
exec uvicorn app.main:app \
    --host "${H3D_HOST:-0.0.0.0}" \
    --port "${H3D_PORT:-8082}" \
    --workers 1 \
    --proxy-headers \
    --timeout-keep-alive 120 \
    --log-level "${H3D_LOG_LEVEL:-info}"
