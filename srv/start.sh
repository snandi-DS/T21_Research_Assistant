#!/usr/bin/env bash
# start.sh — Auto-detecting launcher for T21 Research Assistant
#
# Detects NVIDIA GPU at runtime and selects the appropriate compose stack:
#   GPU available  →  docker-compose.yml + docker-compose.gpu.yml  (CUDA embed + rerank)
#   No GPU         →  docker-compose.yml only                       (CPU embed + rerank)
#
# Usage:
#   ./start.sh              # start (detached)
#   ./start.sh down         # stop all containers
#   ./start.sh restart api  # restart a specific service
#   ./start.sh logs -f api  # tail logs for a service
#   ./start.sh build        # rebuild images

set -euo pipefail

COMPOSE_BASE="-f docker-compose.yml"
COMPOSE_GPU="-f docker-compose.gpu.yml"

# ── GPU detection ─────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "[start.sh] GPU detected: ${GPU_NAME} — enabling CUDA for embed + rerank models"
    COMPOSE_FILES="${COMPOSE_BASE} ${COMPOSE_GPU}"
else
    echo "[start.sh] No GPU detected — using CPU for embed + rerank models"
    COMPOSE_FILES="${COMPOSE_BASE}"
fi

# ── Run compose with detected profile ────────────────────────────────────────
CMD="${1:-up}"

if [[ "${CMD}" == "up" ]]; then
    # shellcheck disable=SC2086
    sudo docker compose ${COMPOSE_FILES} up -d "${@:2}"
elif [[ "${CMD}" == "build" ]]; then
    # shellcheck disable=SC2086
    sudo docker compose ${COMPOSE_FILES} up -d --build "${@:2}"
else
    # Pass all arguments through (down, restart, logs, ps, etc.)
    # shellcheck disable=SC2086
    sudo docker compose ${COMPOSE_FILES} "$@"
fi
