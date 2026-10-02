#!/usr/bin/env bash
# Setup on a RunPod pod (Ubuntu + CUDA image, volume mounted at /workspace). Rerun after every pod
# start: the container disk (Ollama binary, apt packages) is wiped when a pod stops; /workspace is kept.
# Usage:  OLLAMA_VERSION=0.34.3 bash runpod_setup.sh      (use the same version on every pod)
set -euo pipefail
cd /workspace

# 1. Ollama, stored on the volume so models survive a pod restart
apt-get update -qq && apt-get install -y -qq tmux zstd curl git rsync >/dev/null
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION="${OLLAMA_VERSION:-}" sh
# Fixed context on every pod so results don't depend on GPU memory (Ollama's default scales with VRAM).
# One loaded model at a time, so the previous model is unloaded before the next one loads.
export OLLAMA_MODELS=/workspace/ollama-models OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_CONTEXT_LENGTH=16384
mkdir -p "$OLLAMA_MODELS"
pkill ollama 2>/dev/null || true
nohup ollama serve > /workspace/ollama.log 2>&1 &
sleep 5 && ollama -v

# 2. Code (master branch) and Python deps (norag/lc mode does not need torch/colpali)
# Clone, or update an existing clone (the volume survives a stop, so the repo may be stale)
[ -d watcher-mcp-server ] || git clone -b master https://github.com/noura3368/watcher-mcp-server.git
git -C watcher-mcp-server pull -q --ff-only
# Ubuntu 24.04 images block system-wide pip (PEP 668), so use a venv on the volume
[ -d venv ] || python3 -m venv venv
venv/bin/pip install -q requests pydantic ollama Jinja2 pymupdf4llm
echo "Setup done. Start the run with:"
echo "  tmux new -s run"
echo "  cd /workspace/watcher-mcp-server/llm_pipeline/services && /workspace/venv/bin/python run.py --config config_runpod.txt --mode norag"
