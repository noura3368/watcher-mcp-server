#!/usr/bin/env bash
# One-time setup on a RunPod pod (Ubuntu + CUDA image, volume mounted at /workspace).
# Usage:  OLLAMA_VERSION=0.x.y bash runpod_setup.sh      (match the Ollama version on compute6)
set -euo pipefail
cd /workspace

# 1. Ollama, stored on the volume so models survive a pod restart
apt-get update -qq && apt-get install -y -qq tmux zstd curl git >/dev/null
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION="${OLLAMA_VERSION:-}" sh
export OLLAMA_MODELS=/workspace/ollama-models OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=2
mkdir -p "$OLLAMA_MODELS"
pkill ollama 2>/dev/null || true
nohup ollama serve > /workspace/ollama.log 2>&1 &
sleep 5 && ollama -v

# 2. Code (master branch) and Python deps (norag/lc mode does not need torch/colpali)
[ -d watcher-mcp-server ] || git clone -b master https://github.com/noura3368/watcher-mcp-server.git
pip install -q requests pydantic ollama Jinja2 pymupdf4llm
echo "Setup done. Start the run with:  tmux new -s run   (see README steps)"
