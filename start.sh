#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$TMPDIR"
export PATH="$PWD/.venv/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$PWD/.cache/torch_extensions"
export TORCH_HOME="$PWD/.cache/torch"
export MAX_JOBS="${MAX_JOBS:-2}"
if [ ! -f third_party/gaussian-splatting/utils/loss_utils.py ]; then
  if [ -e .git ]; then
    git submodule update --init third_party/gaussian-splatting
  else
    git clone https://github.com/graphdeco-inria/gaussian-splatting.git third_party/gaussian-splatting
    git -C third_party/gaussian-splatting checkout "$(cat third_party/graphdeco-commit.txt)"
  fi
fi
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
if ! .venv/bin/python -c 'import torch' 2>/dev/null; then
  .venv/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
fi
if ! .venv/bin/python -c 'import torchvision' 2>/dev/null; then
  .venv/bin/python -m pip install torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126 --no-deps
fi
if ! .venv/bin/python -c 'import fastapi,uvicorn,multipart,numpy,PIL,httpx,pycolmap' 2>/dev/null; then
  .venv/bin/python -m pip install -r requirements.txt
fi
if ! .venv/bin/python -c 'import scipy,ninja,gsplat' 2>/dev/null; then
  .venv/bin/python -m pip install scipy ninja gsplat==1.5.3
fi
exec .venv/bin/python -m uvicorn studio.server:app --host 127.0.0.1 --port "${PORT:-7860}"
