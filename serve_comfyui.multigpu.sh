#!/usr/bin/env bash
# Experimental Qwen Image 2.1 tensor-parallel run across four GPUs.
# Set COMFYUI_ROOT and QWEN_TP_ROOT when these directories are not in the
# conventional locations. No host-specific path is embedded in this script.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMFYUI_ROOT="${COMFYUI_ROOT:-${HOME}/ComfyUI}"
QWEN_TP_ROOT="${QWEN_TP_ROOT:-${SCRIPT_DIR}}"
cd "${COMFYUI_ROOT}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# If the host exposes a small display adapter alongside the V100s, selecting
# the first four enumerated devices would silently mix incompatible GPUs.  On
# an otherwise standard four-GPU host this still resolves to 0,1,2,3; when
# nvidia-smi is available, prefer the first four cards with at least 12 GiB.
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  detected=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    detected="$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits 2>/dev/null \
      | awk -F',' 'BEGIN { count=0 } { gsub(/[[:space:]]/, "", $2); if (($2 + 0) >= 12288 && count < 4) { printf "%s%s", (count ? "," : ""), $1; count++ } } END { if (count != 4) exit 1 }' \
      || true)"
  fi
  export CUDA_VISIBLE_DEVICES="${detected:-0,1,2,3}"
else
  export CUDA_VISIBLE_DEVICES
fi
export QWEN_TP=${QWEN_TP:-1}
export QWEN_TP_ROOT
export QWEN_TP_DEVICES=${QWEN_TP_DEVICES:-0,1,2,3}
export QWEN_TP_PROFILE=${QWEN_TP_PROFILE:-0}
# Fuse Q/K/V projections into one GEMM and use NCCL/NVLink reduction by
# default. Set either variable to 0/loop for an A/B comparison or rollback.
export QWEN_TP_FUSED_QKV=${QWEN_TP_FUSED_QKV:-1}
export QWEN_TP_REDUCE=${QWEN_TP_REDUCE:-comm}
# Set to 1 for a one-time 1-step numerical comparison against stock ComfyUI.
export QWEN_TP_VALIDATE=${QWEN_TP_VALIDATE:-0}

COMFYUI_PYTHON="${COMFYUI_PYTHON:-./venv/bin/python}"
COMFYUI_LISTEN="${COMFYUI_LISTEN:-127.0.0.1}"
COMFYUI_PORT="${COMFYUI_PORT:-8188}"

exec "${COMFYUI_PYTHON}" main.py \
  --listen "${COMFYUI_LISTEN}" \
  --port "${COMFYUI_PORT}" \
  --fp16-unet \
  --disable-auto-launch
