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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export QWEN_TP=${QWEN_TP:-1}
export QWEN_TP_ROOT
export QWEN_TP_DEVICES=${QWEN_TP_DEVICES:-0,1,2,3}
export QWEN_TP_PROFILE=${QWEN_TP_PROFILE:-0}
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
