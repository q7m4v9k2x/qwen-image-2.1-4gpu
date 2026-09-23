"""ComfyUI loader for the opt-in Qwen tensor-parallel experiment.

The project directory is supplied by ``QWEN_TP_ROOT`` at deployment time.  The
loader deliberately does not contain a workstation-specific path so it can be
copied into any ComfyUI ``custom_nodes`` directory.
"""
import os
import sys
from pathlib import Path


if os.environ.get("QWEN_TP") == "1":
    root_value = os.environ.get("QWEN_TP_ROOT")
    if not root_value:
        raise RuntimeError("QWEN_TP_ROOT must point to the qwen2.1-4gpu project directory")
    root = Path(root_value).expanduser().resolve()
    if not (root / "comfy_extension" / "tensor_parallel.py").is_file():
        raise RuntimeError(f"QWEN_TP_ROOT does not contain comfy_extension: {root}")
    sys.path.insert(0, str(root))
    from comfy_extension.tensor_parallel import install
    install()

NODE_CLASS_MAPPINGS = {}
