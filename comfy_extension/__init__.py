"""Opt-in ComfyUI extension; install this folder through a symlink."""
import os

NODE_CLASS_MAPPINGS = {}
if os.environ.get('QWEN_TP') == '1':
    from .tensor_parallel import install
    install()
