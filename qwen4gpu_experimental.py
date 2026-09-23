"""Experimental Qwen Image 2.1 layer-parallel runner for 4+ GPUs.

This is intentionally separate from the production ComfyUI service. It splits
the 32 DiT blocks across visible CUDA devices and moves activations between
stages. It requires the official Diffusers checkpoint (not ComfyUI GGUF or
int8_convrot files). Use float16 on V100; bfloat16 is not native on Volta.
"""
import argparse
from pathlib import Path
import torch
from diffusers import QwenImage21Pipeline


def split_blocks(blocks, devices):
    count = len(blocks)
    per = (count + len(devices) - 1) // len(devices)
    return [devices[min(i // per, len(devices) - 1)] for i in range(count)]


class FourGPUQwen:
    def __init__(self, model, devices, dtype=torch.float16):
        self.devices = [torch.device(x) for x in devices]
        self.control = self.devices[0]
        self.pipe = QwenImage21Pipeline.from_pretrained(model, torch_dtype=dtype)
        self.pipe.text_encoder.to(self.control)
        self.pipe.vae.to(self.control)
        t = self.pipe.transformer
        for name, module in t.named_children():
            if name != "transformer_blocks":
                module.to(self.control)
        self.block_devices = split_blocks(t.transformer_blocks, self.devices)
        for block, device in zip(t.transformer_blocks, self.block_devices):
            block.to(device)
        self.pipe.transformer = t

    def generate(self, prompt, output, steps=40, width=1024, height=1024, seed=-1):
        if seed < 0:
            seed = torch.seed() % (2**63 - 1)
        generator = torch.Generator(device=self.control).manual_seed(seed)
        image = self.pipe(prompt=prompt, width=width, height=height,
                          num_inference_steps=steps, generator=generator).images[0]
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        return seed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Diffusers Qwen/Qwen-Image-2.1 directory")
    ap.add_argument("--output", default="qwen4gpu-test.png")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=-1)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    devices = [x.strip() for x in args.devices.split(",")]
    if len(devices) < 2:
        raise SystemExit("provide at least two CUDA devices")
    for d in devices:
        if torch.device(d).index >= torch.cuda.device_count():
            raise SystemExit(f"device {d} is not visible")
    seed = FourGPUQwen(args.model, devices).generate(
        args.prompt, args.output, args.steps, args.width, args.height, args.seed)
    print(f"saved={args.output} seed={seed} devices={','.join(devices)}")


if __name__ == "__main__":
    main()
