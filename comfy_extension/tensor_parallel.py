"""Four-device head/MLP tensor parallelism for stock Qwen Image 2.1.

Weights are dequantized once into FP16 shards. The original checkpoint remains
untouched. This implementation is inference-only, for plain text-to-image.
"""
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import time

import torch
import torch.nn.functional as F
try:
    # PyTorch 2.10 keeps this module but no longer exposes it as
    # ``torch.cuda.comm`` until it is imported explicitly.
    from torch.cuda import comm as _cuda_comm
except ImportError:  # pragma: no cover - older/minimal PyTorch builds
    _cuda_comm = None
import comfy.model_management as mm
import comfy.ops as ops
from comfy.ldm.qwen_image21 import model as qm


def dense_weight(module, device):
    if getattr(module, 'weight_function', ()) or getattr(module, 'weight_lowvram_function', None):
        raise ValueError('TP does not support patched/LoRA weights')
    if getattr(module, 'pre_quant_scale', None) is not None:
        raise ValueError('TP does not support pre_quant_scale')
    # Dynamic-VRAM checkpoints may keep a meta tensor or a memory-mapped
    # quantized tensor on the host. Ask ComfyUI's loader for one device copy,
    # then detach a dense FP16 clone so later block eviction cannot invalidate
    # the tensor-parallel shard.
    if hasattr(module, '_v'):
        weight, bias, state = ops.cast_bias_weight(
            module, input=None, dtype=torch.float16, device=device,
            bias_dtype=torch.float16, offloadable=True,
        )
        try:
            value = weight.detach().clone()
        finally:
            ops.uncast_bias_weight(module, weight, bias, state)
    else:
        value = module.weight.to(device=device, dtype=torch.float16)
        if hasattr(value, 'dequantize'):
            value = value.dequantize()
    return value.to(dtype=torch.float16).detach().clone()


class TensorParallel:
    def __init__(self, model, devices):
        self.devices = devices
        self.pool = ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix='qwen-tp')
        self.shards = [[] for _ in devices]
        self.profile = os.environ.get('QWEN_TP_PROFILE', '0') == '1'
        # The stock implementation launches three independent GEMMs for Q/K/V
        # on every rank.  Their input and output shapes are identical, so a
        # single QKV GEMM is both cheaper to launch and easier for cuBLAS to
        # keep busy.  Keep an escape hatch for old checkpoints while this is
        # being rolled out in production.
        self.fused_qkv = os.environ.get('QWEN_TP_FUSED_QKV', '1') != '0'
        # ``reduce_add`` uses NCCL when it is available (NVLink on SXM2), and
        # falls back to the same P2P copies as the old loop otherwise.  The
        # environment switch makes A/B tests and rollback straightforward.
        self.reduce_mode = os.environ.get('QWEN_TP_REDUCE', 'comm').lower()
        self.profile_ms = [0.0 for _ in devices]
        heads = model.transformer_blocks[0].attn.heads
        if heads % len(devices):
            raise ValueError('attention head count must be divisible by device count')
        self.heads = heads // len(devices)
        started = time.monotonic()
        for i, block in enumerate(model.transformer_blocks):
            if not block.img_mlp.fused:
                raise ValueError('TP requires fused gate_up MLP')
            layer = [{} for _ in devices]
            modules = {'o': block.attn.to_out[0], 'gu': block.img_mlp.gate_up, 'down': block.img_mlp.out}
            for name, module in modules.items():
                if module.bias is not None:
                    raise ValueError('TP currently expects bias-free Qwen weights')
                full = dense_weight(module, devices[0])
                if name == 'gu':
                    gate, up = full.chunk(2, dim=0)
                    pieces = [torch.cat((g, u), dim=0) for g, u in zip(gate.chunk(len(devices)), up.chunk(len(devices)))]
                else:
                    pieces = full.chunk(len(devices), dim=1 if name in ('o', 'down') else 0)
                for d, piece, target in zip(layer, pieces, devices):
                    d[name] = piece.to(target).contiguous().clone() if target == devices[0] else piece.to(target).contiguous()
                del pieces, full
            # Q/K/V are laid out identically in Qwen Image.  Concatenate their
            # row shards so attention can issue one larger GEMM per rank.  If
            # a custom checkpoint has incompatible projections, fail early
            # with a useful error instead of silently changing its layout.
            if self.fused_qkv:
                for name in ('q', 'k', 'v'):
                    if getattr(block.attn, 'to_' + name).bias is not None:
                        raise ValueError('TP currently expects bias-free Qwen weights')
                qkv_full = [dense_weight(getattr(block.attn, 'to_' + name), devices[0])
                            for name in ('q', 'k', 'v')]
                if len({tuple(value.shape) for value in qkv_full}) != 1:
                    raise ValueError('QWEN_TP_FUSED_QKV requires matching Q/K/V matrix shapes')
                if any(value.shape[0] % len(devices) for value in qkv_full):
                    raise ValueError('QKV output dimension must be divisible by device count')
                # Shard each projection first, then concatenate the matching
                # local Q/K/V rows.  Chunking the full [Q;K;V] matrix directly
                # would give some ranks Q-only or V-only rows.
                qkv_pieces = [torch.cat(parts, dim=0)
                              for parts in zip(*(value.chunk(len(devices), dim=0)
                                                 for value in qkv_full))]
                for d, piece, target in zip(layer, qkv_pieces, devices):
                    # Detach rank 0 from the full concatenated allocation;
                    # otherwise the view would retain all ranks' rows for
                    # every block and inflate rank 0's resident memory.
                    d['qkv'] = (piece.to(target).contiguous().clone()
                                if target == devices[0]
                                else piece.to(target).contiguous())
                del qkv_pieces, qkv_full
            else:
                for name in ('q', 'k', 'v'):
                    module = getattr(block.attn, 'to_' + name)
                    if module.bias is not None:
                        raise ValueError('TP currently expects bias-free Qwen weights')
                    full = dense_weight(module, devices[0])
                    pieces = full.chunk(len(devices), dim=0)
                    for d, piece, target in zip(layer, pieces, devices):
                        d[name] = piece.to(target).contiguous()
                    del pieces, full
            for d, target in zip(layer, devices):
                d['qn'] = dense_weight(block.attn.norm_q, target).clone()
                d['kn'] = dense_weight(block.attn.norm_k, target).clone()
                d['eps'] = block.attn.norm_q.eps
            for dest, d in zip(self.shards, layer):
                dest.append(d)
            if i % 8 == 7:
                logging.info('QWEN_TP prepared %d/%d layers', i + 1, len(model.transformer_blocks))
        for device in devices:
            torch.cuda.synchronize(device)
        logging.info('QWEN_TP ready: %s; FP16 shards MiB=%s; prepare=%.2fs', devices,
                     [round(sum(w.numel()*w.element_size() for b in rank for w in b.values() if isinstance(w, torch.Tensor))/2**20) for rank in self.shards],
                     time.monotonic() - started)

    @staticmethod
    def attention(x, pe, segments, shard, heads, device):
        with torch.inference_mode(), torch.cuda.device(device):
            b, n, _ = x.shape
            weight_dtype = shard.get('qkv', shard.get('q')).dtype
            x = x.to(dtype=weight_dtype)
            if 'qkv' in shard:
                # Qwen's Q/K/V projections have equal output widths.  Keeping
                # the fused result contiguous makes the three views cheap and
                # avoids another allocation before RoPE.
                qkv = F.linear(x, shard['qkv'])
                q, k, v = qkv.chunk(3, dim=-1)
                q, k, v = [value.contiguous().view(b, n, heads, -1) for value in (q, k, v)]
            else:
                q, k, v = [F.linear(x, shard[name]).view(b, n, heads, -1) for name in ('q', 'k', 'v')]
            q, k = qm.comfy.quant_ops.ck.rms_rope(q, k, pe, shard['qn'], shard['kn'], shard['eps'])
            out = qm.block_causal_attention(segments)(q, k, v, heads)
            return F.linear(out, shard['o'])

    @staticmethod
    def mlp(x, shard, device):
        with torch.inference_mode(), torch.cuda.device(device):
            x = x.to(dtype=shard['gu'].dtype)
            gate, up = F.linear(x, shard['gu']).chunk(2, dim=-1)
            return F.linear(F.silu(gate) * up, shard['down'])

    def parallel(self, fn, x, index, extras=None):
        futures = []
        for rank, device in enumerate(self.devices):
            value = x.to(device, non_blocking=True)
            if extras is None:
                args = value, self.shards[rank][index], device
            else:
                args = value, extras[rank][0], extras[rank][1], self.shards[rank][index], self.heads, device
            futures.append(self.pool.submit(self._run_one, fn, args, rank, device))
        timed_parts = [f.result() for f in futures]
        parts = [item[0] for item in timed_parts]
        for rank, item in enumerate(timed_parts):
            self.profile_ms[rank] += item[1]
        if self.reduce_mode not in {'loop', 'none'} and len(parts) > 1 and _cuda_comm is not None:
            try:
                # torch.cuda.comm.reduce_add selects NCCL on installations
                # with NCCL support and performs the reduction in one peer
                # collective.  On systems without NCCL it uses asynchronous
                # P2P copies internally, still avoiding the Python loop.
                return _cuda_comm.reduce_add(parts, destination=self.devices[0].index)
            except (AttributeError, RuntimeError, AssertionError, ValueError) as exc:
                # Keep the service usable on Windows or builds without NCCL;
                # the warning is emitted once and the old path remains exact.
                if not getattr(self, '_reduce_warned', False):
                    logging.warning('QWEN_TP reduce_add unavailable; falling back to P2P loop: %s', exc)
                    self._reduce_warned = True
        result = parts[0]
        for part in parts[1:]:
            result = result + part.to(self.devices[0], non_blocking=True)
        return result

    def _run_one(self, fn, args, rank, device):
        if not self.profile:
            return fn(*args), 0.0
        with torch.cuda.device(device):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            value = fn(*args)
            end.record()
            end.synchronize()
            return value, start.elapsed_time(end)

    @torch.inference_mode()
    def forward(self, model, x, timesteps, context, transformer_options):
        b, _, h, w = x.shape
        hidden, pe, segments = model.build_sequence(x, context, [], [])
        prefix_len = hidden.shape[1] - h*w
        t = ((timesteps * 1000).to(x.dtype) / 1000).to(x.dtype)
        temb = model.time_text_embed(torch.cat([t, t.new_zeros(1)]), x.dtype)
        s1, g1, s2, g2 = model.modulation(temb).chunk(4, dim=-1)
        mod = (qm._split_rows(s1), qm._split_rows(g1.tanh()), qm._split_rows(s2), qm._split_rows(g2.tanh()), torch.zeros_like(s1[:1, None]))
        scale1, gate1, scale2, gate2, zero = mod
        extras = [(pe.to(d), [(a, b, mask.to(d) if mask is not None else None) for a, b, mask in segments]) for d in self.devices]
        self.profile_ms = [0.0 for _ in self.devices]
        for i, block in enumerate(model.transformer_blocks):
            mm.throw_exception_if_processing_interrupted()
            norm = qm._modulated_norm(block.img_norm1, hidden, scale1, prefix_len, zero)
            attn = self.parallel(self.attention, norm, i, extras)
            hidden = qm._gated_residual(hidden, attn, gate1, prefix_len)
            norm = qm._modulated_norm(block.img_norm2, hidden, scale2, prefix_len, zero)
            mlp = self.parallel(self.mlp, norm, i)
            hidden = qm._gated_residual(hidden, mlp, gate2, prefix_len).clip(-65504, 65504)
        hidden = model.proj_out(model.norm_out(hidden[:, prefix_len:], temb[:-1]))
        if self.profile:
            logging.info('QWEN_TP device_compute_ms=%s', [round(value, 1) for value in self.profile_ms])
        return hidden.transpose(1, 2).reshape(b, model.out_channels, h, w)


def install():
    original = qm.QwenImage21Transformer2DModel._forward
    if getattr(original, '_qwen_tp', False):
        return

    @torch.inference_mode()
    def forward(self, x, timesteps, context, ref_latents=None, image_slots=None, transformer_options=None, **kwargs):
        options = transformer_options or {}
        if ref_latents or options.get('patches') or options.get('patches_replace'):
            # Keep reference-image and extension workflows usable. They use
            # ComfyUI's original implementation, while plain text-to-image
            # requests use the four-device tensor-parallel path below.
            return original(self, x, timesteps, context, ref_latents, image_slots, options, **kwargs)
        if getattr(self, '_qwen_tp_runtime', None) is None:
            devices = [torch.device('cuda', int(i)) for i in os.environ.get('QWEN_TP_DEVICES', '0,1,2,3').split(',')]
            if len(set(devices)) != 4 or devices[0] != x.device:
                raise ValueError('QWEN_TP requires four distinct devices; first must be the Comfy compute device')
            self._qwen_tp_runtime = TensorParallel(self, devices)
        result = self._qwen_tp_runtime.forward(self, x, timesteps, context, options)
        if os.environ.get('QWEN_TP_VALIDATE') == '1' and not getattr(self, '_qwen_tp_validated', False):
            enabled = self.prefix_cache_enabled
            self.prefix_cache_enabled = False
            try:
                golden = original(self, x, timesteps, context, ref_latents, image_slots, options, **kwargs)
            finally:
                self.prefix_cache_enabled = enabled
            delta = result.float() - golden.float()
            relative = (delta.square().mean().sqrt() / golden.float().square().mean().sqrt().clamp_min(1e-8)).item()
            logging.info('QWEN_TP validation relative_RMSE=%.6f max_abs=%.6f', relative, delta.abs().max().item())
            if not torch.isfinite(result).all() or relative > 0.05:
                raise RuntimeError(f'TP output differs from dense reference: relative RMSE={relative:.6f}')
            self._qwen_tp_validated = True
        return result

    forward._qwen_tp = True
    qm.QwenImage21Transformer2DModel._forward = forward
    logging.warning('QWEN_TP enabled: concurrent attention-head and MLP tensor parallel, FP16, four GPUs')
