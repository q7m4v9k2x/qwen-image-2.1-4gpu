# 四卡单图实验记录

GitHub 上目前有三条相关路线：官方 Qwen Image 2.1 的 vLLM-Omni、SGLang-Diffusion，以及社区把 32 个 DiT block 手工分配到多张 GPU 的 Diffusers 原型。第三条最接近 V100 的硬件形态，因此仓库保留了 `qwen4gpu_experimental.py` 作为独立实验入口。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python qwen4gpu_experimental.py \
  --model /models/Qwen-Image-2.1 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --prompt "a ceramic teapot on a wooden table" \
  --output /tmp/qwen4gpu.png
```

这个原型需要官方 Diffusers 目录（`model_index.json`、`transformer/`、`text_encoder/`、`vae/`），不能读取当前 ComfyUI 的 `int8_convrot.safetensors` 作为 Diffusers 模型。它使用 `float16`，因为 V100 没有原生 BF16。当前版本先用于验证模型加载和显存分层，不替换生产 ComfyUI。

参考：

- https://github.com/QwenLM/Qwen-Image-2.1
- https://github.com/vllm-project/vllm-omni
- https://github.com/sgl-project/sglang
- https://gist.github.com/toddtail/11f07c55b2d0146deebbbc412ada34cd

## 2026-09-23 实测结论

已在本地工作站的四张 Tesla V100-SXM2-16GB 上部署 ComfyUI custom node 张量并行
原型。每个 Qwen Image 2.1 DiT block 的 Q/K/V 和 gate/up 输出按 attention heads
切成四份，输出投影和 MLP down 的部分结果跨卡求和。32 层 FP16 分片约占每卡
3.3 GiB；文本编码器和 VAE 仍由 `cuda:0` 承担。

- 512×512、4 steps：四卡约 4 秒；原生单卡约 12 秒。
- 512×512、25 steps：四卡约 24 秒，输出正常。
- 1024×1024、4 steps：四卡约 14 秒，未溢出。
- 同输入 1 step 的 TP 与原生路径相对 RMSE 约 0.0078，最大绝对误差约 0.13。
- `QWEN_TP_PROFILE=1` 时四卡单步 CUDA event 计算时间约 350–370 ms/卡；生产配置
  关闭 profile。

部署文件位于项目目录的 `comfy_extension/tensor_parallel.py`、ComfyUI
`custom_nodes/qwen_tp_custom_node.py` 和 `serve_comfyui.multigpu.sh`。启动脚本通过
`QWEN_TP_ROOT`、`COMFYUI_ROOT` 与 `CUDA_VISIBLE_DEVICES` 配置路径和设备；TP 的
`cuda:0..3` 对应所选 GPU，`QWEN_TP=0` 可回到原生单卡路径。

当前实验只处理纯文生图；参考图、ComfyUI block hook、LoRA/权重 patch 和 prefix
cache 需要回到原生路径。网关接口不变，8190 仍调用 8188。

### latent batch 吞吐试验

在同一工作站直接向 8188 提交相同提示词的 latent batch（512×512、4 steps，
含一次请求启动开销）得到：`n=1` 约 2.7 秒/张，`n=2` 约 2.9 秒/2 张，
`n=4` 约 4.1 秒/4 张。1024×1024、4 steps 的 `n=1/2` 分别约 5.9/7.6 秒；
1024×1024、25 steps、`n=2` 约 39.7 秒且四张卡均完成。批量明显降低了多图
请求的每张图开销，但激活显存随 `分辨率²×n` 增长，不能用它替代高分辨率分块。
网关默认把批量总像素限制在 4MP，生产部署前应继续记录 `n=2/4` 的峰值显存。

### fused/归约与高分辨率复测（2026-09-23）

新版默认开启 fused QKV 和 `torch.cuda.comm.reduce_add`。PyTorch 2.10 需要显式
导入 `torch.cuda.comm`；运行时已加入兼容导入，归约不可用时自动回退 P2P loop。
固定 seed 的 512²、4 steps 冷启动约 12 秒，模型 warm 后约 2.7 秒；旧的非 fused
路径 warm 后约 2.7 秒，fused 主要减少 kernel launch 和归约 Python 开销，在高分辨率
和首次加载时收益更明显。

当前工作站自动排除了 2 GiB 的 Quadro K620，只把四张 V100 映射为进程内
`cuda:0..3`。实测结果：

- 1024²、4 steps：`n=1` 约 4.0 秒，`n=4` 约 15.1 秒（约 3.75 张/单张耗时）。
- 2048²、4 steps：约 18–20 秒；2048²、25 steps：92.2 秒。
- 1024²、8 steps 运行期间四张 V100 的 SM 利用率约 100%、79%、65%、40%；rank0
  还承担文本编码、VAE 和归约，利用率不完全对称，但四卡均有计算负载。
- 2048² 任务结束后显存约 8.6 GiB（rank0）/4.2 GiB（其余卡），当前 16 GiB 卡有余量；
  仍应在长时间连续任务后检查 allocator 碎片。
