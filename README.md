# Qwen Image 2.1 四卡张量并行服务

这是一个面向 ComfyUI 的 Qwen Image 2.1 实验项目：把 DiT block 的计算分布到多张 CUDA GPU，同一请求由多卡共同完成一张图片。项目包含 ComfyUI custom node、异步图片网关和轻量 Web UI。默认只处理纯文生图，参考图、LoRA 和自定义 block hook 会回退到 ComfyUI 原生路径。

## 目录

- `comfy_extension/`：张量并行运行时。
- `qwen_tp_custom_node.py`：复制到 ComfyUI `custom_nodes/` 的可选加载器。
- `gateway.py`：异步 HTTP API 与网页服务，默认端口为 8190。
- `index.html`：生成页面和 API 使用说明。
- `serve_comfyui.multigpu.sh`：Linux 启动示例。
- `bench_comfy.py`：只使用 Python 标准库的 ComfyUI 基准工具。
- `qwen4gpu_experimental.py`：需要 Diffusers checkpoint 的独立研究原型。

生成的图片、远程部署脚本、SSH 主机密钥和 benchmark 输出不属于源码，已由 `.gitignore` 排除。请不要把模型权重或任何凭据提交到仓库。

## 环境要求

- Linux、Python 3.10+、CUDA 和 ComfyUI；V100 等 Volta GPU 使用 FP16。
- ComfyUI 中安装 Qwen Image 2.1 所需的 diffusion model、text encoder 和 VAE。
- 运行网关的 Python 只需要标准库；ComfyUI 虚拟环境需包含其自身依赖。
- 张量并行进程需要至少两张可见 GPU。`CUDA_VISIBLE_DEVICES` 决定物理卡到 `cuda:0..N` 的映射。

## 启用四卡路径

将 `qwen_tp_custom_node.py` 复制到 ComfyUI 的 `custom_nodes/`，并在启动前设置：

```bash
export QWEN_TP=1
export QWEN_TP_ROOT=/path/to/qwen2.1-4gpu
export QWEN_TP_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3
export QWEN_TP_FUSED_QKV=1
export QWEN_TP_REDUCE=comm
```

`QWEN_TP_ROOT` 是本项目目录，不应写死为某台机器的路径。也可以从 ComfyUI 目录运行 `serve_comfyui.multigpu.sh`，通过 `COMFYUI_ROOT` 和 `QWEN_TP_ROOT` 覆盖默认位置。启动脚本未显式设置 `CUDA_VISIBLE_DEVICES` 时，会优先选择显存至少 12 GiB 的前四张卡，避免把小型显示卡混入四卡 TP；也可以手动指定物理卡。将 `QWEN_TP=0` 或移除 custom node 即可回到原生路径。

默认将 Q/K/V 投影合并为一次 GEMM，并用 `torch.cuda.comm.reduce_add` 做跨卡归约；前者可通过 `QWEN_TP_FUSED_QKV=0` 关闭，后者可通过 `QWEN_TP_REDUCE=loop` 强制使用旧的逐卡归约，便于固定 seed 做回归和性能 A/B 测试。没有 NCCL 的构建会自动回退到 P2P 归约。

启动网关：

```bash
COMFY_URL=http://127.0.0.1:8188 \
GATEWAY_HOST=127.0.0.1 GATEWAY_PORT=8190 \
python gateway.py
```

网关应放在反向代理、内网或授权 VPN 后面；不要把 ComfyUI 或网关端口直接暴露到公网。生产部署、队列上限和高分辨率策略见 [DEPLOYMENT.md](DEPLOYMENT.md) 及 [HIGH-RESOLUTION.md](HIGH-RESOLUTION.md)。

## API

提交异步任务：

```bash
curl -X POST http://127.0.0.1:8190/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-image-2.1","prompt":"雨后的森林书屋，温暖灯光，水彩插画","size":"1024x1024","steps":25,"seed":42}'
```

需要一次生成多张相同提示词的图片时可设置 `n`。网关会把它们合并为一个
ComfyUI latent batch，四张卡共同执行一次，完成后在 `data` 中返回多张图片：

```json
{"model":"qwen-image-2.1","prompt":"雨后的森林书屋","size":"512x512","steps":4,"n":4}
```

`n` 默认是 1，最大值和批量总像素预算可通过 `QWEN_MAX_BATCH_SIZE`、
`QWEN_MAX_BATCH_PIXELS` 调整。默认批量预算为 4MP（例如 512²×4 或
1024²×2）；2048² 请求只能使用 `n=1`。批量不会并行提交多个 TP 任务，
因此不会重入共享的四卡运行时，也不会额外加载模型副本。

响应为 HTTP 202，包含 `id`、`seed` 和 `status_url`。使用 `GET /jobs/{id}` 查询状态；返回 `completed` 时，`data[].url` 可直接下载图片。`progress.current` 和 `progress.total` 来自 ComfyUI 采样进度，不能把它当作精确剩余时间。

当前网关接受 256–2048 边长、宽高为 8 的倍数且总像素不超过 4,194,304 的尺寸（例如 `512x512`、`1024x1024`、`1536x1536` 和 `2048x2048`），步数为 1–50，提示词最多 6000 字。默认队列最多保留 8 个任务；四卡 TP 进程使用单 worker 串行生成，每个任务内部由四张卡共同计算。高分辨率请求会占用更长时间和更多显存，建议先用 4 步压测，再提高到生产步数。

## 已知限制与验证

张量并行会增加跨卡同步开销，吞吐和显存占用取决于模型、分辨率、步数及 PCIe/NVLink 拓扑。提交前应使用固定 seed 对原生路径和 TP 路径做图像误差回归，并记录每张卡的显存、利用率和功耗。当前实验记录见 [4GPU-RESEARCH.md](4GPU-RESEARCH.md)，性能和并发建议见 [PERFORMANCE-TUNING.md](PERFORMANCE-TUNING.md)。

## 模型与许可证

本仓库只包含适配代码，不重新分发 Qwen 权重或 ComfyUI。请分别阅读 Qwen Image、ComfyUI 及依赖项目的许可证，并遵守模型使用条款。本项目代码以 MIT 许可证发布，见 [LICENSE](LICENSE)。
