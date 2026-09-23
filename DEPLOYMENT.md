# 部署说明

本文把项目目录、ComfyUI 目录和网络入口分开配置，示例中的地址只代表同机回环接口。实际服务应放在内网或授权 VPN 后，并由现有网关负责 TLS、认证和访问控制。

## 1. 安装 custom node

```bash
export PROJECT_ROOT=/path/to/qwen2.1-4gpu
export COMFYUI_ROOT=/path/to/ComfyUI
cp "$PROJECT_ROOT/qwen_tp_custom_node.py" "$COMFYUI_ROOT/custom_nodes/qwen_tp_custom_node.py"
```

启动 ComfyUI 前设置：

```bash
# 有显示卡混在设备列表时请填写物理 GPU，例如四张 V100 为 1,2,3,4：
# export CUDA_VISIBLE_DEVICES=1,2,3,4
# 也可以不设置，由 serve_comfyui.multigpu.sh 自动选择显存至少 12 GiB 的四张卡。
export QWEN_TP=1
export QWEN_TP_ROOT="$PROJECT_ROOT"
export QWEN_TP_DEVICES=0,1,2,3
export QWEN_TP_FUSED_QKV=1
export QWEN_TP_REDUCE=comm
```

若机器的 GPU 编号不是连续的，请在 `CUDA_VISIBLE_DEVICES` 中先选择物理卡，随后仍用进程内编号 `0,1,2,3`。不需要 TP 时设置 `QWEN_TP=0`。

`QWEN_TP_FUSED_QKV=1` 将每个 rank 的 Q/K/V 投影合并为一次 GEMM；`QWEN_TP_REDUCE=comm` 使用 NCCL/NVLink 归约。固定 seed 做回归时可分别设为 `0` 和 `loop` 与旧实现比较。

## 2. 启动服务

先按 ComfyUI 官方方式启动 8188，再启动网关：

```bash
cd "$COMFYUI_ROOT"
./venv/bin/python main.py --listen 127.0.0.1 --port 8188 --fp16-unet --disable-auto-launch

cd "$PROJECT_ROOT"
COMFY_URL=http://127.0.0.1:8188 \
GATEWAY_HOST=127.0.0.1 GATEWAY_PORT=8190 \
python gateway.py
```

如果反向代理把页面挂在 `/qwen-image-ui` 这类前缀下，同时设置
`QWEN_PUBLIC_BASE_PATH=/qwen-image-ui`；直接访问 8190 时保持为空，API 返回的
`status_url` 才会直接指向 `/jobs/{id}`。

反向代理只转发网关端口，并限制来源网络。网关目前不提供用户鉴权；如果需要给多个调用方使用，请在前置代理加入认证、限流和请求大小限制。

## 3. 健康检查与故障排查

```bash
curl http://127.0.0.1:8190/health
python bench_comfy.py --url http://127.0.0.1:8188 --size 512x512 --steps 4
```

同一提示词需要多张图时可用 `--n` 做一次 latent batch 基准，例如
`python bench_comfy.py --url http://127.0.0.1:8188 --size 512x512 --steps 4 --n 4`。
网关 API 的 `n` 参数使用相同限制；默认总像素预算为 4MP，先从 `n=2` 开始观察
峰值显存，再决定是否提高 `QWEN_MAX_BATCH_PIXELS`。

首次请求会加载模型，耗时明显更长。若 TP 加载失败，检查 `QWEN_TP_ROOT` 是否包含 `comfy_extension/tensor_parallel.py`，以及 ComfyUI 日志中的 CUDA 错误。可以暂时关闭 `QWEN_TP` 回到原生路径进行对照；不要删除模型或用户输出目录。

## 4. 并发与高分辨率

一个 TP runtime 同时只应执行一个采样任务；网关默认保留最多 8 个待处理任务，超出后返回 HTTP 429 和 `Retry-After`。先以 512²、4 steps 做冒烟测试，再逐步增加到 1024²、1536² 和 2048²。当前默认像素上限为 4,194,304、单边上限为 2048；可以用 `QWEN_MAX_PIXELS` 和 `QWEN_MAX_DIMENSION` 收紧限制。每次调整后固定 seed，比较输出误差和四张卡的最大显存，确认没有 OOM 或显存碎片。不要把 `QWEN_GATEWAY_WORKERS` 设置为大于 1，除非每个 worker 使用独立的非 TP ComfyUI 实例。

## 5. 安全与隐私

- SSH 密码、API token、VPN 链接和主机密钥只从环境变量或受限配置读取。
- 不要提交 `*.known_hosts`、远程部署脚本、运行输出、日志或模型权重。
- 对外提供 API 前，配置 TLS、认证、限流、最大请求体和任务保留策略。
