# 高分辨率路线

## 为什么不能直接把上限改到 4K

Qwen Image 2.1 的 latent token 数随输出面积增加。对全局 self-attention 而言，注意力矩阵的工作量和显存大致按 token 数的平方增长；把边长从 2048 提到 4096 会使面积变成 4 倍，注意力部分的理论规模接近 16 倍。四卡张量并行只把每个 head/MLP 分片到不同卡，并没有改变这个复杂度，也不能把四张 16 GB 显存当成一个没有通信成本的大显存池。

因此网关的原生模式应继续把单边和总像素限制在已经实测的范围内。当前工作站的参考点是 2048×2048、25 步约 96 秒；任何放宽限制的改动都必须重新记录每卡峰值显存、端到端时间和 OOM 率。

## 推荐的三个档位

| 模式 | 适用尺寸 | 做法 | 预期特性 |
| --- | --- | --- | --- |
| `native` | ≤ 2048 边长、≤ 4 MP | 现有 Qwen TP KSampler 一次完成 | 结构最一致；四卡共同计算；一个 TP runtime 只能串行执行 |
| `cascade` | 2K 到 4K 或更高 | 先在 1024–1328 边长生成基础图，再分两倍级联做低去噪细化 | 每个阶段的全局 token 数可控；需要 tile/overlap 融合，耗时随阶段数增加 |
| `native_hr_plugin` | 2K–4K 实验 | 在 ComfyUI 中安装高分辨率节点（DyPE/HRDiT 等），对位置编码或注意力做专门处理 | 需要按 Qwen 2.1、TP monkey-patch 和当前模型文件逐项验证；不能默认开启 |

### 级联方案的实现边界

仅使用 `ImageScale` 或 VAE tiled 解码只能减少 VAE 峰值显存，不能降低 DiT 全局注意力的开销。可行的级联必须满足下列条件：

1. 基础阶段在原生分辨率生成并保存 latent/像素图。
2. 放大阶段使用低去噪的 img2img 或专门的 cascade 节点，按固定 tile 大小处理，tile 之间保留 overlap 并做 feather 融合。
3. VAE 编码/解码使用 tiled 版本（例如 `VAEEncodeTiled`/`VAEDecodeTiled`），tile 大小和 overlap 由显存压测决定；这只解决 VAE 峰值，不替代分块 DiT。
4. 每个 tile 的任务仍由一个调度器串行送入 TP runtime；不能让 tile 并发重入同一 `TensorParallel` 对象。若要并发，必须启动独立 ComfyUI/模型副本并给每个副本固定 GPU 集合。

当前网关尚未实现这条级联链路，因此不能通过把 `QWEN_MAX_PIXELS` 调大来冒充 4K 支持。建议以后增加显式参数，例如：

```json
{
  "mode": "cascade",
  "base_size": "1024x1024",
  "target_size": "4096x4096",
  "stages": 2,
  "tile_size": 1024,
  "overlap": 128,
  "denoise": 0.25
}
```

服务端应把 cascade 计为高权重任务，单独限制同时占用的 tile 数，并在进度中报告 `base`, `upscale-1`, `upscale-2` 等阶段。对于不支持的组合返回明确的 4xx 错误，而不是让 ComfyUI 在采样中途 OOM。

## 公开项目调研

截至 2026-09-23，`wildminder/ComfyUI-DyPE` 的 Qwen 示例提供了以下可参考路线：

- DyPE/vision-YARN：对 Qwen 的位置编码做动态外推，示例把基础分辨率设为 1328，并尝试 4096 输出。
- `VAEDecodeTiled`：示例使用 512 tile、64 overlap，解决高分辨率 VAE 解码峰值。
- PixelRush/HiFlow：先生成基础 latent，再做两倍级联和重叠 patch 细化；适合“保留构图、补充细节”的场景。
- HRDiT SPA/HAP：分别处理高分辨率空间错位和稀疏注意力；HAP 需要针对模型和分辨率校准 scope plan。

这些节点是外部 ComfyUI 插件，且可能包装模型的 `_forward`。本项目的 `QWEN_TP` 也会替换 Qwen 的 `_forward`，所以不能直接把插件复制进生产环境。启用前至少要验证：

- 插件识别的是 Qwen Image 2.1（而不是旧 Qwen Image/不同 VAE 格式）；
- 插件包装顺序与 `qwen_tp_custom_node.py` 不冲突；
- 固定 seed 的 dense/TP 相对 RMSE 仍在 0.05 门槛内；
- 2K、3K、4K 的每卡显存、P2P 通信和输出质量均有记录；
- 失败时可以通过环境变量关闭插件并回到 `native`。

参考：

- <https://github.com/wildminder/ComfyUI-DyPE>
- <https://github.com/comfyanonymous/ComfyUI>
- <https://huggingface.co/Comfy-Org/Qwen-Image-2.1>

## 压测矩阵

固定提示词、seed 和 sampler，分开记录采样和 VAE 时间：

| 维度 | 值 |
| --- | --- |
| 原生尺寸 | 512²、1024²、1536²、2048² |
| 级联目标 | 2048²、3072²、4096² |
| 步数 | 4、8、25 |
| 队列深度 | 1、2、4、8 |
| 记录项 | 每卡峰值显存/功耗/SM、P2P、采样步时、VAE 时、总时、OOM/失败率 |

先以 4 步完成矩阵，再对通过的尺寸测试 25 步。高分辨率功能只有在连续 20 次任务无 OOM、无拼接接缝且输出质量通过人工抽检后，才能从实验开关升级为可见 API 模式。
