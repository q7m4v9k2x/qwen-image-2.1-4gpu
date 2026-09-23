# 四卡 TP 性能、并发与高分辨率设计

## 当前实现

`comfy_extension/tensor_parallel.py` 是单个请求内四卡并行：每个 DiT block 的 Q/K/V 和 MLP 分片在四张 V100 上同时计算，再归并到 `cuda:0`。`ThreadPoolExecutor` 只负责同一个 block 的四个 rank，不能安全地把多个请求同时送进同一个 `TensorParallel` 实例。

当前 API 把任务提交给 ComfyUI 队列。默认按 prompt 串行执行，因此并发 HTTP 请求会排队；在 Python HTTP 层直接并发调用不会提高吞吐，绕过队列还可能造成共享模型、CUDA stream 和临时张量竞争。

## 可执行优化顺序

1. 在网关或扩展中增加全局 GPU 生成锁，最多允许一个 TP 任务进入 `TensorParallel.forward`；任务状态保存 `queued/admitted/running/saving/completed/failed/cancelled`、当前 step 和更新时间。建议初始队列上限 8，超出返回 429。
2. 通过 batch 提高吞吐。把相同尺寸、steps 的请求按短时间窗口聚合成 batch，再让四卡一次计算多个样本；当前 `forward` 已读取 batch 维，但 gateway 固定 `batch_size=1`，需要在调度层分桶并把结果拆回各 job。batch=2/4 要在 16GB V100 上实测。
3. 减少每个 block 的同步与拷贝。`parallel()` 目前把完整 hidden state 复制到四张卡，并把后三个结果依次搬回 `cuda:0`；后续可使用每 rank 的 CUDA stream、GPU peer copy 和树形归约，但每次变更必须以固定 seed 做 dense/TP RMSE 回归。
4. 服务启动后做一次 1-step 512² warmup，保持 TP runtime 和权重 shard 常驻，避免首请求解量化；同时关注 CUDA allocator 碎片，不能在请求间重建 `TensorParallel`。

## 高分辨率

激活和 attention 随 latent token 数增长，显存/计算压力主要由 `width * height` 决定。API 应强制宽高为 8 的倍数、限制单边和总像素，并限制 steps；建议默认 512²、768²、1024²，1536² 先单独压测。2048² 及以上不要直接走全局 attention，使用“1024 基础图 + latent/像素分块放大”的两阶段流程；分块需 overlap 和边缘融合，并在文档中标注其全局构图与原生模式不同。

建议增加 `mode: base|hires`、`base_size`、`scale`、`tile_size`、`overlap`，给 hires 任务更低的并发权重。不要只放大像素上限，否则 OOM 会在采样中途发生。

## 准确进度同步

网页应以不超过 1 秒的短轮询查询 `/jobs/{id}`，并直接显示网关从 ComfyUI WebSocket `progress` 事件转发的 `value/max` 采样步数；没有真实步数时只显示“排队中/处理中”，不要按耗时伪造百分比。后续可增加 SSE `/v1/images/generations/{id}/events`，事件包含 `{job_id, phase, step, total, percent}`，网页再改用 `EventSource`，减少轮询开销。

## 验证矩阵

固定 prompt、seed 和 sampler，测试 512²/1024²/1536²、steps 4/25、batch 1/2/4；记录每卡最大显存、功耗、SM、CUDA event 时间、端到端耗时和 OOM 率。再用 1/2/4/8 个客户端提交，确认同一个 TP runtime 不重入。当前 TP 相对 RMSE 门槛保持 0.05。
