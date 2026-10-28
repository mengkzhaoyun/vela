# 变更说明：Vela v0.1.0 多 GPU 流式视频超分引擎初始化

- **版本**：`v0.1.0`
- **日期**：`2026-10-27`
- **负责人**：`Mengsk`

## 1. 核心架构特性

1. **多 GPU GOP 关键帧动态抢占调度（`vela/pipeline/scheduler.py`）**：
   - 自动发现全部 CUDA GPU，按算力（Compute Capability）与显存容量降序排列（如 RTX 5090 优先于 RTX 4090），并根据显存自动适配 `batch_size`（32GB 显卡默认 `14`，24GB 显卡默认 `8`）。
   - 将长视频按 GOP 时间窗（默认 `30.0s`）拆分为独立任务放入共享队列，多张 GPU 并行拉取处理，支持 `.done` 原子标记断点续传。
2. **TensorRT 10.16 FP16 引擎、全显存后处理与零磁盘中间帧管道（`vela/models/trt_engine.py` & `vela/models/upscaler.py` & `vela/pipeline/worker.py`）**：
   - 基于 `spandrel` 支持 `PLKSRArch`（`4xNomosWebPhoto_RealPLKSR.safetensors`），重写 `PLKConv2d` 为非原地 NHWC 连续拼接，优先使用 **TensorRT 10.16 FP16** 序列化引擎（缓存于 `/data/jasna/model_weights/trt-cache/`，自动回退 `torch.compile` NHWC FP16），并将尾段余数帧零填充至固定 `batch_size`。
   - 在 GPU 显存内直接完成 FP16 `Bicubic Antialiased 1920x1080` 下采样与 `FidelityFX CAS` 自适应锐化，通过 `readinto` 零拷贝锁页输入环与 `torch.cuda.Event` 异步输出环送入 `hevc_nvenc`（显式写入 BT.709 色彩矩阵与 VUI 标签，无 NVENC 权限时自动回退 `libx265`）。
   - 对高于 `max_fps`（默认 `30.0`）的源视频采用整数抽取因子 `n = ceil(src_fps / max_fps)`（如 `59.94 -> 29.97fps`，有理数 `19001/634`），并对解码返回码、片段输出帧数与队列异常退出进行强校验。
3. **统一挂载目录与多封装格式兼容（`vela/cli.py` & `vela/pipeline/probe.py` & `vela/pipeline/scheduler.py`）**：
   - 模型目录统一复用 `/data/jasna/model_weights`，视频扫描默认绑定 `/data/jasna/ais:/data`，不在 `/data` 下创建冗余子目录。
   - 原生支持 `.mkv`、`.mp4`、`.avi`、`.mov`、`.wmv`、`.flv`、`.webm`、`.ts`、`.rmvb`、`.vob` 等格式，针对 Matroska (`.mkv`) 修正 `avg_frame_rate` 与 `tags.DURATION` 解析，并在封装 `_1080p.mp4` 时自动处理非 MP4 兼容音轨（自动回退 `AAC 192k`）。

