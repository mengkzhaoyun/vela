# 变更说明：Vela v0.1.0 多 GPU 流式视频超分引擎初始化

- **版本**：`v0.1.0`
- **日期**：`2026-10-27`
- **负责人**：`Mengsk`

## 1. 核心架构特性

1. **多 GPU GOP 关键帧动态抢占调度（`vela/pipeline/scheduler.py`）**：
   - 自动发现全部 CUDA GPU，按算力（Compute Capability）与显存容量降序排列（如 RTX 5090 优先于 RTX 4090），并根据显存自动适配 `batch_size`（32GB 显卡默认 `14`，24GB 显卡默认 `8`）。
   - 将长视频按 GOP 时间窗（默认 `30.0s`）拆分为独立任务放入共享队列，多张 GPU 并行拉取处理，支持 `.done` 原子标记断点续传。
2. **全显存后处理与零磁盘中间帧管道（`vela/models/upscaler.py` & `vela/pipeline/worker.py`）**：
   - 基于 `spandrel` 原生支持 `PLKSRArch`（`4xNomosWebPhoto_RealPLKSR.pth`、`4x-ClearRealityV1.pth`），自动识别 `PLKSRArch` 的 `float32` 精度要求。
   - 在 GPU 显存内直接完成 `Bicubic Antialiased 1920x1080` 下采样与纯 PyTorch 实现的 `FidelityFX CAS` 自适应对比度锐化，随后直接通过管道送入 `hevc_nvenc` / `h264_nvenc`。

