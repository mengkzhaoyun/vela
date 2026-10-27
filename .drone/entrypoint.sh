#!/bin/bash
set -e

# ============================================================
# Vela Multi-GPU Video Super-Resolution Docker Entrypoint
#
# 用法:
#   vela                                # 自动扫描 /data 批量多卡超分至 1080p
#   vela --input a.mkv --output b.mp4   # 指定单文件多卡超分
#   vela info                           # 查看多卡探测与 NVENC 状态
# ============================================================

if [ -d "/app/vela/model_weights" ] && [ ! -e "/data/vela/model_weights" ]; then
    mkdir -p /data/vela
    ln -sfn /app/vela/model_weights /data/vela/model_weights
fi

if [ "$#" -gt 0 ]; then
    exec python3.13 -m vela "$@"
fi

exec python3.13 -m vela \
    --scan-dir "${SCAN_DIR:-/data}" \
    --gpus "${VELA_GPUS:-all}" \
    --model "${VELA_MODEL:-realplksr}" \
    --target-width "${TARGET_WIDTH:-1920}" \
    --target-height "${TARGET_HEIGHT:-1080}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --segment-seconds "${SEGMENT_SECONDS:-30.0}" \
    --cas-strength "${CAS_STRENGTH:-0.25}" \
    --codec "${CODEC:-hevc}" \
    --target-bitrate "${TARGET_BITRATE:-5M}" \
    --encoder-cq "${ENCODER_CQ:-20}" \
    ${EXTRA_ARGS:-}
