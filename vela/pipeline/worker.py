"""Single-GPU zero-disk-frame streaming worker (FFmpeg Decode Pipe -> GPU SR/CAS -> NVENC Pipe)."""
from __future__ import annotations

import functools
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from vela.config import VelaConfig
from vela.models.upscaler import VelaUpscaler
from vela.pipeline.probe import SegmentTask, VideoInfo, parse_bitrate_to_kbps


@functools.lru_cache(maxsize=8)
def has_ffmpeg_encoder(encoder_name: str) -> bool:
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.STDOUT).decode(
            "utf-8", errors="ignore"
        )
        return encoder_name in out
    except Exception:
        return False


def build_video_encoder_args(cfg: VelaConfig, info: VideoInfo, gpu_id: int | None, gpu_name: str = "") -> list[str]:
    target_kbps = parse_bitrate_to_kbps(cfg.target_bitrate)
    low_bitrate_mode = cfg.preserve_low_bitrate and info.bitrate_kbps > 0 and info.bitrate_kbps <= target_kbps

    codec_key = cfg.codec.lower()
    if codec_key in ("hevc", "h265"):
        nvenc_name, amf_name, sw_name = "hevc_nvenc", "hevc_amf", "libx265"
    elif codec_key in ("av1",):
        nvenc_name, amf_name, sw_name = "av1_nvenc", "av1_amf", "libsvtav1"
    else:
        nvenc_name, amf_name, sw_name = "h264_nvenc", "h264_amf", "libx264"

    is_amd = any(k in gpu_name.upper() for k in ("AMD", "RADEON", "GFX"))

    if gpu_id is not None and not is_amd and has_ffmpeg_encoder(nvenc_name):
        args = [
            "-c:v",
            nvenc_name,
            "-gpu",
            str(gpu_id),
            "-preset",
            "p5",
            "-rc",
            "vbr",
            "-cq",
            str(cfg.encoder_cq),
            "-g",
            "60",
            "-pix_fmt",
            "yuv420p",
        ]
        if not low_bitrate_mode and target_kbps > 0:
            args.extend(["-b:v", f"{target_kbps}k", "-maxrate", f"{target_kbps}k", "-bufsize", f"{target_kbps * 2}k"])
        return args

    if gpu_id is not None and is_amd and has_ffmpeg_encoder(amf_name):
        args = [
            "-c:v",
            amf_name,
            "-quality",
            "balanced",
            "-rc",
            "vbr_peak",
            "-qp_i",
            str(cfg.encoder_cq),
            "-qp_p",
            str(cfg.encoder_cq),
            "-g",
            "60",
            "-pix_fmt",
            "yuv420p",
        ]
        if not low_bitrate_mode and target_kbps > 0:
            args.extend(["-b:v", f"{target_kbps}k", "-maxrate", f"{target_kbps}k", "-bufsize", f"{target_kbps * 2}k"])
        return args

    # Software encoder fallback for CPU / non-NVENC hosts
    args = [
        "-c:v",
        sw_name if has_ffmpeg_encoder(sw_name) else "libx264",
        "-preset",
        "fast",
        "-crf",
        str(cfg.encoder_cq),
        "-pix_fmt",
        "yuv420p",
    ]
    return args


def resolve_effective_fps(info: VideoInfo, cfg: VelaConfig) -> tuple[float, bool]:
    """Return (output_fps, needs_fps_filter). Caps >max_fps sources to max_fps (default 30fps)."""
    if cfg.target_fps > 0:
        return cfg.target_fps, True
    if cfg.max_fps > 0 and info.fps > cfg.max_fps + 0.5:
        return cfg.max_fps, True
    return info.fps, False


def process_segment_stream(
    task: SegmentTask,
    info: VideoInfo,
    cfg: VelaConfig,
    upscaler: VelaUpscaler,
    gpu_id: int | None = 0,
    batch_size: int = 8,
    gpu_name: str = "",
    total_segments: int = 1,
) -> int:
    """
    Stream a single GOP segment through FFmpeg Decoder -> VelaUpscaler (VRAM) -> FFmpeg NVENC.
    Returns total processed frames.
    """
    done_marker = task.output_file.with_suffix(".done")
    if done_marker.exists() and task.output_file.exists() and task.output_file.stat().st_size > 1024:
        return 0

    task.output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = task.output_file.with_suffix(".part.mp4")
    tmp_output.unlink(missing_ok=True)

    out_fps, needs_fps_filter = resolve_effective_fps(info, cfg)
    vf_filters: list[str] = []
    if needs_fps_filter:
        vf_filters.append(f"fps={out_fps:g}")
    if cfg.pre_denoise.strip():
        vf_filters.append(cfg.pre_denoise.strip())

    decode_cmd = [
        "ffmpeg",
        "-v",
        "fatal",
        "-ss",
        f"{task.start_sec:.3f}",
        "-t",
        f"{task.duration_sec:.3f}",
        "-i",
        str(info.path),
    ]
    if vf_filters:
        decode_cmd.extend(["-vf", ",".join(vf_filters)])
    decode_cmd.extend(
        [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )

    encode_cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{cfg.target_width}x{cfg.target_height}",
        "-r",
        f"{out_fps:.4f}",
        "-i",
        "pipe:0",
        *build_video_encoder_args(cfg, info, gpu_id, gpu_name=gpu_name),
        str(tmp_output),
    ]

    frame_bytes = info.width * info.height * 3
    dec_proc = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * max(batch_size, 4))
    enc_proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, bufsize=cfg.target_width * cfg.target_height * 3 * 4)

    est_frames = max(1, int(round(task.duration_sec * out_fps)))
    log_step = max(batch_size * 4, est_frames // 4)
    next_log_frames = log_step
    dev_tag = f"CUDA:{gpu_id}" if gpu_id is not None else "CPU"
    seg_t0 = time.time()

    processed_frames = 0
    try:
        assert dec_proc.stdout is not None
        assert enc_proc.stdin is not None

        while True:
            raw_chunk = dec_proc.stdout.read(frame_bytes * batch_size)
            if not raw_chunk:
                break
            n_frames = len(raw_chunk) // frame_bytes
            if n_frames == 0:
                break
            valid_bytes = raw_chunk[: n_frames * frame_bytes]
            arr = np.frombuffer(valid_bytes, dtype=np.uint8).reshape((n_frames, info.height, info.width, 3))
            batch_tensor = torch.from_numpy(arr.copy())

            out_tensor = upscaler.process_batch_rgb24(
                batch_tensor,
                target_w=cfg.target_width,
                target_h=cfg.target_height,
                cas_strength=cfg.cas_strength,
            )
            enc_proc.stdin.write(out_tensor.numpy().tobytes())
            processed_frames += n_frames

            if processed_frames >= next_log_frames and processed_frames < est_frames:
                dt = max(time.time() - seg_t0, 1e-3)
                pct = min(99, int(processed_frames * 100 / est_frames))
                print(
                    f"[VELA][{dev_tag}] Seg {task.idx + 1:03d}/{total_segments:03d} "
                    f"in-progress: {processed_frames}/{est_frames} frames ({pct}%) | {processed_frames / dt:.1f} fps",
                    flush=True,
                )
                next_log_frames += log_step

        enc_proc.stdin.close()
        enc_proc.wait()
        dec_proc.wait()
        if enc_proc.returncode != 0:
            raise RuntimeError(f"FFmpeg encoder exited with code {enc_proc.returncode} on segment {task.idx}")
    finally:
        if dec_proc.poll() is None:
            dec_proc.kill()
        if enc_proc.poll() is None:
            enc_proc.kill()

    tmp_output.replace(task.output_file)
    done_marker.write_text(f"frames={processed_frames}\n", encoding="utf-8")
    return processed_frames
