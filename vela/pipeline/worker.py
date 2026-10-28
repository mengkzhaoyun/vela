"""Single-GPU zero-disk-frame streaming worker (FFmpeg Decode Pipe -> GPU SR/CAS -> NVENC Pipe)."""
from __future__ import annotations

import functools
import math
import queue
import subprocess
import threading
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


@functools.lru_cache(maxsize=16)
def can_use_nvenc(encoder_name: str, gpu_id: int) -> bool:
    if not has_ffmpeg_encoder(encoder_name):
        return False
    try:
        res = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=256x256:r=30:d=0.04",
                "-frames:v",
                "1",
                "-c:v",
                encoder_name,
                "-gpu",
                str(gpu_id),
                "-pix_fmt",
                "yuv420p",
                "-f",
                "null",
                "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        return res.returncode == 0
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

    if gpu_id is not None and not is_amd and can_use_nvenc(nvenc_name, gpu_id):
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


def resolve_effective_fps(info: VideoInfo, cfg: VelaConfig) -> tuple[float, str, bool]:
    """
    Return (output_fps, output_fps_rational, needs_fps_filter).
    Uses integer decimation n = ceil(src_fps / max_fps) for high-FPS sources (e.g. 59.94 -> 29.97)
    to avoid non-integer frame duplication jitter and preserve exact rational timebase.
    """
    if cfg.target_fps > 0:
        return cfg.target_fps, f"{cfg.target_fps:g}", True
    if cfg.max_fps > 0 and info.fps > cfg.max_fps + 0.5:
        n = max(2, math.ceil(info.fps / cfg.max_fps))
        out_fps = info.fps / n
        if "/" in info.fps_rational:
            num_s, den_s = info.fps_rational.split("/", 1)
            if num_s.strip().isdigit() and den_s.strip().isdigit() and int(den_s.strip()) > 0:
                return out_fps, f"{int(num_s.strip())}/{int(den_s.strip()) * n}", True
        return out_fps, f"{out_fps:.6f}".rstrip("0").rstrip("."), True
    rational = info.fps_rational if info.fps_rational and info.fps_rational != "0/0" else f"{info.fps:g}"
    return info.fps, rational, False


def _readinto_full(stream: object, buf_mv: memoryview) -> int:
    """Read directly into a pre-allocated memoryview until full or EOF (zero intermediate bytes copy)."""
    total = 0
    target = len(buf_mv)
    readinto_fn = getattr(stream, "readinto", None)
    if readinto_fn is not None:
        while total < target:
            n = readinto_fn(buf_mv[total:])
            if not n:
                break
            total += int(n)
        return total
    raw = getattr(stream, "read")(target)
    if not raw:
        return 0
    n = len(raw)
    buf_mv[:n] = raw
    return n


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

    out_fps, out_fps_rational, needs_fps_filter = resolve_effective_fps(info, cfg)
    vf_filters: list[str] = []
    if needs_fps_filter:
        vf_filters.append(f"fps={out_fps_rational}")
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
        out_fps_rational,
        "-i",
        "pipe:0",
        "-vf",
        "scale=out_color_matrix=bt709:out_range=tv",
        *build_video_encoder_args(cfg, info, gpu_id, gpu_name=gpu_name),
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
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
    ring_depth = 4
    use_pin = torch.cuda.is_available()
    in_ring = [
        torch.empty((batch_size, info.height, info.width, 3), dtype=torch.uint8, pin_memory=use_pin)
        for _ in range(ring_depth)
    ]
    out_ring = [
        torch.empty((batch_size, cfg.target_height, cfg.target_width, 3), dtype=torch.uint8, pin_memory=use_pin)
        for _ in range(ring_depth)
    ]
    free_in_slots: queue.Queue[int] = queue.Queue()
    free_out_slots: queue.Queue[int] = queue.Queue()
    for idx in range(ring_depth):
        free_in_slots.put(idx)
        free_out_slots.put(idx)

    read_q: queue.Queue[tuple[int, int] | None] = queue.Queue(maxsize=ring_depth)
    write_q: queue.Queue[tuple[torch.Tensor, torch.cuda.Event | None, int] | None] = queue.Queue(maxsize=ring_depth)
    stop_event = threading.Event()
    worker_errors: list[BaseException] = []

    def _put_Abortable(q: queue.Queue, item: object) -> bool:
        while not stop_event.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _get_Abortable(q: queue.Queue) -> tuple[bool, object]:
        while not stop_event.is_set():
            try:
                return True, q.get(timeout=0.2)
            except queue.Empty:
                if worker_errors:
                    stop_event.set()
                    break
                continue
        return False, None

    def _reader_loop() -> None:
        assert dec_proc.stdout is not None
        try:
            while not stop_event.is_set():
                ok, slot_obj = _get_Abortable(free_in_slots)
                if not ok or slot_obj is None:
                    break
                slot = int(slot_obj)
                buf_tensor = in_ring[slot]
                mv = memoryview(buf_tensor.numpy()).cast("B")
                n_read = _readinto_full(dec_proc.stdout, mv)
                if n_read == 0:
                    free_in_slots.put(slot)
                    break
                n_f = n_read // frame_bytes
                if n_f == 0:
                    free_in_slots.put(slot)
                    break
                if not _put_Abortable(read_q, (slot, n_f)):
                    break
        except BaseException as exc:
            worker_errors.append(exc)
            stop_event.set()
        finally:
            _put_Abortable(read_q, None)

    def _writer_loop() -> None:
        assert enc_proc.stdin is not None
        try:
            while not stop_event.is_set():
                ok, item = _get_Abortable(write_q)
                if not ok or item is None:
                    break
                out_slice, cuda_ev, out_slot = item  # type: ignore[misc]
                if cuda_ev is not None:
                    cuda_ev.synchronize()
                enc_proc.stdin.write(memoryview(out_slice.numpy()).cast("B"))
                free_out_slots.put(out_slot)
        except BaseException as exc:
            worker_errors.append(exc)
            stop_event.set()

    reader_thread = threading.Thread(target=_reader_loop, daemon=True)
    writer_thread = threading.Thread(target=_writer_loop, daemon=True)
    reader_thread.start()
    writer_thread.start()

    try:
        while not stop_event.is_set():
            ok, item = _get_Abortable(read_q)
            if not ok or item is None:
                break
            in_slot, n_frames = item  # type: ignore[misc]
            if worker_errors:
                raise worker_errors[0]

            ok_out, out_slot_obj = _get_Abortable(free_out_slots)
            if not ok_out or out_slot_obj is None:
                break
            out_slot = int(out_slot_obj)

            batch_tensor = in_ring[in_slot][:n_frames]
            out_slice, cuda_ev = upscaler.process_batch_rgb24_async(
                batch_tensor,
                out_ring[out_slot],
                target_w=cfg.target_width,
                target_h=cfg.target_height,
                cas_strength=cfg.cas_strength,
            )
            free_in_slots.put(in_slot)
            if not _put_Abortable(write_q, (out_slice, cuda_ev, out_slot)):
                break
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

        _put_Abortable(write_q, None)
        writer_thread.join(timeout=30.0)
        reader_thread.join(timeout=30.0)
        if worker_errors:
            raise worker_errors[0]
        if enc_proc.stdin is not None:
            enc_proc.stdin.close()
        enc_proc.wait()
        dec_proc.wait()
        if dec_proc.returncode != 0:
            raise RuntimeError(f"FFmpeg decoder exited with code {dec_proc.returncode} on segment {task.idx}")
        if enc_proc.returncode != 0:
            raise RuntimeError(f"FFmpeg encoder exited with code {enc_proc.returncode} on segment {task.idx}")
        max_frame_tol = max(4, int(est_frames * 0.05))
        if processed_frames == 0 or abs(processed_frames - est_frames) > max_frame_tol:
            raise RuntimeError(
                f"Segment {task.idx} frame count mismatch: decoded {processed_frames} frames vs expected {est_frames}"
            )
    finally:
        stop_event.set()
        if dec_proc.poll() is None:
            dec_proc.kill()
        if enc_proc.poll() is None:
            enc_proc.kill()

    tmp_output.replace(task.output_file)
    done_marker.write_text(f"frames={processed_frames}\n", encoding="utf-8")
    return processed_frames
