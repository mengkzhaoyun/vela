"""Video stream probing, GPU capability detection, and GOP segment planning."""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    fps_rational: str
    duration: float
    bitrate_kbps: int
    codec_name: str
    has_audio: bool
    audio_codec: str = ""


@dataclass
class SegmentTask:
    idx: int
    start_sec: float
    duration_sec: float
    output_file: Path


@dataclass
class GPUDeviceSpec:
    gpu_id: int
    name: str
    total_vram_gb: float
    compute_capability: tuple[int, int]
    recommended_batch_size: int


def parse_bitrate_to_kbps(val: str) -> int:
    s = val.strip().upper()
    if not s:
        return 5000
    if s.endswith("M"):
        return int(float(s[:-1]) * 1000)
    if s.endswith("K"):
        return int(float(s[:-1]))
    return int(float(s))


def _parse_fps(rate_str: str | None) -> float:
    if not rate_str or rate_str == "N/A":
        return 0.0
    try:
        if "/" in rate_str:
            num, den = rate_str.split("/", 1)
            d = float(den)
            return float(num) / d if d > 0 else 0.0
        return float(rate_str)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _parse_duration(fmt: dict, v_stream: dict) -> float:
    for val in (fmt.get("duration"), v_stream.get("duration")):
        if val is not None and str(val) not in ("", "N/A"):
            try:
                d = float(val)
                if d > 0:
                    return d
            except ValueError:
                pass
    for tags in (v_stream.get("tags") or {}, fmt.get("tags") or {}):
        for k, v in tags.items():
            if k.upper().startswith("DURATION") and isinstance(v, str) and ":" in v:
                try:
                    parts = v.strip().split(":")
                    if len(parts) == 3:
                        return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
                except ValueError:
                    pass
    return 0.0


def probe_video(video_path: Path) -> VideoInfo:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    out = subprocess.check_output(cmd).decode("utf-8", errors="replace")
    data = json.loads(out)

    v_stream = None
    has_audio = False
    audio_codec = ""
    for st in data.get("streams", []):
        if st.get("codec_type") == "video" and v_stream is None:
            v_stream = st
        elif st.get("codec_type") == "audio":
            has_audio = True
            if not audio_codec:
                audio_codec = str(st.get("codec_name", ""))

    if v_stream is None:
        raise RuntimeError(f"No video stream found in {video_path}")

    width = int(v_stream.get("width", 854))
    height = int(v_stream.get("height", 480))
    avg_rate_str = str(v_stream.get("avg_frame_rate") or "")
    r_rate_str = str(v_stream.get("r_frame_rate") or "")
    avg_fps = _parse_fps(avg_rate_str)
    r_fps = _parse_fps(r_rate_str)
    if 1.0 <= avg_fps <= 120.0 and (r_fps <= 0.0 or r_fps > 120.0 or abs(r_fps - avg_fps) > 5.0):
        fps = avg_fps
        fps_str = avg_rate_str
    elif 1.0 <= r_fps <= 120.0:
        fps = r_fps
        fps_str = r_rate_str
    elif avg_fps > 0.0:
        fps = min(avg_fps, 120.0)
        fps_str = avg_rate_str
    else:
        fps = 30.0
        fps_str = "30/1"

    fmt = data.get("format", {})
    duration = _parse_duration(fmt, v_stream)
    raw_br = v_stream.get("bit_rate") or fmt.get("bit_rate") or 0
    bitrate_kbps = int(raw_br) // 1000 if str(raw_br).isdigit() else 2500
    codec_name = str(v_stream.get("codec_name", "h264"))

    return VideoInfo(
        path=video_path,
        width=width,
        height=height,
        fps=fps,
        fps_rational=fps_str,
        duration=duration,
        bitrate_kbps=bitrate_kbps,
        codec_name=codec_name,
        has_audio=has_audio,
        audio_codec=audio_codec,
    )


def detect_gpus(gpu_spec: str = "all", default_batch: int = 8) -> list[GPUDeviceSpec]:
    import torch

    if not torch.cuda.is_available():
        return []

    count = torch.cuda.device_count()
    if gpu_spec.strip().lower() in ("all", "auto", "*"):
        indices = list(range(count))
    else:
        indices = [int(x.strip()) for x in gpu_spec.split(",") if x.strip().isdigit() and int(x.strip()) < count]

    specs: list[GPUDeviceSpec] = []
    for idx in indices:
        props = torch.cuda.get_device_properties(idx)
        vram_gb = props.total_memory / (1024**3)
        cc = (props.major, props.minor)
        # Empirical scan on 854x480 shows B=8 (TRT FP16: 21.3 fps, Compile: 13.4 fps) / B=4 (TRT FP16: 21.4 fps, Compile: 13.7 fps)
        # achieves peak throughput while keeping VRAM footprint and remainder padding overhead minimal.
        if vram_gb >= 30.0:
            rec_batch = max(default_batch, 8)
        elif vram_gb >= 22.0:
            rec_batch = default_batch
        elif vram_gb >= 19.0:
            rec_batch = min(default_batch, 6)
        else:
            rec_batch = min(default_batch, 4)

        specs.append(
            GPUDeviceSpec(
                gpu_id=idx,
                name=props.name,
                total_vram_gb=round(vram_gb, 2),
                compute_capability=cc,
                recommended_batch_size=rec_batch,
            )
        )

    # Sort highest compute capability & VRAM first (e.g. RTX 5090 before RTX 4090)
    specs.sort(key=lambda s: (s.compute_capability, s.total_vram_gb), reverse=True)
    return specs


def plan_segments(duration_sec: float, segment_seconds: float, work_dir: Path) -> list[SegmentTask]:
    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be positive, got {duration_sec}")
    seg_len = max(segment_seconds, 5.0)
    num_segments = max(1, math.ceil(duration_sec / seg_len))
    tasks: list[SegmentTask] = []

    for i in range(num_segments):
        start = round(i * seg_len, 3)
        rem = round(duration_sec - start, 3)
        if rem <= 0.05:
            break
        cur_dur = min(seg_len, rem)
        out_file = work_dir / f"seg_{i:04d}.mp4"
        tasks.append(SegmentTask(idx=i, start_sec=start, duration_sec=cur_dur, output_file=out_file))

    return tasks
