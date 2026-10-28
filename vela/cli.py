"""Command-line interface for Vela Multi-GPU Video Super-Resolution Engine."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vela import __version__
from vela.config import MODEL_ALIASES, VelaConfig
from vela.pipeline.probe import detect_gpus, probe_video
from vela.pipeline.scheduler import MultiGPUScheduler
from vela.pipeline.worker import has_ffmpeg_encoder

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    ".ts",
    ".m2ts",
    ".mts",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".rmvb",
    ".rm",
    ".vob",
    ".asf",
    ".3gp",
    ".f4v",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vela",
        description="Vela — Multi-GPU Real-World Video Super-Resolution Streaming Engine",
    )
    parser.add_argument("command", nargs="?", default=None, help="Optional subcommand: 'info'")
    parser.add_argument("-v", "--version", action="version", version=f"vela {__version__}")
    parser.add_argument("-i", "--input", type=Path, default=None, help="Input video file path")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output video file path")
    parser.add_argument("--scan-dir", type=Path, default=None, help="Scan directory and batch upscale videos < target height")
    parser.add_argument(
        "-m",
        "--model",
        default="realplksr",
        help=f"SR model alias ({', '.join(MODEL_ALIASES.keys())}) or filename",
    )
    parser.add_argument("--model-path", type=Path, default=None, help="Explicit path to .pth / .safetensors model weights")
    parser.add_argument("--gpus", default="all", help="GPU IDs to use: 'all' or comma-separated list like '0,1'")
    parser.add_argument("--target-width", type=int, default=1920, help="Target output width (default: 1920)")
    parser.add_argument("--target-height", type=int, default=1080, help="Target output height (default: 1080)")
    parser.add_argument("--fps", type=float, default=0.0, help="Target FPS (0 = auto cap at --max-fps)")
    parser.add_argument("--max-fps", type=float, default=30.0, help="Max output FPS cap for high-FPS sources (default: 30.0, 0=unlimited)")
    parser.add_argument("--batch-size", type=int, default=8, help="Frames per batch per GPU (default: 8)")
    parser.add_argument("--segment-seconds", type=float, default=30.0, help="GOP segment length in seconds for multi-GPU queue")
    parser.add_argument("--pre-denoise", default="hqdn3d=1.5:1.2:2.5:2.0", help="FFmpeg pre-deblocking filter before SR")
    parser.add_argument("--cas-strength", type=float, default=0.25, help="GPU Contrast Adaptive Sharpening strength (0.0~1.0)")
    parser.add_argument("--codec", default="hevc", choices=["hevc", "h264", "av1"], help="Output video codec (default: hevc)")
    parser.add_argument("--target-bitrate", default="5M", help="Target bitrate for NVENC (default: 5M)")
    parser.add_argument("--encoder-cq", type=int, default=20, help="NVENC CQ / x264 CRF quality value (default: 20)")
    parser.add_argument("--compile", dest="compile", action="store_true", default=True, help="Enable torch.compile graph optimization on CUDA (default: enabled)")
    parser.add_argument("--no-compile", dest="compile", action="store_false", help="Disable torch.compile graph optimization")
    parser.add_argument("--keep-temp", action="store_true", help="Keep .vela_work temporary segment files")
    return parser


def print_system_info(cfg: VelaConfig) -> int:
    gpus = detect_gpus(cfg.gpus, default_batch=cfg.batch_size)
    model_file = cfg.resolve_model_file()
    print(f"Vela Multi-GPU Video Super-Resolution Engine v{__version__}")
    print(f"  Model Alias      : {cfg.model_name} -> {model_file} (exists={model_file.exists()})")
    fps_desc = f"{cfg.target_fps:g}fps" if cfg.target_fps > 0 else f"max {cfg.max_fps:g}fps"
    print(f"  Target Resolution: {cfg.target_width}x{cfg.target_height} @ {fps_desc} | CAS={cfg.cas_strength}")
    print(f"  NVENC Encoders   : hevc_nvenc={has_ffmpeg_encoder('hevc_nvenc')}, h264_nvenc={has_ffmpeg_encoder('h264_nvenc')}")
    if gpus:
        print(f"  Detected GPUs ({len(gpus)}):")
        for g in gpus:
            cc_str = f"SM {g.compute_capability[0]}.{g.compute_capability[1]}"
            print(f"    - GPU {g.gpu_id}: {g.name} | VRAM: {g.total_vram_gb} GB | {cc_str} | Batch: {g.recommended_batch_size}")
    else:
        print("  Detected GPUs    : 0 (CPU fallback mode)")
    return 0


def run_batch_scan(scheduler: MultiGPUScheduler, scan_dir: Path, cfg: VelaConfig) -> int:
    if not scan_dir.exists():
        raise FileNotFoundError(f"Scan directory does not exist: {scan_dir}")

    candidates = sorted(
        p
        for p in scan_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
        and not p.stem.lower().endswith("_1080p")
        and not p.stem.endswith(".muxing")
        and not p.stem.endswith(".part")
        and ".vela_work" not in p.parts
    )
    print(f"[VELA-SCAN] Found {len(candidates)} candidate video files in {scan_dir}")
    processed = 0
    for vid in candidates:
        out_vid = vid.with_name(f"{vid.stem}_1080p.mp4")
        if out_vid.exists():
            print(f"[SKIP] Output already exists: {out_vid.name}")
            continue
        try:
            info = probe_video(vid)
            if info.height >= cfg.target_height:
                print(f"[SKIP] {vid.name} is already {info.width}x{info.height} (>= {cfg.target_height}p)")
                continue
            scheduler.process_video(vid, out_vid)
            processed += 1
        except Exception as exc:
            print(f"[ERROR] Failed to process {vid.name}: {exc}")

    print(f"[VELA-SCAN] Completed batch scan. Upscaled {processed} video(s).")
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = VelaConfig(
        input_path=args.input,
        output_path=args.output,
        scan_dir=args.scan_dir,
        model_name=args.model,
        model_path=args.model_path,
        gpus=args.gpus,
        target_width=args.target_width,
        target_height=args.target_height,
        target_fps=args.fps,
        max_fps=args.max_fps,
        batch_size=args.batch_size,
        segment_seconds=args.segment_seconds,
        pre_denoise=args.pre_denoise,
        cas_strength=args.cas_strength,
        codec=args.codec,
        target_bitrate=args.target_bitrate,
        encoder_cq=args.encoder_cq,
        compile_model=args.compile,
        keep_temp=args.keep_temp,
    )

    if args.command == "info":
        return print_system_info(cfg)

    scheduler = MultiGPUScheduler(cfg)

    if cfg.scan_dir is not None:
        return run_batch_scan(scheduler, cfg.scan_dir, cfg)

    if cfg.input_path is not None:
        if not cfg.input_path.exists():
            raise FileNotFoundError(f"Input video not found: {cfg.input_path}")
        out_path = cfg.output_path or cfg.input_path.with_name(f"{cfg.input_path.stem}_1080p.mp4")
        scheduler.process_video(cfg.input_path, out_path)
        return 0

    parser.print_help()
    return 0

