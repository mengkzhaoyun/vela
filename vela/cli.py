"""Command-line interface for Vela Multi-GPU Video Super-Resolution Engine."""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import traceback
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from vela import __version__
from vela.config import MODEL_ALIASES, VelaConfig
from vela.pipeline.probe import VideoInfo, detect_gpus, parse_bitrate_to_kbps, probe_video
from vela.runtime_log import RunLog, TaskLog, format_bitrate, format_duration, format_fps, format_size

if TYPE_CHECKING:
    from vela.pipeline.scheduler import MultiGPUScheduler

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
    from vela.pipeline.worker import has_ffmpeg_encoder

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


def _input_summary(info: VideoInfo | None) -> str:
    if info is None:
        return "输入=无法获取 视频码率=无法获取 视频时长=无法获取"
    shape = (
        f"{info.measured_height}p({info.measured_width}x{info.measured_height})"
        if info.measured_width and info.measured_height else "无法获取"
    )
    duration = format_duration(info.duration) if info.duration > 0 else "无法获取"
    return (
        f"输入={shape}@{format_fps(info.measured_fps)} "
        f"视频码率={format_bitrate(info.video_bitrate_bps)} 视频时长={duration} "
        f"视频码率来源={info.video_bitrate_source} "
        f"文件总码率={format_bitrate(info.container_bitrate_bps)} "
        f"文件总码率来源={info.container_bitrate_source}"
    )


def _output_summary(output: Path, stats: dict, started_at: float, task: TaskLog) -> str:
    # The scheduler must have produced a final file. A partial/missing file is
    # never a successful result, even when a backend returned without an error.
    size = output.stat().st_size
    if size <= 0:
        raise RuntimeError("Final output file is empty")
    info = None
    try:
        info = probe_video(output)
    except Exception as exc:
        task.emit("处理日志", f"输出元数据无法获取: {exc}")
    video_bitrate = info.video_bitrate_bps if info else None
    container_bitrate = info.container_bitrate_bps if info else None
    container_source = info.container_bitrate_source if info else "无法获取"
    if info and container_bitrate is None and info.duration > 0:
        container_bitrate = int(size * 8 / info.duration)
        container_source = "按文件大小估算"
    task.emit(
        "处理日志",
        f"输出视频码率来源={info.video_bitrate_source if info else '无法获取'} "
        f"输出文件总码率来源={container_source} 大小来源=最终文件字节数",
    )
    resolution = (
        f"{info.measured_width}x{info.measured_height}"
        if info and info.measured_width and info.measured_height else "无法获取"
    )
    elapsed = time.monotonic() - started_at
    frames = stats.get("frames")
    speed = f"{frames / elapsed:.1f}fps" if isinstance(frames, int) and frames >= 0 and elapsed > 0 else "未记录"
    return (
        f"输出={output} 分辨率={resolution} 帧率={format_fps(info.measured_fps if info else None)} "
        f"视频码率={format_bitrate(video_bitrate)} 文件总码率={format_bitrate(container_bitrate)} "
        f"大小={format_size(size)} 处理时长={format_duration(elapsed)} "
        f"处理速度={speed} 处理帧数={frames if frames is not None else '未记录'}"
    )


class _RunInterrupted(KeyboardInterrupt):
    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"收到中断信号 {signum}")


@contextmanager
def _termination_signal():
    # Ctrl+C already raises KeyboardInterrupt. Handle an ordinary SIGTERM as
    # well so the active task and remaining queue can receive a truthful tail.
    def handler(signum, _frame):
        raise _RunInterrupted(signum)

    previous = None
    installed = False
    try:
        previous = signal.signal(signal.SIGTERM, handler)
        installed = True
    except ValueError:  # Library callers may be outside the main thread.
        pass
    try:
        yield
    finally:
        if installed:
            signal.signal(signal.SIGTERM, previous)


def _run_video_tasks(
    scheduler: MultiGPUScheduler | None,
    jobs: list[tuple[Path, Path]],
    cfg: VelaConfig,
    log: RunLog,
    *,
    scan_dir: Path | None = None,
) -> int:
    total = len(jobs)
    batch_t0 = time.monotonic()
    outcomes: dict[int, str] = {}
    exit_code = 0
    timezone = datetime.now().astimezone().strftime("UTC%z")
    log.emit(
        "批次开始",
        f"运行编号={log.path.parent.name} 任务数={total} 执行方式=串行 "
        f"失败策略={'继续下一视频' if scan_dir is not None else '结束运行'} "
        f"时区={timezone} 日志={log.path}",
    )
    log.emit("参数记录", "生效配置=" + json.dumps(asdict(cfg), ensure_ascii=False, default=str))
    try:
        for index, (source, output) in enumerate(jobs, 1):
            name = source.relative_to(scan_dir).as_posix() if scan_dir is not None else source.name
            task = log.task(name, index, total)
            started = False
            t0 = time.monotonic()
            try:
                if scan_dir is not None and output.exists():
                    outcomes[index] = "跳过"
                    task.emit("任务跳过", f"原因=输出已存在 输出={output}")
                    continue
                with task.capture():
                    if not source.is_file():
                        raise FileNotFoundError(f"Input video not found: {source}")
                    info = probe_video(source)
                if scan_dir is not None and info.height >= cfg.target_height:
                    outcomes[index] = "跳过"
                    task.emit("任务跳过", f"原因=输入高度已达到目标 输入={info.width}x{info.height} 目标高度={cfg.target_height}")
                    continue
                t0 = time.monotonic()
                task.emit("任务开始", _input_summary(info))
                started = True
                with task.capture():
                    from vela.pipeline.worker import resolve_effective_fps

                    out_fps, _, _ = resolve_effective_fps(info, cfg)
                    target = parse_bitrate_to_kbps(cfg.target_bitrate)
                    task.emit(
                        "处理目标",
                        f"{cfg.target_height}p({cfg.target_width}x{cfg.target_height})@{format_fps(out_fps)} "
                        f"配置目标视频码率={format_bitrate(target * 1000) if target > 0 else '未设置'} "
                        f"编码={cfg.codec.upper()} 质量配置={cfg.encoder_cq} 模型={cfg.model_name} "
                        "生效编码器及码率上限=见初始化后的编码参数记录",
                    )
                    if scheduler is None:
                        from vela.pipeline.scheduler import MultiGPUScheduler

                        scheduler = MultiGPUScheduler(cfg)
                    stats = scheduler.process_video(source, output, task_log=task, input_info=info)
                    summary = _output_summary(output, stats, t0, task)
                # Flush partial/native output before the terminal record and
                # count success only after log capture has closed cleanly.
                outcomes[index] = "成功"
                task.emit("处理完毕", summary)
            except KeyboardInterrupt as exc:
                exit_code = 128 + getattr(exc, "signum", signal.SIGINT)
                # A signal arriving immediately after a terminal record must
                # not count the same completed/skipped video a second time.
                if index not in outcomes:
                    outcomes[index] = "中断"
                    if not started:
                        task.emit("任务开始", _input_summary(None))
                    task.emit("任务中断", f"原因=收到中断信号 输出=未确认完成 视频码率=无法获取 大小=未确认 处理时长={format_duration(time.monotonic() - t0)}")
                break
            except Exception as exc:
                if index in outcomes:
                    # The task ended, but its terminal log could not be written.
                    # Keep that outcome and fail the batch logging operation.
                    exit_code = 1
                    log.emit("处理日志", traceback.format_exc())
                    break
                outcomes[index] = "失败"
                if not started:
                    task.emit("任务开始", _input_summary(None))
                task.emit("处理日志", traceback.format_exc())
                task.emit(
                    "处理失败",
                    f"原因={exc} 输出=未确认完成 视频码率=无法获取 大小=未确认 "
                    f"处理时长={format_duration(time.monotonic() - t0)} "
                    f"后续={'继续下一视频' if scan_dir is not None else '结束运行'}",
                )
                if scan_dir is None:
                    exit_code = 1
                    break
    except KeyboardInterrupt as exc:
        # Interruption between tasks leaves the remaining jobs unprocessed.
        exit_code = 128 + getattr(exc, "signum", signal.SIGINT)
    except Exception:
        exit_code = 1
        log.emit("处理日志", traceback.format_exc())
    finally:
        counts = {state: sum(value == state for value in outcomes.values())
                  for state in ("成功", "失败", "跳过", "中断")}
        unprocessed = total - sum(counts.values())
        status = (
            "中断" if exit_code >= 128 else "存在失败" if counts["失败"] or exit_code else
            "含跳过" if counts["跳过"] else "全部成功"
        )
        log.emit(
            "批次结束",
            f"状态={status} 总数={total} " + " ".join(f"{key}={value}" for key, value in counts.items()) +
            f" 未处理={unprocessed} 总耗时={format_duration(time.monotonic() - batch_t0)} 进程退出码={exit_code}",
        )
    return exit_code


def run_batch_scan(
    scheduler: MultiGPUScheduler | None,
    scan_dir: Path,
    cfg: VelaConfig,
    *,
    run_log: RunLog | None = None,
) -> int:
    with nullcontext(run_log) if run_log is not None else RunLog() as log:
        assert log is not None
        scan_t0 = time.monotonic()
        try:
            if not scan_dir.is_dir():
                raise FileNotFoundError(f"Scan directory does not exist: {scan_dir}")
            candidates = sorted(
                p for p in scan_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
                and not p.stem.lower().endswith("_1080p")
                and not p.stem.endswith(".muxing") and not p.stem.endswith(".part")
                and ".vela_work" not in p.parts
            )
        except (Exception, KeyboardInterrupt) as exc:
            interrupted = isinstance(exc, KeyboardInterrupt)
            exit_code = 128 + getattr(exc, "signum", signal.SIGINT) if interrupted else 1
            log.emit("批次开始", f"运行编号={log.path.parent.name} 任务数=未确认 时区={datetime.now().astimezone():UTC%z} 日志={log.path}")
            log.emit("处理日志", traceback.format_exc())
            log.emit(
                "批次结束",
                f"状态={'中断' if interrupted else '失败'} 原因=无法建立候选队列 "
                f"总数=未确认 成功=0 失败=0 跳过=0 未处理=未确认 "
                f"总耗时={format_duration(time.monotonic() - scan_t0)} 进程退出码={exit_code}",
            )
            return exit_code
        jobs = [(vid, vid.with_name(f"{vid.stem}_1080p.mp4")) for vid in candidates]
        return _run_video_tasks(scheduler, jobs, cfg, log, scan_dir=scan_dir)


def run_single_video(
    scheduler: MultiGPUScheduler | None,
    input_video: Path,
    output_video: Path,
    cfg: VelaConfig,
    *,
    run_log: RunLog | None = None,
) -> int:
    with nullcontext(run_log) if run_log is not None else RunLog() as log:
        assert log is not None
        return _run_video_tasks(scheduler, [(input_video, output_video)], cfg, log)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
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

    with _termination_signal():
        if cfg.scan_dir is not None:
            return run_batch_scan(None, cfg.scan_dir, cfg)
        if cfg.input_path is not None:
            out_path = cfg.output_path or cfg.input_path.with_name(f"{cfg.input_path.stem}_1080p.mp4")
            return run_single_video(None, cfg.input_path, out_path, cfg)

    parser.print_help()
    return 0
