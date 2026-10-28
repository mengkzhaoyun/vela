"""Multi-GPU Work-Stealing Scheduler (Multi-Process Isolated CUDA Contexts) and Lossless Remuxer."""
from __future__ import annotations

import math
import multiprocessing as mp
from contextlib import nullcontext
import os
import queue
import re
import shutil
import subprocess
import time
import threading
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from vela.config import VelaConfig
from vela.pipeline.probe import GPUDeviceSpec, SegmentTask, VideoInfo, detect_gpus, plan_segments, probe_video
from vela.runtime_log import capture_output

if TYPE_CHECKING:
    from vela.runtime_log import TaskLog


MP4_COMPATIBLE_AUDIO_CODECS = {"aac", "mp3", "ac3", "eac3", "alac"}
_resource_warnings: set[tuple[int | None, str]] = set()


def _normalize_smi_device(value: object) -> str:
    """Adapt CUDA's bare UUID objects to nvidia-smi's GPU UUID spelling."""
    device = (value.decode() if isinstance(value, bytes) else str(value)).strip()
    if re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", device):
        return f"GPU-{device}"
    # Keep numeric selectors, existing GPU-/MIG- IDs, and unknown text intact.
    return device


def _report_worker_resources(torch_module, gpu_id: int | None, stage: str) -> None:
    """Take a best-effort snapshot at an existing worker event, never a timer."""
    tag = f"CUDA:{gpu_id}" if gpu_id is not None else "CPU"
    if gpu_id is None:
        print(f"[VELA][{tag}] 资源快照 阶段={stage} CUDA统计=不适用", flush=True)
        return

    def warn_once(source: str, exc: Exception) -> None:
        key = (gpu_id, source)
        if key not in _resource_warnings:
            _resource_warnings.add(key)
            print(f"[VELA][{tag}] 资源统计不可用 来源={source} 原因={exc}", flush=True)

    torch_values = "进程Torch显存=无法获取"
    smi_device = str(gpu_id)
    try:
        cuda = torch_module.cuda
        torch_values = (
            f"进程Torch已分配={cuda.memory_allocated(gpu_id) / (1 << 20):.1f}MiB "
            f"进程Torch保留={cuda.memory_reserved(gpu_id) / (1 << 20):.1f}MiB "
            f"进程Torch峰值已分配={cuda.max_memory_allocated(gpu_id) / (1 << 20):.1f}MiB"
        )
        # Prefer the physical UUID when CUDA_VISIBLE_DEVICES remaps indices.
        uuid = getattr(cuda.get_device_properties(gpu_id), "uuid", None)
        if uuid:
            smi_device = _normalize_smi_device(uuid)
        else:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if gpu_id < len(visible) and visible[gpu_id].strip():
                smi_device = _normalize_smi_device(visible[gpu_id])
    except Exception as exc:
        warn_once("PyTorch", exc)

    device_values = "整卡资源快照=无法获取"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id", smi_device,
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=2.0, check=True,
        )
        values = [value.strip() for value in result.stdout.strip().split(",")]
        if len(values) != 5:
            raise ValueError("unexpected nvidia-smi field count")
        units = ("%", "MiB", "MiB", "C", "W")
        labels = ("GPU利用率快照", "整卡显存已用", "整卡总显存", "整卡温度", "整卡功耗")
        fields = []
        for label, value, unit in zip(labels, values, units):
            try:
                number = float(value)
                if not math.isfinite(number) or number < 0:
                    raise ValueError
                fields.append(f"{label}={value}{unit}")
            except ValueError:
                fields.append(f"{label}=无法获取")
        device_values = " ".join(fields)
    except Exception as exc:
        warn_once("nvidia-smi", exc)
    print(f"[VELA][{tag}] 资源快照 阶段={stage} {torch_values} {device_values}", flush=True)


def mux_segments_with_audio(segments: list[SegmentTask], src_info: VideoInfo, output_path: Path) -> None:
    """Losslessly concatenate encoded 1080p video segments and remux original audio/chapters."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concat_file = segments[0].output_file.parent / "concat_list.txt"
    lines = [f"file '{seg.output_file.resolve().as_posix()}'" for seg in segments]
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tmp_out = output_path.with_suffix(f".muxing{output_path.suffix}")
    tmp_out.unlink(missing_ok=True)

    is_mp4 = output_path.suffix.lower() in (".mp4", ".m4v", ".mov")
    can_copy_audio = (not is_mp4) or (src_info.audio_codec.lower() in MP4_COMPATIBLE_AUDIO_CODECS)

    def _build_mux_cmd(copy_audio: bool) -> list[str]:
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
        ]
        if src_info.has_audio:
            audio_args = ["-c:a", "copy"] if copy_audio else ["-c:a", "aac", "-b:a", "192k"]
            cmd.extend(
                [
                    "-i",
                    str(src_info.path),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0?",
                    "-c:v",
                    "copy",
                    *audio_args,
                ]
            )
        else:
            cmd.extend(["-map", "0:v:0", "-c:v", "copy"])

        if output_path.suffix.lower() == ".mp4":
            cmd.extend(["-movflags", "+faststart"])
        cmd.append(str(tmp_out))
        return cmd

    def _run_mux(copy_audio: bool) -> None:
        result = subprocess.run(
            _build_mux_cmd(copy_audio), capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
        for output in (result.stdout, result.stderr):
            if output:
                print(output, end="" if output.endswith("\n") else "\n", flush=True)
        result.check_returncode()

    try:
        _run_mux(copy_audio=can_copy_audio)
    except subprocess.CalledProcessError:
        if src_info.has_audio and can_copy_audio:
            tmp_out.unlink(missing_ok=True)
            _run_mux(copy_audio=False)
        else:
            raise
    tmp_out.replace(output_path)


def _gpu_worker_process(
    gpu_spec: GPUDeviceSpec | None,
    cfg: VelaConfig,
    info: VideoInfo,
    model_file: Path,
    total_segments: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    log_queue: mp.Queue | None = None,
    cancel_event: object | None = None,
) -> None:
    """Keep CUDA and Python output isolated, forwarding complete lines to the parent."""
    dev_str = f"cuda:{gpu_spec.gpu_id}" if gpu_spec is not None else "cpu"
    tag = dev_str.upper()

    def forward_log(line: str) -> None:
        assert log_queue is not None
        log_queue.put(line if f"[{tag}]" in line else f"[VELA][{tag}] {line}")

    capture = capture_output(forward_log) if log_queue is not None else nullcontext()
    try:
        with capture:
            try:
                init_t0 = time.monotonic()
                print(f"[VELA][{tag}] Worker 初始化开始 权重={model_file.name}", flush=True)
                # Imports and model initialization can print diagnostics too.
                import torch
                from vela.models.upscaler import VelaUpscaler
                from vela.pipeline.worker import process_segment_stream

                if gpu_spec is not None and torch.cuda.is_available():
                    torch.cuda.set_device(gpu_spec.gpu_id)
                b_size = gpu_spec.recommended_batch_size if gpu_spec is not None else cfg.batch_size
                gid = gpu_spec.gpu_id if gpu_spec is not None else None
                gname = gpu_spec.name if gpu_spec is not None else ""
                upscaler = VelaUpscaler(
                    model_path=model_file if model_file.exists() else None,
                    device=dev_str, compile_model=cfg.compile_model, batch_size=b_size,
                )
                model_loaded = getattr(upscaler, "model", None) is not None
                print(
                    f"[VELA][{tag}] Worker 初始化完成 耗时={time.monotonic() - init_t0:.2f}秒 "
                    f"后端={getattr(upscaler, 'backend', '未记录')} "
                    f"模型已加载={'是' if model_loaded else '否'} batch={b_size}", flush=True,
                )
                if not model_loaded:
                    print(f"[VELA][{tag}] 告警 模型未加载，本轮仅执行缩放与锐化回退", flush=True)
                _report_worker_resources(torch, gid, "初始化完成")
                while cancel_event is None or not cancel_event.is_set():
                    try:
                        seg_task: SegmentTask | None = task_queue.get(timeout=0.5)
                    except queue.Empty:
                        break
                    if seg_task is None:
                        break
                    seg_t0 = time.monotonic()
                    print(
                        f"[VELA][{tag}] Seg {seg_task.idx + 1:03d}/{total_segments:03d} "
                        f"处理开始 起点={seg_task.start_sec:.1f}秒 长度={seg_task.duration_sec:.1f}秒",
                        flush=True,
                    )
                    n_f = process_segment_stream(
                        seg_task, info, cfg, upscaler, gpu_id=gid, batch_size=b_size,
                        gpu_name=gname, total_segments=total_segments, cancel_event=cancel_event,
                    )
                    seg_dt = time.monotonic() - seg_t0
                    if n_f > 0:
                        fps_eff = n_f / max(seg_dt, 1e-3)
                        print(
                            f"[VELA][{dev_str.upper()}] Seg {seg_task.idx + 1:03d}/{total_segments:03d} "
                            f"({seg_task.start_sec:.1f}s + {seg_task.duration_sec:.1f}s) -> "
                            f"{n_f} frames in {seg_dt:.1f}s ({fps_eff:.1f} fps)", flush=True,
                        )
                    else:
                        print(
                            f"[VELA][{tag}] Seg {seg_task.idx + 1:03d}/{total_segments:03d} "
                            f"处理完成 本轮新处理帧数=0 耗时={seg_dt:.2f}秒", flush=True,
                        )
                    _report_worker_resources(torch, gid, f"分段{seg_task.idx + 1}完成")
                    result_queue.put(("ok", seg_task.idx, n_f, None))
            except BaseException as exc:
                traceback.print_exc()
                result_queue.put(("err", None, 0, str(exc)))
    finally:
        # Flush producer feeder threads before the parent considers this task finished.
        result_queue.close()
        result_queue.join_thread()
        if log_queue is not None:
            log_queue.close()
            log_queue.join_thread()


class MultiGPUScheduler:
    """Dispatch GOP segments across isolated GPU workers and collect their logs."""

    def __init__(self, cfg: VelaConfig) -> None:
        self.cfg = cfg
        self.gpu_specs: list[GPUDeviceSpec] = detect_gpus(cfg.gpus, default_batch=cfg.batch_size)

    def process_video(
        self, input_video: Path, output_video: Path, *,
        task_log: TaskLog | None = None, input_info: VideoInfo | None = None,
    ) -> dict[str, float | int]:
        from vela.pipeline.worker import resolve_effective_fps

        t0 = time.monotonic()
        info = input_info if input_info is not None else probe_video(input_video)
        work_dir = output_video.parent / ".vela_work" / input_video.stem
        work_dir.mkdir(parents=True, exist_ok=True)
        segments = plan_segments(info.duration, self.cfg.segment_seconds, work_dir)
        model_file = self.cfg.resolve_model_file()
        gpu_summary = (
            ", ".join(f"GPU {g.gpu_id}: {g.name} ({g.total_vram_gb}GB, batch={g.recommended_batch_size})" for g in self.gpu_specs)
            if self.gpu_specs else "CPU Fallback (0 CUDA GPUs detected)"
        )
        out_fps, out_fps_rational, _ = resolve_effective_fps(info, self.cfg)
        print(f"[VELA] Input: {input_video.name} ({info.width}x{info.height} @ {info.fps:.2f}fps, {info.duration:.1f}s)")
        print(
            f"[VELA] Target: {self.cfg.target_width}x{self.cfg.target_height} @ {out_fps:.2f}fps ({out_fps_rational}) | "
            f"Model: {model_file.name} | Segments: {len(segments)}"
        )
        print(f"[VELA] Workers: {gpu_summary}", flush=True)

        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()
        log_queue = ctx.Queue()
        cancel_event = ctx.Event()
        logs_stopped = threading.Event()
        log_errors: list[BaseException] = []

        def consume_logs() -> None:
            while True:
                try:
                    line = log_queue.get(timeout=0.1)
                except queue.Empty:
                    if logs_stopped.is_set():
                        return
                    continue
                except (EOFError, OSError, ValueError):
                    return
                try:
                    if task_log is None:
                        print(line, flush=True)
                    else:
                        task_log.emit("处理日志", line)
                except BaseException as exc:
                    # Continue draining so workers do not deadlock on a full pipe.
                    log_errors.append(exc)

        log_thread = threading.Thread(target=consume_logs, name="vela-worker-logs", daemon=True)
        log_thread.start()
        target_specs: list[GPUDeviceSpec | None] = list(self.gpu_specs) if self.gpu_specs else [None]
        active_specs = target_specs[: max(1, len(segments))]
        active_devices = ",".join(f"CUDA:{spec.gpu_id}" if spec is not None else "CPU" for spec in active_specs)
        print(
            f"[VELA] 实际调度 workers={len(active_specs)} devices={active_devices} "
            f"分段数={len(segments)} 发现GPU数={len(self.gpu_specs)}", flush=True,
        )
        procs: list[mp.Process] = []
        completed: set[int] = set()
        newly_processed_segments = 0
        total_frames = 0
        successful = False
        try:
            for seg in segments:
                task_queue.put(seg)
            for _ in active_specs:
                task_queue.put(None)
            for spec in active_specs:
                proc = ctx.Process(
                    target=_gpu_worker_process,
                    args=(spec, self.cfg, info, model_file, len(segments), task_queue,
                          result_queue, log_queue, cancel_event),
                )
                proc.start()
                procs.append(proc)

            finished_at: float | None = None
            while True:
                try:
                    status, seg_idx, n_f, err_msg = result_queue.get(timeout=0.1)
                except queue.Empty:
                    failed = next((proc for proc in procs if proc.exitcode not in (None, 0)), None)
                    if failed is not None:
                        raise RuntimeError(f"Vela worker exited with code {failed.exitcode}")
                    if not any(proc.is_alive() for proc in procs):
                        break
                    if finished_at is not None and time.monotonic() - finished_at > 10:
                        raise RuntimeError("Vela worker did not exit after completing its segments")
                    continue
                if status == "err":
                    raise RuntimeError(f"Vela Multi-GPU worker failed: {err_msg or 'Unknown worker error'}")
                if status != "ok" or seg_idx not in range(len(segments)) or seg_idx in completed:
                    raise RuntimeError(f"Invalid or duplicate Vela segment result: {seg_idx}")
                completed.add(seg_idx)
                total_frames += int(n_f)
                if int(n_f) > 0:
                    newly_processed_segments += 1
                elapsed_now = max(time.monotonic() - t0, 1e-3)
                pct = len(completed) * 100.0 / len(segments)
                remaining = len(segments) - len(completed)
                # Cached segments advance completion but provide no throughput
                # sample for this run. Wait for fresh work before estimating.
                eta = (
                    "0.0m" if remaining == 0 else
                    f"{remaining * elapsed_now / newly_processed_segments / 60.0:.1f}m"
                    if newly_processed_segments else "pending"
                )
                print(
                    f"[VELA] Progress: {len(completed)}/{len(segments)} ({pct:.1f}%) | "
                    f"Resumed: {len(completed) - newly_processed_segments} | "
                    f"Frames: {total_frames} | Avg Speed: {total_frames / elapsed_now:.1f} fps | "
                    f"Elapsed: {elapsed_now / 60.0:.1f}m | ETA: {eta}", flush=True,
                )
                if len(completed) == len(segments):
                    finished_at = time.monotonic()
            for proc in procs:
                proc.join()
                if proc.exitcode != 0:
                    raise RuntimeError(f"Vela worker exited with code {proc.exitcode}")
            if len(completed) != len(segments):
                raise RuntimeError(
                    f"Vela workers completed only {len(completed)}/{len(segments)} segments"
                )
            successful = True
        finally:
            if not successful:
                cancel_event.set()
            # Let cooperative cancellation close FFmpeg children and flush logs first.
            deadline = time.monotonic() + 10.0
            while any(proc.is_alive() for proc in procs) and time.monotonic() < deadline:
                # Failed batches can leave peers flushing results. Drain their
                # pipe as well as the separate log stream before joining them.
                try:
                    result_queue.get(timeout=0.05)
                except queue.Empty:
                    pass
            for proc in procs:
                proc.join(timeout=0)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5.0)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=5.0)
            logs_stopped.set()
            log_thread.join(timeout=5.0)
            # Queued task messages may be unconsumed after a failure; do not wait
            # for their producer feeder to write into an abandoned pipe.
            task_queue.cancel_join_thread()
            for q in (task_queue, result_queue, log_queue):
                q.close()
            result_queue.join_thread()
            log_queue.join_thread()
            for proc in procs:
                if not proc.is_alive():
                    proc.close()

        if log_thread.is_alive():
            raise RuntimeError("Vela worker log stream did not finish")
        if log_errors:
            raise RuntimeError(f"Vela could not record worker logs: {log_errors[0]}")
        print(f"[VELA] All {len(segments)} segments completed. Remuxing audio & streams -> {output_video.name} ...")
        mux_segments_with_audio(segments, info, output_video)
        if not self.cfg.keep_temp:
            shutil.rmtree(work_dir, ignore_errors=True)
        elapsed = time.monotonic() - t0
        overall_fps = total_frames / max(elapsed, 1e-3)
        print(f"[VELA] Done! {output_video} ({elapsed:.1f}s, average {overall_fps:.1f} fps)")
        return {
            "segments": len(segments), "frames": total_frames,
            "elapsed_sec": round(elapsed, 2), "avg_fps": round(overall_fps, 2),
        }
