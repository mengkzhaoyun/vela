"""Multi-GPU Work-Stealing Scheduler (Multi-Process Isolated CUDA Contexts) and Lossless Remuxer."""
from __future__ import annotations

import multiprocessing as mp
import queue
import shutil
import subprocess
import time
from pathlib import Path

from vela.config import VelaConfig
from vela.models.upscaler import VelaUpscaler
from vela.pipeline.probe import GPUDeviceSpec, SegmentTask, VideoInfo, detect_gpus, plan_segments, probe_video
from vela.pipeline.worker import process_segment_stream


MP4_COMPATIBLE_AUDIO_CODECS = {"aac", "mp3", "ac3", "eac3", "alac"}


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

    try:
        subprocess.run(_build_mux_cmd(copy_audio=can_copy_audio), check=True)
    except subprocess.CalledProcessError:
        if src_info.has_audio and can_copy_audio:
            tmp_out.unlink(missing_ok=True)
            subprocess.run(_build_mux_cmd(copy_audio=False), check=True)
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
) -> None:
    """Isolated per-GPU OS process worker to eliminate Python GIL and CUDA context lock contention."""
    import torch

    dev_str = f"cuda:{gpu_spec.gpu_id}" if gpu_spec is not None else "cpu"
    if gpu_spec is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_spec.gpu_id)
    b_size = gpu_spec.recommended_batch_size if gpu_spec is not None else cfg.batch_size
    gid = gpu_spec.gpu_id if gpu_spec is not None else None
    gname = gpu_spec.name if gpu_spec is not None else ""

    try:
        upscaler = VelaUpscaler(
            model_path=model_file if model_file.exists() else None,
            device=dev_str,
            compile_model=cfg.compile_model,
        )
        while True:
            try:
                seg_task: SegmentTask | None = task_queue.get(timeout=0.5)
            except queue.Empty:
                break
            if seg_task is None:
                break

            seg_t0 = time.time()
            n_f = process_segment_stream(
                seg_task, info, cfg, upscaler, gpu_id=gid, batch_size=b_size, gpu_name=gname
            )
            seg_dt = time.time() - seg_t0
            if n_f > 0:
                fps_eff = n_f / max(seg_dt, 1e-3)
                print(
                    f"[VELA][{dev_str.upper()}] Seg {seg_task.idx + 1:03d}/{total_segments:03d} "
                    f"({seg_task.start_sec:.1f}s + {seg_task.duration_sec:.1f}s) -> {n_f} frames in {seg_dt:.1f}s ({fps_eff:.1f} fps)",
                    flush=True,
                )
            result_queue.put(("ok", n_f, None))
    except BaseException as exc:
        result_queue.put(("err", 0, str(exc)))


class MultiGPUScheduler:
    """
    Dispatches GOP video segments across all detected GPUs using isolated per-GPU OS processes
    and a shared work-stealing multiprocessing queue.
    """

    def __init__(self, cfg: VelaConfig) -> None:
        self.cfg = cfg
        self.gpu_specs: list[GPUDeviceSpec] = detect_gpus(cfg.gpus, default_batch=cfg.batch_size)

    def process_video(self, input_video: Path, output_video: Path) -> dict[str, float | int]:
        t0 = time.time()
        info = probe_video(input_video)
        work_dir = output_video.parent / ".vela_work" / input_video.stem
        work_dir.mkdir(parents=True, exist_ok=True)

        segments = plan_segments(info.duration, self.cfg.segment_seconds, work_dir)
        model_file = self.cfg.resolve_model_file()

        gpu_summary = (
            ", ".join(f"GPU {g.gpu_id}: {g.name} ({g.total_vram_gb}GB, batch={g.recommended_batch_size})" for g in self.gpu_specs)
            if self.gpu_specs
            else "CPU Fallback (0 CUDA GPUs detected)"
        )
        print(f"[VELA] Input: {input_video.name} ({info.width}x{info.height} @ {info.fps:.2f}fps, {info.duration:.1f}s)")
        print(f"[VELA] Target: {self.cfg.target_width}x{self.cfg.target_height} | Model: {model_file.name} | Segments: {len(segments)}")
        print(f"[VELA] Workers: {gpu_summary}", flush=True)

        ctx = mp.get_context("spawn")
        task_queue: mp.Queue = ctx.Queue()
        result_queue: mp.Queue = ctx.Queue()

        target_specs: list[GPUDeviceSpec | None] = list(self.gpu_specs) if self.gpu_specs else [None]
        # Only spawn as many GPU workers as there are segments
        active_specs = target_specs[: max(1, len(segments))]

        for seg in segments:
            task_queue.put(seg)
        for _ in active_specs:
            task_queue.put(None)

        procs: list[mp.Process] = []
        for spec in active_specs:
            p = ctx.Process(
                target=_gpu_worker_process,
                args=(spec, self.cfg, info, model_file, len(segments), task_queue, result_queue),
            )
            p.start()
            procs.append(p)

        total_frames = 0
        errors: list[str] = []
        completed_count = 0

        while completed_count < len(segments) and any(p.is_alive() for p in procs):
            try:
                status, n_f, err_msg = result_queue.get(timeout=1.0)
                if status == "err":
                    errors.append(err_msg or "Unknown worker error")
                    break
                total_frames += int(n_f)
                completed_count += 1
            except queue.Empty:
                continue

        # Drain any remaining items in result_queue
        while not result_queue.empty():
            try:
                status, n_f, err_msg = result_queue.get_nowait()
                if status == "err":
                    errors.append(err_msg or "Unknown worker error")
                else:
                    total_frames += int(n_f)
            except queue.Empty:
                break

        for p in procs:
            p.join(timeout=10.0)
            if p.is_alive():
                p.terminate()

        if errors:
            raise RuntimeError(f"Vela Multi-GPU worker failed: {errors[0]}")

        print(f"[VELA] All {len(segments)} segments completed. Remuxing audio & streams -> {output_video.name} ...")
        mux_segments_with_audio(segments, info, output_video)

        if not self.cfg.keep_temp:
            shutil.rmtree(work_dir, ignore_errors=True)

        elapsed = time.time() - t0
        overall_fps = total_frames / max(elapsed, 1e-3)
        print(f"[VELA] Done! {output_video} ({elapsed:.1f}s, average {overall_fps:.1f} fps)")
        return {
            "segments": len(segments),
            "frames": total_frames,
            "elapsed_sec": round(elapsed, 2),
            "avg_fps": round(overall_fps, 2),
        }
