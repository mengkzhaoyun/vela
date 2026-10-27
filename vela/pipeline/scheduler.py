"""Multi-GPU Work-Stealing Scheduler and Lossless Audio/Chapter Remuxer."""
from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

from vela.config import VelaConfig
from vela.models.upscaler import VelaUpscaler
from vela.pipeline.probe import GPUDeviceSpec, SegmentTask, VideoInfo, detect_gpus, plan_segments, probe_video
from vela.pipeline.worker import process_segment_stream


def mux_segments_with_audio(segments: list[SegmentTask], src_info: VideoInfo, output_path: Path) -> None:
    """Losslessly concatenate encoded 1080p video segments and remux original audio/chapters."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concat_file = segments[0].output_file.parent / "concat_list.txt"
    lines = [f"file '{seg.output_file.resolve().as_posix()}'" for seg in segments]
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tmp_out = output_path.with_suffix(f".muxing{output_path.suffix}")
    tmp_out.unlink(missing_ok=True)

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
        cmd.extend(
            [
                "-i",
                str(src_info.path),
                "-map",
                "0:v:0",
                "-map",
                "1:a?",
                "-c:v",
                "copy",
                "-c:a",
                "copy",
            ]
        )
    else:
        cmd.extend(["-map", "0:v:0", "-c:v", "copy"])

    if output_path.suffix.lower() == ".mp4":
        cmd.extend(["-movflags", "+faststart"])
    cmd.append(str(tmp_out))

    subprocess.run(cmd, check=True)
    tmp_out.replace(output_path)


class MultiGPUScheduler:
    """
    Dispatches GOP video segments across all detected GPUs using a shared work-stealing queue.
    Faster GPUs (e.g., RTX 5090 paired with RTX 4090) automatically pull more segments.
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
        print(f"[VELA] Workers: {gpu_summary}")

        task_queue: queue.Queue[SegmentTask] = queue.Queue()
        for seg in segments:
            task_queue.put(seg)

        errors: list[BaseException] = []
        total_frames = 0
        lock = threading.Lock()

        def worker_loop(gpu_spec: GPUDeviceSpec | None) -> None:
            nonlocal total_frames
            dev_str = f"cuda:{gpu_spec.gpu_id}" if gpu_spec is not None else "cpu"
            b_size = gpu_spec.recommended_batch_size if gpu_spec is not None else self.cfg.batch_size
            gid = gpu_spec.gpu_id if gpu_spec is not None else None
            gname = gpu_spec.name if gpu_spec is not None else ""

            try:
                upscaler = VelaUpscaler(
                    model_path=model_file if model_file.exists() else None,
                    device=dev_str,
                    compile_model=self.cfg.compile_model,
                )
                while True:
                    try:
                        seg_task = task_queue.get_nowait()
                    except queue.Empty:
                        break
                    seg_t0 = time.time()
                    n_f = process_segment_stream(
                        seg_task, info, self.cfg, upscaler, gpu_id=gid, batch_size=b_size, gpu_name=gname
                    )
                    seg_dt = time.time() - seg_t0
                    with lock:
                        total_frames += n_f
                    if n_f > 0:
                        fps_eff = n_f / max(seg_dt, 1e-3)
                        print(
                            f"[VELA][{dev_str.upper()}] Seg {seg_task.idx + 1:03d}/{len(segments):03d} "
                            f"({seg_task.start_sec:.1f}s + {seg_task.duration_sec:.1f}s) -> {n_f} frames in {seg_dt:.1f}s ({fps_eff:.1f} fps)"
                        )
                    task_queue.task_done()
            except BaseException as exc:
                with lock:
                    errors.append(exc)

        workers: list[threading.Thread] = []
        target_specs: list[GPUDeviceSpec | None] = list(self.gpu_specs) if self.gpu_specs else [None]
        for spec in target_specs:
            t = threading.Thread(target=worker_loop, args=(spec,), daemon=True)
            t.start()
            workers.append(t)

        for t in workers:
            t.join()

        if errors:
            raise RuntimeError(f"Vela Multi-GPU worker failed: {errors[0]}") from errors[0]

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
