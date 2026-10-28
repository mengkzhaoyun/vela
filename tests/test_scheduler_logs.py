"""Exercise worker log forwarding and failure completion without model inference."""
from __future__ import annotations

import io
import os
import queue
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from vela.config import VelaConfig
from vela.pipeline import scheduler, worker
from vela.pipeline.probe import GPUDeviceSpec, SegmentTask, VideoInfo
from vela.runtime_log import RunLog, capture_output


def fake_spawn_worker(spec, cfg, info, model_file, total, tasks, results, logs, cancel):
    """A real spawned producer; no media, torch, model, or GPU is involved."""
    if cfg.model_name == "crash":
        os._exit(7)
    try:
        if cfg.model_name == "cancel":
            cancel.wait(timeout=5)
            logs.put("worker cancellation acknowledged")
            return
        while True:
            task = tasks.get(timeout=1)
            if task is None:
                break
            logs.put(f"worker segment {task.idx}")
            if cfg.model_name == "error":
                logs.put("Traceback: deliberate worker error")
                results.put(("err", None, 0, "deliberate worker error"))
                break
            results.put(("ok", task.idx, 150, None))
        # Specifically verify that output written after the result is not lost.
        logs.put("last worker line")
    finally:
        results.close()
        results.join_thread()
        logs.close()
        logs.join_thread()


class LocalQueue(queue.Queue):
    def close(self):
        pass

    def join_thread(self):
        pass


class InterruptResultQueue:
    """Inject a parent interruption while retaining a real spawn-safe queue."""
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.interrupt = True

    def get(self, *args, **kwargs):
        if self.interrupt:
            self.interrupt = False
            raise KeyboardInterrupt
        return self.wrapped.get(*args, **kwargs)

    def put(self, value):
        self.wrapped.put(value)

    def close(self):
        self.wrapped.close()

    def join_thread(self):
        self.wrapped.join_thread()


class InterruptContext:
    def __init__(self, real):
        self.real = real
        self.count = 0

    def Queue(self):
        self.count += 1
        q = self.real.Queue()
        return InterruptResultQueue(q) if self.count == 2 else q

    def Event(self):
        return self.real.Event()

    def Process(self, **kwargs):
        return self.real.Process(**kwargs)


class SchedulerLoggingTests(unittest.TestCase):
    def info(self, path: Path) -> VideoInfo:
        return VideoInfo(path, 16, 16, 30, "30/1", 10, 5000, "h264", False)

    def make_scheduler(self, model: str = "realplksr"):
        cfg = VelaConfig(model_name=model, segment_seconds=5, keep_temp=True)
        with patch.object(scheduler, "detect_gpus", return_value=[]):
            return scheduler.MultiGPUScheduler(cfg)

    def test_spawned_logs_finish_before_next_task_and_success(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            engine = self.make_scheduler()
            engine.gpu_specs = [GPUDeviceSpec(i, "test", 24, (8, 9), 1) for i in range(2)]
            with RunLog(root / "vela.log", console=io.StringIO()) as log:
                task_log = log.task("input.mp4", 1, 2)
                with patch.object(scheduler, "_gpu_worker_process", fake_spawn_worker), \
                     patch.object(scheduler, "mux_segments_with_audio") as mux, task_log.capture():
                    result = engine.process_video(root / "input.mp4", root / "out.mp4",
                                                  task_log=task_log, input_info=self.info(root / "input.mp4"))
                log.task("second.mp4", 2, 2).emit("任务开始", "next")
            text = (root / "vela.log").read_text(encoding="utf-8")
            self.assertEqual(result["frames"], 300)
            self.assertEqual(mux.call_count, 1)
            self.assertIn("Frames: 300", text)
            self.assertEqual(text.count("last worker line"), 2)
            self.assertLess(text.rindex("last worker line"), text.index("second.mp4"))
            self.assertTrue(all("input.mp4 [1/2]" in line for line in text.splitlines() if "worker" in line))

    def test_crashed_worker_does_not_mux_or_report_done(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            engine = self.make_scheduler("crash")
            lines = []
            with patch.object(scheduler, "_gpu_worker_process", fake_spawn_worker), \
                 patch.object(scheduler, "mux_segments_with_audio") as mux, capture_output(lines.append):
                with self.assertRaisesRegex(RuntimeError, "exited with code 7"):
                    engine.process_video(root / "input.mp4", root / "out.mp4", input_info=self.info(root / "input.mp4"))
            mux.assert_not_called()
            self.assertFalse(any("Done!" in line for line in lines))

    def test_reported_error_keeps_worker_logs_and_does_not_mux(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            engine = self.make_scheduler("error")
            lines = []
            with patch.object(scheduler, "_gpu_worker_process", fake_spawn_worker), \
                 patch.object(scheduler, "mux_segments_with_audio") as mux, capture_output(lines.append):
                with self.assertRaisesRegex(RuntimeError, "deliberate worker error"):
                    engine.process_video(root / "input.mp4", root / "out.mp4", input_info=self.info(root / "input.mp4"))
            mux.assert_not_called()
            self.assertTrue(any("Traceback:" in line for line in lines))
            self.assertIn("last worker line", lines)

    def test_worker_capture_includes_initialization_and_traceback(self):
        torch_fake = types.ModuleType("torch")
        model_fake = types.ModuleType("vela.models.upscaler")
        def fail_model(**kwargs):
            print("model initialization")
            print("model warning", file=sys.stderr)
            raise RuntimeError("model unavailable")
        model_fake.VelaUpscaler = fail_model
        results, logs = LocalQueue(), LocalQueue()
        with patch.dict(sys.modules, {"torch": torch_fake, "vela.models.upscaler": model_fake}):
            scheduler._gpu_worker_process(None, VelaConfig(), self.info(Path("input.mp4")),
                                          Path("missing.pth"), 1, LocalQueue(), results, logs)
        recorded = []
        while True:
            try:
                recorded.append(logs.get_nowait())
            except queue.Empty:
                break
        self.assertIn("model initialization", recorded)
        self.assertIn("model warning", recorded)
        self.assertTrue(any("Traceback (most recent call last)" in line for line in recorded))
        self.assertEqual(results.get_nowait()[0], "err")

    def test_interruption_drains_cancelled_worker_before_returning(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            engine = self.make_scheduler("cancel")
            context = InterruptContext(scheduler.mp.get_context("spawn"))
            lines = []
            with patch.object(scheduler, "_gpu_worker_process", fake_spawn_worker), \
                 patch.object(scheduler.mp, "get_context", return_value=context), \
                 patch.object(scheduler, "mux_segments_with_audio") as mux, capture_output(lines.append):
                with self.assertRaises(KeyboardInterrupt):
                    engine.process_video(root / "input.mp4", root / "out.mp4", input_info=self.info(root / "input.mp4"))
            mux.assert_not_called()
            self.assertIn("worker cancellation acknowledged", lines)

    def test_actual_encoder_parameters_distinguish_quality_only_fallback(self):
        worker._last_encoder_args.clear()
        lines = []
        with capture_output(lines.append):
            args = ["-c:v", "hevc_nvenc", "-rc", "vbr", "-cq", "20", "-b:v", "5000k", "-maxrate", "5000k"]
            worker._report_encoder_args(args, 0, "test")
            worker._report_encoder_args(args, 0, "test")
            worker._report_encoder_args(["-c:v", "libx265", "-crf", "20"], 0, "test")
        self.assertEqual(len(lines), 2)
        self.assertIn("生效目标视频码率=5.000Mbps", lines[0])
        self.assertIn("编码器=libx265", lines[1])
        self.assertIn("CRF=20", lines[1])
        self.assertIn("生效目标视频码率=未设置", lines[1])
        self.assertIn("码率上限=未设置", lines[1])

    def test_reused_segment_reports_no_new_encoding(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "seg_0000.mp4"
            output.write_bytes(b"x" * 2048)
            output.with_suffix(".done").write_text("frames=150\n", encoding="utf-8")
            segment = SegmentTask(0, 0, 5, output)
            lines = []
            with patch.dict(sys.modules, {"torch": types.ModuleType("torch")}), capture_output(lines.append):
                count = worker.process_segment_stream(segment, self.info(Path("input.mp4")), VelaConfig(), object())
            self.assertEqual(count, 0)
            self.assertTrue(any("本轮新处理帧数=0 编码器本轮未运行" in line for line in lines))

    def test_subprocess_stderr_is_drained_without_separate_files(self):
        lines = []
        code = "import sys; sys.stderr.write(('diagnostic ' + 'x'*1000 + '\\n')*100); sys.stderr.flush()"
        with capture_output(lines.append):
            with worker._ffmpeg_processes([sys.executable, "-c", code], [sys.executable, "-c", code], 1024, 1024) as (dec, enc):
                enc.stdin.close()
                dec.stdout.read()
                dec.wait(timeout=10)
                enc.wait(timeout=10)
        self.assertEqual(len(lines), 200)
        self.assertEqual(sum(line.startswith("[FFmpeg decoder]") for line in lines), 100)
        self.assertEqual(sum(line.startswith("[FFmpeg encoder]") for line in lines), 100)


if __name__ == "__main__":
    unittest.main()
