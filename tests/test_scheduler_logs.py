"""Exercise worker log forwarding and failure completion without model inference."""
from __future__ import annotations

import io
import os
import queue
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

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
            n_frames = 0 if cfg.model_name == "resume" and task.idx == 0 else 150
            results.put(("ok", task.idx, n_frames, None))
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

    def test_resumed_segments_advance_progress_without_shortening_fresh_work_eta(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            engine = self.make_scheduler("resume")
            info = self.info(root / "input.mp4")
            info.duration = 15  # Three segments: cached, fresh, fresh.
            ticks = iter([0.0])
            clock = types.SimpleNamespace(monotonic=lambda: next(ticks, 120.0))
            lines = []
            with patch.object(scheduler, "_gpu_worker_process", fake_spawn_worker), \
                 patch.object(scheduler, "mux_segments_with_audio") as mux, \
                 patch.object(scheduler, "time", clock), capture_output(lines.append):
                result = engine.process_video(root / "input.mp4", root / "out.mp4", input_info=info)
            progress = [line for line in lines if "[VELA] Progress:" in line]
            self.assertEqual(len(progress), 3)
            self.assertIn("Progress: 1/3", progress[0])
            self.assertIn("Resumed: 1 | Frames: 0", progress[0])
            self.assertIn("ETA: pending", progress[0])
            self.assertIn("Progress: 2/3", progress[1])
            self.assertIn("Resumed: 1 | Frames: 150", progress[1])
            # 120 seconds for one fresh segment with one left, not 60 seconds
            # obtained by incorrectly treating the cached segment as work.
            self.assertIn("ETA: 2.0m", progress[1])
            self.assertIn("Avg Speed: 1.2 fps", progress[1])
            self.assertIn("Progress: 3/3", progress[2])
            self.assertIn("Resumed: 1 | Frames: 300", progress[2])
            self.assertIn("ETA: 0.0m", progress[2])
            self.assertEqual(result["frames"], 300)
            self.assertEqual(mux.call_count, 1)

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
        self.assertIn("[VELA][CPU] model initialization", recorded)
        self.assertIn("[VELA][CPU] model warning", recorded)
        self.assertTrue(any("Traceback (most recent call last)" in line for line in recorded))
        self.assertTrue(all(line.startswith("[VELA][CPU]") for line in recorded))
        self.assertEqual(results.get_nowait()[0], "err")

    def test_worker_reports_backend_lifecycle_and_tags_library_output(self):
        torch_fake = types.ModuleType("torch")
        torch_fake.cuda = types.SimpleNamespace(is_available=lambda: True, set_device=Mock())
        model_fake = types.ModuleType("vela.models.upscaler")

        def load_model(**kwargs):
            print("[VELA-TRT] library initialization")
            return types.SimpleNamespace(backend="tensorrt-fp16", model=object())

        model_fake.VelaUpscaler = load_model
        tasks, results, logs = LocalQueue(), LocalQueue(), LocalQueue()
        tasks.put(SegmentTask(0, 0, 5, Path("seg.mp4")))
        tasks.put(None)

        def process(*args, **kwargs):
            print("[VELA][CUDA:3] already attributed")
            return 150

        spec = GPUDeviceSpec(3, "fixture", 24, (8, 9), 8)
        with patch.dict(sys.modules, {"torch": torch_fake, "vela.models.upscaler": model_fake}), \
             patch.object(worker, "process_segment_stream", side_effect=process), \
             patch.object(scheduler, "_report_worker_resources") as resources:
            scheduler._gpu_worker_process(spec, VelaConfig(), self.info(Path("input.mp4")),
                                          Path("fixture.pth"), 1, tasks, results, logs)
        recorded = []
        while not logs.empty():
            recorded.append(logs.get_nowait())
        text = "\n".join(recorded)
        self.assertIn("Worker 初始化开始", text)
        self.assertIn("Worker 初始化完成", text)
        self.assertIn("后端=tensorrt-fp16 模型已加载=是 batch=8", text)
        self.assertIn("Seg 001/001 处理开始", text)
        self.assertIn("150 frames in", text)
        self.assertIn("[VELA][CUDA:3] [VELA-TRT] library initialization", recorded)
        self.assertIn("[VELA][CUDA:3] already attributed", recorded)
        self.assertNotIn("[VELA][CUDA:3] [VELA][CUDA:3]", text)
        self.assertEqual(resources.call_count, 2)
        self.assertEqual(results.get_nowait(), ("ok", 0, 150, None))

    def test_resource_snapshot_distinguishes_process_and_whole_gpu(self):
        cuda = types.SimpleNamespace(
            memory_allocated=lambda _: 128 << 20,
            memory_reserved=lambda _: 256 << 20,
            max_memory_allocated=lambda _: 512 << 20,
            get_device_properties=lambda _: types.SimpleNamespace(uuid="GPU-fixture"),
        )
        lines = []
        response = subprocess.CompletedProcess([], 0, "75, 1024, 24564, 60, 145.5\n", "")
        with patch.object(scheduler.subprocess, "run", return_value=response) as query, capture_output(lines.append):
            scheduler._report_worker_resources(types.SimpleNamespace(cuda=cuda), 3, "分段1完成")
        self.assertEqual(query.call_args.kwargs["timeout"], 2.0)
        self.assertIn("GPU-fixture", query.call_args.args[0])
        text = "\n".join(lines)
        for field in ("[CUDA:3]", "阶段=分段1完成", "进程Torch已分配=128.0MiB",
                      "进程Torch保留=256.0MiB", "进程Torch峰值已分配=512.0MiB",
                      "GPU利用率快照=75%", "整卡显存已用=1024MiB", "整卡功耗=145.5W"):
            self.assertIn(field, text)

    def test_resource_statistics_unavailable_do_not_fail_or_repeat_warnings(self):
        scheduler._resource_warnings.clear()
        lines = []
        with patch.object(scheduler.subprocess, "run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 2)), \
             capture_output(lines.append):
            scheduler._report_worker_resources(object(), 2, "初始化完成")
            scheduler._report_worker_resources(object(), 2, "分段1完成")
            scheduler._report_worker_resources(object(), None, "初始化完成")
        self.assertEqual(sum("资源统计不可用 来源=PyTorch" in line for line in lines), 1)
        self.assertEqual(sum("资源统计不可用 来源=nvidia-smi" in line for line in lines), 1)
        self.assertEqual(sum("整卡资源快照=无法获取" in line for line in lines), 2)
        self.assertTrue(any("[CPU]" in line and "CUDA统计=不适用" in line for line in lines))
        scheduler._resource_warnings.clear()

    def test_resource_query_normalizes_cuda_uuid_and_preserves_device_mapping(self):
        bare = "01234567-89ab-cdef-0123-456789abcdef"

        class CUuuid:
            def __str__(self):
                return bare

        cases = (
            (CUuuid(), "7,6", f"GPU-{bare}"),
            (bare.encode(), "7,6", f"GPU-{bare}"),
            (f"GPU-{bare}", "7,6", f"GPU-{bare}"),
            (f"MIG-GPU-{bare}/1/0", "7,6", f"MIG-GPU-{bare}/1/0"),
            (None, "5,2", "2"),
            (None, f"5,GPU-{bare}", f"GPU-{bare}"),
            (None, f"5,{bare}", f"GPU-{bare}"),
            (None, "", "1"),
            ("not-a-uuid", "7,6", "not-a-uuid"),
        )
        response = subprocess.CompletedProcess([], 0, "75, 1024, 24564, 60, 145.5\n", "")
        for uuid, visible, expected in cases:
            with self.subTest(uuid=str(uuid), visible=visible):
                cuda = types.SimpleNamespace(
                    memory_allocated=lambda _: 0,
                    memory_reserved=lambda _: 0,
                    max_memory_allocated=lambda _: 0,
                    get_device_properties=lambda _: types.SimpleNamespace(uuid=uuid),
                )
                lines = []
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visible}), \
                     patch.object(scheduler.subprocess, "run", return_value=response) as query, \
                     capture_output(lines.append):
                    scheduler._report_worker_resources(types.SimpleNamespace(cuda=cuda), 1, "初始化完成")
                self.assertEqual(query.call_args.args[0][2], expected)
                self.assertTrue(any("GPU利用率快照=75%" in line for line in lines))

    def test_active_worker_count_is_limited_by_available_segments(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            engine = self.make_scheduler()
            engine.gpu_specs = [GPUDeviceSpec(i, "fixture", 24, (8, 9), 8) for i in range(8)]
            lines = []
            with patch.object(scheduler, "_gpu_worker_process", fake_spawn_worker), \
                 patch.object(scheduler, "mux_segments_with_audio"), capture_output(lines.append):
                engine.process_video(root / "input.mp4", root / "out.mp4", input_info=self.info(root / "input.mp4"))
            dispatch = next(line for line in lines if "实际调度" in line)
            self.assertIn("workers=2 devices=CUDA:0,CUDA:1 分段数=2 发现GPU数=8", dispatch)

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
