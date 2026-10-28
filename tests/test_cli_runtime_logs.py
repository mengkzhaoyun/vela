"""Exercise CLI task and batch logs without starting a model or reading media."""
from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vela import cli
from vela.config import VelaConfig
from vela.pipeline.probe import VideoInfo
from vela.runtime_log import RunLog, TaskLog


class CliRuntimeLogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.log_path = self.root / "batch" / "vela.log"
        self.console = io.StringIO()
        self.cfg = VelaConfig(target_fps=60.0, max_fps=60.0, target_bitrate="5M")

    def source(self, relative: str) -> Path:
        path = self.inputs / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"input fixture; metadata is mocked")
        return path

    @staticmethod
    def metadata(path: Path) -> VideoInfo:
        is_output = path.stem.endswith("_1080p")
        width, height = (1920, 1080) if is_output else (854, 480)
        bitrate = 2_200_000 if is_output else 5_000_000
        return VideoInfo(
            path=path,
            width=width,
            height=height,
            fps=30.0 if is_output else 60.0,
            fps_rational="30/1" if is_output else "60/1",
            duration=60.0,
            bitrate_kbps=bitrate // 1000,
            codec_name="hevc" if is_output else "h264",
            has_audio=True,
            audio_codec="aac",
            video_bitrate_bps=bitrate,
            container_bitrate_bps=2_360_000 if is_output else 5_160_000,
            video_bitrate_source="视频流bit_rate",
            container_bitrate_source="容器bit_rate",
            # Deliberately differs from the processing field and target FPS.
            measured_fps=30000 / 1001 if is_output else 60.0,
            measured_width=width,
            measured_height=height,
        )

    @staticmethod
    def success(input_video: Path, output_video: Path, *, task_log, input_info) -> dict:
        if input_info.path != input_video:
            raise AssertionError("The task received another video's input metadata")
        output_video.write_bytes(b"x" * 123_456)
        print(f"[VELA] Done! {output_video.name}")
        print("fixture stderr diagnostic", file=sys.stderr)
        return {"frames": 1800, "elapsed_sec": 75.0, "avg_fps": 24.0}

    def read_log(self) -> str:
        text = self.log_path.read_text(encoding="utf-8")
        self.assertEqual(text, self.console.getvalue())
        self.assertEqual(list(self.log_path.parent.iterdir()), [self.log_path])
        for line in text.splitlines():
            self.assertRegex(line, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} Vela .+ \[(?:批次|\d+/\d+)\] ")
        return text

    def batch(self, scheduler, *, probe=None) -> tuple[int, str]:
        with patch("vela.cli.probe_video", side_effect=probe or self.metadata):
            with RunLog(self.log_path, console=self.console) as log:
                code = cli.run_batch_scan(scheduler, self.inputs, self.cfg, run_log=log)
        return code, self.read_log()

    def single(self, scheduler, source: Path, *, probe=None) -> tuple[int, str]:
        output = source.with_name(f"{source.stem}_1080p.mp4")
        with patch("vela.cli.probe_video", side_effect=probe or self.metadata):
            with RunLog(self.log_path, console=self.console) as log:
                code = cli.run_single_video(scheduler, source, output, self.cfg, run_log=log)
        return code, self.read_log()

    def test_two_same_named_videos_are_serial_and_share_one_log(self) -> None:
        first = self.source("left/sample.mp4")
        second = self.source("right/sample.mp4")
        scheduler = SimpleNamespace(process_video=Mock(side_effect=self.success))

        code, log = self.batch(scheduler)

        self.assertEqual(code, 0)
        self.assertEqual([call.args[0] for call in scheduler.process_video.call_args_list], [first, second])
        first_label = "Vela left/sample.mp4 [1/2]"
        second_label = "Vela right/sample.mp4 [2/2]"
        for label in (first_label, second_label):
            self.assertIn(f"{label} 任务开始", log)
            self.assertIn(f"{label} 处理目标", log)
            self.assertIn(f"{label} 处理日志 [VELA] Done!", log)
            self.assertIn(f"{label} 处理日志 fixture stderr diagnostic", log)
            self.assertIn(f"{label} 处理完毕", log)
        self.assertLess(log.index(f"{first_label} 处理完毕"), log.index(f"{second_label} 任务开始"))
        summary = next(line for line in log.splitlines() if "批次结束" in line)
        for field in ("总数=2", "成功=2", "失败=0", "跳过=0", "未处理=0", "进程退出码=0"):
            self.assertIn(field, summary)
        for line in log.splitlines():
            if "进程退出码=" in line:
                self.assertIn("[批次]", line)

    def test_failure_continues_to_skip_and_success_with_separate_counts(self) -> None:
        failed = self.source("a-failed.mp4")
        skipped = self.source("b-skipped.mp4")
        good = self.source("c-good.mp4")
        existing = skipped.with_name("b-skipped_1080p.mp4")
        existing.write_bytes(b"existing output must be preserved")

        def process(input_video, output_video, **kwargs):
            if input_video == failed:
                print("failure detail on stderr", file=sys.stderr)
                raise RuntimeError("simulated encoder failure")
            return self.success(input_video, output_video, **kwargs)

        scheduler = SimpleNamespace(process_video=Mock(side_effect=process))
        code, log = self.batch(scheduler)

        self.assertEqual(code, 0)  # Preserve native scan exit-code behavior.
        self.assertEqual([call.args[0] for call in scheduler.process_video.call_args_list], [failed, good])
        self.assertIn("Vela a-failed.mp4 [1/3] 处理失败", log)
        self.assertIn("simulated encoder failure", log)
        self.assertIn("Vela a-failed.mp4 [1/3] 处理日志 failure detail on stderr", log)
        self.assertIn("Vela b-skipped.mp4 [2/3] 任务跳过", log)
        self.assertNotIn("Vela b-skipped.mp4 [2/3] 任务开始", log)
        self.assertEqual(existing.read_bytes(), b"existing output must be preserved")
        self.assertIn("Vela c-good.mp4 [3/3] 处理完毕", log)
        summary = next(line for line in log.splitlines() if "批次结束" in line)
        for field in ("状态=存在失败", "总数=3", "成功=1", "失败=1", "跳过=1", "未处理=0", "进程退出码=0"):
            self.assertIn(field, summary)

    def test_keyboard_interrupt_preserves_finished_task_and_unstarted_queue(self) -> None:
        first = self.source("a.mp4")
        second = self.source("b.mp4")
        self.source("c.mp4")

        def process(input_video, output_video, **kwargs):
            if input_video == second:
                raise KeyboardInterrupt()
            return self.success(input_video, output_video, **kwargs)

        scheduler = SimpleNamespace(process_video=Mock(side_effect=process))
        code, log = self.batch(scheduler)

        self.assertEqual(code, 130)
        self.assertEqual([call.args[0] for call in scheduler.process_video.call_args_list], [first, second])
        self.assertIn("Vela a.mp4 [1/3] 处理完毕", log)
        self.assertIn("Vela b.mp4 [2/3] 任务中断", log)
        self.assertNotIn("Vela c.mp4 [3/3] 任务开始", log)
        summary = next(line for line in log.splitlines() if "批次结束" in line)
        for field in ("状态=中断", "总数=3", "成功=1", "中断=1", "未处理=1", "进程退出码=130"):
            self.assertIn(field, summary)

    def test_single_video_uses_final_metadata_instead_of_configured_target(self) -> None:
        source = self.source("sample.mp4")
        scheduler = SimpleNamespace(process_video=Mock(side_effect=self.success))

        code, log = self.single(scheduler, source)

        self.assertEqual(code, 0)
        result = next(line for line in log.splitlines() if "处理完毕" in line)
        for field in ("Vela sample.mp4 [1/1]", "输出=", "分辨率=1920x1080", "视频码率=2.200Mbps", "文件总码率=2.360Mbps", "大小=0.12MB", "处理时长=", "处理速度=", "处理帧数=1800"):
            self.assertIn(field, result)
        self.assertRegex(result, r"帧率=29\.97\d?fps")
        self.assertNotIn("帧率=60fps", result)
        self.assertNotIn("视频码率=5.000Mbps", result)
        self.assertNotIn("进程退出码", result)
        self.assertIn("进程退出码=0", log.splitlines()[-1])

    def test_output_probe_failure_keeps_success_but_marks_unknown_metrics(self) -> None:
        source = self.source("sample.mp4")

        def probe(path):
            if path.stem.endswith("_1080p"):
                raise RuntimeError("simulated metadata unavailable")
            return self.metadata(path)

        scheduler = SimpleNamespace(process_video=Mock(side_effect=self.success))
        code, log = self.single(scheduler, source, probe=probe)

        self.assertEqual(code, 0)
        result = next(line for line in log.splitlines() if "处理完毕" in line)
        self.assertIn("视频码率=无法获取", result)
        self.assertIn("帧率=无法获取", result)
        self.assertIn("大小=0.12MB", result)
        self.assertIn("simulated metadata unavailable", log)
        self.assertNotIn("处理失败", log)

    def test_single_video_failure_returns_nonzero_and_keeps_traceback(self) -> None:
        source = self.source("sample.mp4")
        scheduler = SimpleNamespace(process_video=Mock(side_effect=RuntimeError("encoder broke")))

        code, log = self.single(scheduler, source)

        self.assertEqual(code, 1)
        self.assertIn("Vela sample.mp4 [1/1] 处理失败", log)
        self.assertIn("encoder broke", log)
        self.assertIn("Traceback (most recent call last)", log)
        self.assertNotIn("处理完毕", log)
        self.assertIn("进程退出码=1", log.splitlines()[-1])

    def test_returning_without_output_file_cannot_be_logged_as_success(self) -> None:
        source = self.source("sample.mp4")
        scheduler = SimpleNamespace(process_video=Mock(return_value={"frames": 1800, "elapsed_sec": 75.0, "avg_fps": 24.0}))

        code, log = self.single(scheduler, source)

        self.assertEqual(code, 1)
        self.assertIn("处理失败", log)
        self.assertNotIn("处理完毕", log)
        self.assertIn("成功=0", log.splitlines()[-1])
        self.assertIn("失败=1", log.splitlines()[-1])

    def test_real_cli_missing_input_creates_one_default_log_and_exits_one(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repository) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(
            [sys.executable, "-m", "vela", "-i", "missing.mp4"],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )

        self.assertEqual(completed.returncode, 1, completed.stderr)
        log_root = self.root / ".tmp" / "restoration-order" / "private" / "runtime-logs"
        logs = list(log_root.glob("*/vela.log"))
        self.assertEqual(len(logs), 1)
        self.assertEqual([p for p in log_root.rglob("*") if p.is_file()], logs)
        log = logs[0].read_text(encoding="utf-8")
        self.assertEqual(log, completed.stdout)
        self.assertEqual(completed.stderr, "")
        self.assertIn("Vela missing.mp4 [1/1] 任务开始", log)
        self.assertIn("Vela missing.mp4 [1/1] 处理失败", log)
        self.assertIn("[批次] 批次结束", log)
        self.assertIn("进程退出码=1", log.splitlines()[-1])
        self.assertNotIn("处理完毕", log)

    def test_interrupt_during_candidate_scan_leaves_truthful_batch_tail(self) -> None:
        scheduler = SimpleNamespace(process_video=Mock())
        with patch("pathlib.Path.rglob", side_effect=KeyboardInterrupt):
            code, log = self.batch(scheduler)

        self.assertEqual(code, 130)
        scheduler.process_video.assert_not_called()
        self.assertIn("[批次] 批次开始", log)
        summary = next(line for line in log.splitlines() if "批次结束" in line)
        for field in ("状态=中断", "总数=未确认", "成功=0", "未处理=未确认", "进程退出码=130"):
            self.assertIn(field, summary)
        self.assertNotIn("处理完毕", log)
        self.assertNotIn("任务开始", log)

    def test_partial_and_native_stderr_are_flushed_before_success_summary(self) -> None:
        source = self.source("sample.mp4")

        def process(input_video, output_video, **kwargs):
            stats = self.success(input_video, output_video, **kwargs)
            sys.stdout.write("partial tail")
            os.write(2, b"native tail\n")
            return stats

        scheduler = SimpleNamespace(process_video=Mock(side_effect=process))
        code, log = self.single(scheduler, source)

        self.assertEqual(code, 0)
        terminal = log.index("Vela sample.mp4 [1/1] 处理完毕")
        for tail in ("partial tail", "native tail"):
            record = f"Vela sample.mp4 [1/1] 处理日志 {tail}"
            self.assertEqual(log.count(record), 1)
            self.assertLess(log.index(record), terminal)
        self.assertIn("成功=1", log.splitlines()[-1])
        self.assertIn("失败=0", log.splitlines()[-1])

    def assert_interrupt_after_terminal_record(self, event: str) -> None:
        first = self.source("a.mp4")
        self.source("b.mp4")
        skipped = event == "任务跳过"
        if skipped:
            first.with_name("a_1080p.mp4").write_bytes(b"existing output")
        scheduler = SimpleNamespace(process_video=Mock(side_effect=self.success))
        original_emit = TaskLog.emit

        def emit_then_interrupt(task, actual_event, message):
            original_emit(task, actual_event, message)
            if task.index == 1 and actual_event == event:
                raise KeyboardInterrupt()

        with patch.object(TaskLog, "emit", new=emit_then_interrupt):
            code, log = self.batch(scheduler)

        self.assertEqual(code, 130)
        self.assertEqual(scheduler.process_video.call_count, 0 if skipped else 1)
        self.assertEqual(log.count(f"Vela a.mp4 [1/2] {event}"), 1)
        self.assertNotIn("任务中断", log)
        self.assertNotIn("Vela b.mp4 [2/2] 任务开始", log)
        summary = next(line for line in log.splitlines() if "批次结束" in line)
        for field in (
            "状态=中断", "总数=2", f"成功={0 if skipped else 1}",
            "失败=0", f"跳过={1 if skipped else 0}", "中断=0",
            "未处理=1", "进程退出码=130",
        ):
            self.assertIn(field, summary)

    def test_interrupt_after_success_record_does_not_reclassify_completed_task(self) -> None:
        self.assert_interrupt_after_terminal_record("处理完毕")

    def test_interrupt_after_skip_record_does_not_count_the_task_twice(self) -> None:
        self.assert_interrupt_after_terminal_record("任务跳过")


if __name__ == "__main__":
    unittest.main()
