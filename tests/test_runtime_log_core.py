"""File and capture behavior of the native single-file runtime logger."""
from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from vela.runtime_log import RunLog, capture_output, format_bitrate, format_duration, format_fps, format_size


class RuntimeLogCoreTests(unittest.TestCase):
    def test_batch_and_serial_tasks_share_one_exclusive_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run" / "vela.log"
            console = io.StringIO()
            with RunLog(path, console=console) as log:
                log.emit("批次开始", "任务数=2")
                log.task("a.mp4", 1, 2).emit("任务开始", "输入=480p")
                log.task("a.mp4", 1, 2).emit("处理完毕", "输出=a_1080p.mp4")
                log.task("b.mp4", 2, 2).emit("任务开始", "输入=480p")
                log.task("b.mp4", 2, 2).emit("处理失败", "原因=sample error")
                log.emit("批次结束", "成功=1 失败=1")
                with self.assertRaises(FileExistsError):
                    RunLog(path)
                self.assertEqual(path.read_text(encoding="utf-8"), console.getvalue())
            self.assertEqual([p.name for p in Path(directory).rglob("*") if p.is_file()], ["vela.log"])
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 6)
            self.assertIn("Vela a.mp4 [1/2] 处理完毕", lines[2])
            self.assertIn("Vela b.mp4 [2/2] 处理失败", lines[4])
            self.assertIn("Vela - [批次] 批次结束", lines[-1])
            with self.assertRaises(ValueError):
                log.emit("处理日志", "after close")

    def test_default_path_is_unique_per_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch("vela.runtime_log.Path", wraps=Path) as path_type:
            # Redirect only the default relative root without changing process cwd.
            def rooted_path(value: str | Path) -> Path:
                value = Path(value)
                return Path(directory) / value if not value.is_absolute() else value
            path_type.side_effect = rooted_path
            with RunLog(console=io.StringIO()) as first, RunLog(console=io.StringIO()) as second:
                self.assertNotEqual(first.path, second.path)
                self.assertEqual(first.path.name, "vela.log")
                self.assertRegex(first.path.parent.name, r"^\d{8}-\d{6}-[0-9a-f]{12}$")
            self.assertEqual(len(list(Path(directory).rglob("vela.log"))), 2)

    def test_capture_stdout_stderr_progress_and_partial_on_exception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vela.log"
            original_stdout, original_stderr = sys.stdout, sys.stderr
            console = io.StringIO()
            with RunLog(path, console=console) as log:
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    with log.task("sample.mp4", 1, 1).capture():
                        print("stdout line")
                        print("stderr line", file=sys.stderr)
                        sys.stdout.write("progress 1\rprogress 2\r\n")
                        sys.stderr.write("partial error")
                        raise RuntimeError("test failure")
                self.assertIs(sys.stdout, original_stdout)
                self.assertIs(sys.stderr, original_stderr)
            bodies = [line.split("处理日志 ", 1)[1] for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(bodies, ["stdout line", "stderr line", "progress 1", "progress 2", "partial error"])
            self.assertEqual(console.getvalue(), path.read_text(encoding="utf-8"))

    def test_capture_partial_writes_do_not_mix_threads(self) -> None:
        records: list[str] = []
        barrier = threading.Barrier(4)
        with capture_output(records.append):
            def writer(index: int) -> None:
                sys.stdout.write(f"worker-{index}")
                barrier.wait(timeout=5)
                sys.stdout.write(" done\n")
            workers = [threading.Thread(target=writer, args=(index,)) for index in range(4)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=5)
                self.assertFalse(worker.is_alive())
        self.assertCountEqual(records, [f"worker-{index} done" for index in range(4)])

    def test_multiline_errors_and_filename_control_characters_keep_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vela.log"
            with RunLog(path, console=io.StringIO()) as log:
                log.task("sample\nforged\tname.mp4", 1, 1).emit("处理日志", "Traceback\n  frame\nRuntimeError\n")
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            for line in lines:
                self.assertRegex(line, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} Vela sample\\nforged\\tname.mp4 \[1/1\] 处理日志 ")

    def test_explicit_flush_and_closed_console_does_not_lose_file(self) -> None:
        records: list[str] = []
        with capture_output(records.append):
            sys.stdout.write("partial")
            sys.stdout.flush()
            self.assertEqual(records, ["partial"])
            sys.stdout.write("\n")
        self.assertEqual(records, ["partial"])

        class BrokenConsole(io.StringIO):
            def write(self, text: str) -> int:
                raise BrokenPipeError("reader detached")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vela.log"
            with RunLog(path, console=BrokenConsole()) as log:
                log.emit("批次结束", "成功=1")
            self.assertIn("成功=1", path.read_text(encoding="utf-8"))

    def test_formats_keep_units_unknowns_and_fractional_fps(self) -> None:
        self.assertEqual(format_bitrate(5_000_000), "5.000Mbps")
        self.assertEqual(format_bitrate(None), "无法获取")
        self.assertEqual(format_bitrate(0), "无法获取")
        self.assertEqual(format_size(38_700_000), "38.70MB")
        self.assertEqual(format_size(6_700_000_000), "6.70GB")
        self.assertEqual(format_size(None), "未记录")
        self.assertEqual(format_duration(6240), "1小时44分")
        self.assertEqual(format_duration(81), "1分21秒")
        self.assertEqual(format_duration(0.25), "0.25秒")
        self.assertEqual(format_duration(59.999), "1分")
        self.assertEqual(format_duration(0), "0秒")
        self.assertEqual(format_fps(59.94), "59.94fps")
        self.assertEqual(format_fps(None), "无法获取")
        self.assertEqual(format_fps(float("nan")), "无法获取")

    def test_native_stderr_is_prefixed_and_written_to_the_same_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vela.log"
            console = io.StringIO()
            with RunLog(path, console=console) as log:
                with log.task("sample.mp4", 1, 1).capture():
                    os.write(2, "native 诊断\rprogress 2\npartial".encode("utf-8"))
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertTrue(all("Vela sample.mp4 [1/1] 处理日志 " in line for line in lines))
            self.assertTrue(lines[0].endswith("native 诊断"))
            self.assertTrue(lines[-1].endswith("partial"))
            self.assertEqual(console.getvalue(), path.read_text(encoding="utf-8"))

    def test_nested_native_capture_and_exception_restore_fd_and_python_streams(self) -> None:
        outer: list[str] = []
        inner: list[str] = []
        fds_before = [(os.fstat(fd), os.get_inheritable(fd)) for fd in (1, 2)]
        string_stdout, string_stderr = io.StringIO(), io.StringIO()
        with patch.object(sys, "stdout", string_stdout), patch.object(sys, "stderr", string_stderr):
            with capture_output(outer.append):
                os.write(2, b"outer before\n")
                os.write(1, b"outer stdout\n")
                with self.assertRaisesRegex(RuntimeError, "intentional"):
                    with capture_output(inner.append):
                        os.write(2, b"inner partial")
                        os.write(1, b"inner stdout\n")
                        print("inner Python", file=sys.stderr)
                        raise RuntimeError("intentional")
                os.write(2, b"outer after\n")
            self.assertIs(sys.stdout, string_stdout)
            self.assertIs(sys.stderr, string_stderr)
        self.assertCountEqual(outer, ["outer before", "outer after", "outer stdout"])
        self.assertCountEqual(inner, ["inner partial", "inner Python", "inner stdout"])
        for fd, (before, inheritable) in zip((1, 2), fds_before):
            after = os.fstat(fd)
            self.assertEqual((before.st_dev, before.st_ino), (after.st_dev, after.st_ino))
            self.assertEqual(os.get_inheritable(fd), inheritable)
        self.assertFalse(any(thread.name.startswith("vela-native-") for thread in threading.enumerate()))

    def test_native_capture_can_be_disabled_without_touching_fd2(self) -> None:
        records: list[str] = []
        with patch("vela.runtime_log.os.dup", side_effect=AssertionError("fd 2 should not be touched")):
            with capture_output(records.append, capture_native_stderr=False):
                print("Python only", file=sys.stderr)
        self.assertEqual(records, ["Python only"])

    def test_owned_console_duplicate_does_not_close_callers_stream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "console.txt").open("w", encoding="utf-8") as console:
                with RunLog(root / "vela.log", console=console) as log:
                    with log.task("sample.mp4", 1, 1).capture():
                        os.write(1, b"native stdout\n")
                        os.write(2, b"native stderr\n")
                self.assertFalse(console.closed)
                self.assertEqual((root / "vela.log").read_text(encoding="utf-8"), (root / "console.txt").read_text(encoding="utf-8"))
                console.write("still open\n")

    def test_real_cli_native_stdout_stderr_match_file_without_tee_recursion(self) -> None:
        # Exercise the real command dispatcher in a fresh process with an
        # injected probe failure, so no video is decoded or model is loaded.
        script = """
import os
import sys
from vela import cli

def probe_fixture(_path):
    os.write(1, b'native stdout fixture\\n')
    os.write(2, b'native stderr fixture\\n')
    print('Python stdout fixture')
    print('Python stderr fixture', file=sys.stderr)
    raise RuntimeError('fixture metadata failure')

cli.probe_video = probe_fixture
raise SystemExit(cli.main(['-i', 'sample.mp4']))
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.mp4").write_bytes(b"not a video; probe is injected")
            env = os.environ.copy()
            repository = Path(__file__).resolve().parents[1]
            env["PYTHONPATH"] = str(repository) + os.pathsep + env.get("PYTHONPATH", "")
            env["PYTHONIOENCODING"] = "utf-8"
            completed = subprocess.run(
                [sys.executable, "-c", script], cwd=root, env=env,
                capture_output=True, text=True, encoding="utf-8", timeout=15, check=False,
            )
            self.assertEqual(completed.returncode, 1, completed.stderr)
            logs = list((root / ".tmp/restoration-order/private/runtime-logs").glob("*/vela.log"))
            self.assertEqual(len(logs), 1)
            text = logs[0].read_text(encoding="utf-8")
            self.assertEqual(text, completed.stdout)
            self.assertEqual(completed.stderr, "")
            for message in ("native stdout fixture", "native stderr fixture", "Python stdout fixture", "Python stderr fixture"):
                self.assertEqual(text.count(f"Vela sample.mp4 [1/1] 处理日志 {message}\n"), 1)
            self.assertIn("[批次] 批次结束", text)


if __name__ == "__main__":
    unittest.main()
