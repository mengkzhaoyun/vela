"""Single-file batch logging for native Vela runs and worker output."""
from __future__ import annotations

import codecs
import io
import math
import os
import sys
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import TextIO


# Redirecting process streams/fds is global. Nested captures on the owning
# thread are allowed; concurrent contexts must not restore each other's fds.
_CAPTURE_LOCK = threading.RLock()


def format_bitrate(bps: int | None) -> str:
    """Display measured bit/s without converting unknown values into zero."""
    if bps is None or not math.isfinite(bps) or bps <= 0:
        return "无法获取"
    return f"{bps / 1_000_000:.3f}Mbps"


def format_size(size_bytes: int | None) -> str:
    """Use decimal units consistently with the Mbps display."""
    if size_bytes is None or not math.isfinite(size_bytes) or size_bytes < 0:
        return "未记录"
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.2f}GB"
    return f"{size_bytes / 1_000_000:.2f}MB"


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "未记录"
    # Round before carrying into minutes/hours (59.999 seconds -> 1 minute).
    centiseconds = round(seconds * 100)
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    seconds_text = f"{remainder / 100:.2f}".rstrip("0").rstrip(".")
    parts = []
    if hours:
        parts.append(f"{hours}小时")
    if minutes:
        parts.append(f"{minutes}分")
    if remainder or not parts:
        parts.append(f"{seconds_text}秒")
    return "".join(parts)


def format_fps(fps: float | None) -> str:
    if fps is None or not math.isfinite(fps) or fps <= 0:
        return "无法获取"
    return f"{fps:.3f}".rstrip("0").rstrip(".") + "fps"


def _label(value: object) -> str:
    """Prevent a filename or event label from creating a forged log record."""
    escaped = []
    for char in str(value):
        if char == "\n":
            escaped.append(r"\n")
        elif char == "\r":
            escaped.append(r"\r")
        elif char == "\t":
            escaped.append(r"\t")
        elif ord(char) < 32 or 127 <= ord(char) < 160 or char in "\u2028\u2029":
            escaped.append(f"\\u{ord(char):04x}")
        else:
            escaped.append(char)
    return "".join(escaped)


class RunLog:
    """Own the one log file; workers send their output to this process."""

    def __init__(self, log_file: Path | None = None, console: TextIO | None = None) -> None:
        if log_file is None:
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:12]
            log_file = Path(".tmp/restoration-order/private/runtime-logs") / run_id / "vela.log"
        self.path = Path(log_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("x", encoding="utf-8", newline="\n", buffering=1)
        self._console = sys.stdout if console is None else console
        self._owns_console = False
        try:
            console_fd = self._console.fileno()
        except (AttributeError, io.UnsupportedOperation, OSError, ValueError):
            # StringIO/test consoles are already independent of native fds.
            pass
        else:
            # Keep an independent handle to the original console. Native
            # stdout/stderr capture must never recapture our prefixed tee.
            duplicate: int | None = None
            try:
                self._console.flush()
                duplicate = os.dup(console_fd)
                self._console = os.fdopen(
                    duplicate, "w", encoding=self._console.encoding or "utf-8",
                    errors=self._console.errors or "replace", newline="\n", buffering=1,
                )
                self._owns_console = True
            except BaseException:
                if duplicate is not None:
                    os.close(duplicate)
                self._file.close()
                raise
        self._lock = threading.RLock()

    def __enter__(self) -> RunLog:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            try:
                self._file.close()
            finally:
                if self._owns_console:
                    try:
                        self._console.close()
                    except (BrokenPipeError, UnicodeEncodeError):
                        pass

    def emit(
        self,
        event: str,
        message: str,
        filename: str = "-",
        index: int = 0,
        total: int = 0,
    ) -> None:
        with self._lock:
            if self._file.closed:
                raise ValueError("Cannot write to a closed Vela run log")
            task = f"{index}/{total}" if index else "批次"
            prefix = f"{datetime.now():%Y-%m-%d %H:%M:%S} Vela {_label(filename)} [{task}] {_label(event)}"
            lines = str(message).replace("\r\n", "\n").replace("\r", "\n").split("\n")
            if len(lines) > 1 and not lines[-1]:
                lines.pop()
            for line in lines:
                record = f"{prefix} {line}\n"
                self._file.write(record)
                self._file.flush()
                try:
                    self._console.write(record)
                    self._console.flush()
                except (BrokenPipeError, UnicodeEncodeError):
                    # A detached or legacy console must not discard the UTF-8 log.
                    pass

    def task(self, filename: str, index: int, total: int) -> TaskLog:
        if index < 1 or total < index:
            raise ValueError("Task index must be between 1 and total")
        return TaskLog(self, filename, index, total)


class TaskLog:
    def __init__(self, run: RunLog, filename: str, index: int, total: int) -> None:
        self._run = run
        self.filename = filename
        self.index = index
        self.total = total

    def emit(self, event: str, message: str) -> None:
        self._run.emit(event, message, self.filename, self.index, self.total)

    @contextmanager
    def capture(self) -> Iterator[None]:
        with capture_output(lambda line: self.emit("处理日志", line)):
            yield


class _LineCapture(io.TextIOBase):
    """Buffer partial writes separately per thread, forwarding complete lines."""

    def __init__(self, emit: Callable[[str], None]) -> None:
        super().__init__()
        self._emit = emit
        self._buffers: dict[int, str] = {}
        self._lock = threading.RLock()

    @property
    def encoding(self) -> str:
        return "utf-8"

    @property
    def errors(self) -> str:
        return "replace"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("write() argument must be str")
        with self._lock:
            thread_id = threading.get_ident()
            buffered = self._buffers.get(thread_id, "") + text
            lines = buffered.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            self._buffers[thread_id] = lines.pop()
            for line in lines:
                if line:
                    self._emit(line)
        return len(text)

    def flush(self) -> None:
        # Explicit flush must expose partial messages immediately. Empty writes
        # and the LF following an already-flushed CR do not create blank records.
        with self._lock:
            for thread_id, line in list(self._buffers.items()):
                if line:
                    self._buffers[thread_id] = ""
                    self._emit(line)


@contextmanager
def _capture_native_stream(stream_fd: int, emit: Callable[[str], None]) -> Iterator[None]:
    """Capture C-library output that bypasses Python's stream objects."""
    stream_name = "stdout" if stream_fd == 1 else "stderr"
    saved_fd: int | None = None
    read_fd: int | None = None
    write_fd: int | None = None
    saved_inheritable = True
    redirected = False
    reader: threading.Thread | None = None
    errors: list[BaseException] = []
    stop = threading.Event()

    def read_stream(fd: int) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        lines = _LineCapture(emit)
        try:
            with os.fdopen(fd, "rb", buffering=0) as source:
                while chunk := source.read(8192):
                    if stop.is_set():
                        return
                    # Continue draining after a sink error, otherwise a full
                    # pipe could deadlock the native library writing to it.
                    if not errors:
                        try:
                            lines.write(decoder.decode(chunk))
                        except BaseException as exc:
                            errors.append(exc)
                if not errors and not stop.is_set():
                    lines.write(decoder.decode(b"", final=True))
                    lines.flush()
        except BaseException as exc:
            errors.append(exc)

    try:
        # This intentionally does not call sys.std*.fileno(): test runners
        # and notebooks may replace the Python stream with a StringIO object.
        saved_fd = os.dup(stream_fd)
        saved_inheritable = os.get_inheritable(stream_fd)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, stream_fd)
        redirected = True
        os.close(write_fd)
        write_fd = None
        reader = threading.Thread(target=read_stream, args=(read_fd,), name=f"vela-native-{stream_name}", daemon=True)
        reader.start()
        read_fd = None  # The reader owns and closes this descriptor.
        yield
    finally:
        pending_exception = sys.exc_info()[0] is not None
        try:
            if redirected and saved_fd is not None:
                # Replacing the fd closes our pipe writer, making EOF available
                # before joining. An inner capture restores the outer pipe.
                os.dup2(saved_fd, stream_fd, inheritable=saved_inheritable)
        finally:
            if saved_fd is not None:
                os.close(saved_fd)
            if write_fd is not None:
                os.close(write_fd)
            if read_fd is not None:
                os.close(read_fd)
            if reader is not None and reader.ident is not None:
                reader.join(timeout=5.0)
                if reader.is_alive():
                    # A surviving child may still own an inherited pipe end.
                    # Do not hang teardown or emit into a subsequently closed
                    # log; the daemon closes its reader when it next receives
                    # data/EOF. Native child processes should be joined first.
                    stop.set()
                    if not pending_exception:
                        raise RuntimeError(f"Native {stream_name} capture did not close; a child process still owns {stream_name}")
        if errors and not pending_exception:
            raise RuntimeError(f"Failed to capture native {stream_name}") from errors[0]


@contextmanager
def capture_output(
    emit: Callable[[str], None], *, capture_native_stderr: bool = True,
    capture_native_stdout: bool | None = None,
) -> Iterator[None]:
    """Capture Python/native streams; emit must use an independent sink.

    RunLog owns a duplicate console fd so its tee cannot feed back into this
    capture. Queue/list callbacks are also safe. For compatibility, disabling
    native stderr disables native stdout unless explicitly overridden.
    """
    with _CAPTURE_LOCK:
        if capture_native_stdout is None:
            capture_native_stdout = capture_native_stderr
        if os.name == "posix" and (capture_native_stdout or capture_native_stderr):
            # Queue/Event creation starts a process-wide resource tracker on
            # POSIX. It outlives individual video tasks and inherits stderr.
            # Start it before redirecting fd 2 so it cannot keep a task's
            # capture pipe open after all GPU workers have already exited.
            from multiprocessing import resource_tracker

            resource_tracker.ensure_running()
        stdout = _LineCapture(emit)
        stderr = _LineCapture(emit)
        native_stdout = _capture_native_stream(1, emit) if capture_native_stdout else nullcontext()
        native_stderr = _capture_native_stream(2, emit) if capture_native_stderr else nullcontext()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr), native_stdout, native_stderr:
                yield
        finally:
            stdout.flush()
            stderr.flush()
