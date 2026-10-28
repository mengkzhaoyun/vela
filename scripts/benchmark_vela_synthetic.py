"""Measure Vela on generated RGB tensors; no media input or video output.

Run inside the Vela GPU environment. --dry-run needs only the standard library,
does not inspect weights, and prints a configuration with measured=false.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time


class BenchmarkError(Exception):
    """A diagnostic safe to publish without local paths or identifiers."""


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def nonnegative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=nonnegative_int, default=0)
    parser.add_argument("--width", type=positive_int, default=854)
    parser.add_argument("--height", type=positive_int, default=480)
    parser.add_argument("--target-width", type=positive_int, default=1920)
    parser.add_argument("--target-height", type=positive_int, default=1080)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--warmup", type=positive_int, default=2)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--backend", choices=("auto", "eager", "compile", "trt"), default="auto")
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def configuration(args: argparse.Namespace) -> dict:
    return {
        "gpu_index": args.gpu,
        "requested_backend": args.backend,
        "warmup_batches": args.warmup,
        "repeat_batches": args.repeats,
        "seed": args.seed,
        "cas_strength": 0.0,
        "synthetic_input": True,
    }


def workload(args: argparse.Namespace) -> dict:
    return {
        "batch_size": args.batch_size,
        "input_width": args.width,
        "input_height": args.height,
        "target_width": args.target_width,
        "target_height": args.target_height,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_benchmark(args: argparse.Namespace) -> dict:
    if args.model_path is None or not args.model_path.is_file() or args.model_path.stat().st_size == 0:
        raise BenchmarkError("An existing, nonempty model file is required.")
    if args.output is None:
        raise BenchmarkError("An output JSON destination is required.")
    if args.output.exists():
        raise BenchmarkError("Output already exists; choose a new report destination.")

    # Resolve this checkout before importing Vela; do not use an unrelated install.
    repo = Path(__file__).resolve().parent.parent
    if not (repo / "vela" / "__init__.py").is_file():
        raise BenchmarkError("The script must be located in the Vela checkout.")
    sys.path.insert(0, str(repo))
    import torch
    import vela
    from vela.models import upscaler as upscaler_module

    if not torch.cuda.is_available():
        raise BenchmarkError("CUDA is unavailable; CPU fallback is disabled.")
    if args.gpu >= torch.cuda.device_count():
        raise BenchmarkError("The requested GPU index is unavailable.")
    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    model_digest = file_sha256(args.model_path)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    inputs = torch.randint(
        0, 256, (args.batch_size, args.height, args.width, 3),
        dtype=torch.uint8, generator=generator, pin_memory=True,
    )

    previous_backend = os.environ.get("VELA_BACKEND")
    os.environ["VELA_BACKEND"] = args.backend
    try:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        upscaler = upscaler_module.VelaUpscaler(
            model_path=args.model_path, device=device,
            compile_model=args.backend != "eager", batch_size=args.batch_size,
        )
        torch.cuda.synchronize(device)
        setup_seconds = time.perf_counter() - started
        if upscaler.device.type != "cuda" or upscaler.model is None:
            raise BenchmarkError("A real CUDA model was not loaded; resize-only fallback is disabled.")
        required = {
            "eager": "pytorch-eager", "compile": "torch-compile-fp16", "trt": "tensorrt-fp16",
        }.get(args.backend)
        if required and upscaler.backend != required:
            raise BenchmarkError("The requested inference backend is unavailable; fallback is disabled.")

        expected_shape = (args.batch_size, args.target_height, args.target_width, 3)

        def timed_batch() -> float:
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = upscaler.process_batch_rgb24(
                inputs, target_w=args.target_width, target_h=args.target_height, cas_strength=0.0,
            )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            if tuple(output.shape) != expected_shape or output.dtype != torch.uint8 or output.device.type != "cpu":
                raise BenchmarkError("Output shape, dtype, or transfer validation failed.")
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise BenchmarkError("Invalid timing sample; no result was recorded.")
            return elapsed

        warmup_seconds = sum(timed_batch() for _ in range(args.warmup))
        torch.cuda.reset_peak_memory_stats(device)
        repeats_seconds = [timed_batch() for _ in range(args.repeats)]
        median_seconds = statistics.median(repeats_seconds)
        config = configuration(args)
        config["actual_backend"] = upscaler.backend
        return {
            "schema_version": 1, "benchmark": "vela_synthetic_sr", "measured": True,
            "device": {"type": "cuda", "name": torch.cuda.get_device_name(device),
                       "memory_total_bytes": properties.total_memory},
            "software": {"python": platform.python_version(), "torch": torch.__version__,
                         "cuda": torch.version.cuda, "vela": vela.__version__},
            "config": config, "workload": workload(args),
            "timings": {"setup_seconds": setup_seconds, "warmup_seconds": warmup_seconds,
                        "repeats_seconds": repeats_seconds, "median_seconds": median_seconds},
            "metrics": {"units_per_second": args.batch_size / median_seconds, "unit": "frames"},
            "memory": {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)},
            "provenance": {"script_sha256": file_sha256(Path(__file__)), "model_sha256": model_digest,
                           "vela_upscaler_sha256": file_sha256(Path(upscaler_module.__file__))},
            "limitations": [
                "Generated RGB tensors only; no media files are read or written.",
                "Component throughput includes host/device transfers, SR, resize, and uint8 conversion; CAS is disabled.",
                "Decode, encode, scheduling, and disk I/O are excluded; FPS is not end-to-end video throughput.",
                "Setup measures model construction; lazy compilation and cache warmup occur in the warmup batches.",
                "Memory peaks cover timed batches, include resident model memory, and exclude setup/warmup peaks.",
                "PyTorch allocator peaks exclude TensorRT/native allocations and are not total driver memory usage.",
                "The model-file hash does not verify the weights inside a pre-existing TensorRT engine cache.",
                "This benchmark does not measure perceptual quality, temporal stability, or pipeline-order superiority.",
            ],
        }
    finally:
        if previous_backend is None:
            os.environ.pop("VELA_BACKEND", None)
        else:
            os.environ["VELA_BACKEND"] = previous_backend


def write_report(path: Path, report: dict) -> None:
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise BenchmarkError("Output already exists; choose a new report destination.")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=".benchmark-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        # Hard-link publication is atomic and refuses to replace an existing report.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seed > 2**63 - 1:
        parser.error("seed must be at most 2**63-1")
    if args.dry_run:
        print(json.dumps({"schema_version": 1, "benchmark": "vela_synthetic_sr", "measured": False,
                          "config": configuration(args), "workload": workload(args)}, allow_nan=False))
        return 0
    if args.model_path is None or args.output is None:
        parser.error("actual measurements require --model-path and --output")
    previous_log_level = logging.root.manager.disable
    try:
        # Third-party diagnostics can contain paths; publish only our structured result.
        logging.disable(logging.CRITICAL)
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            report = run_benchmark(args)
        write_report(args.output, report)
        print(json.dumps({"benchmark": report["benchmark"], "measured": True,
                          "median_seconds": report["timings"]["median_seconds"],
                          "frames_per_second": report["metrics"]["units_per_second"]}, allow_nan=False))
        return 0
    except BenchmarkError as error:
        print(json.dumps({"status": "failed", "message": str(error)}), file=sys.stderr)
        return 1
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "message": "Benchmark failed; no new measurement report was published."}), file=sys.stderr)
        return 1
    finally:
        logging.disable(previous_log_level)


if __name__ == "__main__":
    raise SystemExit(main())
