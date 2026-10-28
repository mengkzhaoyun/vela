"""Measure LTX denoise with random placeholder weights and synthetic latents.

No video input, learned restoration weights, VAE decoding, or image output is used.
Run --dry-run to inspect the workload without importing GPU libraries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path


MODELS = ("distilled2", "distilled4", "distilled", "undistilled")
LIMITATIONS = [
    "Random placeholder weights measure computation, not restoration quality.",
    "Detection, VAE encoding/decoding, compositing and file codecs are excluded.",
    "Windows per second is component throughput, not full-video frames per second.",
    "Setup includes imports, device initialization and placeholder construction, not learned-weight file loading.",
    "Warmup includes full sampler calls and any cache-dependent compilation; it is not a clean-cache compile benchmark.",
    "Peak allocated/reserved bytes cover the PyTorch allocator; native/runtime allocations may be outside it.",
]


class BenchmarkError(RuntimeError):
    """An actionable error whose message contains no local configuration paths."""


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jasna-root", type=Path, help="Optional Jasna source checkout; otherwise use the installed package")
    parser.add_argument("--output", type=Path, help="New JSON result file; required unless --dry-run")
    parser.add_argument("--gpu", type=nonnegative_int, default=0, help="Visible CUDA device index (default: 0)")
    parser.add_argument("--model", choices=MODELS, default="distilled2")
    parser.add_argument("--canvas", type=int, choices=(512, 768), default=512)
    parser.add_argument("--windows", type=positive_int, default=3)
    parser.add_argument("--warmup", type=positive_int, default=1)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Print the configuration only; do not import GPU libraries or write results")
    return parser


def configuration(args: argparse.Namespace) -> dict:
    return {
        "gpu": args.gpu,
        "model": args.model,
        "canvas": args.canvas,
        "windows": args.windows,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "quant": "int8",
        "weights": "random-placeholder",
    }


def measure(args: argparse.Namespace) -> dict:
    setup_started = time.perf_counter()
    try:
        import torch
    except ImportError as exc:
        raise BenchmarkError("PyTorch is unavailable in the selected Python environment.") from exc
    if not torch.cuda.is_available():
        raise BenchmarkError("A CUDA GPU is required; CPU fallback is disabled.")
    if args.gpu >= torch.cuda.device_count():
        raise BenchmarkError("The requested visible CUDA device does not exist.")
    if args.seed + args.windows - 1 > 2**63 - 1:
        raise BenchmarkError("Seed plus window count exceeds the supported range.")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    if args.jasna_root is not None:
        root = args.jasna_root.resolve()
        if not (root / "jasna" / "__init__.py").is_file():
            raise BenchmarkError("The supplied Jasna source root does not contain the package.")
        sys.path.insert(0, str(root))
    try:
        import jasna
        from jasna._suppress_noise import install as install_noise_filters
        from jasna.accelerator import device_free_bytes
        from jasna.engine_paths import configure_torch_compile_cache
        from jasna.ltx import transformer as architecture
        from jasna.ltx.model_files import LtxModelFiles, PlaceholderTensors
        from jasna.ltx.plan import LATENT_FRAMES, WINDOW_HOP
        from jasna.ltx.sampler import LtxTransformer, TransformerSource
    except ImportError as exc:
        raise BenchmarkError("Jasna or a required benchmark dependency is unavailable; use its complete Python environment.") from exc

    install_noise_filters()
    configure_torch_compile_cache()
    placeholder = LtxModelFiles.placeholder(args.model, fast=False).transformer
    if not isinstance(placeholder, PlaceholderTensors) or placeholder.part != "int8":
        raise BenchmarkError("The installed Jasna did not provide the required random INT8 placeholder.")

    transformer = None
    try:
        transformer = LtxTransformer(TransformerSource(placeholder), device, device_free_bytes)
        side = args.canvas // architecture.LATENT_SPATIAL_SCALE
        expected_shape = (1, 128, LATENT_FRAMES, side, side)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        references = [
            torch.randn(expected_shape, device=device, generator=generator, dtype=torch.bfloat16)
            for _ in range(args.windows)
        ]
        seeds = list(range(args.seed, args.seed + args.windows))
        steps = len(transformer.conditions)
        if not steps:
            raise BenchmarkError("The sampler has no denoising steps.")
        torch.cuda.synchronize(device)
        setup_seconds = time.perf_counter() - setup_started

        def timed_chain() -> float:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            outputs = transformer.denoise_chain(references, seeds, WINDOW_HOP, lambda _n: None)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise BenchmarkError("The measured duration is invalid.")
            if len(outputs) != args.windows or any(tuple(item.shape) != expected_shape for item in outputs):
                raise BenchmarkError("The sampler output does not match the requested workload.")
            if any(not bool(torch.isfinite(item).all().item()) for item in outputs):
                raise BenchmarkError("The sampler produced non-finite latent values.")
            return elapsed

        warmup_seconds = sum(timed_chain() for _ in range(args.warmup))
        torch.cuda.reset_peak_memory_stats(device)
        repeat_seconds = [timed_chain() for _ in range(args.repeats)]
        median = statistics.median(repeat_seconds)
        config = configuration(args)
        config.update({
            "steps": steps,
            "stg": bool(transformer.stg_scale),
            "backend": transformer.block_backend,
            "resident_blocks": int(transformer.blocks.resident),
            "total_blocks": int(architecture.NUM_BLOCKS),
            "window_hop": int(WINDOW_HOP),
        })
        return {
            "schema_version": 1,
            "benchmark": "ltx_trial_denoise",
            "measured": True,
            "device": {
                "type": "cuda",
                "name": torch.cuda.get_device_name(device),
                "memory_total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            },
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "jasna": str(jasna.__version__),
            },
            "config": config,
            "timings": {
                "setup_seconds": setup_seconds,
                "warmup_seconds": warmup_seconds,
                "repeats_seconds": repeat_seconds,
                "median_seconds": median,
            },
            "workload": {"canvas": args.canvas, "windows": args.windows, "latent_frames": int(LATENT_FRAMES)},
            "metrics": {"units_per_second": args.windows / median, "unit": "windows"},
            "memory": {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "native_vram_bytes_at_end": int(transformer.native_vram_bytes()),
            },
            "provenance": {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "limitations": LIMITATIONS,
        }
    finally:
        if transformer is not None:
            transformer.close()


def write_result(path: Path, report: dict) -> None:
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
        # Publish without replacing a report created by another concurrent run.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seed + args.windows - 1 > 2**63 - 1:
        parser.error("--seed plus --windows exceeds the supported range")
    if args.dry_run:
        print(json.dumps({"schema_version": 1, "benchmark": "ltx_trial_denoise", "measured": False,
                          "config": configuration(args), "limitations": LIMITATIONS}, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required for a measured run")
    try:
        if args.output.exists():
            raise BenchmarkError("The output already exists; choose a new result file.")
        report = measure(args)
        write_result(args.output, report)
    except BenchmarkError as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Benchmark failed ({type(exc).__name__}); no new result was written.", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
