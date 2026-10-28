"""Validate component benchmark reports and produce a numerical review sheet.

Only measured reports are accepted. This does not compare image quality or infer
end-to-end video throughput from components with different units.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


BENCHMARKS = {
    "ltx_trial_denoise": ("LTX random-weight denoise", "windows"),
    "vela_synthetic_sr": ("Vela synthetic-input SR", "frames"),
}


def number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def validate(report: object) -> dict:
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError("unsupported report schema")
    if report.get("measured") is not True:
        raise ValueError("dry-run or unmeasured reports cannot be summarized as results")
    kind = report.get("benchmark")
    if not isinstance(kind, str) or kind not in BENCHMARKS:
        raise ValueError("unknown benchmark")
    for section in ("device", "config", "timings", "workload", "metrics", "memory"):
        if not isinstance(report.get(section), dict):
            raise ValueError(f"missing {section} section")
    if report["device"].get("type") != "cuda":
        raise ValueError("a measured GPU report must identify a CUDA device")
    times = report["timings"].get("repeats_seconds")
    if not isinstance(times, list) or not times:
        raise ValueError("at least one measured repeat is required")
    times = [number(value, "repeat duration", positive=True) for value in times]
    median = number(report["timings"].get("median_seconds"), "median", positive=True)
    if not math.isclose(median, statistics.median(times), rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("reported median does not match the repeats")
    expected_unit = BENCHMARKS[kind][1]
    if report["metrics"].get("unit") != expected_unit:
        raise ValueError("component throughput unit is inconsistent")
    count_key = "windows" if kind == "ltx_trial_denoise" else "batch_size"
    count = number(report["workload"].get(count_key), count_key, positive=True)
    if not count.is_integer():
        raise ValueError("workload count must be an integer")
    rate = number(report["metrics"].get("units_per_second"), "throughput", positive=True)
    if not math.isclose(rate, count / median, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("throughput does not match measured work and duration")
    settings = []
    if kind == "ltx_trial_denoise":
        canvas = report["workload"].get("canvas")
        if type(canvas) is not int or canvas not in (512, 768):
            raise ValueError("unsupported LTX canvas")
        steps = number(report["config"].get("steps"), "sampler steps", positive=True)
        if not steps.is_integer():
            raise ValueError("sampler steps must be an integer")
        if report["config"].get("weights") != "random-placeholder" or report["config"].get("quant") != "int8":
            raise ValueError("LTX trial configuration does not match this benchmark")
        settings = [("Canvas", f"{canvas} px"), ("Sampler steps", str(int(steps))), ("Weights", "Random INT8 placeholder")]
    else:
        backend = report["config"].get("actual_backend")
        if backend not in ("tensorrt-fp16", "torch-compile-fp16", "pytorch-eager"):
            raise ValueError("unsupported Vela backend")
        if report["config"].get("synthetic_input") is not True:
            raise ValueError("Vela report must identify synthetic input")
        dimensions = []
        for key in ("input_width", "input_height", "target_width", "target_height"):
            dimension = number(report["workload"].get(key), key, positive=True)
            if not dimension.is_integer():
                raise ValueError("image dimensions must be integers")
            dimensions.append(int(dimension))
        settings = [("Backend", backend), ("Input", f"{dimensions[0]} × {dimensions[1]}"),
                    ("Target", f"{dimensions[2]} × {dimensions[3]}")]
    return {
        "kind": kind,
        "title": BENCHMARKS[kind][0],
        "unit": expected_unit,
        "count": int(count),
        "settings": settings,
        "repeats": len(times),
        "minimum": min(times),
        "maximum": max(times),
        "median": median,
        "rate": rate,
        "setup": number(report["timings"].get("setup_seconds"), "setup duration"),
        "warmup": number(report["timings"].get("warmup_seconds"), "warmup duration"),
        "total_memory": number(report["device"].get("memory_total_bytes"), "device capacity", positive=True),
        "allocated": number(report["memory"].get("peak_allocated_bytes"), "peak allocation"),
        "reserved": number(report["memory"].get("peak_reserved_bytes"), "peak reservation"),
    }


def render(rows: list[dict]) -> str:
    lines = [
        "# Component performance review", "",
        "Measured component timings only. LTX windows/s and Vela frames/s have different scopes;",
        "they must not be ranked as alternative pipelines or converted into a quality verdict.", "",
    ]
    for index, row in enumerate(rows, 1):
        lines.extend([
            f"## {index}. {row['title']}", "",
            "| Metric | Value |", "| --- | --- |",
            *[f"| {key} | {value} |" for key, value in row["settings"]],
            f"| Setup | {row['setup']:.3f} s |",
            f"| Warmup total | {row['warmup']:.3f} s |",
            f"| Work per repeat | {row['count']} {row['unit']} |",
            f"| Measured repeats | {row['repeats']} |",
            f"| Median | {row['median']:.6f} s |",
            f"| Range | {row['minimum']:.6f}–{row['maximum']:.6f} s |",
            f"| Throughput | {row['rate']:.3f} {row['unit']}/s |",
            f"| Device capacity | {row['total_memory'] / 2**30:.3f} GiB |",
            f"| Peak allocated | {row['allocated'] / 2**30:.3f} GiB |",
            f"| Peak reserved | {row['reserved'] / 2**30:.3f} GiB |", "",
        ])
    lines.extend([
        f"## {len(rows) + 1}. Review boundary", "",
        "- These measurements exclude complete video restoration, detection, and file encoding.",
        "- LTX uses placeholder weights and synthetic latent inputs; no image-quality result is produced.",
        "- Vela uses a loaded SR model with synthetic RGB input; its timing includes the transfers documented by that runner.",
        "- GPU execution is not human acceptance. The reviewer must inspect the original configuration and limitations.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for index, path in enumerate(args.reports, 1):
        try:
            report = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            parser.error(f"could not read valid JSON for report #{index}")
        try:
            rows.append(validate(report))
        except ValueError as error:
            parser.error(f"report #{index}: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(rows), encoding="utf-8", newline="\n")
    print(json.dumps({"reports": len(rows), "quality_evaluated": False, "end_to_end_measured": False}))


if __name__ == "__main__":
    main()
