"""Synthetic resampling diagnostic; no private media or neural inference.

Run with: python scripts/diagnose_resampling_order.py --run-date 2026-10-28
Dependencies: numpy, opencv-python, Pillow.

The direct target mapping is a comparison reference, not ground truth for a
restoration model. OpenCV CPU interpolation approximates the geometric paths;
it is not a byte-exact implementation of another application's GPU kernels.
"""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import platform
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import PIL


CANVAS = 512
SCALE = 2.25
SOURCE_SIDES = (96, 160, 224, 320, 480)
FREQUENCIES = (8, 24, 48, 64, 96)


def quantize(image: np.ndarray) -> np.ndarray:
    return np.rint(image).clip(0, 255).astype(np.uint8)


def map_canvas(canvas: np.ndarray, side: int) -> np.ndarray:
    interpolation = cv2.INTER_CUBIC if side > CANVAS else cv2.INTER_AREA
    return quantize(cv2.resize(canvas.astype(np.float32), (side, side), interpolation=interpolation))


def round_trip(canvas: np.ndarray, source_side: int, target_side: int) -> np.ndarray:
    pasted = map_canvas(canvas, source_side)
    return quantize(cv2.resize(pasted.astype(np.float32), (target_side, target_side), interpolation=cv2.INTER_CUBIC))


def sinusoid(cycles: int) -> np.ndarray:
    x = (np.arange(CANVAS, dtype=np.float64) + 0.5) / CANVAS
    row = quantize(127.5 + 100.0 * np.sin(2.0 * np.pi * cycles * x))
    return np.broadcast_to(row, (CANVAS, CANVAS)).copy()


def fitted_amplitude(image: np.ndarray, cycles: int) -> float:
    row = image.astype(np.float64).mean(axis=0)
    side = image.shape[1]
    edge = max(4, int(side * 0.05))
    x = (np.arange(side, dtype=np.float64) + 0.5) / side
    basis = np.stack([np.sin(2 * np.pi * cycles * x), np.cos(2 * np.pi * cycles * x), np.ones(side)], axis=1)
    coefficients = np.linalg.lstsq(basis[edge:-edge], row[edge:-edge], rcond=None)[0]
    return float(np.hypot(coefficients[0], coefficients[1]))


def test_chart() -> np.ndarray:
    y, x = np.mgrid[0:CANVAS, 0:CANVAS].astype(np.float64)
    xn, yn = (x + 0.5) / CANVAS, (y + 0.5) / CANVAS
    chart = np.empty((CANVAS, CANVAS), np.float64)
    top_left = (x < CANVAS / 2) & (y < CANVAS / 2)
    top_right = (x >= CANVAS / 2) & (y < CANVAS / 2)
    bottom_left = (x < CANVAS / 2) & (y >= CANVAS / 2)
    bottom_right = (x >= CANVAS / 2) & (y >= CANVAS / 2)
    chart[top_left] = (127.5 + 100 * np.sin(2 * np.pi * 64 * xn))[top_left]
    chart[top_right] = (127.5 + 90 * np.sin(2 * np.pi * (48 * xn + 32 * yn)))[top_right]
    chart[bottom_left] = (127.5 + 100 * np.sin(2 * np.pi * (8 * xn + 100 * xn * xn)))[bottom_left]
    chart[bottom_right] = (40 + 150 * (xn + yn) / 2 + 25 * np.sin(2 * np.pi * 12 * yn))[bottom_right]
    return quantize(chart)


def error_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(delta * delta))
    return {"rmse_u8": float(np.sqrt(mse)), "psnr_db": float(10 * np.log10(255 * 255 / mse))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".tmp/restoration-order/public-evidence"))
    parser.add_argument("--run-date", default=date.today().isoformat(), type=lambda value: date.fromisoformat(value).isoformat())
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    canvas = test_chart()
    Image.fromarray(canvas).save(output / "synthetic_canvas.png")
    cases = []
    comparisons = []
    for source_side in SOURCE_SIDES:
        target_side = round(source_side * SCALE)
        direct = map_canvas(canvas, target_side)
        via_source = round_trip(canvas, source_side, target_side)
        case = {
            "source_roi_side": source_side,
            "target_roi_side": target_side,
            "target_scale": SCALE,
            "chart_difference_vs_direct_mapping": error_metrics(via_source, direct),
            "frequency_response": [],
        }
        for cycles in FREQUENCIES:
            wave = sinusoid(cycles)
            wave_direct = map_canvas(wave, target_side)
            wave_via_source = round_trip(wave, source_side, target_side)
            amplitude_direct = fitted_amplitude(wave_direct, cycles)
            amplitude_via_source = fitted_amplitude(wave_via_source, cycles)
            case["frequency_response"].append({
                "cycles_per_canvas_width": cycles,
                "below_source_nyquist": cycles < source_side / 2,
                "below_target_nyquist": cycles < target_side / 2,
                "direct_target_amplitude": amplitude_direct,
                "via_source_amplitude": amplitude_via_source,
                "amplitude_ratio_vs_direct": amplitude_via_source / amplitude_direct,
            })
        cases.append(case)
        if source_side in (160, 224):
            Image.fromarray(direct).save(output / f"roi_{source_side}_direct_target.png")
            Image.fromarray(via_source).save(output / f"roi_{source_side}_via_source.png")
            comparisons.append((source_side, target_side, direct, via_source))

    panel_width, panel_height = 1060, 1180
    panel = Image.new("RGB", (panel_width, panel_height), "#f4f4f4")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default(size=20)
    draw.text((20, 15), "Synthetic resampling only; no model inference", font=font, fill="#111111")
    draw.text((20, 45), "Direct target mapping", font=font, fill="#111111")
    draw.text((540, 45), "Source-size paste, then bicubic", font=font, fill="#111111")
    y = 85
    for source_side, target_side, direct, via_source in comparisons:
        draw.text((20, y), f"Canvas 512; source ROI {source_side}; target ROI {target_side}", font=font, fill="#111111")
        y += 32
        # Display native pixels, without rescaling evidence images.
        panel.paste(Image.fromarray(direct).convert("RGB"), (20, y))
        panel.paste(Image.fromarray(via_source).convert("RGB"), (540, y))
        y += target_side + 25
    draw.text((20, y), "Reference = direct geometric mapping, not a neural-model ground truth.", font=font, fill="#111111")
    panel.crop((0, 0, panel_width, y + 45)).save(output / "comparison.png")
    report = {
        "experiment": "fixed_canvas_resampling_loss",
        "run_date": args.run_date,
        "canvas_side": CANVAS,
        "synthetic_only": True,
        "neural_model_inference": False,
        "remote_gpu_used": False,
        "methods": {
            "direct": "512 canvas -> target ROI; AREA when shrinking, bicubic when enlarging",
            "via_source": "512 canvas -> source ROI using AREA -> target ROI using bicubic",
            "quantization": "round and clip to uint8 after each mapping",
            "frequency_metric": "least-squares sine/cosine amplitude at known frequency, excluding 5 percent edges",
            "chart_error_reference": "direct target mapping of the same synthetic 512 canvas",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "pillow": PIL.__version__,
            "device": "CPU",
        },
        "limitations": [
            "The same hypothetical restored canvas is reused in both paths; changed model conditioning and crops are not represented.",
            "Bicubic is not RealPLKSR; no neural restoration or super-resolution weights are used.",
            "OpenCV CPU interpolation is not byte-exact to a GPU implementation.",
            "The diagnostic does not measure detection accuracy, temporal consistency, encoding loss, latency, or VRAM.",
            "These results cannot determine the best end-to-end model order for a real video.",
        ],
        "cases": cases,
    }
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"cases": len(cases), "outputs": ["results.json", "comparison.png"], "neural_model_inference": False}))


if __name__ == "__main__":
    main()
