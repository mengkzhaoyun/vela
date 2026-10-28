"""Unit tests for Vela Multi-GPU VSR pipeline operators and GOP planner."""
from __future__ import annotations

from pathlib import Path

import torch

from vela.cli import VIDEO_EXTENSIONS, build_parser
from vela.config import VelaConfig
from vela.models.upscaler import VelaUpscaler, gpu_cas_sharpen, gpu_resize_to_target
from vela.pipeline.probe import VideoInfo, _parse_duration, _parse_fps, parse_bitrate_to_kbps, plan_segments
from vela.pipeline.worker import resolve_effective_fps


def test_plan_segments_exact_and_remainder(tmp_path: Path) -> None:
    tasks = plan_segments(duration_sec=65.0, segment_seconds=30.0, work_dir=tmp_path)
    assert len(tasks) == 3
    assert tasks[0].start_sec == 0.0 and tasks[0].duration_sec == 30.0
    assert tasks[1].start_sec == 30.0 and tasks[1].duration_sec == 30.0
    assert tasks[2].start_sec == 60.0 and tasks[2].duration_sec == 5.0


def test_gpu_resize_and_cas_sharpen() -> None:
    dummy = torch.rand((2, 3, 480, 854), dtype=torch.float32)
    resized = gpu_resize_to_target(dummy, target_h=1080, target_w=1920)
    assert resized.shape == (2, 3, 1080, 1920)
    assert float(resized.min()) >= 0.0 and float(resized.max()) <= 1.0

    sharpened = gpu_cas_sharpen(resized, strength=0.35)
    assert sharpened.shape == (2, 3, 1080, 1920)
    assert float(sharpened.min()) >= 0.0 and float(sharpened.max()) <= 1.0


def test_upscaler_fallback_batch_rgb24() -> None:
    upscaler = VelaUpscaler(model_path=None, device="cpu", batch_size=8)
    raw_batch = torch.randint(0, 256, (2, 240, 426, 3), dtype=torch.uint8)
    out = upscaler.process_batch_rgb24(raw_batch, target_w=1920, target_h=1080, cas_strength=0.25)
    assert out.shape == (2, 1080, 1920, 3)
    assert out.dtype == torch.uint8

    out_buf = torch.empty((8, 1080, 1920, 3), dtype=torch.uint8)
    out_slice, ev = upscaler.process_batch_rgb24_async(raw_batch, out_buf, target_w=1920, target_h=1080, cas_strength=0.25)
    assert out_slice.shape == (2, 1080, 1920, 3)
    assert ev is None


def test_integer_fps_decimation() -> None:
    info_60 = VideoInfo(
        path=Path("MKZ-033.mkv"),
        width=854,
        height=480,
        fps=19001.0 / 317.0,
        fps_rational="19001/317",
        duration=100.0,
        bitrate_kbps=2130,
        codec_name="h264",
        has_audio=True,
    )
    cfg = VelaConfig(max_fps=30.0)
    out_fps, out_rat, capped = resolve_effective_fps(info_60, cfg)
    assert capped is True
    assert out_rat == "19001/634"
    assert abs(out_fps - 29.97003) < 0.001


def test_cli_and_bitrate_parser() -> None:
    assert parse_bitrate_to_kbps("5M") == 5000
    assert parse_bitrate_to_kbps("2500K") == 2500
    parser = build_parser()
    args = parser.parse_args(["--input", "MKZ-033.mkv", "--gpus", "0,1", "--model", "realplksr"])
    assert args.gpus == "0,1"
    assert args.model == "realplksr"
    cfg = VelaConfig(model_name="realplksr")
    assert cfg.resolve_model_file().name == "4xNomosWebPhoto_RealPLKSR.pth"
    for ext in (".mkv", ".mp4", ".avi", ".wmv", ".mov", ".flv", ".webm", ".ts", ".rmvb", ".vob", ".mpg"):
        assert ext in VIDEO_EXTENSIONS
    assert abs(_parse_fps("24000/1001") - 23.976) < 0.01
    assert _parse_duration({"duration": "N/A"}, {"tags": {"DURATION": "01:02:03.500000000"}}) == 3723.5


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        test_plan_segments_exact_and_remainder(Path(td))
    test_gpu_resize_and_cas_sharpen()
    test_upscaler_fallback_batch_rgb24()
    test_integer_fps_decimation()
    test_cli_and_bitrate_parser()
    print("[PASS] All 5 Vela pipeline unit tests passed!")

