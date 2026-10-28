"""Verify log metadata keeps observations separate from pipeline defaults."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from vela.pipeline.probe import VideoInfo, probe_video


class LogMetadataTests(unittest.TestCase):
    def probe(self, stream: dict, fmt: dict | None = None) -> VideoInfo:
        data = {"streams": [{"codec_type": "video", **stream}], "format": fmt or {}}
        with patch("vela.pipeline.probe.subprocess.check_output", return_value=json.dumps(data).encode()):
            return probe_video(Path("sample.mp4"))

    def test_observed_stream_and_container_bitrates_stay_separate(self) -> None:
        info = self.probe(
            {"width": 854, "height": 480, "avg_frame_rate": "60000/1001", "r_frame_rate": "60/1", "bit_rate": "5000000"},
            {"bit_rate": "5160000"},
        )
        self.assertEqual(info.video_bitrate_bps, 5_000_000)
        self.assertEqual(info.container_bitrate_bps, 5_160_000)
        self.assertEqual(info.video_bitrate_source, "视频流bit_rate")
        self.assertEqual(info.container_bitrate_source, "容器bit_rate")
        self.assertEqual((info.measured_width, info.measured_height), (854, 480))
        self.assertAlmostEqual(info.measured_fps, 60000 / 1001)
        self.assertEqual(info.fps, 60.0)  # Processing selection remains unchanged.
        self.assertEqual(info.bitrate_kbps, 5000)

    def test_missing_metadata_does_not_expose_processing_defaults_as_observations(self) -> None:
        info = self.probe({})
        self.assertEqual((info.width, info.height, info.fps, info.bitrate_kbps), (854, 480, 30.0, 0))
        self.assertIsNone(info.measured_width)
        self.assertIsNone(info.measured_height)
        self.assertIsNone(info.measured_fps)
        self.assertIsNone(info.video_bitrate_bps)
        self.assertIsNone(info.container_bitrate_bps)
        self.assertEqual(info.video_bitrate_source, "无法获取")

    def test_container_fallback_is_not_reported_as_video_bitrate(self) -> None:
        info = self.probe({}, {"bit_rate": "2500000"})
        self.assertEqual(info.bitrate_kbps, 2500)
        self.assertIsNone(info.video_bitrate_bps)
        self.assertEqual(info.container_bitrate_bps, 2_500_000)
        self.assertEqual(info.video_bitrate_source, "无法获取")

    def test_stream_tags_have_explicit_sources_and_priority(self) -> None:
        for tags, expected_source in (
            ({"BPS": "3200000", "BPS-eng": "3100000"}, "视频流标签BPS"),
            ({"BPS": "N/A", "BPS-eng": "3200000"}, "视频流标签BPS-eng"),
        ):
            with self.subTest(tags=tags):
                info = self.probe({"bit_rate": "N/A", "tags": tags}, {"bit_rate": "3400000"})
                self.assertEqual(info.video_bitrate_bps, 3_200_000)
                self.assertEqual(info.video_bitrate_source, expected_source)
                self.assertEqual(info.bitrate_kbps, 2500)  # Preserve legacy encoder policy.
        info = self.probe({"bit_rate": "5000000", "tags": {"BPS": "3200000"}})
        self.assertEqual(info.video_bitrate_bps, 5_000_000)
        self.assertEqual(info.video_bitrate_source, "视频流bit_rate")

    def test_invalid_bitrates_are_unknown_instead_of_measured_zero_or_defaults(self) -> None:
        for value in ("N/A", "NaN", "inf", "-1", "0", True):
            with self.subTest(value=value):
                info = self.probe({"bit_rate": value}, {"bit_rate": value})
                self.assertIsNone(info.video_bitrate_bps)
                self.assertIsNone(info.container_bitrate_bps)

    def test_invalid_fractional_fps_is_unknown_or_uses_valid_metadata(self) -> None:
        for value in ("N/A", "NaN", "inf", "-1", "0/0", "30/0", "bad/rate"):
            with self.subTest(value=value):
                info = self.probe({"avg_frame_rate": value, "r_frame_rate": "0/0"})
                self.assertIsNone(info.measured_fps)
        info = self.probe({"avg_frame_rate": "N/A", "r_frame_rate": "24000/1001"})
        self.assertAlmostEqual(info.measured_fps, 24000 / 1001)

    def test_observed_fps_does_not_inherit_processing_cap(self) -> None:
        info = self.probe({"avg_frame_rate": "240/1", "r_frame_rate": "0/0"})
        self.assertEqual(info.fps, 120.0)
        self.assertEqual(info.measured_fps, 240.0)

    def test_old_construction_remains_compatible(self) -> None:
        info = VideoInfo(Path("sample.mp4"), 854, 480, 30.0, "30/1", 60.0, 5000, "h264", False)
        self.assertIsNone(info.video_bitrate_bps)
        self.assertIsNone(info.measured_fps)
        self.assertEqual(info.video_bitrate_source, "未记录")


if __name__ == "__main__":
    unittest.main()
