"""Protect FFmpeg option scope at the segment decoder boundary."""
from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from vela.config import VelaConfig
from vela.pipeline import worker
from vela.pipeline.probe import SegmentTask, VideoInfo


class DecoderCaptured(Exception):
    """Stop before launching FFmpeg or allocating inference buffers."""


class SegmentDurationTests(unittest.TestCase):
    def test_segment_duration_limits_decoder_output_after_input_seek(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            info = VideoInfo(
                root / "input.mp4", 854, 480, 60000 / 1001, "60000/1001",
                8432.172, 2745, "hevc", True,
            )
            cfg = VelaConfig(max_fps=0, target_fps=0)
            for idx, start, duration in ((10, 300, 30), (281, 8430, 2.172)):
                with self.subTest(start=start, duration=duration):
                    task = SegmentTask(idx, start, duration, root / f"seg_{idx:04d}.mp4")
                    with patch.dict(sys.modules, {"torch": types.ModuleType("torch")}), \
                         patch.object(worker, "build_video_encoder_args", return_value=["-c:v", "test"]), \
                         patch.object(worker, "_report_encoder_args"), \
                         patch.object(worker, "_ffmpeg_processes", side_effect=DecoderCaptured) as processes:
                        with self.assertRaises(DecoderCaptured):
                            worker.process_segment_stream(task, info, cfg, object())

                    decode_cmd, encode_cmd = processes.call_args.args[:2]
                    input_pos = decode_cmd.index("-i")
                    seek_pos = decode_cmd.index("-ss")
                    duration_pos = decode_cmd.index("-t")
                    # -ss remains an input option; -t must constrain the output.
                    self.assertLess(seek_pos, input_pos)
                    self.assertGreater(duration_pos, input_pos + 1)
                    self.assertEqual(decode_cmd[seek_pos + 1], f"{start:.3f}")
                    self.assertEqual(decode_cmd[duration_pos + 1], f"{duration:.3f}")
                    self.assertEqual(decode_cmd.count("-t"), 1)
                    self.assertEqual(encode_cmd[encode_cmd.index("-r") + 1], "60000/1001")


if __name__ == "__main__":
    unittest.main()
