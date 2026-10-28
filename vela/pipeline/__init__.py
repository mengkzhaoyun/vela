"""Multi-GPU streaming pipeline, video probing, and lossless remuxing."""

from vela.pipeline.probe import SegmentTask, VideoInfo, detect_gpus, plan_segments, probe_video


def __getattr__(name: str):
    # Probing and log setup must work before loading the GPU runtime.
    if name == "MultiGPUScheduler":
        from vela.pipeline.scheduler import MultiGPUScheduler

        return MultiGPUScheduler
    raise AttributeError(name)

__all__ = ["MultiGPUScheduler", "SegmentTask", "VideoInfo", "detect_gpus", "plan_segments", "probe_video"]

