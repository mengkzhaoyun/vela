"""Configuration and hardware discovery for Vela Multi-GPU VSR Engine."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

MODEL_ALIASES: dict[str, str] = {
    "realplksr": "4xNomosWebPhoto_RealPLKSR.pth",
    "clearreality": "4x-ClearRealityV1.pth",
    "siax": "nmkdSiaxCX_200k.safetensors",
}

DEFAULT_MODEL_SEARCH_DIRS: tuple[str, ...] = (
    "/data/vela/model_weights",
    "/app/vela/model_weights",
    "/data/jasna/model_weights",
    "/app/sglang/model_weights",
    "model_weights",
)


@dataclass
class VelaConfig:
    input_path: Path | None = None
    output_path: Path | None = None
    scan_dir: Path | None = None
    model_name: str = field(default_factory=lambda: os.environ.get("VELA_MODEL", "realplksr"))
    model_path: Path | None = None
    gpus: str = field(default_factory=lambda: os.environ.get("VELA_GPUS", "all"))
    target_width: int = field(default_factory=lambda: int(os.environ.get("TARGET_WIDTH", "1920")))
    target_height: int = field(default_factory=lambda: int(os.environ.get("TARGET_HEIGHT", "1080")))
    target_fps: float = field(default_factory=lambda: float(os.environ.get("TARGET_FPS", "0")))
    batch_size: int = field(default_factory=lambda: int(os.environ.get("BATCH_SIZE", "8")))
    segment_seconds: float = field(default_factory=lambda: float(os.environ.get("SEGMENT_SECONDS", "30.0")))
    pre_denoise: str = field(default_factory=lambda: os.environ.get("PRE_DENOISE", "hqdn3d=1.5:1.2:2.5:2.0"))
    cas_strength: float = field(default_factory=lambda: float(os.environ.get("CAS_STRENGTH", "0.25")))
    codec: str = field(default_factory=lambda: os.environ.get("CODEC", "hevc"))
    target_bitrate: str = field(default_factory=lambda: os.environ.get("TARGET_BITRATE", "5M"))
    encoder_cq: int = field(default_factory=lambda: int(os.environ.get("ENCODER_CQ", "20")))
    preserve_low_bitrate: bool = field(
        default_factory=lambda: os.environ.get("PRESERVE_LOW_BITRATE", "true").lower() in ("1", "true", "yes")
    )
    compile_model: bool = field(
        default_factory=lambda: os.environ.get("COMPILE_MODEL", "false").lower() in ("1", "true", "yes")
    )
    keep_temp: bool = False

    def resolve_model_file(self) -> Path:
        if self.model_path and self.model_path.exists():
            return self.model_path

        env_model_path = os.environ.get("VELA_MODEL_PATH")
        if env_model_path and Path(env_model_path).exists():
            return Path(env_model_path)

        filename = MODEL_ALIASES.get(self.model_name.lower(), self.model_name)
        direct = Path(filename)
        if direct.exists():
            return direct

        for d in DEFAULT_MODEL_SEARCH_DIRS:
            candidate = Path(d) / filename
            if candidate.exists():
                return candidate.resolve()
            st_candidate = Path(d) / f"{Path(filename).stem}.safetensors"
            if st_candidate.exists():
                return st_candidate.resolve()

        return Path("model_weights") / filename

