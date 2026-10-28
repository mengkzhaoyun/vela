"""Spandrel / PLKSR GPU Upscaler + VRAM Antialiased Downscaler + GPU FidelityFX CAS."""
from __future__ import annotations

import os
import types
from pathlib import Path

import torch
import torch.nn.functional as F


def gpu_resize_to_target(frames_bchw: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Antialiased bicubic downscale/upscale directly in GPU VRAM."""
    _, _, h, w = frames_bchw.shape
    if h == target_h and w == target_w:
        return frames_bchw
    out = F.interpolate(
        frames_bchw,
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return out.clamp_(0.0, 1.0)


def gpu_cas_sharpen(frames_bchw: torch.Tensor, strength: float = 0.25) -> torch.Tensor:
    """
    Pure PyTorch GPU implementation of Contrast Adaptive Sharpening (CAS).
    Operates on (B, C, H, W) tensor in [0.0, 1.0] without leaving VRAM.
    """
    if strength <= 0.0:
        return frames_bchw

    padded = F.pad(frames_bchw, (1, 1, 1, 1), mode="replicate")
    # 5-tap cross neighborhood: top (b), left (d), center (e), right (f), bottom (h)
    b = padded[:, :, 0:-2, 1:-1]
    d = padded[:, :, 1:-1, 0:-2]
    e = frames_bchw
    f = padded[:, :, 1:-1, 2:]
    h = padded[:, :, 2:, 1:-1]

    mn = torch.minimum(torch.minimum(torch.minimum(d, e), torch.minimum(f, b)), h)
    mx = torch.maximum(torch.maximum(torch.maximum(d, e), torch.maximum(f, b)), h)

    # Adaptive sharpening weight per pixel based on local contrast
    amp = torch.clamp(torch.minimum(mn, 2.0 - mx) / (mx + 1e-5), min=0.0, max=1.0)
    amp = torch.sqrt(amp)

    peak = -1.0 / (8.0 - 3.0 * float(min(max(strength, 0.0), 1.0)))
    w = amp * peak
    rcp_weight = 1.0 / (1.0 + 4.0 * w)

    sharpened = ((b + d + f + h) * w + e) * rcp_weight
    return sharpened.clamp_(0.0, 1.0)


def _patch_plksr_for_compile_and_nhwc(model: torch.nn.Module, use_channels_last: bool = True) -> None:
    """
    Replace PLKConv2d's in-place non-contiguous slice assignment:
        x[:, :self.idx] = self.conv(x[:, :self.idx])
    with an out-of-place NHWC-contiguous concat so TorchInductor fuses all 28 PLKBlocks
    and cuDNN stays on its fast NHWC TensorCore path (+85% throughput).
    """
    for mod in model.modules():
        if mod.__class__.__name__ == "PLKConv2d":
            if use_channels_last:

                def _fast_lk_nhwc(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
                    x1 = x[:, : self.idx].contiguous(memory_format=torch.channels_last)
                    return torch.cat([self.conv(x1), x[:, self.idx :]], dim=1)

                mod.forward = types.MethodType(_fast_lk_nhwc, mod)
            else:

                def _fast_lk_nchw(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
                    x1 = x[:, : self.idx].contiguous()
                    return torch.cat([self.conv(x1), x[:, self.idx :]], dim=1)

                mod.forward = types.MethodType(_fast_lk_nchw, mod)


class VelaUpscaler:
    """GPU-resident video super-resolution model wrapper."""

    def __init__(
        self,
        model_path: Path | None,
        device: str | torch.device = "cuda:0",
        compile_model: bool = True,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.model_path = model_path
        self.dtype = torch.float32
        self.use_channels_last = False
        self.model = None
        self._eager_model = None
        self._compiled_batch_size: int | None = None

        if model_path is not None and model_path.exists():
            from spandrel import ModelLoader

            if self.device.type == "cuda":
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
                if not cache_dir:
                    cache_dir = str(model_path.resolve().parent / "torch-compile-cache")
                    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
                Path(cache_dir).mkdir(parents=True, exist_ok=True)

            loader = ModelLoader(device=self.device)
            resolved_path = model_path.resolve()
            descriptor = loader.load_from_file(str(resolved_path))
            descriptor.eval()

            if self.device.type == "cuda":
                self.use_channels_last = True
                _patch_plksr_for_nhwc = True
                descriptor.model.to(device=self.device, dtype=torch.float16, memory_format=torch.channels_last)
                self.dtype = torch.float16
            else:
                _patch_plksr_for_nhwc = False
                descriptor.model.to(device=self.device, dtype=torch.float32)
                self.dtype = torch.float32

            _patch_plksr_for_compile_and_nhwc(descriptor.model, use_channels_last=_patch_plksr_for_nhwc)
            self._eager_model = descriptor.model

            if compile_model and hasattr(torch, "compile") and self.device.type == "cuda":
                self.model = torch.compile(descriptor.model, mode="default")
            else:
                self.model = descriptor.model

    @torch.inference_mode()
    def process_batch_rgb24(
        self,
        raw_rgb_uint8: torch.Tensor,
        target_w: int = 1920,
        target_h: int = 1080,
        cas_strength: float = 0.25,
    ) -> torch.Tensor:
        """
        Input: (B, H, W, 3) uint8 tensor on CPU/pinned memory.
        Output: (B, target_h, target_w, 3) uint8 tensor on CPU ready for NVENC pipe.
        """
        b_cur = raw_rgb_uint8.shape[0]
        if self._compiled_batch_size is None:
            self._compiled_batch_size = b_cur

        gpu_u8 = raw_rgb_uint8.to(device=self.device, non_blocking=True)
        x = gpu_u8.permute(0, 3, 1, 2).to(dtype=self.dtype).mul_(1.0 / 255.0)
        if self.use_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        if self.model is not None:
            # Avoid triggering a second torch.compile recompilation on the final remainder batch of a segment
            active_model = self.model if b_cur == self._compiled_batch_size else (self._eager_model or self.model)
            sr = active_model(x)
        else:
            sr = x

        sr_1080 = gpu_resize_to_target(sr, target_h=target_h, target_w=target_w)
        if cas_strength > 0.0:
            sr_1080 = gpu_cas_sharpen(sr_1080, strength=cas_strength)

        out_uint8 = (
            (sr_1080 * 255.0 + 0.5)
            .clamp_(0, 255)
            .to(dtype=torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
        )
        return out_uint8

