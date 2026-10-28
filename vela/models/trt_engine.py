"""TensorRT FP16 / INT8 Engine Builder and High-Throughput Async Runtime for Vela."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def is_tensorrt_available() -> bool:
    try:
        import onnx  # noqa: F401
        import tensorrt  # noqa: F401

        return torch.cuda.is_available()
    except ImportError:
        return False


class _FileLock:
    """Cross-process file lock so multi-GPU workers do not build the same TRT engine simultaneously."""

    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        self._fd: int | None = None

    def __enter__(self) -> _FileLock:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o666)
        try:
            import fcntl

            fcntl.flock(self._fd, fcntl.LOCK_EX)
        except ImportError:
            pass
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._fd is not None:
            try:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except ImportError:
                pass
            os.close(self._fd)
            self._fd = None


class VelaTRTEngine:
    """
    Per-GPU TensorRT FP16 execution wrapper for Spandrel SR models (e.g. RealPLKSR).
    Caches compiled engines by (model_stem, SM architecture, batch_size, height, width, precision)
    under `<model_dir>/trt-cache/`.
    """

    def __init__(
        self,
        pytorch_model: torch.nn.Module,
        model_path: Path,
        device: torch.device,
        scale: int = 4,
        precision: str = "fp16",
    ) -> None:
        import tensorrt as trt

        self.trt = trt
        self.pytorch_model = pytorch_model
        self.model_path = model_path.resolve()
        self.device = device
        self.scale = scale
        self.precision = precision.lower()
        self.cache_dir = self.model_path.parent / "trt-cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        gpu_idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(gpu_idx)
        self.sm_tag = f"sm{props.major}{props.minor}"
        self.stream = torch.cuda.Stream(device=device)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        self._loaded_shape: tuple[int, int, int] | None = None
        self._engine: Any = None
        self._context: Any = None

    def _engine_path(self, batch: int, h: int, w: int) -> Path:
        stem = self.model_path.stem.lower()
        if "realplksr" in stem:
            stem = "realplksr"
        return self.cache_dir / f"{stem}_{self.sm_tag}_b{batch}_{h}x{w}_{self.precision}.engine"

    def _onnx_path(self, h: int, w: int) -> Path:
        stem = self.model_path.stem.lower()
        if "realplksr" in stem:
            stem = "realplksr"
        return self.cache_dir / f"{stem}_dyn_{h}x{w}.onnx"

    def _timing_cache_path(self) -> Path:
        return self.cache_dir / f"timing_{self.sm_tag}.cache"

    def _ensure_engine(self, batch: int, h: int, w: int) -> None:
        if self._loaded_shape == (batch, h, w) and self._context is not None:
            return

        trt = self.trt
        eng_file = self._engine_path(batch, h, w)
        onnx_file = self._onnx_path(h, w)
        lock_file = self.cache_dir / f"{eng_file.name}.lock"

        if not eng_file.exists():
            with _FileLock(lock_file):
                if not eng_file.exists():
                    self._build_engine(batch, h, w, onnx_file, eng_file)

        self._engine = self.runtime.deserialize_cuda_engine(eng_file.read_bytes())
        self._context = self._engine.create_execution_context()
        self._context.set_input_shape("input", (batch, 3, h, w))
        self._loaded_shape = (batch, h, w)

    def _build_engine(self, batch: int, h: int, w: int, onnx_file: Path, eng_file: Path) -> None:
        trt = self.trt
        torch.cuda.set_device(self.device)

        if not onnx_file.exists():
            print(f"[VELA-TRT] Exporting ONNX graph ({h}x{w}) -> {onnx_file.name} ...", flush=True)
            m_fp32 = self.pytorch_model.to(device=self.device, dtype=torch.float32, memory_format=torch.contiguous_format)
            dummy = torch.zeros((batch, 3, h, w), device=self.device, dtype=torch.float32)
            tmp_onnx = onnx_file.with_suffix(f".{os.getpid()}.onnx")
            with torch.inference_mode():
                torch.onnx.export(
                    m_fp32,
                    (dummy,),
                    str(tmp_onnx),
                    input_names=["input"],
                    output_names=["output"],
                    opset_version=17,
                    do_constant_folding=True,
                    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                    dynamo=False,
                )
            tmp_onnx.replace(onnx_file)
            self.pytorch_model.to(device=self.device, dtype=torch.float16, memory_format=torch.channels_last)

        print(
            f"[VELA-TRT] Building TensorRT {self.precision.upper()} engine for {self.sm_tag.upper()} "
            f"(B={batch}, {w}x{h}) -> {eng_file.name} ...",
            flush=True,
        )
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        with open(onnx_file, "rb") as f:
            if not parser.parse(f.read()):
                errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError(f"TensorRT ONNX parse failed: {'; '.join(errs)}")

        network.get_input(0).dtype = trt.DataType.HALF
        network.get_output(0).dtype = trt.DataType.HALF

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
        config.set_flag(trt.BuilderFlag.FP16)

        tc_file = self._timing_cache_path()
        tc_bytes = tc_file.read_bytes() if tc_file.exists() else b""
        timing_cache = config.create_timing_cache(tc_bytes)
        config.set_timing_cache(timing_cache, ignore_mismatch=False)

        profile = builder.create_optimization_profile()
        profile.set_shape("input", (batch, 3, h, w), (batch, 3, h, w), (batch, 3, h, w))
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT failed to build serialized network")

        try:
            new_tc = config.get_timing_cache()
            if new_tc is not None:
                tc_file.write_bytes(bytes(new_tc.serialize()))
        except Exception:
            pass

        tmp_eng = eng_file.with_suffix(f".{os.getpid()}.engine")
        tmp_eng.write_bytes(bytes(serialized))
        tmp_eng.replace(eng_file)
        print(f"[VELA-TRT] Engine ready: {eng_file.name}", flush=True)

    def __call__(self, x_bchw_fp16: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x_bchw_fp16.shape
        self._ensure_engine(b, h, w)

        x_contig = x_bchw_fp16.contiguous()
        y_out = torch.empty(
            (b, c, h * self.scale, w * self.scale),
            device=self.device,
            dtype=torch.float16,
        )

        cur_stream = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(cur_stream)
        with torch.cuda.stream(self.stream):
            in_dtype = self._engine.get_tensor_dtype("input")
            out_dtype = self._engine.get_tensor_dtype("output")
            if in_dtype == self.trt.DataType.FLOAT:
                x_buf = x_contig.float()
            else:
                x_buf = x_contig
            if out_dtype == self.trt.DataType.FLOAT:
                y_buf = torch.empty_like(y_out, dtype=torch.float32)
            else:
                y_buf = y_out

            self._context.set_tensor_address("input", x_buf.data_ptr())
            self._context.set_tensor_address("output", y_buf.data_ptr())
            self._context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            if out_dtype == self.trt.DataType.FLOAT:
                y_out.copy_(y_buf.to(torch.float16))

        cur_stream.wait_stream(self.stream)
        return y_out
