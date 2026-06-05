"""
Parallel Processing Package

CUDA paths are automatically configured via .venv/Lib/site-packages/cuda_setup.pth
when the virtual environment is activated.
"""

from .cuda_device_driver import (
    CudaDeviceDriver,
    get_driver_device,
    is_cuda_available,
)
from .numba_config import (
    NumbaConfig,
    get_numba_config,
)
from .cupy_config import (
    CupyConfig,
    get_cupy_config,
)
from .bm3d_interface import (
    DenoiseBackend,
    BenchmarkResult,
    compare_backends,
    psnr,
    mae,
)
from .bm3d_baseline import Bm3dBaselineBackend
from .bm3d_cupy import CupyBm3dBackend

__all__ = [
    "CudaDeviceDriver",
    "get_driver_device", 
    "is_cuda_available",
    "NumbaConfig",
    "get_numba_config",
    "CupyConfig",
    "get_cupy_config",
    "DenoiseBackend",
    "BenchmarkResult",
    "compare_backends",
    "psnr",
    "mae",
    "Bm3dBaselineBackend",
    "CupyBm3dBackend",
]
