"""Starter CuPy backend scaffold for BM3D development.

This module provides a drop-in backend compatible with the benchmark harness.
Current behavior supports two modes:
1. "reference": delegates filtering to the bm3d package while managing CuPy transfer.
2. "stub": runs a lightweight GPU mean filter placeholder for iteration speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np
from bm3d import BM3DStages, bm3d


@dataclass
class CupyBm3dBackend:
    """Starter backend for a future full CuPy BM3D implementation."""

    profile: str = "np"
    mode: str = "reference"

    @property
    def name(self) -> str:
        """Human-readable backend name."""
        return f"cupy-bm3d[{self.mode}]"

    def _validate_input(self, noisy: np.ndarray) -> np.ndarray:
        noisy_img = np.asarray(noisy)
        if noisy_img.ndim != 2:
            raise ValueError(f"Expected a 2D grayscale image, got shape={noisy_img.shape}")

        return np.clip(noisy_img.astype(np.float32, copy=False), 0.0, 1.0)

    def _stub_gpu_filter(self, noisy: np.ndarray, sigma: float) -> np.ndarray:
        """Placeholder GPU pipeline with a sigma-scaled smoothing pass.

        TODO: replace this with full BM3D stages on GPU:
        1. Block matching
        2. 3D transform + hard-thresholding
        3. Wiener refinement
        4. Aggregation
        """
        x_gpu = cp.asarray(noisy)
        radius = 1 if sigma < 0.1 else 2

        # Fast local mean-like smoothing using rolling neighborhood accumulation.
        acc = cp.zeros_like(x_gpu)
        count = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                acc += cp.roll(cp.roll(x_gpu, dy, axis=0), dx, axis=1)
                count += 1

        denoised = acc / float(count)
        return cp.asnumpy(cp.clip(denoised, 0.0, 1.0))

    def denoise(self, noisy: np.ndarray, sigma: float) -> np.ndarray:
        """Denoise image with starter CuPy backend contract."""
        noisy_img = self._validate_input(noisy)

        if self.mode == "stub":
            return self._stub_gpu_filter(noisy_img, sigma).astype(noisy_img.dtype, copy=False)

        if self.mode != "reference":
            raise ValueError(f"Unsupported mode={self.mode!r}; use 'reference' or 'stub'")

        # Reference mode keeps output aligned to the package baseline while
        # preserving GPU/CPU transfer points for future replacement.
        noisy_gpu = cp.asarray(noisy_img)
        noisy_cpu = cp.asnumpy(noisy_gpu)
        denoised = bm3d(
            noisy_cpu,
            sigma_psd=float(sigma),
            profile=self.profile,
            stage_arg=BM3DStages.ALL_STAGES,
        )
        denoised_gpu = cp.asarray(np.asarray(denoised, dtype=np.float32))
        return cp.asnumpy(cp.clip(denoised_gpu, 0.0, 1.0)).astype(noisy_img.dtype, copy=False)
