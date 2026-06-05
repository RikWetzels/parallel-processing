"""Adapter around the third-party bm3d package."""

from __future__ import annotations

import numpy as np
from bm3d import BM3DStages, bm3d


class Bm3dBaselineBackend:
    """Baseline backend using the reference bm3d package implementation."""

    def __init__(self, profile: str = "np"):
        self._profile = profile

    @property
    def name(self) -> str:
        """Human-readable backend name."""
        return f"bm3d-package[{self._profile}]"

    def denoise(self, noisy: np.ndarray, sigma: float) -> np.ndarray:
        """Run BM3D denoising and preserve input shape and dtype contract."""
        noisy_img = np.asarray(noisy)
        if noisy_img.ndim != 2:
            raise ValueError(f"Expected a 2D grayscale image, got shape={noisy_img.shape}")

        input_dtype = noisy_img.dtype
        noisy_float = np.clip(noisy_img.astype(np.float32, copy=False), 0.0, 1.0)
        denoised = bm3d(
            noisy_float,
            sigma_psd=float(sigma),
            profile=self._profile,
            stage_arg=BM3DStages.ALL_STAGES,
        )

        return np.clip(np.asarray(denoised), 0.0, 1.0).astype(input_dtype, copy=False)
