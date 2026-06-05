"""Shared interfaces and metrics for BM3D backends."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np


class DenoiseBackend(Protocol):
    """Common interface for denoising backends."""

    @property
    def name(self) -> str:
        """Human-readable backend name."""

    def denoise(self, noisy: np.ndarray, sigma: float) -> np.ndarray:
        """Return denoised image for a noisy grayscale input in [0, 1]."""


@dataclass(frozen=True)
class BenchmarkResult:
    """Quality and latency metrics for a backend run."""

    backend: str
    elapsed_seconds: float
    psnr_db: float
    mae_value: float


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """Convert to 2D float32 array in [0, 1]."""
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape={arr.shape}")

    arr = arr.astype(np.float32, copy=False)
    return np.clip(arr, 0.0, 1.0)


def psnr(reference: np.ndarray, estimate: np.ndarray, max_value: float = 1.0) -> float:
    """Compute peak signal-to-noise ratio in dB."""
    ref = _normalize_image(reference)
    est = _normalize_image(estimate)
    mse = float(np.mean((ref - est) ** 2))

    if mse == 0.0:
        return float("inf")

    return float(20.0 * np.log10(max_value) - 10.0 * np.log10(mse))


def mae(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute mean absolute error."""
    ref = _normalize_image(reference)
    est = _normalize_image(estimate)
    return float(np.mean(np.abs(ref - est)))


def compare_backends(
    clean: np.ndarray,
    noisy: np.ndarray,
    sigma: float,
    backends: list[DenoiseBackend],
    runs: int = 3,
) -> tuple[list[BenchmarkResult], dict[str, np.ndarray]]:
    """Benchmark multiple backends on identical input and return results."""
    if runs < 1:
        raise ValueError("runs must be >= 1")
    if not backends:
        raise ValueError("backends must contain at least one backend")

    clean_img = _normalize_image(clean)
    noisy_img = _normalize_image(noisy)

    results: list[BenchmarkResult] = []
    outputs: dict[str, np.ndarray] = {}

    for backend in backends:
        output: np.ndarray | None = None
        timings: list[float] = []

        for _ in range(runs):
            start = perf_counter()
            candidate = backend.denoise(noisy_img, sigma)
            elapsed = perf_counter() - start
            timings.append(elapsed)
            output = _normalize_image(candidate)

        assert output is not None
        outputs[backend.name] = output
        results.append(
            BenchmarkResult(
                backend=backend.name,
                elapsed_seconds=min(timings),
                psnr_db=psnr(clean_img, output),
                mae_value=mae(clean_img, output),
            )
        )

    return results, outputs
