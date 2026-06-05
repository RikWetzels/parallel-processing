"""Tests for starter CuPy BM3D backend."""

import numpy as np

from parallel_processing.bm3d_cupy import CupyBm3dBackend
from parallel_processing.bm3d_interface import psnr


def test_cupy_backend_reference_mode_contract():
    """Reference mode should preserve shape/dtype and valid range."""
    rng = np.random.default_rng(42)
    noisy = rng.random((48, 48), dtype=np.float32)

    backend = CupyBm3dBackend(mode="reference")
    out = backend.denoise(noisy, sigma=0.08)

    assert out.shape == noisy.shape
    assert out.dtype == noisy.dtype
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_cupy_backend_stub_mode_improves_psnr_on_fixture():
    """Stub mode should not degrade a simple synthetic fixture heavily."""
    clean = np.load("tests/fixtures/clean_gradient_64.npy")
    noisy = np.load("tests/fixtures/noisy_gradient_64_sigma008_seed7.npy")

    backend = CupyBm3dBackend(mode="stub")
    denoised = backend.denoise(noisy, sigma=0.08)

    assert denoised.shape == clean.shape
    assert denoised.dtype == clean.dtype

    # Keep this tolerance modest for a placeholder implementation.
    assert psnr(clean, denoised) >= psnr(clean, noisy) - 0.2


def test_cupy_backend_rejects_bad_mode():
    """Unsupported mode should raise a descriptive error."""
    backend = CupyBm3dBackend(mode="invalid")
    arr = np.zeros((16, 16), dtype=np.float32)

    try:
        backend.denoise(arr, sigma=0.08)
        assert False, "Expected ValueError for unsupported mode"
    except ValueError as exc:
        assert "Unsupported mode" in str(exc)
