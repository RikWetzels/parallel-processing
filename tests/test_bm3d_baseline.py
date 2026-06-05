"""Tests for BM3D baseline adapter and comparison helpers."""

import numpy as np

from parallel_processing.bm3d_baseline import Bm3dBaselineBackend
from parallel_processing.bm3d_interface import compare_backends, psnr


class TestBm3dBaselineBackend:
    """Test suite for BM3D baseline backend."""

    def test_denoise_preserves_shape_and_dtype(self):
        """Baseline output should preserve input image contract."""
        rng = np.random.default_rng(123)
        noisy = rng.random((64, 64), dtype=np.float32)

        backend = Bm3dBaselineBackend()
        denoised = backend.denoise(noisy=noisy, sigma=0.08)

        assert denoised.shape == noisy.shape
        assert denoised.dtype == noisy.dtype
        assert float(denoised.min()) >= 0.0
        assert float(denoised.max()) <= 1.0

    def test_denoise_rejects_non_2d_input(self):
        """Baseline should reject non-grayscale arrays."""
        backend = Bm3dBaselineBackend()

        invalid = np.zeros((32, 32, 3), dtype=np.float32)

        try:
            backend.denoise(invalid, sigma=0.08)
            assert False, "Expected ValueError for non-2D input"
        except ValueError:
            assert True


class TestCompareBackends:
    """Test suite for backend comparison helper."""

    def test_compare_backends_returns_metrics(self):
        """Comparison helper should produce at least one result."""
        size = 64
        x = np.linspace(0.0, 1.0, size, dtype=np.float32)
        clean = np.tile(x, (size, 1)).astype(np.float32)

        rng = np.random.default_rng(7)
        noisy = np.clip(clean + rng.normal(0.0, 0.08, size=clean.shape).astype(np.float32), 0.0, 1.0)

        baseline = Bm3dBaselineBackend()
        results, outputs = compare_backends(clean, noisy, sigma=0.08, backends=[baseline], runs=1)

        assert len(results) == 1
        assert baseline.name in outputs
        assert outputs[baseline.name].shape == clean.shape

        denoised_psnr = psnr(clean, outputs[baseline.name])
        noisy_psnr = psnr(clean, noisy)

        assert denoised_psnr >= noisy_psnr
