"""Benchmark script for BM3D backends.

Run with:
    uv run python scripts/benchmark_bm3d_backends.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from parallel_processing.bm3d_baseline import Bm3dBaselineBackend
from parallel_processing.bm3d_interface import DenoiseBackend, compare_backends


def synthetic_image(size: int = 256) -> np.ndarray:
    """Create a deterministic textured image in [0, 1]."""
    x = np.linspace(0.0, 1.0, size, dtype=np.float32)
    y = np.linspace(0.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    gradient = 0.5 * xx + 0.5 * yy
    sinusoid = 0.2 * np.sin(12.0 * np.pi * xx) * np.cos(8.0 * np.pi * yy)
    checker = 0.1 * (((np.floor(xx * 16) + np.floor(yy * 16)) % 2.0) * 2.0 - 1.0)

    return np.clip(gradient + sinusoid + checker, 0.0, 1.0).astype(np.float32)


def load_image(path: Path) -> np.ndarray:
    """Load a 2D image from .npy file."""
    if path.suffix.lower() != ".npy":
        raise ValueError("Only .npy input is supported in this script")

    image = np.load(path)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D array in {path}, got shape={image.shape}")

    return np.clip(image.astype(np.float32), 0.0, 1.0)


def try_load_cupy_backend(mode: str = "reference") -> DenoiseBackend | None:
    """Load custom CuPy backend if implemented by the user."""
    try:
        from parallel_processing.bm3d_cupy import CupyBm3dBackend

        return CupyBm3dBackend(mode=mode)
    except Exception:
        return None


def save_outputs(output_dir: Path, clean: np.ndarray, noisy: np.ndarray, outputs: dict[str, np.ndarray]) -> None:
    """Persist arrays so backend outputs can be visually and numerically inspected."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "clean.npy", clean)
    np.save(output_dir / "noisy.npy", noisy)

    for backend_name, image in outputs.items():
        safe_name = backend_name.replace("[", "_").replace("]", "").replace("/", "_").replace(" ", "_")
        np.save(output_dir / f"denoised_{safe_name}.npy", image)


def save_metrics(output_dir: Path, sigma: float, runs: int, results) -> None:
    """Write machine-readable benchmark metrics for experiment tracking."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["backend", "elapsed_seconds", "psnr_db", "mae_value"])
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "backend": row.backend,
                    "elapsed_seconds": f"{row.elapsed_seconds:.8f}",
                    "psnr_db": f"{row.psnr_db:.8f}",
                    "mae_value": f"{row.mae_value:.8f}",
                }
            )

    summary = {
        "sigma": float(sigma),
        "runs": int(runs),
        "results": [
            {
                "backend": row.backend,
                "elapsed_seconds": row.elapsed_seconds,
                "psnr_db": row.psnr_db,
                "mae_value": row.mae_value,
            }
            for row in results
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare BM3D package vs custom CuPy backend")
    parser.add_argument("--sigma", type=float, default=0.08, help="Noise standard deviation in [0, 1]")
    parser.add_argument("--size", type=int, default=256, help="Synthetic image size")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs per backend")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for noise")
    parser.add_argument("--input", type=Path, default=None, help="Optional .npy clean image path")
    parser.add_argument(
        "--cupy-mode",
        type=str,
        default="reference",
        choices=["reference", "stub"],
        help="CupyBm3dBackend mode",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to save outputs and metrics (npy/csv/json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    clean = load_image(args.input) if args.input else synthetic_image(size=args.size)

    rng = np.random.default_rng(args.seed)
    noise = rng.normal(0.0, args.sigma, size=clean.shape).astype(np.float32)
    noisy = np.clip(clean + noise, 0.0, 1.0)

    backends: list[DenoiseBackend] = [Bm3dBaselineBackend()]
    cupy_backend = try_load_cupy_backend(mode=args.cupy_mode)
    if cupy_backend is not None:
        backends.append(cupy_backend)

    results, outputs = compare_backends(clean=clean, noisy=noisy, sigma=args.sigma, backends=backends, runs=args.runs)

    print("Backend comparison (lower time/MAE is better, higher PSNR is better):")
    for result in results:
        print(
            f"- {result.backend}: "
            f"time={result.elapsed_seconds:.4f}s "
            f"PSNR={result.psnr_db:.3f}dB "
            f"MAE={result.mae_value:.6f}"
        )

    if args.output_dir is not None:
        save_outputs(args.output_dir, clean, noisy, outputs)
        save_metrics(args.output_dir, args.sigma, args.runs, results)
        print(f"Saved arrays and metrics to: {args.output_dir}")


if __name__ == "__main__":
    main()
