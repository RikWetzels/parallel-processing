# Parallel Processing Project

A Python project with CUDA support for parallel processing tasks.

## Prerequisites

- Python 3.12 or higher
- NVIDIA CUDA Toolkit 13.1 (or compatible version)
- NVIDIA GPU with CUDA support
- UV package manager (will be installed automatically by setup script)

## CUDA Environment Setup

This project requires specific CUDA environment variables to be configured. There are multiple ways to set these up:

### Option 1: Batch Script (Windows - Recommended)

Run the provided batch script before executing any CUDA-dependent code:

```cmd
scripts\setup_cuda_env.bat
```

This will set up all required environment variables for the current terminal session.

### Required Environment Variables

The following environment variables are automatically configured when you activate the virtual environment:

- **CUDA_PATH**: Path to CUDA installation (default: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`)
- **CUDA_TILE_ENABLE_CRASH_DUMP**: Enable crash dumps for debugging (default: `1`)
- **CUDA_TILE_COMPILER_TIMEOUT_SEC**: Compiler timeout in seconds (default: `300`)
- **CUDA_TILE_LOGS**: Logging level for cuTile (default: `CUTILEIR`)
- **CUDA_TILE_TEMP_DIR**: Temporary directory for CUDA tile files

## Quick Start (First Time Setup)

Run the all-in-one setup script:

```cmd
scripts\setup_all.bat
```

This will:
1. Install UV package manager (if not already installed)
2. Configure CUDA environment variables
3. Install project dependencies (`uv sync`)
4. Patch virtual environment activation to include CUDA settings

After setup, activate the virtual environment:

```cmd
.venv\Scripts\activate.bat
```

Now CUDA environment variables are automatically set whenever you activate the venv!

## Installation

### Manual Setup

If you prefer to set up components individually:

```cmd
# 1. Install/configure UV
scripts\setup_uv.bat
# Or for PowerShell with auto-install:
.\scripts\setup_uv.ps1

# 2. Setup CUDA environment
scripts\setup_cuda_env.bat

# 3. Install project dependencies
uv sync
```

## Usage

Simply activate the virtual environment and run your code:

```cmd
# Activate virtual environment
.venv\Scripts\activate.bat

# Run your Python code
python src\main.py
```

Or use `uv run` without activation:

```cmd
uv run python src\main.py
```

## Development

```bash
# Run tests
pytest
```

## BM3D Baseline and Backend Comparison

This project now includes a baseline adapter around the installed `bm3d` package,
plus a shared backend interface for comparing results with your own implementation.

### Baseline adapter

- `parallel_processing.bm3d_baseline.Bm3dBaselineBackend`

### Shared comparison helpers

- `parallel_processing.bm3d_interface.DenoiseBackend`
- `parallel_processing.bm3d_interface.compare_backends`
- `parallel_processing.bm3d_interface.psnr`
- `parallel_processing.bm3d_interface.mae`

### Run the comparison benchmark

```cmd
uv run python scripts\benchmark_bm3d_backends.py --runs 3 --size 256
```

Use the CuPy starter backend in placeholder mode:

```cmd
uv run python scripts\benchmark_bm3d_backends.py --cupy-mode stub --runs 3 --size 256
```

Optional input image (must be a 2D `.npy` array in `[0, 1]`):

```cmd
uv run python scripts\benchmark_bm3d_backends.py --input path\to\clean_image.npy --sigma 0.08 --runs 3
```

Save all outputs and metrics for experiment tracking:

```cmd
uv run python scripts\benchmark_bm3d_backends.py --runs 3 --output-dir outputs\bm3d_run_001
```

This writes:

- `clean.npy`
- `noisy.npy`
- `denoised_<backend>.npy`
- `metrics.csv`
- `metrics.json`

### Add your CuPy backend

`src\parallel_processing\bm3d_cupy.py` now includes a starter
`CupyBm3dBackend` with:

- `mode="reference"`: delegates to package BM3D with explicit CuPy transfer points
- `mode="stub"`: lightweight GPU placeholder filter for rapid iteration

For your full implementation, replace the stub stage logic with BM3D stages:

- `name` property
- `denoise(noisy: np.ndarray, sigma: float) -> np.ndarray`

Fixture files are available for deterministic testing:

- `tests\fixtures\clean_gradient_64.npy`
- `tests\fixtures\noisy_gradient_64_sigma008_seed7.npy`

## Troubleshooting

### CUDA DLL Not Found

If you encounter errors about missing `nvrtc64_130_0.dll`, ensure:
1. CUDA 13.1 is installed
2. The batch script has been run: `scripts\setup_cuda_env.bat`
3. The CUDA_PATH environment variable points to the correct installation

### TileCompilerTimeoutError

If compilation times out, increase the timeout value:
- In `.env`: Set `CUDA_TILE_COMPILER_TIMEOUT_SEC=600`
- In Python: `setup_cuda_environment(tile_timeout=600)`
