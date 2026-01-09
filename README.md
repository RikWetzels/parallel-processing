# Parallel Processing Project

A Python project with CUDA support for parallel processing tasks.

## Prerequisites

- Python 3.12 or higher
- NVIDIA CUDA Toolkit 13.1 (or compatible version)
- NVIDIA GPU with CUDA support

## CUDA Environment Setup

This project requires specific CUDA environment variables to be configured. There are multiple ways to set these up:

### Option 1: Batch Script (Windows - Recommended)

Run the provided batch script before executing any CUDA-dependent code:

```cmd
scripts\setup_cuda_env.bat
```

This will set up all required environment variables for the current terminal session.

### Option 2: Python Configuration

Use the provided Python module to configure CUDA programmatically:

```python
from cuda_config import setup_cuda_environment

# Use default configuration
setup_cuda_environment()

# Or load from .env file
from cuda_config import load_cuda_env_from_file
load_cuda_env_from_file()
```

### Option 3: Environment File

1. Copy `.env.example` to `.env`:
   ```cmd
   copy .env.example .env
   ```

2. Edit `.env` to match your system's CUDA installation path

3. Load the environment in your Python code using the `cuda_config` module

### Required Environment Variables

The following environment variables are configured:

- **CUDA_PATH**: Path to CUDA installation (default: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`)
- **CUDA_TILE_ENABLE_CRASH_DUMP**: Enable crash dumps for debugging (default: `1`)
- **CUDA_TILE_COMPILER_TIMEOUT_SEC**: Compiler timeout in seconds (default: `300`)
- **CUDA_TILE_LOGS**: Logging level for cuTile (default: `CUTILEIR`)
- **CUDA_TILE_TEMP_DIR**: Temporary directory for CUDA tile files

## Installation

```bash
# Install dependencies (using uv or pip)
uv sync
```

## Usage

```python
# Import your modules
from cuda_config import setup_cuda_environment
from main import main

# Configure CUDA environment
setup_cuda_environment()

# Run your code
main()
```

## Development

```bash
# Run tests
pytest
```

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
