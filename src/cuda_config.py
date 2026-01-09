"""
CUDA Environment Configuration Module

This module provides utilities to set up CUDA environment variables
programmatically from Python code.
"""

import os
from pathlib import Path


def setup_cuda_environment(
    cuda_path: str = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
    tile_timeout: int = 300,
    enable_crash_dump: bool = True,
    tile_logs: str = "CUTILEIR",
    temp_dir: str | None = None
) -> None:
    """
    Set up CUDA environment variables for the current process.
    
    Args:
        cuda_path: Path to CUDA installation directory
        tile_timeout: Compiler timeout in seconds
        enable_crash_dump: Enable crash dumps for TileCompiler errors
        tile_logs: cuTile logging level (e.g., "CUTILEIR")
        temp_dir: Temporary directory for CUDA tile files (defaults to system temp)
    """
    # Set CUDA_PATH
    os.environ["CUDA_PATH"] = cuda_path
    
    # Add CUDA bin directory to PATH
    cuda_bin = os.path.join(cuda_path, "bin", "x64")
    if cuda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{cuda_bin};{os.environ.get('PATH', '')}"
    
    # cuTile environment variables
    os.environ["CUDA_TILE_ENABLE_CRASH_DUMP"] = "1" if enable_crash_dump else "0"
    os.environ["CUDA_TILE_COMPILER_TIMEOUT_SEC"] = str(tile_timeout)
    os.environ["CUDA_TILE_LOGS"] = tile_logs
    
    # Set temporary directory
    if temp_dir is None:
        temp_dir = os.path.join(os.environ.get("TEMP", os.environ.get("TMP", r"C:\Temp")), "cuda_tile")
    os.environ["CUDA_TILE_TEMP_DIR"] = temp_dir
    
    print(f"CUDA environment configured:")
    print(f"  CUDA_PATH: {os.environ['CUDA_PATH']}")
    print(f"  CUDA_TILE_ENABLE_CRASH_DUMP: {os.environ['CUDA_TILE_ENABLE_CRASH_DUMP']}")
    print(f"  CUDA_TILE_COMPILER_TIMEOUT_SEC: {os.environ['CUDA_TILE_COMPILER_TIMEOUT_SEC']}")
    print(f"  CUDA_TILE_LOGS: {os.environ['CUDA_TILE_LOGS']}")
    print(f"  CUDA_TILE_TEMP_DIR: {os.environ['CUDA_TILE_TEMP_DIR']}")


def load_cuda_env_from_file(env_file: str = ".env") -> None:
    """
    Load CUDA environment variables from a .env file.
    
    Args:
        env_file: Path to the .env file (relative to project root)
    """
    env_path = Path(env_file)
    
    if not env_path.exists():
        print(f"Warning: {env_file} not found. Using default configuration.")
        setup_cuda_environment()
        return
    
    # Simple .env parser (or use python-dotenv package)
    env_vars = {}
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
    
    # Apply environment variables
    cuda_path = env_vars.get('CUDA_PATH', r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1")
    
    setup_cuda_environment(
        cuda_path=cuda_path,
        tile_timeout=int(env_vars.get('CUDA_TILE_COMPILER_TIMEOUT_SEC', 300)),
        enable_crash_dump=env_vars.get('CUDA_TILE_ENABLE_CRASH_DUMP', '1') == '1',
        tile_logs=env_vars.get('CUDA_TILE_LOGS', 'CUTILEIR'),
        temp_dir=env_vars.get('CUDA_TILE_TEMP_DIR')
    )


if __name__ == "__main__":
    # Example usage
    setup_cuda_environment()
