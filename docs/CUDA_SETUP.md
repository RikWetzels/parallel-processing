# CUDA Setup Documentation

## Automatic DLL Path Configuration

This project automatically configures CUDA DLL paths when the virtual environment is activated. You don't need to add any imports or setup code to your files.

### How It Works

1. **`.pth` File**: When Python starts in this virtual environment, it automatically processes [.venv/Lib/site-packages/cuda_setup.pth](.venv/Lib/site-packages/cuda_setup.pth)
2. **DLL Directory**: The `.pth` file adds the CUDA bin directory to Windows DLL search paths using `os.add_dll_directory()`
3. **Environment Variables**: Sets Numba CUDA configuration variables

### What's Configured

- CUDA bin directory: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64`
- Environment variables:
  - `NUMBA_ENABLE_CUDASIM=0` (disable simulation mode)
  - `NUMBA_CUDA_LOG_LEVEL=WARNING`

### Usage

Simply activate your virtual environment and start coding:

```python
# No setup needed - just import and use!
import cupy as cp
from numba import cuda

print(f"CUDA available: {cuda.is_available()}")
print(f"Devices: {cp.cuda.runtime.getDeviceCount()}")
```

This works identically in:
- Python scripts (`python src/main.py`)
- Jupyter notebooks
- Interactive Python sessions
- Tests

### Troubleshooting

If CUDA libraries are not found:

1. **Check CUDA installation path**: Verify that CUDA is installed at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`
2. **Update `.pth` file**: If your CUDA is in a different location, edit [.venv/Lib/site-packages/cuda_setup.pth](.venv/Lib/site-packages/cuda_setup.pth)
3. **Restart Python**: Changes to `.pth` files require restarting the Python interpreter

### For Different Environments

If you recreate your virtual environment, you'll need to recreate the `.pth` file. The content should be:

```python
import os, sys; (os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64") if sys.platform == 'win32' and hasattr(os, 'add_dll_directory') and os.path.exists(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64") else None, os.environ.setdefault('NUMBA_ENABLE_CUDASIM', '0'), os.environ.setdefault('NUMBA_CUDA_LOG_LEVEL', 'WARNING'))
```
