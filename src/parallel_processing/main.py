import cupy as cp
# from cupyx.scipy.fft import dctn

import numba
from numba import jit, cuda, config


def multiply(a, b):
    return a * b

def main():
    print("Hello from parallel-processing!")


if __name__ == "__main__":
    # Check CUDA availability
    print(f"Numba CUDA available: {numba.cuda.is_available()}")
    print(f"CuPy CUDA available: {cp.cuda.is_available()}")
    print(f"CUDA devices found: {cp.cuda.runtime.getDeviceCount()}")
    
    main()
