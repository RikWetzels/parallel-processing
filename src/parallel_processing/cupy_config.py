"""CuPy Configuration Information."""

import cupy as cp
import numpy as np
import sys
from typing import Optional, Tuple


class CupyConfig:
    """Wrapper class for CuPy configuration and runtime information."""
    
    def __init__(self):
        """Initialize CuPy configuration wrapper."""
        pass
    
    @property
    def version(self) -> str:
        """Get CuPy version."""
        return cp.__version__
    
    @property
    def numpy_version(self) -> str:
        """Get NumPy version."""
        return np.__version__
    
    @property
    def python_version(self) -> str:
        """Get Python version."""
        return sys.version.split()[0]
    
    @property
    def cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            return cp.cuda.is_available()
        except:
            return False
    
    @property
    def device_count(self) -> int:
        """Get number of CUDA devices."""
        try:
            return cp.cuda.runtime.getDeviceCount()
        except:
            return 0
    
    @property
    def cuda_driver_version(self) -> Optional[Tuple[int, int]]:
        """Get CUDA Driver version as (major, minor) tuple."""
        try:
            version = cp.cuda.runtime.driverGetVersion()
            major = version // 1000
            minor = (version % 1000) // 10
            return (major, minor)
        except:
            return None
    
    @property
    def cuda_runtime_version(self) -> Optional[Tuple[int, int]]:
        """Get CUDA Runtime version as (major, minor) tuple."""
        try:
            version = cp.cuda.runtime.runtimeGetVersion()
            major = version // 1000
            minor = (version % 1000) // 10
            return (major, minor)
        except:
            return None
    
    @property
    def cuda_path(self) -> Optional[str]:
        """Get CUDA installation path."""
        try:
            return cp.cuda.get_cuda_path()
        except:
            return None
    
    @property
    def current_device_id(self) -> int:
        """Get current device ID."""
        try:
            return cp.cuda.Device().id
        except:
            return 0
    
    @property
    def memory_pool_enabled(self) -> bool:
        """Check if memory pool is enabled."""
        try:
            pool = cp.get_default_memory_pool()
            return pool is not None
        except:
            return False
    
    @property
    def pinned_memory_pool_enabled(self) -> bool:
        """Check if pinned memory pool is enabled."""
        try:
            pool = cp.get_default_pinned_memory_pool()
            return pool is not None
        except:
            return False
    
    def get_memory_pool_stats(self) -> dict:
        """Get memory pool statistics."""
        try:
            pool = cp.get_default_memory_pool()
            return {
                'used_bytes': pool.used_bytes(),
                'total_bytes': pool.total_bytes(),
                'free_blocks': pool.n_free_blocks(),
            }
        except:
            return {}
    
    def get_device_memory_info(self, device_id: Optional[int] = None) -> dict:
        """Get memory information for a device."""
        try:
            if device_id is not None:
                with cp.cuda.Device(device_id):
                    free, total = cp.cuda.runtime.memGetInfo()
            else:
                free, total = cp.cuda.runtime.memGetInfo()
            
            return {
                'free_bytes': free,
                'total_bytes': total,
                'used_bytes': total - free,
                'free_gb': free / 1024**3,
                'total_gb': total / 1024**3,
                'used_gb': (total - free) / 1024**3,
            }
        except:
            return {}
    
    @property
    def has_cudnn(self) -> bool:
        """Check if cuDNN is available."""
        try:
            import cudnn
            return True
        except ImportError:
            return False
    
    @property
    def has_cutensor(self) -> bool:
        """Check if cuTENSOR is available."""
        try:
            import cutensor
            return True
        except ImportError:
            return False
    
    @property
    def cudnn_version(self) -> Optional[str]:
        """Get cuDNN version if available."""
        try:
            import cudnn
            return cudnn.__version__
        except (ImportError, AttributeError):
            return None
    
    @property
    def cutensor_version(self) -> Optional[str]:
        """Get cuTENSOR version if available."""
        try:
            import cutensor
            return cutensor.__version__
        except (ImportError, AttributeError):
            return None
    
    def print_info(self) -> None:
        """Print comprehensive CuPy configuration."""
        print("=" * 60)
        print("CuPy Configuration and CUDA Environment")
        print("=" * 60)
        
        # Version information
        print("Version Information:")
        print(f"  Python Version: {self.python_version}")
        print(f"  NumPy Version: {self.numpy_version}")
        print(f"  CuPy Version: {self.version}")
        
        # CUDA availability
        print("\nCUDA Availability:")
        print(f"  CUDA Available: {self.cuda_available}")
        print(f"  Device Count: {self.device_count}")
        
        if self.cuda_available:
            # CUDA versions
            driver_ver = self.cuda_driver_version
            runtime_ver = self.cuda_runtime_version
            
            print("\nCUDA Versions:")
            if driver_ver:
                print(f"  CUDA Driver Version: {driver_ver[0]}.{driver_ver[1]}")
            if runtime_ver:
                print(f"  CUDA Runtime Version: {runtime_ver[0]}.{runtime_ver[1]}")
            
            # CUDA path
            if self.cuda_path:
                print(f"\nCUDA Installation:")
                print(f"  CUDA Path: {self.cuda_path}")
            
            # Current device
            print(f"\nCurrent Device:")
            print(f"  Device ID: {self.current_device_id}")
            
            # Memory information
            mem_info = self.get_device_memory_info()
            if mem_info:
                print("\nDevice Memory:")
                print(f"  Total: {mem_info['total_gb']:.2f} GB")
                print(f"  Free: {mem_info['free_gb']:.2f} GB")
                print(f"  Used: {mem_info['used_gb']:.2f} GB")
            
            # Memory pool
            print("\nMemory Pool:")
            print(f"  Memory Pool Enabled: {self.memory_pool_enabled}")
            print(f"  Pinned Memory Pool Enabled: {self.pinned_memory_pool_enabled}")
            
            if self.memory_pool_enabled:
                pool_stats = self.get_memory_pool_stats()
                if pool_stats:
                    print(f"  Pool Used: {pool_stats['used_bytes'] / 1024**3:.2f} GB")
                    print(f"  Pool Total: {pool_stats['total_bytes'] / 1024**3:.2f} GB")
                    print(f"  Free Blocks: {pool_stats['free_blocks']}")
        
        # Additional libraries
        print("\nAdditional CUDA Libraries:")
        print(f"  cuDNN: {'Available' if self.has_cudnn else 'Not installed'}", end="")
        if self.cudnn_version:
            print(f" ({self.cudnn_version})")
        else:
            print()
        
        print(f"  cuTENSOR: {'Available' if self.has_cutensor else 'Not installed'}", end="")
        if self.cutensor_version:
            print(f" ({self.cutensor_version})")
        else:
            print()
        
        print("=" * 60)
    
    def __repr__(self) -> str:
        """String representation."""
        driver_ver = self.cuda_driver_version
        driver_str = f"{driver_ver[0]}.{driver_ver[1]}" if driver_ver else "N/A"
        return f"CupyConfig(version='{self.version}', cuda_driver='{driver_str}', devices={self.device_count})"
    
    def __str__(self) -> str:
        """Human-readable string."""
        return f"CuPy {self.version} (CUDA devices: {self.device_count})"


def get_cupy_config() -> CupyConfig:
    """
    Get CuPy configuration.
    
    Returns:
        CupyConfig instance
    """
    return CupyConfig()
