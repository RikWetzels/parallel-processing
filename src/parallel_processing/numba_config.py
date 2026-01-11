"""Numba Configuration Information."""

import numba
from numba import config
from typing import Optional


class NumbaConfig:
    """Wrapper class for Numba configuration and build information."""
    
    def __init__(self):
        """Initialize Numba configuration wrapper."""
        self._config = config
    
    @property
    def version(self) -> str:
        """Get Numba version."""
        return numba.__version__
    
    @property
    def threading_layer(self) -> str:
        """Get default threading layer."""
        return self._config.THREADING_LAYER
    
    @property
    def num_threads(self) -> int:
        """Get number of threads."""
        return self._config.NUMBA_NUM_THREADS
    
    @property
    def threading_layer_priority(self) -> Optional[str]:
        """Get threading layer priority if available."""
        if hasattr(self._config, 'THREADING_LAYER_PRIORITY'):
            return str(self._config.THREADING_LAYER_PRIORITY)
        return None
    
    @property
    def has_parallel_support(self) -> bool:
        """Check if parallel support is available."""
        try:
            from numba.np.ufunc import parallel
            return True
        except:
            return False
    
    @property
    def has_tbb(self) -> bool:
        """Check if TBB (Threading Building Blocks) is available."""
        try:
            import numba.core.runtime.nrtdynmod
            priority = self.threading_layer_priority
            if priority:
                return 'tbb' in priority.lower()
            return False
        except:
            return False
    
    @property
    def has_openmp(self) -> bool:
        """Check if OpenMP is available."""
        try:
            priority = self.threading_layer_priority
            if priority:
                return 'omp' in priority.lower()
            return False
        except:
            return False
    
    @property
    def active_thread_count(self) -> Optional[int]:
        """Get active thread count if available."""
        try:
            from numba.np.ufunc.parallel import get_thread_count
            return get_thread_count()
        except:
            return None
    
    @property
    def llvm_version(self) -> str:
        """Get LLVM version."""
        try:
            if hasattr(self._config, 'LLVM_VERSION'):
                return str(self._config.LLVM_VERSION)
            else:
                return str(numba.llvmlite.__version__)
        except:
            try:
                import llvmlite
                return f"{llvmlite.__version__} (binding: {llvmlite.binding.llvm_version_info})"
            except:
                return "Unknown"
    
    @property
    def cuda_available(self) -> bool:
        """Check if CUDA is available for Numba."""
        try:
            from numba import cuda
            return cuda.is_available()
        except:
            return False
    
    @property
    def cuda_log_level(self) -> Optional[str]:
        """Get CUDA log level if set."""
        if hasattr(self._config, 'NUMBA_CUDA_LOG_LEVEL'):
            return self._config.NUMBA_CUDA_LOG_LEVEL
        return None
    
    @property
    def enable_cudasim(self) -> bool:
        """Check if CUDA simulation mode is enabled."""
        if hasattr(self._config, 'ENABLE_CUDASIM'):
            return bool(self._config.ENABLE_CUDASIM)
        return False
    
    def print_info(self) -> None:
        """Print comprehensive Numba configuration."""
        print("=" * 60)
        print("Numba Build Configuration")
        print("=" * 60)
        
        print(f"Numba Version: {self.version}")
        
        print("\nThreading Layer Options:")
        print(f"  Default Threading Layer: {self.threading_layer}")
        print(f"  Number of Threads: {self.num_threads}")
        
        print("\nThreading Backend Support:")
        print(f"  Parallel support: {'Available' if self.has_parallel_support else 'Not available'}")
        print(f"  TBB (Threading Building Blocks): {'Available' if self.has_tbb else 'Not detected'}")
        print(f"  OpenMP: {'Available' if self.has_openmp else 'Not detected'}")
        
        if self.threading_layer_priority:
            print("\nThreading Configuration:")
            print(f"  Threading Layer Priority: {self.threading_layer_priority}")
        
        if self.active_thread_count is not None:
            print("\nCurrent Threading Backend:")
            print(f"  Active threads: {self.active_thread_count}")
        
        print("\nLLVM Configuration:")
        print(f"  LLVM Version: {self.llvm_version}")
        
        print("\nCUDA Configuration:")
        print(f"  CUDA Available: {self.cuda_available}")
        if self.cuda_log_level:
            print(f"  CUDA Log Level: {self.cuda_log_level}")
        print(f"  CUDA Simulation Mode: {self.enable_cudasim}")
        
        print("=" * 60)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"NumbaConfig(version='{self.version}', threading_layer='{self.threading_layer}')"
    
    def __str__(self) -> str:
        """Human-readable string."""
        return f"Numba {self.version} (Threading: {self.threading_layer})"


def get_numba_config() -> NumbaConfig:
    """
    Get Numba configuration.
    
    Returns:
        NumbaConfig instance
    """
    return NumbaConfig()
