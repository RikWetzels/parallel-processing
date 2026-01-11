"""CUDA Device Information using direct driver API (cuda-python)."""

from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart
from typing import Optional


class CudaDeviceDriver:
    """Wrapper for CUDA device using low-level driver API only."""
    
    def __init__(self, device_id: int = 0):
        """
        Initialize CUDA device using driver API.
        
        Args:
            device_id: CUDA device ID (default: 0)
        
        Raises:
            RuntimeError: If CUDA is not available or initialization fails
        """
        # Initialize CUDA driver
        err, = cuda.cuInit(0)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA driver initialization failed: {err}")
        
        # Get device count
        err, device_count = cuda.cuDeviceGetCount()
        if err != cuda.CUresult.CUDA_SUCCESS or device_count == 0:
            raise RuntimeError("No CUDA devices found")
        
        if device_id >= device_count:
            raise RuntimeError(f"Device {device_id} not found (only {device_count} devices available)")
        
        # Get device handle
        err, self._device = cuda.cuDeviceGet(device_id)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Failed to get device {device_id}: {err}")
        
        self.device_id = device_id
    
    def _get_attribute(self, attrib: cuda.CUdevice_attribute) -> int:
        """Get device attribute value."""
        err, value = cuda.cuDeviceGetAttribute(attrib, self._device)
        if err != cuda.CUresult.CUDA_SUCCESS:
            return 0
        return value
    
    @property
    def name(self) -> str:
        """Get device name."""
        err, name = cuda.cuDeviceGetName(128, self._device)
        if err != cuda.CUresult.CUDA_SUCCESS:
            return "Unknown"
        return name.decode('utf-8') if isinstance(name, bytes) else name
    
    @property
    def compute_capability(self) -> tuple[int, int]:
        """Get compute capability as (major, minor)."""
        major = self._get_attribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        minor = self._get_attribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        return (major, minor)
    
    @property
    def total_memory(self) -> int:
        """Get total device memory in bytes."""
        err, total_mem = cuda.cuDeviceTotalMem(self._device)
        if err != cuda.CUresult.CUDA_SUCCESS:
            return 0
        return total_mem
    
    @property
    def free_memory(self) -> int:
        """Get free device memory in bytes (requires runtime API)."""
        err, free, total = cudart.cudaMemGetInfo()
        if err != cudart.cudaError_t.cudaSuccess:
            return 0
        return free
    
    @property
    def used_memory(self) -> int:
        """Get used device memory in bytes."""
        return self.total_memory - self.free_memory
    
    @property
    def multiprocessor_count(self) -> int:
        """Get number of streaming multiprocessors (SMs)."""
        return self._get_attribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    
    @property
    def max_threads_per_block(self) -> int:
        """Get maximum threads per block."""
        return self._get_attribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    
    @property
    def max_block_dim_x(self) -> int:
        """Get maximum block dimension X."""
        return self._get_attribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
    
    @property
    def max_block_dim_y(self) -> int:
        """Get maximum block dimension Y."""
        return self._get_attribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)
    
    @property
    def max_block_dim_z(self) -> int:
        """Get maximum block dimension Z."""
        return self._get_attribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)
    
    @property
    def max_block_dimensions(self) -> tuple[int, int, int]:
        """Get maximum block dimensions."""
        return (self.max_block_dim_x, self.max_block_dim_y, self.max_block_dim_z)
    
    @property
    def max_grid_dim_x(self) -> int:
        """Get maximum grid dimension X."""
        return self._get_attribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
    
    @property
    def max_grid_dim_y(self) -> int:
        """Get maximum grid dimension Y."""
        return self._get_attribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
    
    @property
    def max_grid_dim_z(self) -> int:
        """Get maximum grid dimension Z."""
        return self._get_attribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)
    
    @property
    def max_grid_dimensions(self) -> tuple[int, int, int]:
        """Get maximum grid dimensions."""
        return (self.max_grid_dim_x, self.max_grid_dim_y, self.max_grid_dim_z)
    
    @property
    def warp_size(self) -> int:
        """Get warp size."""
        return self._get_attribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE)
    
    @property
    def max_threads_per_multiprocessor(self) -> int:
        """Get maximum threads per multiprocessor."""
        return self._get_attribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    
    @property
    def max_shared_memory_per_block(self) -> int:
        """Get maximum shared memory per block in bytes."""
        return self._get_attribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    
    @property
    def total_constant_memory(self) -> int:
        """Get total constant memory in bytes."""
        return self._get_attribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY)
    
    @property
    def l2_cache_size(self) -> int:
        """Get L2 cache size in bytes."""
        return self._get_attribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
    
    @property
    def clock_rate(self) -> int:
        """Get clock rate in kHz."""
        return self._get_attribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE)
    
    @property
    def memory_clock_rate(self) -> int:
        """Get memory clock rate in kHz."""
        return self._get_attribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
    
    @property
    def memory_bus_width(self) -> int:
        """Get memory bus width in bits."""
        return self._get_attribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
    
    @property
    def cuda_cores_per_sm(self) -> int:
        """Estimate CUDA cores per SM based on compute capability."""
        major, minor = self.compute_capability
        
        # CUDA cores per SM varies by architecture
        cores_per_sm_map = {
            (3, 0): 192,  # Kepler
            (3, 5): 192,
            (3, 7): 192,
            (5, 0): 128,  # Maxwell
            (5, 2): 128,
            (6, 0): 64,   # Pascal
            (6, 1): 128,
            (7, 0): 64,   # Volta
            (7, 5): 64,   # Turing
            (8, 0): 64,   # Ampere
            (8, 6): 128,
            (8, 9): 128,  # Ada Lovelace
            (9, 0): 128,  # Hopper
        }
        
        return cores_per_sm_map.get((major, minor), 128)  # Default estimate
    
    def print_info(self) -> None:
        """Print comprehensive device information."""
        print("=" * 60)
        print("NVIDIA GPU Device Information (Driver API)")
        print("=" * 60)
        
        # Device name and compute capability
        print(f"Device Name: {self.name}")
        print(f"Compute Capability: {self.compute_capability}")
        
        # Memory information
        print(f"\nMemory Information:")
        print(f"  Total Memory: {self.total_memory / 1024**3:.2f} GB ({self.total_memory / 1024**2:.0f} MB)")
        print(f"  Free Memory:  {self.free_memory / 1024**3:.2f} GB ({self.free_memory / 1024**2:.0f} MB)")
        print(f"  Used Memory:  {self.used_memory / 1024**3:.2f} GB ({self.used_memory / 1024**2:.0f} MB)")
        print(f"  Memory Bus Width: {self.memory_bus_width} bits")
        
        # Hardware specs
        print(f"\nHardware Specifications:")
        print(f"  Multiprocessors (SMs): {self.multiprocessor_count}")
        print(f"  CUDA Cores per SM: ~{self.cuda_cores_per_sm}")
        print(f"  Total CUDA Cores: ~{self.multiprocessor_count * self.cuda_cores_per_sm}")
        print(f"  Max Threads per Block: {self.max_threads_per_block}")
        print(f"  Max Block Dimensions: {self.max_block_dimensions[0]} x {self.max_block_dimensions[1]} x {self.max_block_dimensions[2]}")
        print(f"  Max Grid Dimensions: {self.max_grid_dimensions[0]} x {self.max_grid_dimensions[1]} x {self.max_grid_dimensions[2]}")
        print(f"  Warp Size: {self.warp_size}")
        print(f"  Max Threads per SM: {self.max_threads_per_multiprocessor}")
        
        # Performance metrics
        print(f"\nPerformance Metrics:")
        print(f"  Max Shared Memory per Block: {self.max_shared_memory_per_block / 1024:.0f} KB")
        print(f"  Total Constant Memory: {self.total_constant_memory / 1024:.0f} KB")
        print(f"  L2 Cache Size: {self.l2_cache_size / 1024:.0f} KB")
        print(f"  Clock Rate: {self.clock_rate / 1000:.2f} MHz")
        print(f"  Memory Clock Rate: {self.memory_clock_rate / 1000:.2f} MHz")
        
        print("=" * 60)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"CudaDriverDevice(id={self.device_id}, name='{self.name}', compute_capability={self.compute_capability})"
    
    def __str__(self) -> str:
        """Human-readable string."""
        return f"{self.name} (Compute {self.compute_capability[0]}.{self.compute_capability[1]})"


def get_driver_device(device_id: int = 0) -> CudaDeviceDriver:
    """
    Get CUDA device using driver API only.
    
    Args:
        device_id: Device ID (default: 0)
    
    Returns:
        CudaDriverDevice instance
    """
    return CudaDeviceDriver(device_id)


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        err, = cuda.cuInit(0)
        if err != cuda.CUresult.CUDA_SUCCESS:
            return False
        err, count = cuda.cuDeviceGetCount()
        return err == cuda.CUresult.CUDA_SUCCESS and count > 0
    except:
        return False
