"""Tests for CUDA Device Driver wrapper."""

import pytest

from parallel_processing import CudaDeviceDriver, get_driver_device, is_cuda_available


class TestCudaAvailability:
    """Test CUDA availability checking."""
    
    def test_is_cuda_available(self):
        """Test that CUDA availability check works."""
        result = is_cuda_available()
        assert isinstance(result, bool)
        
    def test_is_cuda_available_matches_device_creation(self):
        """Test that availability check matches device creation."""
        if is_cuda_available():
            # Should not raise
            device = CudaDeviceDriver(0)
            assert device is not None
        else:
            # Should raise RuntimeError
            with pytest.raises(RuntimeError):
                CudaDeviceDriver(0)


class TestCudaDeviceDriver:
    """Test CudaDeviceDriver class."""
    
    @pytest.fixture
    def device(self):
        """Fixture to get CUDA device, skip if not available."""
        if not is_cuda_available():
            pytest.skip("CUDA not available")
        return CudaDeviceDriver(0)
    
    def test_device_initialization(self, device):
        """Test device initialization."""
        assert device is not None
        assert device.device_id == 0
        assert hasattr(device, '_device')
    
    def test_device_id_out_of_range(self):
        """Test that invalid device ID raises error."""
        if not is_cuda_available():
            pytest.skip("CUDA not available")
        
        with pytest.raises(RuntimeError, match="not found"):
            CudaDeviceDriver(999)
    
    def test_device_name(self, device):
        """Test getting device name."""
        name = device.name
        assert isinstance(name, str)
        assert len(name) > 0
        # Typical NVIDIA GPU names contain these
        assert any(word in name for word in ['NVIDIA', 'GeForce', 'Tesla', 'Quadro', 'RTX', 'GTX'])
    
    def test_compute_capability(self, device):
        """Test getting compute capability."""
        major, minor = device.compute_capability
        assert isinstance(major, int)
        assert isinstance(minor, int)
        assert major >= 3  # Minimum supported compute capability
        assert 0 <= minor <= 9
    
    def test_total_memory(self, device):
        """Test getting total memory."""
        total_mem = device.total_memory
        assert isinstance(total_mem, int)
        assert total_mem > 0
        # Modern GPUs have at least 1GB
        assert total_mem >= 1024 * 1024 * 1024
    
    def test_free_memory(self, device):
        """Test getting free memory."""
        free_mem = device.free_memory
        assert isinstance(free_mem, int)
        assert free_mem >= 0
        # Free memory should be less than or equal to total
        assert free_mem <= device.total_memory
    
    def test_used_memory(self, device):
        """Test getting used memory."""
        used_mem = device.used_memory
        assert isinstance(used_mem, int)
        assert used_mem >= 0
        # Used + free should equal total
        assert abs((used_mem + device.free_memory) - device.total_memory) < 1024 * 1024  # Within 1MB tolerance
    
    def test_multiprocessor_count(self, device):
        """Test getting SM count."""
        sm_count = device.multiprocessor_count
        assert isinstance(sm_count, int)
        assert sm_count > 0
        # Modern GPUs have at least 1 SM
        assert sm_count >= 1
    
    def test_cuda_cores_per_sm(self, device):
        """Test getting CUDA cores per SM."""
        cores = device.cuda_cores_per_sm
        assert isinstance(cores, int)
        assert cores > 0
        # Reasonable range for CUDA cores per SM
        assert 32 <= cores <= 256
    
    def test_max_threads_per_block(self, device):
        """Test max threads per block."""
        max_threads = device.max_threads_per_block
        assert isinstance(max_threads, int)
        assert max_threads >= 512  # Minimum for compute 3.0+
        assert max_threads <= 2048  # Common maximum
    
    def test_max_block_dimensions(self, device):
        """Test max block dimensions."""
        dims = device.max_block_dimensions
        assert isinstance(dims, tuple)
        assert len(dims) == 3
        assert all(isinstance(d, int) for d in dims)
        assert all(d > 0 for d in dims)
        # x should be at least 512
        assert dims[0] >= 512
    
    def test_max_grid_dimensions(self, device):
        """Test max grid dimensions."""
        dims = device.max_grid_dimensions
        assert isinstance(dims, tuple)
        assert len(dims) == 3
        assert all(isinstance(d, int) for d in dims)
        assert all(d > 0 for d in dims)
        # Grid dimensions should be very large
        assert dims[0] >= 65535
    
    def test_warp_size(self, device):
        """Test warp size."""
        warp_size = device.warp_size
        assert isinstance(warp_size, int)
        # All NVIDIA GPUs use warp size 32
        assert warp_size == 32
    
    def test_max_threads_per_multiprocessor(self, device):
        """Test max threads per SM."""
        max_threads = device.max_threads_per_multiprocessor
        assert isinstance(max_threads, int)
        assert max_threads > 0
        # Typical range
        assert max_threads >= 1024
    
    def test_max_shared_memory_per_block(self, device):
        """Test max shared memory per block."""
        shared_mem = device.max_shared_memory_per_block
        assert isinstance(shared_mem, int)
        assert shared_mem > 0
        # At least 16KB for compute 3.0+
        assert shared_mem >= 16 * 1024
    
    def test_total_constant_memory(self, device):
        """Test total constant memory."""
        const_mem = device.total_constant_memory
        assert isinstance(const_mem, int)
        assert const_mem > 0
        # Typically 64KB
        assert const_mem >= 64 * 1024
    
    def test_l2_cache_size(self, device):
        """Test L2 cache size."""
        l2_cache = device.l2_cache_size
        assert isinstance(l2_cache, int)
        # Some older GPUs might not have L2 cache
        assert l2_cache >= 0
    
    def test_clock_rate(self, device):
        """Test clock rate."""
        clock_rate = device.clock_rate
        assert isinstance(clock_rate, int)
        assert clock_rate > 0
        # Reasonable range in kHz (500 MHz to 3 GHz)
        assert 500_000 <= clock_rate <= 3_000_000
    
    def test_memory_clock_rate(self, device):
        """Test memory clock rate."""
        mem_clock = device.memory_clock_rate
        assert isinstance(mem_clock, int)
        assert mem_clock > 0
    
    def test_memory_bus_width(self, device):
        """Test memory bus width."""
        bus_width = device.memory_bus_width
        assert isinstance(bus_width, int)
        assert bus_width > 0
        # Common bus widths: 64, 128, 192, 256, 384, 512 bits
        assert bus_width in [64, 96, 128, 192, 256, 320, 384, 512]
    
    def test_print_info(self, device, capsys):
        """Test print_info method."""
        device.print_info()
        captured = capsys.readouterr()
        
        # Check that output contains expected sections
        assert "NVIDIA GPU Device Information" in captured.out
        assert "Device Name:" in captured.out
        assert "Compute Capability:" in captured.out
        assert "Memory Information:" in captured.out
        assert "Hardware Specifications:" in captured.out
        assert "Performance Metrics:" in captured.out
    
    def test_repr(self, device):
        """Test __repr__ method."""
        repr_str = repr(device)
        assert "CudaDriverDevice" in repr_str
        assert "id=0" in repr_str
        assert device.name.strip() in repr_str  # Strip whitespace from name
    
    def test_str(self, device):
        """Test __str__ method."""
        str_repr = str(device)
        assert device.name in str_repr
        assert "Compute" in str_repr


class TestGetDriverDevice:
    """Test get_driver_device helper function."""
    
    def test_get_driver_device(self):
        """Test get_driver_device function."""
        if not is_cuda_available():
            pytest.skip("CUDA not available")
        
        device = get_driver_device(0)
        assert isinstance(device, CudaDeviceDriver)
        assert device.device_id == 0
    
    def test_get_driver_device_invalid_id(self):
        """Test get_driver_device with invalid ID."""
        if not is_cuda_available():
            pytest.skip("CUDA not available")
        
        with pytest.raises(RuntimeError):
            get_driver_device(999)


class TestMemoryConsistency:
    """Test memory reporting consistency."""
    
    @pytest.fixture
    def device(self):
        """Fixture to get CUDA device."""
        if not is_cuda_available():
            pytest.skip("CUDA not available")
        return CudaDeviceDriver(0)
    
    def test_memory_values_are_consistent(self, device):
        """Test that memory values are self-consistent."""
        total = device.total_memory
        free = device.free_memory
        used = device.used_memory
        
        # All should be non-negative
        assert total >= 0
        assert free >= 0
        assert used >= 0
        
        # Free should not exceed total
        assert free <= total
        
        # Used + Free should approximately equal Total (within 1% tolerance)
        assert abs((used + free) - total) <= total * 0.01
    
    def test_memory_values_change_correctly(self, device):
        """Test that free memory can change (due to allocations)."""
        free1 = device.free_memory
        free2 = device.free_memory
        
        # Both calls should return valid values
        assert free1 >= 0
        assert free2 >= 0
        
        # Should be close but may vary slightly
        assert abs(free1 - free2) <= device.total_memory * 0.1  # Within 10%


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
