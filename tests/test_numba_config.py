"""Tests for Numba Configuration wrapper."""

import pytest
import numpy as np
from parallel_processing import NumbaConfig, get_numba_config


class TestNumbaConfig:
    """Test NumbaConfig class."""
    
    @pytest.fixture
    def config(self):
        """Fixture to get Numba config."""
        return get_numba_config()
    
    def test_config_initialization(self, config):
        """Test config initialization."""
        assert config is not None
        assert isinstance(config, NumbaConfig)
        assert hasattr(config, '_config')
    
    def test_version(self, config):
        """Test getting Numba version."""
        version = config.version
        assert isinstance(version, str)
        assert len(version) > 0
        # Version should have format like "0.60.0"
        assert '.' in version
    
    def test_threading_layer(self, config):
        """Test getting threading layer."""
        layer = config.threading_layer
        assert isinstance(layer, str)
        # Common threading layers
        assert layer in ['default', 'safe', 'forksafe', 'threadsafe', 'workqueue', 'tbb', 'omp']
    
    def test_num_threads(self, config):
        """Test getting number of threads."""
        num = config.num_threads
        assert isinstance(num, int)
        assert num > 0
        # Should be at least 1 and reasonable (not more than 1024)
        assert 1 <= num <= 1024
    
    def test_threading_layer_priority(self, config):
        """Test getting threading layer priority."""
        priority = config.threading_layer_priority
        # Can be None or a string
        assert priority is None or isinstance(priority, str)
    
    def test_has_parallel_support(self, config):
        """Test parallel support check."""
        has_parallel = config.has_parallel_support
        assert isinstance(has_parallel, bool)
    
    def test_has_tbb(self, config):
        """Test TBB availability check."""
        has_tbb = config.has_tbb
        assert isinstance(has_tbb, bool)
    
    def test_has_openmp(self, config):
        """Test OpenMP availability check."""
        has_openmp = config.has_openmp
        assert isinstance(has_openmp, bool)
    
    def test_active_thread_count(self, config):
        """Test getting active thread count."""
        count = config.active_thread_count
        # Can be None or an integer
        assert count is None or isinstance(count, int)
        if count is not None:
            assert count > 0
    
    def test_llvm_version(self, config):
        """Test getting LLVM version."""
        version = config.llvm_version
        assert isinstance(version, str)
        assert len(version) > 0
        # Should not be "Unknown" in a proper installation
        # but we allow it in case llvmlite is not available
        assert version != ""
    
    def test_cuda_available(self, config):
        """Test CUDA availability check."""
        cuda_avail = config.cuda_available
        assert isinstance(cuda_avail, bool)
    
    def test_cuda_log_level(self, config):
        """Test CUDA log level."""
        log_level = config.cuda_log_level
        # Can be None or a string
        assert log_level is None or isinstance(log_level, str)
    
    def test_enable_cudasim(self, config):
        """Test CUDA simulation mode check."""
        cudasim = config.enable_cudasim
        assert isinstance(cudasim, bool)
    
    def test_print_info(self, config, capsys):
        """Test print_info method."""
        config.print_info()
        captured = capsys.readouterr()
        
        # Check that output contains expected sections
        assert "Numba Build Configuration" in captured.out
        assert "Numba Version:" in captured.out
        assert "Threading Layer Options:" in captured.out
        assert "Threading Backend Support:" in captured.out
        assert "LLVM Configuration:" in captured.out
        assert "CUDA Configuration:" in captured.out
    
    def test_repr(self, config):
        """Test __repr__ method."""
        repr_str = repr(config)
        assert "NumbaConfig" in repr_str
        assert "version=" in repr_str
        assert "threading_layer=" in repr_str
        assert config.version in repr_str
    
    def test_str(self, config):
        """Test __str__ method."""
        str_repr = str(config)
        assert "Numba" in str_repr
        assert config.version in str_repr
        assert config.threading_layer in str_repr


class TestGetNumbaConfig:
    """Test get_numba_config helper function."""
    
    def test_get_numba_config(self):
        """Test get_numba_config function."""
        config = get_numba_config()
        assert isinstance(config, NumbaConfig)
    
    def test_get_numba_config_multiple_calls(self):
        """Test that multiple calls work independently."""
        config1 = get_numba_config()
        config2 = get_numba_config()
        
        # Should both be valid NumbaConfig instances
        assert isinstance(config1, NumbaConfig)
        assert isinstance(config2, NumbaConfig)
        
        # Should have same version
        assert config1.version == config2.version


class TestNumbaConfigProperties:
    """Test relationships between NumbaConfig properties."""
    
    @pytest.fixture
    def config(self):
        """Fixture to get Numba config."""
        return get_numba_config()
    
    def test_threading_consistency(self, config):
        """Test that threading properties are consistent."""
        # If we have parallel support, we should have a threading layer
        if config.has_parallel_support:
            assert config.threading_layer is not None
            assert config.num_threads > 0
    
    def test_tbb_openmp_mutual_exclusion(self, config):
        """Test that TBB and OpenMP are typically mutually exclusive."""
        # Note: They could theoretically both be available, but typically
        # only one is active at a time
        if config.has_tbb or config.has_openmp:
            assert config.threading_layer_priority is not None
    
    def test_cuda_consistency(self, config):
        """Test CUDA-related properties are consistent."""
        # If CUDA simulation is enabled, CUDA might still be unavailable
        if config.enable_cudasim:
            # Simulation mode doesn't require real CUDA
            assert isinstance(config.cuda_available, bool)
    
    def test_version_format(self, config):
        """Test that version strings follow expected format."""
        # Numba version should be X.Y.Z format
        version_parts = config.version.split('.')
        assert len(version_parts) >= 2
        # Major and minor should be numeric
        assert version_parts[0].isdigit()
        assert version_parts[1].isdigit()


class TestNumbaConfigOutput:
    """Test NumbaConfig output formatting."""
    
    @pytest.fixture
    def config(self):
        """Fixture to get Numba config."""
        return get_numba_config()
    
    def test_print_info_contains_version(self, config, capsys):
        """Test that print_info includes version."""
        config.print_info()
        captured = capsys.readouterr()
        assert config.version in captured.out
    
    def test_print_info_contains_threading_info(self, config, capsys):
        """Test that print_info includes threading info."""
        config.print_info()
        captured = capsys.readouterr()
        assert str(config.num_threads) in captured.out
        assert config.threading_layer in captured.out
    
    def test_print_info_contains_llvm(self, config, capsys):
        """Test that print_info includes LLVM info."""
        config.print_info()
        captured = capsys.readouterr()
        assert "LLVM" in captured.out
    
    def test_print_info_formatting(self, config, capsys):
        """Test that print_info has proper formatting."""
        config.print_info()
        captured = capsys.readouterr()
        
        # Check for separator lines
        assert "=" * 60 in captured.out
        # Check for proper indentation
        assert "  " in captured.out


class TestNumbaCudaFunctionality:
    """Test Numba CUDA functionality."""
    
    def test_numba_jit_compilation(self):
        """Test basic Numba JIT compilation."""
        from numba import jit
        
        @jit(nopython=True)
        def test_function(x):
            return x * 2 + 1
        
        result = test_function(5)
        assert result == 11
    
    def test_cuda_availability(self):
        """Test CUDA availability check."""
        from numba import cuda
        
        # Should return a boolean
        is_available = cuda.is_available()
        assert isinstance(is_available, bool)
    
    @pytest.mark.skipif(
        not get_numba_config().cuda_available,
        reason="CUDA not available"
    )
    def test_cuda_simple_kernel(self):
        """Test simple CUDA kernel execution."""
        from numba import cuda
        
        @cuda.jit
        def add_kernel(x, y, out):
            idx = cuda.grid(1)
            if idx < out.size:
                out[idx] = x[idx] + y[idx]
        
        # Test the kernel with device arrays
        n = 100000
        x_host = np.arange(n).astype(np.float32)
        y_host = np.ones(n, dtype=np.float32)
        
        # Transfer to GPU
        x_device = cuda.to_device(x_host)
        y_device = cuda.to_device(y_host)
        out_device = cuda.device_array(n, dtype=np.float32)
        
        threadsperblock = 256
        blockspergrid = (n + threadsperblock - 1) // threadsperblock
        add_kernel[blockspergrid, threadsperblock](x_device, y_device, out_device)
        
        # Copy result back to host
        out = out_device.copy_to_host()
        
        # Verify results
        expected = x_host + y_host
        np.testing.assert_array_almost_equal(out, expected, decimal=5)
        
        # Check first few values explicitly
        assert np.allclose(out[:5], [1., 2., 3., 4., 5.])
    
    @pytest.mark.skipif(
        not get_numba_config().cuda_available,
        reason="CUDA not available"
    )
    def test_cuda_grid_configuration(self):
        """Test CUDA kernel grid configuration."""
        from numba import cuda
        
        @cuda.jit
        def dummy_kernel(out):
            idx = cuda.grid(1)
            if idx < out.size:
                out[idx] = idx
        
        # Use larger array for better GPU utilization (modern GPUs have thousands of cores)
        n = 1_000_000
        out_device = cuda.device_array(n, dtype=np.int32)
        
        threadsperblock = 256
        blockspergrid = (n + threadsperblock - 1) // threadsperblock
        
        # Verify grid configuration
        assert threadsperblock == 256
        assert blockspergrid == (n + 255) // 256
        
        dummy_kernel[blockspergrid, threadsperblock](out_device)
        out = out_device.copy_to_host()
        
        # Verify output (check first/last elements to avoid large array comparison)
        assert out[0] == 0
        assert out[-1] == n - 1
        # Check a sample in the middle
        assert out[n//2] == n//2


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
