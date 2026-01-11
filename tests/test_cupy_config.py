"""Tests for CuPy configuration module."""

import pytest
import cupy as cp
from parallel_processing.cupy_config import CupyConfig, get_cupy_config


class TestCupyConfig:
    """Test suite for CupyConfig class."""
    
    def test_initialization(self):
        """Test CupyConfig initialization."""
        config = CupyConfig()
        assert config is not None
        assert isinstance(config, CupyConfig)
    
    def test_version(self):
        """Test version property."""
        config = CupyConfig()
        assert isinstance(config.version, str)
        assert config.version == cp.__version__
        assert len(config.version) > 0
    
    def test_numpy_version(self):
        """Test numpy_version property."""
        config = CupyConfig()
        assert isinstance(config.numpy_version, str)
        assert len(config.numpy_version) > 0
    
    def test_python_version(self):
        """Test python_version property."""
        config = CupyConfig()
        assert isinstance(config.python_version, str)
        assert len(config.python_version) > 0
        # Should be in format like "3.12.12"
        parts = config.python_version.split('.')
        assert len(parts) >= 2
    
    def test_cuda_available(self):
        """Test cuda_available property."""
        config = CupyConfig()
        assert isinstance(config.cuda_available, bool)
    
    def test_device_count(self):
        """Test device_count property."""
        config = CupyConfig()
        assert isinstance(config.device_count, int)
        assert config.device_count >= 0
    
    def test_cuda_driver_version(self):
        """Test cuda_driver_version property."""
        config = CupyConfig()
        driver_version = config.cuda_driver_version
        
        if driver_version is not None:
            assert isinstance(driver_version, tuple)
            assert len(driver_version) == 2
            major, minor = driver_version
            assert isinstance(major, int)
            assert isinstance(minor, int)
            assert major >= 0
            assert minor >= 0
    
    def test_cuda_runtime_version(self):
        """Test cuda_runtime_version property."""
        config = CupyConfig()
        runtime_version = config.cuda_runtime_version
        
        if runtime_version is not None:
            assert isinstance(runtime_version, tuple)
            assert len(runtime_version) == 2
            major, minor = runtime_version
            assert isinstance(major, int)
            assert isinstance(minor, int)
            assert major >= 0
            assert minor >= 0
    
    def test_cuda_path(self):
        """Test cuda_path property."""
        config = CupyConfig()
        cuda_path = config.cuda_path
        
        if cuda_path is not None:
            assert isinstance(cuda_path, str)
            assert len(cuda_path) > 0
    
    def test_current_device_id(self):
        """Test current_device_id property."""
        config = CupyConfig()
        device_id = config.current_device_id
        
        assert isinstance(device_id, int)
        assert device_id >= 0
        
        if config.device_count > 0:
            assert device_id < config.device_count
    
    def test_memory_pool_enabled(self):
        """Test memory_pool_enabled property."""
        config = CupyConfig()
        assert isinstance(config.memory_pool_enabled, bool)
    
    def test_pinned_memory_pool_enabled(self):
        """Test pinned_memory_pool_enabled property."""
        config = CupyConfig()
        assert isinstance(config.pinned_memory_pool_enabled, bool)
    
    def test_get_memory_pool_stats(self):
        """Test get_memory_pool_stats method."""
        config = CupyConfig()
        stats = config.get_memory_pool_stats()
        
        assert isinstance(stats, dict)
        
        if stats:  # If memory pool is available
            assert 'used_bytes' in stats
            assert 'total_bytes' in stats
            assert 'free_blocks' in stats
            assert isinstance(stats['used_bytes'], int)
            assert isinstance(stats['total_bytes'], int)
            assert isinstance(stats['free_blocks'], int)
            assert stats['used_bytes'] >= 0
            assert stats['total_bytes'] >= 0
            assert stats['free_blocks'] >= 0
    
    def test_get_device_memory_info(self):
        """Test get_device_memory_info method."""
        config = CupyConfig()
        
        if config.cuda_available and config.device_count > 0:
            mem_info = config.get_device_memory_info()
            
            assert isinstance(mem_info, dict)
            assert 'free_bytes' in mem_info
            assert 'total_bytes' in mem_info
            assert 'used_bytes' in mem_info
            assert 'free_gb' in mem_info
            assert 'total_gb' in mem_info
            assert 'used_gb' in mem_info
            
            assert mem_info['free_bytes'] >= 0
            assert mem_info['total_bytes'] > 0
            assert mem_info['used_bytes'] >= 0
            assert mem_info['free_gb'] >= 0
            assert mem_info['total_gb'] > 0
            assert mem_info['used_gb'] >= 0
            
            # Check consistency
            assert mem_info['free_bytes'] + mem_info['used_bytes'] == mem_info['total_bytes']
    
    def test_get_device_memory_info_specific_device(self):
        """Test get_device_memory_info with specific device."""
        config = CupyConfig()
        
        if config.cuda_available and config.device_count > 0:
            mem_info = config.get_device_memory_info(device_id=0)
            
            assert isinstance(mem_info, dict)
            assert 'total_bytes' in mem_info
            assert mem_info['total_bytes'] > 0
    
    def test_has_cudnn(self):
        """Test has_cudnn property."""
        config = CupyConfig()
        assert isinstance(config.has_cudnn, bool)
    
    def test_has_cutensor(self):
        """Test has_cutensor property."""
        config = CupyConfig()
        assert isinstance(config.has_cutensor, bool)
    
    def test_cudnn_version(self):
        """Test cudnn_version property."""
        config = CupyConfig()
        cudnn_ver = config.cudnn_version
        
        if cudnn_ver is not None:
            assert isinstance(cudnn_ver, str)
            assert len(cudnn_ver) > 0
    
    def test_cutensor_version(self):
        """Test cutensor_version property."""
        config = CupyConfig()
        cutensor_ver = config.cutensor_version
        
        if cutensor_ver is not None:
            assert isinstance(cutensor_ver, str)
            assert len(cutensor_ver) > 0
    
    def test_print_info(self):
        """Test print_info method."""
        config = CupyConfig()
        # Should not raise an exception
        config.print_info()
    
    def test_repr(self):
        """Test __repr__ method."""
        config = CupyConfig()
        repr_str = repr(config)
        
        assert isinstance(repr_str, str)
        assert "CupyConfig" in repr_str
        assert "version" in repr_str
        assert config.version in repr_str
    
    def test_str(self):
        """Test __str__ method."""
        config = CupyConfig()
        str_repr = str(config)
        
        assert isinstance(str_repr, str)
        assert "CuPy" in str_repr
        assert config.version in str_repr


class TestGetCupyConfig:
    """Test suite for get_cupy_config function."""
    
    def test_get_cupy_config_returns_instance(self):
        """Test that get_cupy_config returns CupyConfig instance."""
        config = get_cupy_config()
        assert isinstance(config, CupyConfig)
    
    def test_get_cupy_config_creates_new_instance(self):
        """Test that get_cupy_config creates a new instance each time."""
        config1 = get_cupy_config()
        config2 = get_cupy_config()
        
        # Should be different instances
        assert config1 is not config2
        
        # But should have same configuration
        assert config1.version == config2.version
        assert config1.device_count == config2.device_count


class TestCupyConfigProperties:
    """Test suite for CupyConfig property consistency."""
    
    def test_version_consistency(self):
        """Test that version is consistent across calls."""
        config = CupyConfig()
        version1 = config.version
        version2 = config.version
        
        assert version1 == version2
    
    def test_device_count_consistency(self):
        """Test that device_count is consistent across calls."""
        config = CupyConfig()
        count1 = config.device_count
        count2 = config.device_count
        
        assert count1 == count2
    
    def test_cuda_versions_consistency(self):
        """Test that CUDA versions are consistent across calls."""
        config = CupyConfig()
        
        driver1 = config.cuda_driver_version
        driver2 = config.cuda_driver_version
        assert driver1 == driver2
        
        runtime1 = config.cuda_runtime_version
        runtime2 = config.cuda_runtime_version
        assert runtime1 == runtime2


class TestCupyConfigOutput:
    """Test suite for CupyConfig output methods."""
    
    def test_print_info_contains_version_info(self, capsys):
        """Test that print_info contains version information."""
        config = CupyConfig()
        config.print_info()
        
        captured = capsys.readouterr()
        assert "CuPy Configuration" in captured.out
        assert "Version Information" in captured.out
        assert config.version in captured.out
    
    def test_print_info_contains_cuda_info(self, capsys):
        """Test that print_info contains CUDA information."""
        config = CupyConfig()
        config.print_info()
        
        captured = capsys.readouterr()
        assert "CUDA Availability" in captured.out
        assert str(config.device_count) in captured.out
    
    def test_print_info_contains_memory_info(self, capsys):
        """Test that print_info contains memory information."""
        config = CupyConfig()
        
        if config.cuda_available and config.device_count > 0:
            config.print_info()
            
            captured = capsys.readouterr()
            assert "Memory" in captured.out


class TestCupyConfigIntegration:
    """Integration tests for CupyConfig."""
    
    def test_cupy_config_with_cuda_available(self):
        """Test CupyConfig when CUDA is available."""
        config = CupyConfig()
        
        if config.cuda_available:
            assert config.device_count > 0
            assert config.cuda_driver_version is not None
            assert config.cuda_runtime_version is not None
            
            mem_info = config.get_device_memory_info()
            assert mem_info
            assert mem_info['total_bytes'] > 0
    
    def test_memory_pool_stats_when_enabled(self):
        """Test memory pool stats when pool is enabled."""
        config = CupyConfig()
        
        if config.memory_pool_enabled:
            stats = config.get_memory_pool_stats()
            assert stats
            assert 'used_bytes' in stats
            assert 'total_bytes' in stats
    
    def test_all_properties_accessible(self):
        """Test that all properties are accessible without errors."""
        config = CupyConfig()
        
        # Access all properties
        _ = config.version
        _ = config.numpy_version
        _ = config.python_version
        _ = config.cuda_available
        _ = config.device_count
        _ = config.cuda_driver_version
        _ = config.cuda_runtime_version
        _ = config.cuda_path
        _ = config.current_device_id
        _ = config.memory_pool_enabled
        _ = config.pinned_memory_pool_enabled
        _ = config.has_cudnn
        _ = config.has_cutensor
        _ = config.cudnn_version
        _ = config.cutensor_version
        
        # Access all methods
        _ = config.get_memory_pool_stats()
        _ = config.get_device_memory_info()
        
        # No exceptions should be raised
        assert True
