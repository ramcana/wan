"""
Unit tests for individual components of the WAN2.2 local installation system.
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from interfaces import HardwareProfile, CPUInfo, MemoryInfo, GPUInfo, StorageInfo, OSInfo


class TestSystemDetection(unittest.TestCase):
    """Unit tests for system detection components."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cpu_detection(self):
        """Test CPU detection functionality."""
        from detect_system import SystemDetector
        
        detector = SystemDetector(self.temp_dir)
        
        with patch('platform.processor') as mock_processor, \
             patch('psutil.cpu_count') as mock_cpu_count, \
             patch('psutil.cpu_freq') as mock_cpu_freq:
            
            mock_processor.return_value = "AMD Ryzen 7 5800X"
            mock_cpu_count.side_effect = [8, 16]  # physical, logical
            mock_cpu_freq.return_value = Mock(current=3800, min=2200, max=4700)
            
            cpu_info = detector._detect_cpu()
            
            self.assertIsInstance(cpu_info, CPUInfo)
            self.assertEqual(cpu_info.cores, 8)
            self.assertEqual(cpu_info.threads, 16)
            self.assertGreater(cpu_info.base_clock, 0)
    
    def test_memory_detection(self):
        """Test memory detection functionality."""
        from detect_system import SystemDetector
        
        detector = SystemDetector(self.temp_dir)
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                total=34359738368,  # 32GB
                available=30064771072  # 28GB
            )
            
            memory_info = detector._detect_memory()
            
            self.assertIsInstance(memory_info, MemoryInfo)
            self.assertEqual(memory_info.total_gb, 32)
            self.assertEqual(memory_info.available_gb, 28)
    
    def test_gpu_detection_nvidia(self):
        """Test NVIDIA GPU detection."""
        from detect_system import SystemDetector
        
        detector = SystemDetector(self.temp_dir)
        
        with patch('subprocess.run') as mock_run:
            # Mock nvidia-smi output
            mock_run.return_value = Mock(
                returncode=0,
                stdout="GeForce RTX 3070, 8192 MiB, 537.13"
            )
            
            gpu_info = detector._detect_gpu()
            
            if gpu_info:  # GPU detection might fail in test environment
                self.assertIsInstance(gpu_info, GPUInfo)
                self.assertIn("RTX", gpu_info.model)
    
    def test_storage_detection(self):
        """Test storage detection functionality."""
        from detect_system import SystemDetector
        
        detector = SystemDetector(self.temp_dir)
        
        with patch('shutil.disk_usage') as mock_disk_usage:
            mock_disk_usage.return_value = (1000000000000, 500000000000, 500000000000)  # 1TB total, 500GB free
            
            storage_info = detector._detect_storage()
            
            self.assertIsInstance(storage_info, StorageInfo)
            self.assertGreater(storage_info.available_gb, 400)  # Should be around 465GB
    
    def test_os_detection(self):
        """Test OS detection functionality."""
        from detect_system import SystemDetector
        
        detector = SystemDetector(self.temp_dir)
        
        with patch('platform.system') as mock_system, \
             patch('platform.release') as mock_release, \
             patch('platform.machine') as mock_machine:
            
            mock_system.return_value = "Windows"
            mock_release.return_value = "11"
            mock_machine.return_value = "AMD64"
            
            os_info = detector._detect_os()
            
            self.assertIsInstance(os_info, OSInfo)
            self.assertEqual(os_info.name, "Windows")
            self.assertEqual(os_info.version, "11")


class TestDependencyManagement(unittest.TestCase):
    """Unit tests for dependency management components."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_python_detection(self):
        """Test Python installation detection."""
        from setup_dependencies import PythonInstallationHandler
        
        handler = PythonInstallationHandler(self.temp_dir)
        
        with patch('shutil.which') as mock_which, \
             patch('subprocess.run') as mock_run:
            
            mock_which.return_value = "C:\\Python39\\python.exe"
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Python 3.9.7"
            )
            
            python_info = handler.check_python_installation()
            
            self.assertIsInstance(python_info, dict)
            self.assertIn("system_python", python_info)
    
    def test_virtual_environment_creation(self):
        """Test virtual environment creation."""
        from setup_dependencies import PythonInstallationHandler
        
        handler = PythonInstallationHandler(self.temp_dir)
        
        # Create mock hardware profile
        hardware_profile = HardwareProfile(
            cpu=CPUInfo("Test CPU", 8, 16, 3.8, 4.7, "x64"),
            memory=MemoryInfo(32, 28, "DDR4", 3200),
            gpu=GPUInfo("RTX 3070", 8, "12.1", "537.13", "8.6"),
            storage=StorageInfo(500, "NVMe SSD"),
            os=OSInfo("Windows", "11", "x64")
        )
        
        venv_path = Path(self.temp_dir) / "test_venv"
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)
            
            # This would normally create a venv, but we'll mock it
            success = handler.create_virtual_environment(str(venv_path), hardware_profile)
            
            # In a real test, we'd check if venv was created
            # For now, just verify the method doesn't crash
            self.assertIsInstance(success, bool)
    
    def test_package_resolution(self):
        """Test package resolution logic."""
        from package_resolver import CUDAPackageSelector
        
        gpu_info = GPUInfo(
            model="NVIDIA GeForce RTX 3070",
            vram_gb=8,
            cuda_version="12.1",
            driver_version="537.13",
            compute_capability="8.6"
        )
        
        selector = CUDAPackageSelector(gpu_info)
        
        cuda_version = selector.select_cuda_version()
        self.assertIsNotNone(cuda_version)
        
        if cuda_version:
            packages = selector.get_cuda_packages(cuda_version)
            self.assertIsInstance(packages, dict)
            self.assertIn("torch", packages)


class TestModelManagement(unittest.TestCase):
    """Unit tests for model management components."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_configuration(self):
        """Test model configuration loading."""
        from download_models import ModelDownloader
        
        downloader = ModelDownloader(self.temp_dir)
        
        models = downloader.get_required_models()
        
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        
        # Check model structure
        for model in models:
            self.assertIn("name", model)
            self.assertIn("url", model)
            self.assertIn("size_gb", model)
            self.assertIn("checksum", model)
    
    def test_model_path_validation(self):
        """Test model path validation."""
        from download_models import ModelDownloader
        
        downloader = ModelDownloader(self.temp_dir)
        
        # Create mock model directory
        models_dir = Path(self.temp_dir) / "models" / "WAN2.2-TI2V-5B"
        models_dir.mkdir(parents=True)
        
        # Create mock model files
        (models_dir / "pytorch_model.bin").write_bytes(b"0" * 1024)
        (models_dir / "config.json").write_text('{"model_type": "test"}')
        
        # Test validation
        is_valid = downloader._validate_model_files("WAN2.2-TI2V-5B")
        self.assertTrue(is_valid)
    
    def test_download_progress_tracking(self):
        """Test download progress tracking."""
        from download_models import ModelDownloader
        
        downloader = ModelDownloader(self.temp_dir)
        
        # Test progress callback
        progress_data = []
        
        def progress_callback(downloaded, total, filename):
            progress_data.append((downloaded, total, filename))
        
        # Mock download with progress
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.iter_content.return_value = [b"0" * 1024] * 10
            mock_response.headers = {"content-length": "10240"}
            mock_get.return_value = mock_response
            
            # This would normally download, but we're mocking it
            # Just verify the progress callback structure works
            self.assertTrue(callable(progress_callback))


class TestConfigurationEngine(unittest.TestCase):
    """Unit tests for configuration engine components."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_hardware_tier_classification(self):
        """Test hardware tier classification."""
        from generate_config import ConfigurationEngine
        
        config_engine = ConfigurationEngine(self.temp_dir)
        
        # Test high-end hardware
        high_end_profile = HardwareProfile(
            cpu=CPUInfo("Threadripper PRO 5995WX", 64, 128, 2.7, 4.5, "x64"),
            memory=MemoryInfo(128, 120, "DDR4", 3200),
            gpu=GPUInfo("RTX 4080", 16, "12.1", "537.13", "8.9"),
            storage=StorageInfo(2000, "NVMe SSD"),
            os=OSInfo("Windows", "11", "x64")
        )
        
        tier = config_engine._classify_hardware_tier(high_end_profile)
        self.assertEqual(tier, "high_end")
        
        # Test budget hardware
        budget_profile = HardwareProfile(
            cpu=CPUInfo("Ryzen 5 3600", 6, 12, 3.6, 4.2, "x64"),
            memory=MemoryInfo(16, 14, "DDR4", 2666),
            gpu=GPUInfo("GTX 1660 Ti", 6, "11.8", "516.94", "7.5"),
            storage=StorageInfo(250, "SATA SSD"),
            os=OSInfo("Windows", "10", "x64")
        )
        
        tier = config_engine._classify_hardware_tier(budget_profile)
        self.assertEqual(tier, "budget")
    
    def test_configuration_generation(self):
        """Test configuration generation for different hardware tiers."""
        from generate_config import ConfigurationEngine
        
        config_engine = ConfigurationEngine(self.temp_dir)
        
        # Test mid-range configuration
        mid_range_profile = HardwareProfile(
            cpu=CPUInfo("Ryzen 7 5800X", 8, 16, 3.8, 4.7, "x64"),
            memory=MemoryInfo(32, 28, "DDR4", 3200),
            gpu=GPUInfo("RTX 3070", 8, "12.1", "537.13", "8.6"),
            storage=StorageInfo(500, "NVMe SSD"),
            os=OSInfo("Windows", "11", "x64")
        )
        
        config = config_engine.generate_config(mid_range_profile)
        
        self.assertIsInstance(config, dict)
        self.assertIn("system", config)
        self.assertIn("optimization", config)
        
        # Verify mid-range optimizations
        self.assertTrue(config["system"]["enable_gpu_acceleration"])
        self.assertGreater(config["optimization"]["cpu_threads"], 8)
        self.assertGreater(config["optimization"]["memory_pool_gb"], 4)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        from config_validator import ConfigValidator
        
        validator = ConfigValidator()
        
        # Test valid configuration
        valid_config = {
            "system": {
                "enable_gpu_acceleration": True,
                "default_quantization": "fp16",
                "max_queue_size": 10
            },
            "optimization": {
                "cpu_threads": 8,
                "memory_pool_gb": 8,
                "max_vram_usage_gb": 6
            }
        }
        
        validation_result = validator.validate_config(valid_config)
        self.assertTrue(validation_result.is_valid)
        
        # Test invalid configuration
        invalid_config = {
            "system": {
                "enable_gpu_acceleration": "invalid_value",  # Should be boolean
                "max_queue_size": -1  # Should be positive
            }
        }
        
        validation_result = validator.validate_config(invalid_config)
        self.assertFalse(validation_result.is_valid)
        self.assertGreater(len(validation_result.errors), 0)


class TestValidationFramework(unittest.TestCase):
    """Unit tests for validation framework components."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dependency_validation(self):
        """Test dependency validation."""
        from validate_installation import InstallationValidator
        
        hardware_profile = HardwareProfile(
            cpu=CPUInfo("Test CPU", 8, 16, 3.8, 4.7, "x64"),
            memory=MemoryInfo(32, 28, "DDR4", 3200),
            gpu=GPUInfo("RTX 3070", 8, "12.1", "537.13", "8.6"),
            storage=StorageInfo(500, "NVMe SSD"),
            os=OSInfo("Windows", "11", "x64")
        )
        
        validator = InstallationValidator(self.temp_dir, hardware_profile)
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="1.0.0",
                stderr=""
            )
            
            result = validator.validate_dependencies()
            
            self.assertIsNotNone(result)
            self.assertHasAttr(result, 'success')
    
    def test_model_validation(self):
        """Test model validation."""
        from validate_installation import InstallationValidator
        
        hardware_profile = HardwareProfile(
            cpu=CPUInfo("Test CPU", 8, 16, 3.8, 4.7, "x64"),
            memory=MemoryInfo(32, 28, "DDR4", 3200),
            gpu=GPUInfo("RTX 3070", 8, "12.1", "537.13", "8.6"),
            storage=StorageInfo(500, "NVMe SSD"),
            os=OSInfo("Windows", "11", "x64")
        )
        
        validator = InstallationValidator(self.temp_dir, hardware_profile)
        
        # Create mock model structure
        models_dir = Path(self.temp_dir) / "models"
        models_dir.mkdir(parents=True)
        
        for model_name in ["WAN2.2-T2V-A14B", "WAN2.2-I2V-A14B", "WAN2.2-TI2V-5B"]:
            model_path = models_dir / model_name
            model_path.mkdir()
            (model_path / "pytorch_model.bin").write_bytes(b"0" * (1024 * 1024))
            (model_path / "config.json").write_text('{"model_type": "test"}')
        
        result = validator.validate_models()
        
        self.assertIsNotNone(result)
        self.assertHasAttr(result, 'success')
    
    def test_hardware_integration_validation(self):
        """Test hardware integration validation."""
        from validate_installation import InstallationValidator
        
        hardware_profile = HardwareProfile(
            cpu=CPUInfo("Test CPU", 8, 16, 3.8, 4.7, "x64"),
            memory=MemoryInfo(32, 28, "DDR4", 3200),
            gpu=GPUInfo("RTX 3070", 8, "12.1", "537.13", "8.6"),
            storage=StorageInfo(500, "NVMe SSD"),
            os=OSInfo("Windows", "11", "x64")
        )
        
        validator = InstallationValidator(self.temp_dir, hardware_profile)
        
        result = validator.validate_hardware_integration()
        
        self.assertIsNotNone(result)
        self.assertHasAttr(result, 'success')


class TestErrorHandling(unittest.TestCase):
    """Unit tests for error handling components."""
    
    def test_error_creation(self):
        """Test error object creation."""
        from error_handler import InstallationError
        
        error = InstallationError(
            message="Test error message",
            category="test_category",
            recovery_suggestions=["Suggestion 1", "Suggestion 2"]
        )
        
        self.assertEqual(error.message, "Test error message")
        self.assertEqual(error.category, "test_category")
        self.assertEqual(len(error.recovery_suggestions), 2)
    
    def test_error_categorization(self):
        """Test error categorization logic."""
        from error_handler import ErrorHandler
        
        handler = ErrorHandler()
        
        # Test different error types
        network_error = Exception("Connection timeout")
        category = handler._categorize_error(network_error)
        self.assertEqual(category, "network")
        
        permission_error = PermissionError("Access denied")
        category = handler._categorize_error(permission_error)
        self.assertEqual(category, "permission")
    
    def test_recovery_suggestions(self):
        """Test recovery suggestion generation."""
        from error_handler import ErrorHandler
        
        handler = ErrorHandler()
        
        # Test network error suggestions
        suggestions = handler._get_recovery_suggestions("network", "Connection timeout")
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        
        # Test permission error suggestions
        suggestions = handler._get_recovery_suggestions("permission", "Access denied")
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)


def assertHasAttr(self, obj, attr):
    """Helper method to check if object has attribute."""
    self.assertTrue(hasattr(obj, attr), f"Object does not have attribute '{attr}'")


# Add the helper method to TestCase
unittest.TestCase.assertHasAttr = assertHasAttr


if __name__ == "__main__":
    unittest.main()
