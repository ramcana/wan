"""
Integration tests for the WAN2.2 local installation system.
Tests cross-component functionality and complete workflows.
"""

import os
import sys
import json
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from interfaces import HardwareProfile, CPUInfo, MemoryInfo, GPUInfo, StorageInfo, OSInfo


class TestDetectionToConfigurationIntegration(unittest.TestCase):
    """Test integration between system detection and configuration generation."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_high_end_detection_to_config(self):
        """Test high-end hardware detection to configuration flow."""
        from detect_system import SystemDetector
        from generate_config import ConfigurationEngine
        
        # Mock high-end hardware detection
        with patch('detect_system.SystemDetector._detect_cpu') as mock_cpu, \
             patch('detect_system.SystemDetector._detect_memory') as mock_memory, \
             patch('detect_system.SystemDetector._detect_gpu') as mock_gpu, \
             patch('detect_system.SystemDetector._detect_storage') as mock_storage, \
             patch('detect_system.SystemDetector._detect_os') as mock_os:
            
            # High-end hardware profile
            mock_cpu.return_value = CPUInfo("AMD Ryzen Threadripper PRO 5995WX", 64, 128, 2.7, 4.5, "x64")
            mock_memory.return_value = MemoryInfo(128, 120, "DDR4", 3200)
            mock_gpu.return_value = GPUInfo("NVIDIA GeForce RTX 4080", 16, "12.1", "537.13", "8.9")
            mock_storage.return_value = StorageInfo(2000, "NVMe SSD")
            mock_os.return_value = OSInfo("Windows", "11", "x64")
            
            # Detect hardware
            detector = SystemDetector(self.temp_dir)
            hardware_profile = detector.detect_hardware()
            
            # Generate configuration
            config_engine = ConfigurationEngine(self.temp_dir)
            config = config_engine.generate_config(hardware_profile)
            
            # Verify high-end optimizations
            self.assertGreaterEqual(config["optimization"]["cpu_threads"], 32)
            self.assertGreaterEqual(config["optimization"]["memory_pool_gb"], 16)
            self.assertTrue(config["system"]["enable_gpu_acceleration"])
            self.assertEqual(config["system"]["default_quantization"], "bf16")
            self.assertFalse(config["system"]["enable_model_offload"])
    
    def test_budget_detection_to_config(self):
        """Test budget hardware detection to configuration flow."""
        from detect_system import SystemDetector
        from generate_config import ConfigurationEngine
        
        # Mock budget hardware detection
        with patch('detect_system.SystemDetector._detect_cpu') as mock_cpu, \
             patch('detect_system.SystemDetector._detect_memory') as mock_memory, \
             patch('detect_system.SystemDetector._detect_gpu') as mock_gpu, \
             patch('detect_system.SystemDetector._detect_storage') as mock_storage, \
             patch('detect_system.SystemDetector._detect_os') as mock_os:
            
            # Budget hardware profile
            mock_cpu.return_value = CPUInfo("AMD Ryzen 5 3600", 6, 12, 3.6, 4.2, "x64")
            mock_memory.return_value = MemoryInfo(16, 14, "DDR4", 2666)
            mock_gpu.return_value = GPUInfo("NVIDIA GeForce GTX 1660 Ti", 6, "11.8", "516.94", "7.5")
            mock_storage.return_value = StorageInfo(250, "SATA SSD")
            mock_os.return_value = OSInfo("Windows", "10", "x64")
            
            # Detect hardware
            detector = SystemDetector(self.temp_dir)
            hardware_profile = detector.detect_hardware()
            
            # Generate configuration
            config_engine = ConfigurationEngine(self.temp_dir)
            config = config_engine.generate_config(hardware_profile)
            
            # Verify budget optimizations
            self.assertLessEqual(config["optimization"]["cpu_threads"], 16)
            self.assertLessEqual(config["optimization"]["memory_pool_gb"], 8)
            self.assertTrue(config["system"]["enable_gpu_acceleration"])
            self.assertEqual(config["system"]["default_quantization"], "fp16")
            self.assertTrue(config["system"]["enable_model_offload"])


class TestDependencyToValidationIntegration(unittest.TestCase):
    """Test integration between dependency management and validation."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dependency_installation_to_validation(self):
        """Test dependency installation to validation flow."""
        from setup_dependencies import DependencyManager
        from validate_installation import InstallationValidator
        from base_classes import ConsoleProgressReporter
        
        # Create mock virtual environment structure
        venv_dir = Path(self.temp_dir) / "venv"
        scripts_dir = venv_dir / "Scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "python.exe").touch()
        (scripts_dir / "pip.exe").touch()
        
        # Create mock site-packages
        site_packages = venv_dir / "Lib" / "site-packages"
        site_packages.mkdir(parents=True)
        
        # Mock installed packages
        for package in ["torch", "transformers", "numpy", "pillow"]:
            package_dir = site_packages / package
            package_dir.mkdir()
            (package_dir / "__init__.py").touch()
        
        hardware_profile = HardwareProfile(
            cpu=CPUInfo("AMD Ryzen 7 5800X", 8, 16, 3.8, 4.7, "x64"),
            memory=MemoryInfo(32, 28, "DDR4", 3200),
            gpu=GPUInfo("NVIDIA GeForce RTX 3070", 8, "12.1", "537.13", "8.6"),
            storage=StorageInfo(500, "NVMe SSD"),
            os=OSInfo("Windows", "11", "x64")
        )
        
        # Test dependency validation
        validator = InstallationValidator(self.temp_dir, hardware_profile)
        
        with patch('subprocess.run') as mock_run:
            # Mock successful package version checks
            mock_run.return_value = Mock(
                returncode=0,
                stdout="2.0.0",
                stderr=""
            )
            
            result = validator.validate_dependencies()
            
            self.assertIsNotNone(result)
            self.assertTrue(hasattr(result, 'success'))
            
            if result.details:
                dependencies = result.details.get("dependencies", [])
                self.assertGreater(len(dependencies), 0)
    
    def test_cuda_dependency_validation(self):
        """Test CUDA-specific dependency validation."""
        from validate_installation import InstallationValidator
        
        # Create mock CUDA environment
        venv_dir = Path(self.temp_dir) / "venv"
        scripts_dir = venv_dir / "Scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "python.exe").touch()
        
        site_packages = venv_dir / "Lib" / "site-packages"
        site_packages.mkdir(parents=True)
        
        # Mock CUDA packages
        for package in ["torch", "torchvision", "torchaudio"]:
            package_dir = site_packages / package
            package_dir.mkdir()
            (package_dir / "__init__.py").touch()
        
        hardware_profile = HardwareProfile(
            cpu=CPUInfo("AMD Ryzen 7 5800X", 8, 16, 3.8, 4.7, "x64"),
            memory=MemoryInfo(32, 28, "DDR4", 3200),
            gpu=GPUInfo("NVIDIA GeForce RTX 3070", 8, "12.1", "537.13", "8.6"),
            storage=StorageInfo(500, "NVMe SSD"),
            os=OSInfo("Windows", "11", "x64")
        )
        
        validator = InstallationValidator(self.temp_dir, hardware_profile)
        
        with patch('subprocess.run') as mock_run:
            # Mock CUDA availability check
            mock_run.side_effect = [
                Mock(returncode=0, stdout="2.0.0+cu121", stderr=""),  # torch version
                Mock(returncode=0, stdout="True", stderr=""),  # CUDA available
                Mock(returncode=0, stdout="12.1", stderr="")   # CUDA version
            ]
            
            result = validator.validate_hardware_integration()
            
            self.assertIsNotNone(result)
            self.assertTrue(hasattr(result, 'success'))


class TestModelToConfigurationIntegration(unittest.TestCase):
    """Test integration between model management and configuration."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_size_to_config_optimization(self):
        """Test model size influence on configuration optimization."""
        from download_models import ModelDownloader
        from generate_config import ConfigurationEngine
        
        # Create mock model structure
        models_dir = Path(self.temp_dir) / "models"
        models_dir.mkdir(parents=True)
        
        # Create large model files to simulate WAN2.2 models
        for model_name in ["WAN2.2-T2V-A14B", "WAN2.2-I2V-A14B", "WAN2.2-TI2V-5B"]:
            model_path = models_dir / model_name
            model_path.mkdir()
            
            # Create large model file (simulate 14B parameter model)
            model_file = model_path / "pytorch_model.bin"
            model_file.write_bytes(b"0" * (1024 * 1024 * 1024))  # 1GB file
            
            (model_path / "config.json").write_text(json.dumps({
                "model_type": "wan22",
                "num_parameters": "14B" if "A14B" in model_name else "5B"
            }))
        
        downloader = ModelDownloader(self.temp_dir)
        config_engine = ConfigurationEngine(self.temp_dir)
        
        # Test with budget hardware (should enable model offloading)
        budget_profile = HardwareProfile(
            cpu=CPUInfo("AMD Ryzen 5 3600", 6, 12, 3.6, 4.2, "x64"),
            memory=MemoryInfo(16, 14, "DDR4", 2666),
            gpu=GPUInfo("NVIDIA GeForce GTX 1660 Ti", 6, "11.8", "516.94", "7.5"),
            storage=StorageInfo(250, "SATA SSD"),
            os=OSInfo("Windows", "10", "x64")
        )
        
        config = config_engine.generate_config(budget_profile)
        
        # With large models and limited VRAM, should enable offloading
        self.assertTrue(config["system"]["enable_model_offload"])
        self.assertLessEqual(config["optimization"]["max_vram_usage_gb"], 5)
    
    def test_model_validation_to_config_adjustment(self):
        """Test model validation results affecting configuration."""
        from download_models import ModelDownloader
        from validate_installation import InstallationValidator
        from generate_config import ConfigurationEngine
        
        # Create partial model structure (missing some models)
        models_dir = Path(self.temp_dir) / "models"
        models_dir.mkdir(parents=True)
        
        # Only create one model (simulate incomplete download)
        model_path = models_dir / "WAN2.2-TI2V-5B"
        model_path.mkdir()
        (model_path / "pytorch_model.bin").write_bytes(b"0" * (1024 * 1024 * 100))
        (model_path / "config.json").write_text('{"model_type": "wan22"}')
        
        hardware_profile = HardwareProfile(
            cpu=CPUInfo("AMD Ryzen 7 5800X", 8, 16, 3.8, 4.7, "x64"),
            memory=MemoryInfo(32, 28, "DDR4", 3200),
            gpu=GPUInfo("NVIDIA GeForce RTX 3070", 8, "12.1", "537.13", "8.6"),
            storage=StorageInfo(500, "NVMe SSD"),
            os=OSInfo("Windows", "11", "x64")
        )
        
        validator = InstallationValidator(self.temp_dir, hardware_profile)
        model_result = validator.validate_models()
        
        # Should detect missing models
        self.assertIsNotNone(model_result)
        
        if model_result.details:
            models = model_result.details.get("models", [])
            missing_models = [m for m in models if not m.get("exists", False)]
            self.assertGreater(len(missing_models), 0)


class TestFullInstallationFlow(unittest.TestCase):
    """Test complete installation flow integration."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_installation_simulation(self):
        """Test complete installation flow simulation."""
        from main_installer import MainInstaller
        from base_classes import ConsoleProgressReporter
        
        # Create basic directory structure
        for subdir in ["scripts", "resources", "models", "logs"]:
            (Path(self.temp_dir) / subdir).mkdir(parents=True)
        
        # Create mock requirements file
        req_file = Path(self.temp_dir) / "resources" / "requirements.txt"
        req_file.write_text("torch>=2.0.0\ntransformers>=4.30.0\nnumpy>=1.24.0")
        
        progress_reporter = ConsoleProgressReporter()
        installer = MainInstaller(self.temp_dir, progress_reporter)
        
        # Test installation phases
        phases = installer.get_installation_phases()
        
        self.assertIsInstance(phases, list)
        self.assertGreater(len(phases), 0)
        
        # Verify phase structure
        required_phases = ["detection", "dependencies", "models", "configuration", "validation"]
        phase_names = [phase["name"] for phase in phases]
        
        for required_phase in required_phases:
            self.assertIn(required_phase, phase_names)
    
    def test_installation_state_management(self):
        """Test installation state management throughout the flow."""
        from installation_flow_controller import InstallationFlowController
        from base_classes import ConsoleProgressReporter
        
        progress_reporter = ConsoleProgressReporter()
        controller = InstallationFlowController(self.temp_dir, progress_reporter)
        
        # Test state initialization
        state = controller.get_installation_state()
        
        self.assertIsNotNone(state)
        self.assertEqual(state.phase, "not_started")
        self.assertEqual(state.progress, 0.0)
        
        # Test state updates
        controller.update_installation_state("detection", 0.1, "Detecting hardware...")
        
        updated_state = controller.get_installation_state()
        self.assertEqual(updated_state.phase, "detection")
        self.assertEqual(updated_state.progress, 0.1)
        self.assertEqual(updated_state.current_task, "Detecting hardware...")


class TestErrorRecoveryIntegration(unittest.TestCase):
    """Test error recovery integration across components."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_network_error_recovery(self):
        """Test network error recovery during model download."""
        from download_models import ModelDownloader
        from error_handler import ErrorHandler
        
        downloader = ModelDownloader(self.temp_dir)
        error_handler = ErrorHandler()
        
        # Simulate network error
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Connection timeout")
            
            try:
                # This should trigger error handling
                downloader._download_single_model("WAN2.2-TI2V-5B")
            except Exception as e:
                # Test error categorization
                category = error_handler._categorize_error(e)
                self.assertEqual(category, "network")
                
                # Test recovery suggestions
                suggestions = error_handler._get_recovery_suggestions(category, str(e))
                self.assertIsInstance(suggestions, list)
                self.assertGreater(len(suggestions), 0)
    
    def test_rollback_integration(self):
        """Test rollback integration with error handling."""
        from rollback_manager import RollbackManager
        from error_handler import ErrorHandler
        
        rollback_manager = RollbackManager(self.temp_dir)
        error_handler = ErrorHandler()
        
        # Create mock installation state
        state_file = Path(self.temp_dir) / "installation_state.json"
        state_data = {
            "phase": "models",
            "progress": 0.5,
            "completed_steps": ["detection", "dependencies"]
        }
        
        state_file.write_text(json.dumps(state_data))
        
        # Test snapshot creation
        snapshot_name = f"before_models_{int(time.time())}"
        success = rollback_manager.create_snapshot(snapshot_name)
        
        # In a real scenario, this would create actual snapshots
        # For testing, we just verify the method doesn't crash
        self.assertIsInstance(success, bool)
        
        # Test snapshot listing
        snapshots = rollback_manager.list_snapshots()
        self.assertIsInstance(snapshots, list)
    
    def test_validation_error_diagnosis(self):
        """Test validation error diagnosis and recovery."""
        from validate_installation import InstallationValidator
        from functionality_tester import FunctionalityTester
        
        hardware_profile = HardwareProfile(
            cpu=CPUInfo("AMD Ryzen 7 5800X", 8, 16, 3.8, 4.7, "x64"),
            memory=MemoryInfo(32, 28, "DDR4", 3200),
            gpu=GPUInfo("NVIDIA GeForce RTX 3070", 8, "12.1", "537.13", "8.6"),
            storage=StorageInfo(500, "NVMe SSD"),
            os=OSInfo("Windows", "11", "x64")
        )
        
        validator = InstallationValidator(self.temp_dir, hardware_profile)
        tester = FunctionalityTester(self.temp_dir, hardware_profile)
        
        # Create mock test results with errors
        from functionality_tester import TestResult
        
        test_results = [
            TestResult("cuda_test", False, 10.0, "CUDA out of memory"),
            TestResult("model_load_test", False, 5.0, "Model checkpoint not found"),
            TestResult("basic_test", True, 2.0)
        ]
        
        # Test error diagnosis
        diagnosis = tester.diagnose_errors(test_results)
        
        self.assertIsInstance(diagnosis, dict)
        self.assertIn("hardware_limitations", diagnosis)
        self.assertIn("configuration_problems", diagnosis)
        self.assertIn("suggested_fixes", diagnosis)
        
        # Verify error categorization
        self.assertGreater(len(diagnosis["hardware_limitations"]), 0)
        self.assertGreater(len(diagnosis["configuration_problems"]), 0)


if __name__ == "__main__":
    unittest.main()
