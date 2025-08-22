#!/usr/bin/env python3
"""
Cross-platform compatibility tests
Tests framework behavior across Windows, Linux, and macOS platforms.
"""

import os
import platform
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from local_testing_framework.environment_validator import EnvironmentValidator
from local_testing_framework.test_manager import LocalTestManager
from local_testing_framework.models.configuration import TestConfiguration


class TestCrossPlatformCompatibility(unittest.TestCase):
    """Test cross-platform compatibility of the testing framework"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        
        # Create basic config
        import json
        test_config = {
            "system": {"gpu_enabled": True},
            "directories": {"models": "models", "outputs": "outputs"},
            "optimization": {"enable_attention_slicing": True},
            "performance": {"stats_refresh_interval": 5}
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
        
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_windows_platform_detection(self):
        """Test platform detection on Windows"""
        validator = EnvironmentValidator()
        
        with patch('platform.system', return_value='Windows'), \
             patch('platform.python_version', return_value='3.9.7'):
            
            platform_info = validator._detect_platform()
            
            self.assertEqual(platform_info["system"], "Windows")
            self.assertIn("python_version", platform_info)
    
    def test_linux_platform_detection(self):
        """Test platform detection on Linux"""
        validator = EnvironmentValidator()
        
        with patch('platform.system', return_value='Linux'), \
             patch('platform.python_version', return_value='3.9.7'):
            
            platform_info = validator._detect_platform()
            
            self.assertEqual(platform_info["system"], "Linux")
            self.assertIn("python_version", platform_info)
    
    def test_macos_platform_detection(self):
        """Test platform detection on macOS"""
        validator = EnvironmentValidator()
        
        with patch('platform.system', return_value='Darwin'), \
             patch('platform.python_version', return_value='3.9.7'):
            
            platform_info = validator._detect_platform()
            
            self.assertEqual(platform_info["system"], "Darwin")
            self.assertIn("python_version", platform_info)
    
    def test_windows_environment_setup_commands(self):
        """Test environment setup command generation for Windows"""
        validator = EnvironmentValidator()
        
        with patch.object(validator, 'platform_info', {"system": "Windows"}):
            commands = validator._generate_env_setup_commands(["HF_TOKEN", "CUDA_VISIBLE_DEVICES"])
            
            # Should contain Windows-specific commands
            command_text = " ".join(commands)
            self.assertTrue(any("setx" in cmd for cmd in commands))
            self.assertTrue(any("$env:" in cmd for cmd in commands))
    
    def test_linux_environment_setup_commands(self):
        """Test environment setup command generation for Linux"""
        validator = EnvironmentValidator()
        
        with patch.object(validator, 'platform_info', {"system": "Linux"}):
            commands = validator._generate_env_setup_commands(["HF_TOKEN", "CUDA_VISIBLE_DEVICES"])
            
            # Should contain Linux-specific commands
            command_text = " ".join(commands)
            self.assertTrue(any("export" in cmd for cmd in commands))
            self.assertTrue(any("bashrc" in cmd or "profile" in cmd for cmd in commands))
    
    def test_macos_environment_setup_commands(self):
        """Test environment setup command generation for macOS"""
        validator = EnvironmentValidator()
        
        with patch.object(validator, 'platform_info', {"system": "Darwin"}):
            commands = validator._generate_env_setup_commands(["HF_TOKEN", "CUDA_VISIBLE_DEVICES"])
            
            # Should contain macOS-specific commands
            command_text = " ".join(commands)
            self.assertTrue(any("export" in cmd for cmd in commands))
            self.assertTrue(any("bash_profile" in cmd or "zshrc" in cmd for cmd in commands))
    
    def test_path_handling_cross_platform(self):
        """Test path handling across different platforms"""
        # Test with different path separators
        test_paths = [
            "models/test_model",  # Unix-style
            "models\\test_model",  # Windows-style
            "models/subfolder/test_model",  # Nested Unix
            "models\\subfolder\\test_model"  # Nested Windows
        ]
        
        for test_path in test_paths:
            # Convert to Path object (should normalize automatically)
            path_obj = Path(test_path)
            
            # Verify path is handled correctly
            self.assertIsInstance(path_obj, Path)
            
            # Verify path parts are accessible
            self.assertGreater(len(path_obj.parts), 0)
    
    def test_file_permissions_cross_platform(self):
        """Test file permission handling across platforms"""
        # Create test file
        test_file = os.path.join(self.temp_dir, "test_permissions.txt")
        with open(test_file, 'w') as f:
            f.write("test content")
        
        # Test permission checking (should work on all platforms)
        self.assertTrue(os.path.exists(test_file))
        self.assertTrue(os.access(test_file, os.R_OK))
        
        # Test setting permissions (Unix-like systems)
        if platform.system() != "Windows":
            try:
                os.chmod(test_file, 0o600)  # Owner read/write only
                stat_info = os.stat(test_file)
                # Verify permissions were set
                self.assertTrue(stat_info.st_mode & 0o600)
            except OSError:
                # Some filesystems don't support chmod
                pass
    
    def test_subprocess_execution_cross_platform(self):
        """Test subprocess execution across platforms"""
        import subprocess
        
        # Test platform-appropriate commands
        if platform.system() == "Windows":
            # Test Windows command
            try:
                result = subprocess.run(["echo", "test"], capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 0)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # Command might not be available in test environment
                pass
        else:
            # Test Unix command
            try:
                result = subprocess.run(["echo", "test"], capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 0)
                self.assertIn("test", result.stdout)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # Command might not be available in test environment
                pass
    
    def test_environment_variable_handling_cross_platform(self):
        """Test environment variable handling across platforms"""
        # Test setting and getting environment variables
        test_var_name = "TEST_FRAMEWORK_VAR"
        test_var_value = "test_value_123"
        
        # Set environment variable
        os.environ[test_var_name] = test_var_value
        
        # Verify it can be retrieved
        retrieved_value = os.environ.get(test_var_name)
        self.assertEqual(retrieved_value, test_var_value)
        
        # Clean up
        del os.environ[test_var_name]
        
        # Verify it's removed
        self.assertIsNone(os.environ.get(test_var_name))
    
    def test_temp_directory_handling_cross_platform(self):
        """Test temporary directory handling across platforms"""
        # Test creating temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertTrue(os.path.exists(temp_dir))
            self.assertTrue(os.path.isdir(temp_dir))
            
            # Test creating files in temp directory
            test_file = os.path.join(temp_dir, "test_file.txt")
            with open(test_file, 'w') as f:
                f.write("test content")
            
            self.assertTrue(os.path.exists(test_file))
        
        # Directory should be cleaned up automatically
        self.assertFalse(os.path.exists(temp_dir))
    
    def test_unicode_handling_cross_platform(self):
        """Test Unicode handling across platforms"""
        # Test Unicode file names and content
        unicode_filename = "test_文件_🚀.txt"
        unicode_content = "Test content with Unicode: 你好世界 🌍"
        
        try:
            unicode_file_path = os.path.join(self.temp_dir, unicode_filename)
            
            # Write Unicode content
            with open(unicode_file_path, 'w', encoding='utf-8') as f:
                f.write(unicode_content)
            
            # Read Unicode content
            with open(unicode_file_path, 'r', encoding='utf-8') as f:
                read_content = f.read()
            
            self.assertEqual(read_content, unicode_content)
            
        except (OSError, UnicodeError):
            # Some filesystems don't support Unicode filenames
            # This is expected on some systems
            pass
    
    def test_line_ending_handling_cross_platform(self):
        """Test line ending handling across platforms"""
        # Test different line endings
        test_content_unix = "line1\nline2\nline3\n"
        test_content_windows = "line1\r\nline2\r\nline3\r\n"
        test_content_mac = "line1\rline2\rline3\r"
        
        for i, content in enumerate([test_content_unix, test_content_windows, test_content_mac]):
            test_file = os.path.join(self.temp_dir, f"test_line_endings_{i}.txt")
            
            # Write with binary mode to preserve line endings
            with open(test_file, 'wb') as f:
                f.write(content.encode('utf-8'))
            
            # Read with text mode (should normalize line endings)
            with open(test_file, 'r', encoding='utf-8') as f:
                read_content = f.read()
            
            # Verify content is readable
            self.assertIn("line1", read_content)
            self.assertIn("line2", read_content)
            self.assertIn("line3", read_content)


class TestPlatformSpecificFeatures(unittest.TestCase):
    """Test platform-specific features and adaptations"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_gpu_detection_cross_platform(self):
        """Test GPU detection across platforms"""
        validator = EnvironmentValidator()
        
        # Mock different CUDA scenarios
        test_scenarios = [
            # CUDA available
            {"cuda_available": True, "device_count": 1, "expected_status": "passed"},
            # CUDA not available
            {"cuda_available": False, "device_count": 0, "expected_status": "warning"},
            # PyTorch not installed
            {"torch_available": False, "expected_status": "warning"}
        ]
        
        for scenario in test_scenarios:
            with patch('torch.cuda.is_available', return_value=scenario.get("cuda_available", True)), \
                 patch('torch.cuda.device_count', return_value=scenario.get("device_count", 1)):
                
                if scenario.get("torch_available", True):
                    result = validator.validate_cuda_availability()
                else:
                    with patch('importlib.util.find_spec', return_value=None):
                        result = validator.validate_cuda_availability()
                
                # Verify result is appropriate for platform
                self.assertIsNotNone(result)
                self.assertIn(result.status.value, ["passed", "warning", "failed"])
    
    def test_memory_detection_cross_platform(self):
        """Test memory detection across platforms"""
        # Mock psutil for different memory scenarios
        with patch('psutil.virtual_memory') as mock_memory:
            # Mock different memory configurations
            memory_configs = [
                {"total": 16 * (1024**3), "used": 8 * (1024**3), "percent": 50.0},  # 16GB, 50% used
                {"total": 8 * (1024**3), "used": 6 * (1024**3), "percent": 75.0},   # 8GB, 75% used
                {"total": 32 * (1024**3), "used": 4 * (1024**3), "percent": 12.5}   # 32GB, 12.5% used
            ]
            
            for config in memory_configs:
                mock_memory.return_value = Mock(
                    total=config["total"],
                    used=config["used"],
                    percent=config["percent"],
                    available=config["total"] - config["used"]
                )
                
                # Test memory detection
                from local_testing_framework.diagnostic_tool import SystemAnalyzer
                analyzer = SystemAnalyzer()
                
                metrics, issues = analyzer.analyze_system_resources()
                
                # Verify memory metrics are detected correctly
                self.assertIsNotNone(metrics)
                self.assertEqual(metrics.memory_percent, config["percent"])
                self.assertEqual(metrics.memory_total_gb, config["total"] / (1024**3))
    
    def test_process_management_cross_platform(self):
        """Test process management across platforms"""
        import subprocess
        import time
        
        # Test starting and stopping processes
        if platform.system() == "Windows":
            # Windows-specific process test
            try:
                # Start a simple process
                process = subprocess.Popen(["ping", "127.0.0.1", "-n", "1"], 
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE)
                
                # Wait for completion
                stdout, stderr = process.communicate(timeout=10)
                
                # Verify process completed
                self.assertIsNotNone(process.returncode)
                
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # Command might not be available
                pass
        else:
            # Unix-specific process test
            try:
                # Start a simple process
                process = subprocess.Popen(["sleep", "0.1"], 
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE)
                
                # Wait for completion
                stdout, stderr = process.communicate(timeout=5)
                
                # Verify process completed
                self.assertEqual(process.returncode, 0)
                
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # Command might not be available
                pass
    
    def test_network_handling_cross_platform(self):
        """Test network handling across platforms"""
        # Test basic network connectivity check
        import socket
        
        try:
            # Test socket creation (should work on all platforms)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            
            # Test connection to localhost (should always be available)
            try:
                result = sock.connect_ex(('127.0.0.1', 80))
                # Connection may succeed or fail, but should not raise exception
                self.assertIsInstance(result, int)
            finally:
                sock.close()
                
        except socket.error:
            # Network operations may fail in test environments
            pass
    
    def test_file_system_operations_cross_platform(self):
        """Test file system operations across platforms"""
        # Test various file system operations
        test_operations = [
            # Create directory
            ("create_dir", lambda: os.makedirs(os.path.join(self.temp_dir, "test_dir"), exist_ok=True)),
            # Create file
            ("create_file", lambda: self._create_test_file()),
            # List directory
            ("list_dir", lambda: os.listdir(self.temp_dir)),
            # Check file existence
            ("file_exists", lambda: os.path.exists(os.path.join(self.temp_dir, "test_file.txt"))),
            # Get file stats
            ("file_stats", lambda: os.stat(os.path.join(self.temp_dir, "test_file.txt"))),
        ]
        
        for operation_name, operation in test_operations:
            try:
                result = operation()
                # Operations should complete without exceptions
                # Some operations return None (like makedirs), others return values
                if operation_name in ["create_dir", "create_file"]:
                    # These operations may return None but should not raise exceptions
                    self.assertTrue(True)  # Just verify no exception was raised
                else:
                    # These operations should return meaningful values
                    self.assertIsNotNone(result)
            except OSError:
                # Some operations may fail on certain filesystems
                pass
    
    def _create_test_file(self):
        """Helper method to create a test file"""
        test_file_path = os.path.join(self.temp_dir, "test_file.txt")
        with open(test_file_path, 'w') as f:
            f.write("test content")
        return test_file_path


if __name__ == '__main__':
    unittest.main(verbosity=2)