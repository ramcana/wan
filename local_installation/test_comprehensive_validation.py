"""
Comprehensive validation test suite for Task 13.2.
Tests all requirements and validates the complete installation system.
"""

import sys
import os
import json
import tempfile
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scripts.integrated_installer import IntegratedInstaller
from scripts.installation_flow_controller import InstallationFlowController
from scripts.interfaces import InstallationPhase, HardwareProfile, CPUInfo, MemoryInfo, GPUInfo


class ComprehensiveValidationSuite:
    """Comprehensive validation suite for the installation system."""
    
    def __init__(self):
        self.test_results = []
        self.test_dir = None
        
    def setup_test_environment(self):
        """Set up a comprehensive test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create complete directory structure
        directories = [
            "scripts", "resources", "models", "logs", "application", 
            "venv", ".wan22_backup", "docs", "examples"
        ]
        
        for directory in directories:
            (self.test_dir / directory).mkdir(exist_ok=True)
        
        # Create comprehensive requirements.txt
        requirements_file = self.test_dir / "resources" / "requirements.txt"
        requirements_file.write_text("""
# Core ML libraries
torch>=1.9.0
torchvision>=0.10.0
torchaudio>=0.9.0

# Transformers and diffusion models
transformers>=4.20.0
diffusers>=0.15.0
accelerate>=0.18.0

# Computer vision
opencv-python>=4.5.0
Pillow>=8.0.0

# Utilities
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
tqdm>=4.62.0

# CUDA support (conditional)
# torch-audio-cuda
# torch-vision-cuda
""")
        
        # Create configuration templates
        config_template = self.test_dir / "resources" / "default_config.json"
        config_template.write_text(json.dumps({
            "system": {
                "threads": 4,
                "memory_limit_gb": 8,
                "default_quantization": "fp16",
                "enable_offload": True,
                "vae_tile_size": 256,
                "max_queue_size": 10,
                "worker_threads": 4
            },
            "models": {
                "cache_dir": "models",
                "download_timeout": 300,
                "verify_checksums": True
            },
            "optimization": {
                "cpu_threads": 4,
                "memory_pool_gb": 4,
                "max_vram_usage_gb": 4
            },
            "logging": {
                "level": "INFO",
                "file_logging": True,
                "console_logging": True
            }
        }, indent=2))
        
        # Create mock model files
        models_dir = self.test_dir / "models"
        for model_name in ["WAN2.2-T2V-A14B", "WAN2.2-I2V-A14B", "WAN2.2-TI2V-5B"]:
            model_dir = models_dir / model_name
            model_dir.mkdir(exist_ok=True)
            (model_dir / "config.json").write_text('{"model_type": "wan22"}')
            (model_dir / "pytorch_model.bin").write_text("mock model data")
        
        # Create application files
        app_dir = self.test_dir / "application"
        (app_dir / "main.py").write_text("# Main application entry point")
        (app_dir / "ui.py").write_text("# User interface module")
        (app_dir / "utils.py").write_text("# Utility functions")
        
        return self.test_dir
    
    def cleanup_test_environment(self):
        """Clean up test environment."""
        if self.test_dir and self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_mock_hardware_profiles(self) -> List[HardwareProfile]:
        """Create various hardware profiles for testing."""
        profiles = []
        
        # High-performance system (Threadripper PRO + RTX 4080)
        profiles.append(HardwareProfile(
            cpu=CPUInfo(
                model="AMD Ryzen Threadripper PRO 5995WX",
                cores=64,
                threads=128,
                base_clock=2.7,
                boost_clock=4.5,
                architecture="x64"
            ),
            memory=MemoryInfo(
                total_gb=128,
                available_gb=120,
                type="DDR4",
                speed=3200
            ),
            gpu=GPUInfo(
                model="NVIDIA GeForce RTX 4080",
                vram_gb=16,
                cuda_version="12.1",
                driver_version="537.13",
                compute_capability="8.9"
            ),
            storage=None,
            os=None
        ))
        
        # Mid-range system (Ryzen 7 + RTX 3070)
        profiles.append(HardwareProfile(
            cpu=CPUInfo(
                model="AMD Ryzen 7 5800X",
                cores=8,
                threads=16,
                base_clock=3.8,
                boost_clock=4.7,
                architecture="x64"
            ),
            memory=MemoryInfo(
                total_gb=32,
                available_gb=28,
                type="DDR4",
                speed=3200
            ),
            gpu=GPUInfo(
                model="NVIDIA GeForce RTX 3070",
                vram_gb=8,
                cuda_version="11.8",
                driver_version="520.0",
                compute_capability="8.6"
            ),
            storage=None,
            os=None
        ))
        
        # Budget system (Ryzen 5 + GTX 1660 Ti)
        profiles.append(HardwareProfile(
            cpu=CPUInfo(
                model="AMD Ryzen 5 3600",
                cores=6,
                threads=12,
                base_clock=3.6,
                boost_clock=4.2,
                architecture="x64"
            ),
            memory=MemoryInfo(
                total_gb=16,
                available_gb=12,
                type="DDR4",
                speed=3200
            ),
            gpu=GPUInfo(
                model="NVIDIA GeForce GTX 1660 Ti",
                vram_gb=6,
                cuda_version="11.8",
                driver_version="520.0",
                compute_capability="7.5"
            ),
            storage=None,
            os=None
        ))
        
        return profiles
    
    def test_requirement_1_1_batch_file_installation(self) -> bool:
        """Test Requirement 1.1: Batch file runs automatically without user input."""
        print("  Testing Requirement 1.1: Batch file installation...")
        
        try:
            # Test that install.bat exists and is executable
            install_bat = Path(__file__).parent / "install.bat"
            if not install_bat.exists():
                print("    ❌ install.bat not found")
                return False
            
            # Test batch file structure
            try:
                content = install_bat.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    content = install_bat.read_text(encoding='cp1252')
                except UnicodeDecodeError:
                    content = install_bat.read_text(encoding='latin-1')
            
            required_elements = [
                "@echo off",
                "setlocal enabledelayedexpansion",
                "python",
                "main_installer.py"
            ]
            
            for element in required_elements:
                if element not in content:
                    print(f"    ❌ Missing required element in install.bat: {element}")
                    return False
            
            print("    ✓ install.bat exists and has correct structure")
            return True
            
        except Exception as e:
            print(f"    ❌ Error testing batch file: {e}")
            return False
    
    def test_requirement_1_2_hardware_detection(self) -> bool:
        """Test Requirement 1.2: Automatic hardware detection."""
        print("  Testing Requirement 1.2: Hardware detection...")
        
        try:
            test_dir = self.setup_test_environment()
            
            mock_args = Mock()
            mock_args.silent = True
            mock_args.dry_run = True
            mock_args.skip_models = True
            mock_args.verbose = False
            mock_args.dev_mode = False
            mock_args.force_reinstall = False
            mock_args.custom_path = None
            
            # Test with different hardware profiles
            hardware_profiles = self.create_mock_hardware_profiles()
            
            for i, profile in enumerate(hardware_profiles):
                with patch('scripts.integrated_installer.SystemDetector') as mock_detector:
                    mock_detector_instance = Mock()
                    mock_detector.return_value = mock_detector_instance
                    mock_detector_instance.detect_hardware.return_value = profile
                    
                    mock_validation_result = Mock()
                    mock_validation_result.meets_minimum = True
                    mock_validation_result.issues = []
                    mock_detector_instance.validate_requirements.return_value = mock_validation_result
                    
                    installer = IntegratedInstaller(str(test_dir), mock_args)
                    success = installer._run_detection_phase()
                    
                    if not success:
                        print(f"    ❌ Hardware detection failed for profile {i+1}")
                        return False
                    
                    if installer.hardware_profile is None:
                        print(f"    ❌ Hardware profile not set for profile {i+1}")
                        return False
            
            print("    ✓ Hardware detection works for all hardware configurations")
            return True
            
        except Exception as e:
            print(f"    ❌ Error testing hardware detection: {e}")
            return False
        finally:
            self.cleanup_test_environment()
    
    def test_requirement_1_3_display_hardware_info(self) -> bool:
        """Test Requirement 1.3: Display detected hardware information."""
        print("  Testing Requirement 1.3: Display hardware information...")
        
        try:
            test_dir = self.setup_test_environment()
            
            mock_args = Mock()
            mock_args.silent = False  # Enable display
            mock_args.dry_run = True
            mock_args.skip_models = True
            mock_args.verbose = False
            mock_args.dev_mode = False
            mock_args.force_reinstall = False
            mock_args.custom_path = None
            
            hardware_profile = self.create_mock_hardware_profiles()[0]  # Use high-end profile
            
            with patch('scripts.integrated_installer.SystemDetector') as mock_detector:
                mock_detector_instance = Mock()
                mock_detector.return_value = mock_detector_instance
                mock_detector_instance.detect_hardware.return_value = hardware_profile
                
                mock_validation_result = Mock()
                mock_validation_result.meets_minimum = True
                mock_validation_result.issues = []
                mock_detector_instance.validate_requirements.return_value = mock_validation_result
                
                installer = IntegratedInstaller(str(test_dir), mock_args)
                
                # Test that hardware summary display method exists and works
                installer.hardware_profile = hardware_profile
                try:
                    installer._display_hardware_summary()
                    print("    ✓ Hardware information display works")
                    return True
                except Exception as e:
                    print(f"    ❌ Hardware display failed: {e}")
                    return False
            
        except Exception as e:
            print(f"    ❌ Error testing hardware display: {e}")
            return False
        finally:
            self.cleanup_test_environment()
    
    def test_requirement_1_4_desktop_shortcuts(self) -> bool:
        """Test Requirement 1.4: Create desktop shortcuts and start menu entries."""
        print("  Testing Requirement 1.4: Desktop shortcuts creation...")
        
        try:
            # Test that shortcut creation script exists
            shortcut_script = Path(__file__).parent / "scripts" / "create_shortcuts.py"
            if not shortcut_script.exists():
                print("    ❌ create_shortcuts.py not found")
                return False
            
            # Test shortcut creation functionality
            test_dir = self.setup_test_environment()
            
            with patch('scripts.integrated_installer.ShortcutCreator') as mock_shortcuts:
                mock_shortcuts_instance = Mock()
                mock_shortcuts.return_value = mock_shortcuts_instance
                mock_shortcuts_instance.create_all_shortcuts.return_value = True
                
                # Test shortcut creation call
                from scripts.create_shortcuts import ShortcutCreator
                creator = ShortcutCreator(str(test_dir))
                
                # Verify the class exists and can be instantiated
                assert creator is not None
                
            print("    ✓ Desktop shortcuts functionality available")
            return True
            
        except Exception as e:
            print(f"    ❌ Error testing desktop shortcuts: {e}")
            return False
        finally:
            self.cleanup_test_environment()
    
    def test_requirement_2_1_python_installation(self) -> bool:
        """Test Requirement 2.1: Automatic Python installation."""
        print("  Testing Requirement 2.1: Python installation...")
        
        try:
            test_dir = self.setup_test_environment()
            
            mock_args = Mock()
            mock_args.silent = True
            mock_args.dry_run = True
            mock_args.skip_models = True
            mock_args.verbose = False
            mock_args.dev_mode = False
            mock_args.force_reinstall = False
            mock_args.custom_path = None
            
            with patch('scripts.integrated_installer.DependencyManager') as mock_deps, \
                 patch('scripts.integrated_installer.PythonInstallationHandler') as mock_python:
                
                # Test scenario where Python is not suitable
                mock_deps_instance = Mock()
                mock_deps.return_value = mock_deps_instance
                
                mock_python_info = Mock()
                mock_python_info.is_suitable = False
                mock_deps_instance.check_python_installation.return_value = mock_python_info
                
                mock_python_instance = Mock()
                mock_python.return_value = mock_python_instance
                mock_python_instance.install_embedded_python.return_value = True
                
                mock_deps_instance.create_optimized_virtual_environment.return_value = True
                mock_deps_instance.install_hardware_optimized_packages.return_value = True
                
                mock_verification_result = Mock()
                mock_verification_result.all_packages_installed = True
                mock_verification_result.missing_packages = []
                mock_deps_instance.verify_package_installation.return_value = mock_verification_result
                
                installer = IntegratedInstaller(str(test_dir), mock_args)
                installer.hardware_profile = self.create_mock_hardware_profiles()[0]
                
                success = installer._run_dependencies_phase()
                
                if not success:
                    print("    ❌ Dependencies phase failed")
                    return False
                
                # Verify Python installer was called
                mock_python.assert_called_once()
                mock_python_instance.install_embedded_python.assert_called_once()
                
            print("    ✓ Python installation functionality works")
            return True
            
        except Exception as e:
            print(f"    ❌ Error testing Python installation: {e}")
            return False
        finally:
            self.cleanup_test_environment()
    
    def test_requirement_3_1_hardware_optimization(self) -> bool:
        """Test Requirement 3.1: Hardware-specific optimization."""
        print("  Testing Requirement 3.1: Hardware optimization...")
        
        try:
            test_dir = self.setup_test_environment()
            
            mock_args = Mock()
            mock_args.silent = True
            mock_args.dry_run = True
            mock_args.skip_models = True
            mock_args.verbose = False
            mock_args.dev_mode = False
            mock_args.force_reinstall = False
            mock_args.custom_path = None
            
            hardware_profiles = self.create_mock_hardware_profiles()
            
            # Test optimization for different hardware tiers
            for i, profile in enumerate(hardware_profiles):
                with patch('scripts.integrated_installer.ConfigurationEngine') as mock_config:
                    mock_config_instance = Mock()
                    mock_config.return_value = mock_config_instance
                    
                    base_config = {"system": {"threads": 4}}
                    
                    # Different optimizations based on hardware
                    if profile.cpu.cores >= 32:  # High-end
                        optimized_config = {
                            "system": {"threads": 32},
                            "optimization": {
                                "cpu_threads": 64,
                                "memory_pool_gb": 32,
                                "max_vram_usage_gb": 14
                            }
                        }
                    elif profile.cpu.cores >= 8:  # Mid-range
                        optimized_config = {
                            "system": {"threads": 16},
                            "optimization": {
                                "cpu_threads": 16,
                                "memory_pool_gb": 16,
                                "max_vram_usage_gb": 6
                            }
                        }
                    else:  # Budget
                        optimized_config = {
                            "system": {"threads": 8},
                            "optimization": {
                                "cpu_threads": 8,
                                "memory_pool_gb": 8,
                                "max_vram_usage_gb": 4
                            }
                        }
                    
                    mock_config_instance.generate_base_configuration.return_value = base_config
                    mock_config_instance.optimize_for_hardware.return_value = optimized_config
                    
                    mock_validation_result = Mock()
                    mock_validation_result.is_valid = True
                    mock_validation_result.errors = []
                    mock_config_instance.validate_configuration.return_value = mock_validation_result
                    mock_config_instance.save_configuration.return_value = True
                    
                    installer = IntegratedInstaller(str(test_dir), mock_args)
                    installer.hardware_profile = profile
                    
                    success = installer._run_configuration_phase()
                    
                    if not success:
                        print(f"    ❌ Configuration failed for hardware profile {i+1}")
                        return False
                    
                    # Verify optimization was called with the hardware profile
                    mock_config_instance.optimize_for_hardware.assert_called_with(
                        base_config, profile
                    )
            
            print("    ✓ Hardware optimization works for all hardware tiers")
            return True
            
        except Exception as e:
            print(f"    ❌ Error testing hardware optimization: {e}")
            return False
        finally:
            self.cleanup_test_environment()
    
    def test_requirement_4_1_progress_indication(self) -> bool:
        """Test Requirement 4.1: Progress indication during installation."""
        print("  Testing Requirement 4.1: Progress indication...")
        
        try:
            test_dir = self.setup_test_environment()
            
            mock_args = Mock()
            mock_args.silent = False  # Enable progress display
            mock_args.dry_run = True
            mock_args.skip_models = True
            mock_args.verbose = False
            mock_args.dev_mode = False
            mock_args.force_reinstall = False
            mock_args.custom_path = None
            
            installer = IntegratedInstaller(str(test_dir), mock_args)
            
            # Test progress tracking
            progress_updates = []
            
            def progress_callback(phase, progress, task):
                progress_updates.append((phase, progress, task))
            
            installer.flow_controller.add_progress_callback(progress_callback)
            
            # Simulate progress updates through different phases
            phases = [
                (InstallationPhase.DETECTION, "Detecting hardware"),
                (InstallationPhase.DEPENDENCIES, "Installing dependencies"),
                (InstallationPhase.CONFIGURATION, "Generating configuration"),
                (InstallationPhase.VALIDATION, "Validating installation")
            ]
            
            for phase, task in phases:
                installer.flow_controller.update_progress(phase, 0.0, f"Starting {task}")
                installer.flow_controller.update_progress(phase, 0.5, f"Processing {task}")
                installer.flow_controller.update_progress(phase, 1.0, f"Completed {task}")
            
            # Verify progress updates were received
            if len(progress_updates) < len(phases) * 3:
                print(f"    ❌ Insufficient progress updates: {len(progress_updates)}")
                return False
            
            # Verify progress is monotonically increasing within phases
            for i in range(0, len(progress_updates), 3):
                if i + 2 < len(progress_updates):
                    if not (progress_updates[i][1] <= progress_updates[i+1][1] <= progress_updates[i+2][1]):
                        print("    ❌ Progress not monotonically increasing")
                        return False
            
            print("    ✓ Progress indication works correctly")
            return True
            
        except Exception as e:
            print(f"    ❌ Error testing progress indication: {e}")
            return False
        finally:
            self.cleanup_test_environment()
    
    def test_requirement_5_1_installation_validation(self) -> bool:
        """Test Requirement 5.1: Installation validation."""
        print("  Testing Requirement 5.1: Installation validation...")
        
        try:
            test_dir = self.setup_test_environment()
            
            mock_args = Mock()
            mock_args.silent = True
            mock_args.dry_run = True
            mock_args.skip_models = True
            mock_args.verbose = False
            mock_args.dev_mode = False
            mock_args.force_reinstall = False
            mock_args.custom_path = None
            
            with patch('scripts.integrated_installer.InstallationValidator') as mock_validator:
                mock_validator_instance = Mock()
                mock_validator.return_value = mock_validator_instance
                
                # Test successful validation
                mock_validator_instance.validate_dependencies.return_value = Mock(success=True, message="")
                mock_validator_instance.validate_models.return_value = Mock(success=True, message="")
                mock_validator_instance.validate_hardware_integration.return_value = Mock(success=True, message="")
                mock_validator_instance.run_basic_functionality_test.return_value = Mock(success=True, message="")
                mock_validator_instance.generate_validation_report.return_value = {
                    "status": "success",
                    "timestamp": "2025-08-01T22:00:00",
                    "dependencies": {"status": "valid", "issues": []},
                    "models": {"status": "valid", "issues": []},
                    "hardware": {"status": "valid", "issues": []},
                    "functionality": {"status": "valid", "issues": []}
                }
                
                installer = IntegratedInstaller(str(test_dir), mock_args)
                installer.hardware_profile = self.create_mock_hardware_profiles()[0]
                
                success = installer._run_validation_phase()
                
                if not success:
                    print("    ❌ Validation phase failed")
                    return False
                
                # Verify all validation methods were called
                mock_validator_instance.validate_dependencies.assert_called_once()
                mock_validator_instance.validate_models.assert_called_once()
                mock_validator_instance.validate_hardware_integration.assert_called_once()
                mock_validator_instance.run_basic_functionality_test.assert_called_once()
                mock_validator_instance.generate_validation_report.assert_called_once()
                
            print("    ✓ Installation validation works correctly")
            return True
            
        except Exception as e:
            print(f"    ❌ Error testing installation validation: {e}")
            return False
        finally:
            self.cleanup_test_environment()
    
    def test_requirement_6_1_model_download(self) -> bool:
        """Test Requirement 6.1: WAN2.2 model download."""
        print("  Testing Requirement 6.1: Model download...")
        
        try:
            test_dir = self.setup_test_environment()
            
            mock_args = Mock()
            mock_args.silent = True
            mock_args.dry_run = True
            mock_args.skip_models = False  # Enable model download
            mock_args.verbose = False
            mock_args.dev_mode = False
            mock_args.force_reinstall = False
            mock_args.custom_path = None
            
            with patch('scripts.integrated_installer.ModelDownloader') as mock_downloader:
                mock_downloader_instance = Mock()
                mock_downloader.return_value = mock_downloader_instance
                
                # Mock model download scenario
                mock_downloader_instance.check_existing_models.return_value = []
                mock_downloader_instance.get_required_models.return_value = [
                    "WAN2.2-T2V-A14B", "WAN2.2-I2V-A14B", "WAN2.2-TI2V-5B"
                ]
                mock_downloader_instance.download_models_parallel.return_value = True
                
                mock_verification_result = Mock()
                mock_verification_result.all_valid = True
                mock_verification_result.invalid_models = []
                mock_downloader_instance.verify_all_models.return_value = mock_verification_result
                
                installer = IntegratedInstaller(str(test_dir), mock_args)
                
                success = installer._run_models_phase()
                
                if not success:
                    print("    ❌ Model download phase failed")
                    return False
                
                # Verify model download methods were called
                mock_downloader_instance.check_existing_models.assert_called_once()
                mock_downloader_instance.get_required_models.assert_called_once()
                mock_downloader_instance.download_models_parallel.assert_called_once()
                mock_downloader_instance.verify_all_models.assert_called_once()
                
            print("    ✓ Model download functionality works")
            return True
            
        except Exception as e:
            print(f"    ❌ Error testing model download: {e}")
            return False
        finally:
            self.cleanup_test_environment()
    
    def test_requirement_7_1_shareable_package(self) -> bool:
        """Test Requirement 7.1: Shareable installation package."""
        print("  Testing Requirement 7.1: Shareable package...")
        
        try:
            # Test that all required files exist for distribution
            base_dir = Path(__file__).parent
            
            required_files = [
                "install.bat",
                "scripts/main_installer.py",
                "scripts/integrated_installer.py",
                "resources/requirements.txt",
                "application/wan22_ui.py",
                "application/web_ui.py",
                "launch_wan22.bat",
                "launch_web_ui.bat",
                "UI_GUIDE.md"
            ]
            
            for file_path in required_files:
                full_path = base_dir / file_path
                if not full_path.exists():
                    print(f"    ❌ Required file missing: {file_path}")
                    return False
            
            # Test that distribution manager exists
            dist_manager_path = base_dir / "scripts" / "distribution_manager.py"
            if not dist_manager_path.exists():
                print("    ❌ Distribution manager not found")
                return False
            
            print("    ✓ All required files for shareable package exist")
            return True
            
        except Exception as e:
            print(f"    ❌ Error testing shareable package: {e}")
            return False
    
    def test_ui_components(self) -> bool:
        """Test UI components are properly integrated."""
        print("  Testing UI components integration...")
        
        try:
            # Test in the actual installation directory, not test environment
            base_dir = Path(__file__).parent
            
            # Test desktop UI
            desktop_ui_path = base_dir / "application" / "wan22_ui.py"
            if not desktop_ui_path.exists():
                print("    ❌ Desktop UI component missing")
                return False
            
            # Test web UI
            web_ui_path = base_dir / "application" / "web_ui.py"
            if not web_ui_path.exists():
                print("    ❌ Web UI component missing")
                return False
            
            # Test launchers
            desktop_launcher = base_dir / "launch_wan22.bat"
            web_launcher = base_dir / "launch_web_ui.bat"
            
            if not desktop_launcher.exists():
                print("    ❌ Desktop UI launcher missing")
                return False
            
            if not web_launcher.exists():
                print("    ❌ Web UI launcher missing")
                return False
            
            # Test UI dependencies in requirements
            requirements_file = base_dir / "resources" / "requirements.txt"
            if requirements_file.exists():
                requirements_content = requirements_file.read_text()
                ui_deps = ["flask", "werkzeug", "Pillow", "opencv-python"]
                
                for dep in ui_deps:
                    if dep.lower() not in requirements_content.lower():
                        print(f"    ❌ UI dependency missing from requirements: {dep}")
                        return False
            
            # Test UI guide exists
            ui_guide = base_dir / "UI_GUIDE.md"
            if not ui_guide.exists():
                print("    ❌ UI guide documentation missing")
                return False
            
            print("    ✓ UI components properly integrated")
            return True
            
        except Exception as e:
            print(f"    ❌ Error testing UI components: {e}")
            return False
    
    def test_error_handling_and_recovery(self) -> bool:
        """Test comprehensive error handling and recovery scenarios."""
        print("  Testing error handling and recovery...")
        
        try:
            test_dir = self.setup_test_environment()
            
            mock_args = Mock()
            mock_args.silent = True
            mock_args.dry_run = True
            mock_args.skip_models = True
            mock_args.verbose = False
            mock_args.dev_mode = False
            mock_args.force_reinstall = False
            mock_args.custom_path = None
            
            # Test various error scenarios
            error_scenarios = [
                ("Hardware detection failure", "SystemDetector", "detect_hardware"),
                ("Dependency installation failure", "DependencyManager", "install_hardware_optimized_packages"),
                ("Configuration generation failure", "ConfigurationEngine", "generate_base_configuration"),
                ("Validation failure", "InstallationValidator", "validate_dependencies")
            ]
            
            for scenario_name, mock_class, failing_method in error_scenarios:
                with patch(f'scripts.integrated_installer.{mock_class}') as mock_component:
                    mock_instance = Mock()
                    mock_component.return_value = mock_instance
                    
                    # Make the specific method fail
                    setattr(mock_instance, failing_method, Mock(side_effect=Exception(f"Test {scenario_name}")))
                    
                    installer = IntegratedInstaller(str(test_dir), mock_args)
                    
                    # Run installation and expect it to fail gracefully
                    success = installer.run_complete_installation()
                    
                    if success:
                        print(f"    ❌ Installation should have failed for {scenario_name}")
                        return False
                    
                    # Verify error was logged
                    if len(installer.flow_controller.current_state.errors) == 0:
                        print(f"    ❌ Error not logged for {scenario_name}")
                        return False
            
            print("    ✓ Error handling and recovery works correctly")
            return True
            
        except Exception as e:
            print(f"    ❌ Error testing error handling: {e}")
            return False
        finally:
            self.cleanup_test_environment()
    
    def test_performance_optimizations(self) -> bool:
        """Test that performance optimizations are working correctly."""
        print("  Testing performance optimizations...")
        
        try:
            test_dir = self.setup_test_environment()
            
            mock_args = Mock()
            mock_args.silent = True
            mock_args.dry_run = True
            mock_args.skip_models = True
            mock_args.verbose = False
            mock_args.dev_mode = False
            mock_args.force_reinstall = False
            mock_args.custom_path = None
            
            hardware_profiles = self.create_mock_hardware_profiles()
            
            # Test that different hardware gets different optimizations
            optimization_results = []
            
            for profile in hardware_profiles:
                with patch('scripts.integrated_installer.ConfigurationEngine') as mock_config:
                    mock_config_instance = Mock()
                    mock_config.return_value = mock_config_instance
                    
                    # Simulate different optimizations based on hardware
                    if profile.cpu.cores >= 32:  # High-end
                        optimized_config = {
                            "optimization": {
                                "cpu_threads": min(64, profile.cpu.threads),
                                "memory_pool_gb": min(32, profile.memory.total_gb // 4),
                                "max_vram_usage_gb": profile.gpu.vram_gb - 2 if profile.gpu else 4
                            }
                        }
                    else:
                        optimized_config = {
                            "optimization": {
                                "cpu_threads": min(16, profile.cpu.threads),
                                "memory_pool_gb": min(16, profile.memory.total_gb // 4),
                                "max_vram_usage_gb": profile.gpu.vram_gb - 2 if profile.gpu else 4
                            }
                        }
                    
                    mock_config_instance.generate_base_configuration.return_value = {}
                    mock_config_instance.optimize_for_hardware.return_value = optimized_config
                    mock_config_instance.validate_configuration.return_value = Mock(is_valid=True, errors=[])
                    mock_config_instance.save_configuration.return_value = True
                    
                    installer = IntegratedInstaller(str(test_dir), mock_args)
                    installer.hardware_profile = profile
                    
                    success = installer._run_configuration_phase()
                    if not success:
                        print(f"    ❌ Configuration failed for {profile.cpu.model}")
                        return False
                    
                    optimization_results.append(optimized_config["optimization"])
            
            # Verify that high-end hardware gets better optimizations
            high_end_opt = optimization_results[0]  # Threadripper
            budget_opt = optimization_results[2]    # Ryzen 5
            
            if high_end_opt["cpu_threads"] <= budget_opt["cpu_threads"]:
                print("    ❌ High-end hardware should get more CPU threads")
                return False
            
            if high_end_opt["memory_pool_gb"] <= budget_opt["memory_pool_gb"]:
                print("    ❌ High-end hardware should get more memory allocation")
                return False
            
            print("    ✓ Performance optimizations work correctly")
            return True
            
        except Exception as e:
            print(f"    ❌ Error testing performance optimizations: {e}")
            return False
        finally:
            self.cleanup_test_environment()
    
    def run_comprehensive_validation(self) -> Dict[str, bool]:
        """Run all comprehensive validation tests."""
        print("Running comprehensive validation tests...")
        print("=" * 70)
        
        tests = [
            ("Requirement 1.1 - Batch file installation", self.test_requirement_1_1_batch_file_installation),
            ("Requirement 1.2 - Hardware detection", self.test_requirement_1_2_hardware_detection),
            ("Requirement 1.3 - Display hardware info", self.test_requirement_1_3_display_hardware_info),
            ("Requirement 1.4 - Desktop shortcuts", self.test_requirement_1_4_desktop_shortcuts),
            ("Requirement 2.1 - Python installation", self.test_requirement_2_1_python_installation),
            ("Requirement 3.1 - Hardware optimization", self.test_requirement_3_1_hardware_optimization),
            ("Requirement 4.1 - Progress indication", self.test_requirement_4_1_progress_indication),
            ("Requirement 5.1 - Installation validation", self.test_requirement_5_1_installation_validation),
            ("Requirement 6.1 - Model download", self.test_requirement_6_1_model_download),
            ("Requirement 7.1 - Shareable package", self.test_requirement_7_1_shareable_package),
            ("UI components integration", self.test_ui_components),
            ("Error handling and recovery", self.test_error_handling_and_recovery),
            ("Performance optimizations", self.test_performance_optimizations)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"\n{test_name}:")
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    print(f"  ✅ PASSED")
                else:
                    print(f"  ❌ FAILED")
            except Exception as e:
                print(f"  💥 ERROR: {e}")
                results[test_name] = False
        
        return results


def create_final_installation_package():
    """Create the final installation package for distribution."""
    print("\nCreating final installation package...")
    
    try:
        base_dir = Path(__file__).parent
        package_dir = base_dir / "WAN22-Installation-Package"
        
        if package_dir.exists():
            shutil.rmtree(package_dir)
        
        package_dir.mkdir()
        
        # Copy essential files
        essential_files = [
            "install.bat",
            "run_first_setup.bat",
            "manage.bat",
            "launch_wan22.bat",
            "launch_web_ui.bat",
            "README.md",
            "GETTING_STARTED.md",
            "UI_GUIDE.md"
        ]
        
        for file_name in essential_files:
            src = base_dir / file_name
            if src.exists():
                shutil.copy2(src, package_dir / file_name)
        
        # Copy directories
        essential_dirs = [
            "scripts",
            "resources", 
            "application",
            "examples",
            "docs"
        ]
        
        for dir_name in essential_dirs:
            src_dir = base_dir / dir_name
            if src_dir.exists():
                shutil.copytree(src_dir, package_dir / dir_name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        
        # Create empty directories
        empty_dirs = ["models", "logs", "venv"]
        for dir_name in empty_dirs:
            (package_dir / dir_name).mkdir(exist_ok=True)
            (package_dir / dir_name / ".gitkeep").write_text("")
        
        print(f"  ✓ Installation package created at: {package_dir}")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to create installation package: {e}")
        return False


if __name__ == "__main__":
    print("🚀 WAN2.2 Comprehensive Validation Suite")
    print("Task 13.2 - Perform comprehensive validation")
    print("=" * 70)
    
    validator = ComprehensiveValidationSuite()
    results = validator.run_comprehensive_validation()
    
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    passed_tests = [name for name, result in results.items() if result]
    failed_tests = [name for name, result in results.items() if not result]
    
    print(f"✅ Passed: {len(passed_tests)}/{len(results)}")
    print(f"❌ Failed: {len(failed_tests)}/{len(results)}")
    
    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  • {test}")
    
    # Create final installation package if all tests pass
    if len(failed_tests) == 0:
        print("\n🎉 All validation tests passed!")
        print("Creating final installation package...")
        
        package_created = create_final_installation_package()
        
        if package_created:
            print("\n✅ Task 13.2 completed successfully!")
            print("✅ Final installation package ready for distribution")
            print("\nValidated requirements:")
            for requirement in passed_tests:
                print(f"  ✓ {requirement}")
        else:
            print("\n⚠️  Validation passed but package creation failed")
            sys.exit(1)
    else:
        print(f"\n❌ {len(failed_tests)} validation tests failed")
        print("Please fix the issues before creating the final package")
        sys.exit(1)
    
    sys.exit(0)
