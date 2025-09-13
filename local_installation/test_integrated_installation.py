"""
Test script for integrated installation system.
Tests the complete installation workflow from start to finish.
This validates Task 13.1 - Integration of all components.
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scripts.integrated_installer import IntegratedInstaller
from scripts.interfaces import InstallationPhase, HardwareProfile, CPUInfo, MemoryInfo, GPUInfo
from scripts.installation_flow_controller import InstallationFlowController


class TestIntegratedInstallation(unittest.TestCase):
    """Test the integrated installation system."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.installation_path = str(self.test_dir)
        
        # Create required directories
        (self.test_dir / "scripts").mkdir(exist_ok=True)
        (self.test_dir / "resources").mkdir(exist_ok=True)
        (self.test_dir / "models").mkdir(exist_ok=True)
        (self.test_dir / "logs").mkdir(exist_ok=True)
        
        # Create mock requirements.txt
        requirements_file = self.test_dir / "resources" / "requirements.txt"
        requirements_file.write_text("torch>=1.9.0\ntorchvision>=0.10.0\n")
        
        # Mock arguments
        self.mock_args = Mock()
        self.mock_args.silent = True
        self.mock_args.verbose = False
        self.mock_args.dev_mode = False
        self.mock_args.skip_models = True  # Skip for testing
        self.mock_args.force_reinstall = False
        self.mock_args.dry_run = True  # Use dry run for testing
        self.mock_args.custom_path = None
        
        # Create mock hardware profile
        self.mock_hardware_profile = HardwareProfile(
            cpu=CPUInfo(
                model="Test CPU",
                cores=8,
                threads=16,
                base_clock=3.0,
                boost_clock=4.0,
                architecture="x64"
            ),
            memory=MemoryInfo(
                total_gb=16,
                available_gb=12,
                type="DDR4",
                speed=3200
            ),
            gpu=GPUInfo(
                model="Test GPU",
                vram_gb=8,
                cuda_version="11.8",
                driver_version="520.0",
                compute_capability="8.6"
            ),
            storage=None,
            os=None
        )
    
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_integrated_installer_initialization(self):
        """Test that the integrated installer initializes correctly."""
        installer = IntegratedInstaller(self.installation_path, self.mock_args)
        
        self.assertIsNotNone(installer.flow_controller)
        self.assertIsNotNone(installer.error_handler)
        self.assertIsNotNone(installer.user_guidance)
        self.assertIsNotNone(installer.progress_reporter)
        
        self.assertEqual(installer.installation_path, Path(self.installation_path))
        self.assertFalse(installer.installation_successful)
    
    @patch('scripts.integrated_installer.SystemDetector')
    def test_detection_phase_integration(self, mock_detector):
        """Test that the detection phase integrates correctly."""
        # Setup mocks
        mock_detector_instance = Mock()
        mock_detector.return_value = mock_detector_instance
        mock_detector_instance.detect_hardware.return_value = self.mock_hardware_profile
        
        mock_validation_result = Mock()
        mock_validation_result.meets_minimum = True
        mock_validation_result.issues = []
        mock_detector_instance.validate_requirements.return_value = mock_validation_result
        
        # Create installer and run detection phase
        installer = IntegratedInstaller(self.installation_path, self.mock_args)
        success = installer._run_detection_phase()
        
        self.assertTrue(success)
        self.assertIsNotNone(installer.hardware_profile)
        self.assertEqual(installer.hardware_profile.cpu.model, "Test CPU")
        
        # Verify system detector was called
        mock_detector.assert_called_once_with(self.installation_path)
        mock_detector_instance.detect_hardware.assert_called_once()
    
    @patch('scripts.integrated_installer.DependencyManager')
    @patch('scripts.integrated_installer.PythonInstallationHandler')
    def test_dependencies_phase_integration(self, mock_python_installer, mock_dependency_manager):
        """Test that the dependencies phase integrates correctly."""
        # Setup mocks
        mock_dep_manager_instance = Mock()
        mock_dependency_manager.return_value = mock_dep_manager_instance
        
        mock_python_info = Mock()
        mock_python_info.is_suitable = True
        mock_dep_manager_instance.check_python_installation.return_value = mock_python_info
        mock_dep_manager_instance.create_optimized_virtual_environment.return_value = True
        mock_dep_manager_instance.install_hardware_optimized_packages.return_value = True
        
        mock_verification_result = Mock()
        mock_verification_result.all_packages_installed = True
        mock_verification_result.missing_packages = []
        mock_dep_manager_instance.verify_package_installation.return_value = mock_verification_result
        
        # Create installer with hardware profile
        installer = IntegratedInstaller(self.installation_path, self.mock_args)
        installer.hardware_profile = self.mock_hardware_profile
        
        success = installer._run_dependencies_phase()
        
        self.assertTrue(success)
        
        # Verify dependency manager was called correctly
        mock_dependency_manager.assert_called_once_with(
            self.installation_path,
            hardware_profile=self.mock_hardware_profile
        )
    
    @patch('scripts.integrated_installer.ModelDownloader')
    def test_models_phase_integration_skip(self, mock_downloader):
        """Test that the models phase correctly skips when requested."""
        installer = IntegratedInstaller(self.installation_path, self.mock_args)
        
        # Test with skip_models=True
        success = installer._run_models_phase()
        
        self.assertTrue(success)
        # Verify downloader was not called when skipping
        mock_downloader.assert_not_called()
    
    @patch('scripts.integrated_installer.ModelDownloader')
    def test_models_phase_integration_download(self, mock_downloader):
        """Test that the models phase integrates correctly when downloading."""
        # Setup args to not skip models
        self.mock_args.skip_models = False
        
        # Setup mocks
        mock_downloader_instance = Mock()
        mock_downloader.return_value = mock_downloader_instance
        mock_downloader_instance.check_existing_models.return_value = []
        mock_downloader_instance.get_required_models.return_value = ["model1", "model2"]
        mock_downloader_instance.download_models_parallel.return_value = True
        
        mock_verification_result = Mock()
        mock_verification_result.all_valid = True
        mock_verification_result.invalid_models = []
        mock_downloader_instance.verify_all_models.return_value = mock_verification_result
        
        installer = IntegratedInstaller(self.installation_path, self.mock_args)
        success = installer._run_models_phase()
        
        self.assertTrue(success)
        
        # Verify downloader was called correctly
        mock_downloader.assert_called_once_with(self.installation_path)
        mock_downloader_instance.download_models_parallel.assert_called_once()
    
    @patch('scripts.integrated_installer.ConfigurationEngine')
    def test_configuration_phase_integration(self, mock_config_engine):
        """Test that the configuration phase integrates correctly."""
        # Setup mocks
        mock_config_instance = Mock()
        mock_config_engine.return_value = mock_config_instance
        
        base_config = {"system": {"threads": 4}}
        optimized_config = {"system": {"threads": 8}, "optimization": {"cpu_threads": 8}}
        
        mock_config_instance.generate_base_configuration.return_value = base_config
        mock_config_instance.optimize_for_hardware.return_value = optimized_config
        
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.errors = []
        mock_config_instance.validate_configuration.return_value = mock_validation_result
        mock_config_instance.save_configuration.return_value = True
        
        # Create installer with hardware profile
        installer = IntegratedInstaller(self.installation_path, self.mock_args)
        installer.hardware_profile = self.mock_hardware_profile
        
        success = installer._run_configuration_phase()
        
        self.assertTrue(success)
        
        # Verify configuration engine was called correctly
        mock_config_engine.assert_called_once_with(self.installation_path)
        mock_config_instance.optimize_for_hardware.assert_called_once_with(
            base_config, self.mock_hardware_profile
        )
    
    @patch('scripts.integrated_installer.InstallationValidator')
    def test_validation_phase_integration(self, mock_validator):
        """Test that the validation phase integrates correctly."""
        # Setup mocks
        mock_validator_instance = Mock()
        mock_validator.return_value = mock_validator_instance
        
        # Mock validation results
        mock_dep_result = Mock()
        mock_dep_result.success = True
        mock_dep_result.message = ""
        mock_validator_instance.validate_dependencies.return_value = mock_dep_result
        
        mock_model_result = Mock()
        mock_model_result.success = True
        mock_model_result.message = ""
        mock_validator_instance.validate_models.return_value = mock_model_result
        
        mock_hardware_result = Mock()
        mock_hardware_result.success = True
        mock_hardware_result.message = ""
        mock_validator_instance.validate_hardware_integration.return_value = mock_hardware_result
        
        mock_func_result = Mock()
        mock_func_result.success = True
        mock_func_result.message = ""
        mock_validator_instance.run_basic_functionality_test.return_value = mock_func_result
        
        mock_validator_instance.generate_validation_report.return_value = {"status": "success"}
        
        # Create installer with hardware profile
        installer = IntegratedInstaller(self.installation_path, self.mock_args)
        installer.hardware_profile = self.mock_hardware_profile
        
        success = installer._run_validation_phase()
        
        self.assertTrue(success)
        
        # Verify validator was called correctly
        mock_validator.assert_called_once_with(self.installation_path)
    
    def test_flow_controller_integration(self):
        """Test that the flow controller integrates correctly."""
        installer = IntegratedInstaller(self.installation_path, self.mock_args)
        
        # Test progress callback integration
        self.assertEqual(len(installer.flow_controller.progress_callbacks), 1)
        
        # Test state management
        installer.flow_controller.initialize_state(self.installation_path)
        state = installer.flow_controller.load_state()
        
        self.assertIsNotNone(state)
        self.assertEqual(state.phase, InstallationPhase.DETECTION)
        self.assertEqual(state.installation_path, self.installation_path)
    
    @patch('scripts.integrated_installer.SystemDetector')
    @patch('scripts.integrated_installer.DependencyManager')
    @patch('scripts.integrated_installer.ConfigurationEngine')
    @patch('scripts.integrated_installer.InstallationValidator')
    @patch('scripts.integrated_installer.PostInstallationSetup')
    @patch('scripts.integrated_installer.ShortcutCreator')
    def test_complete_installation_workflow(self, mock_shortcut, mock_post_install, 
                                          mock_validator, mock_config, mock_deps, mock_detector):
        """Test the complete installation workflow integration."""
        # Setup all mocks for successful installation
        self._setup_successful_mocks(
            mock_detector, mock_deps, mock_config, mock_validator, 
            mock_post_install, mock_shortcut
        )
        
        installer = IntegratedInstaller(self.installation_path, self.mock_args)
        success = installer.run_complete_installation()
        
        self.assertTrue(success)
        self.assertTrue(installer.installation_successful)
        
        # Verify all phases were called
        mock_detector.assert_called()
        mock_deps.assert_called()
        mock_config.assert_called()
        mock_validator.assert_called()
        mock_post_install.assert_called()
        mock_shortcut.assert_called()
    
    def _setup_successful_mocks(self, mock_detector, mock_deps, mock_config, 
                               mock_validator, mock_post_install, mock_shortcut):
        """Setup mocks for successful installation."""
        # System detector mocks
        mock_detector_instance = Mock()
        mock_detector.return_value = mock_detector_instance
        mock_detector_instance.detect_hardware.return_value = self.mock_hardware_profile
        
        mock_validation_result = Mock()
        mock_validation_result.meets_minimum = True
        mock_validation_result.issues = []
        mock_detector_instance.validate_requirements.return_value = mock_validation_result
        
        # Dependency manager mocks
        mock_deps_instance = Mock()
        mock_deps.return_value = mock_deps_instance
        
        mock_python_info = Mock()
        mock_python_info.is_suitable = True
        mock_deps_instance.check_python_installation.return_value = mock_python_info
        mock_deps_instance.create_optimized_virtual_environment.return_value = True
        mock_deps_instance.install_hardware_optimized_packages.return_value = True
        
        mock_verification_result = Mock()
        mock_verification_result.all_packages_installed = True
        mock_verification_result.missing_packages = []
        mock_deps_instance.verify_package_installation.return_value = mock_verification_result
        
        # Configuration engine mocks
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        mock_config_instance.generate_base_configuration.return_value = {"system": {}}
        mock_config_instance.validate_configuration.return_value = Mock(is_valid=True, errors=[])
        mock_config_instance.save_configuration.return_value = True
        
        # Validator mocks
        mock_validator_instance = Mock()
        mock_validator.return_value = mock_validator_instance
        mock_validator_instance.validate_dependencies.return_value = Mock(success=True, message="")
        mock_validator_instance.validate_models.return_value = Mock(success=True, message="")
        mock_validator_instance.validate_hardware_integration.return_value = Mock(success=True, message="")
        mock_validator_instance.run_basic_functionality_test.return_value = Mock(success=True, message="")
        mock_validator_instance.generate_validation_report.return_value = {"status": "success"}
        
        # Post-install setup mocks
        mock_post_install_instance = Mock()
        mock_post_install.return_value = mock_post_install_instance
        mock_post_install_instance.run_complete_post_install_setup.return_value = True
        
        # Shortcut creator mocks
        mock_shortcut_instance = Mock()
        mock_shortcut.return_value = mock_shortcut_instance
        mock_shortcut_instance.create_all_shortcuts.return_value = True
    
    def test_error_handling_integration(self):
        """Test that error handling is properly integrated."""
        installer = IntegratedInstaller(self.installation_path, self.mock_args)
        
        # Test error handler integration
        self.assertIsNotNone(installer.error_handler)
        self.assertIsNotNone(installer.user_guidance)
        
        # Test error handling in phase
        with patch.object(installer, '_run_detection_phase', side_effect=Exception("Test error")):
            success = installer._run_all_phases()
            self.assertFalse(success)
    
    def test_snapshot_integration(self):
        """Test that snapshot functionality is properly integrated."""
        installer = IntegratedInstaller(self.installation_path, self.mock_args)
        
        # Test snapshot creation
        try:
            installer._create_initial_snapshot()
            # Should not raise exception even if no files to backup
        except Exception as e:
            self.fail(f"Snapshot creation failed: {e}")
        
        # Test snapshot listing
        snapshots = installer.flow_controller.list_snapshots()
        self.assertIsInstance(snapshots, list)


class TestInstallationFlowIntegration(unittest.TestCase):
    """Test the installation flow controller integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.installation_path = str(self.test_dir)
        
        # Create required directories
        (self.test_dir / "logs").mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_flow_controller_initialization(self):
        """Test flow controller initialization."""
        controller = InstallationFlowController(
            self.installation_path, 
            dry_run=True
        )
        
        self.assertIsNotNone(controller.rollback_manager)
        self.assertEqual(len(controller.progress_callbacks), 0)
        self.assertTrue(controller.dry_run)
    
    def test_state_management_integration(self):
        """Test state management integration."""
        controller = InstallationFlowController(
            self.installation_path, 
            dry_run=True
        )
        
        # Initialize state
        state = controller.initialize_state(self.installation_path)
        self.assertIsNotNone(state)
        self.assertEqual(state.phase, InstallationPhase.DETECTION)
        
        # Update progress
        controller.update_progress(
            InstallationPhase.DEPENDENCIES, 0.5, "Installing packages"
        )
        
        self.assertEqual(controller.current_state.phase, InstallationPhase.DEPENDENCIES)
        self.assertGreater(controller.current_state.progress, 0)
    
    def test_error_and_warning_integration(self):
        """Test error and warning handling integration."""
        controller = InstallationFlowController(
            self.installation_path, 
            dry_run=True
        )
        
        controller.initialize_state(self.installation_path)
        
        # Add error
        controller.add_error("Test error")
        self.assertEqual(len(controller.current_state.errors), 1)
        self.assertEqual(controller.current_state.errors[0], "Test error")
        
        # Add warning
        controller.add_warning("Test warning")
        self.assertEqual(len(controller.current_state.warnings), 1)
        self.assertEqual(controller.current_state.warnings[0], "Test warning")


def run_integration_tests():
    """Run all integration tests."""
    print("Running integrated installation tests...")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestIntegratedInstallation))
    suite.addTest(unittest.makeSuite(TestInstallationFlowIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
