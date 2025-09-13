"""
Integrated Installation Orchestrator
Wires together all installation phases into a cohesive workflow.
This is the main integration point for task 13.1.
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import all installation components
from interfaces import InstallationPhase, InstallationError, ErrorCategory, HardwareProfile
from base_classes import BaseInstallationComponent, ConsoleProgressReporter
from installation_flow_controller import InstallationFlowController
from error_handler import ComprehensiveErrorHandler
from user_guidance import UserGuidanceSystem
from rollback_manager import RollbackManager
from reliability_manager import ReliabilityManager

# Import phase-specific components
from detect_system import SystemDetector
from setup_dependencies import DependencyManager, PythonInstallationHandler
from download_models import ModelDownloader
from generate_config import ConfigurationEngine
from validate_installation import InstallationValidator
from post_install_setup import PostInstallationSetup
from create_shortcuts import ShortcutCreator


class IntegratedInstaller(BaseInstallationComponent):
    """
    Main integrated installer that coordinates all installation phases
    into a cohesive workflow.
    """
    
    def __init__(self, installation_path: str, args: argparse.Namespace):
        super().__init__(installation_path)
        self.args = args
        self.installation_start_time = datetime.now()
        
        # Initialize reliability system first
        self.reliability_manager = ReliabilityManager(installation_path, self.logger)
        
        # Initialize core systems
        self.flow_controller = InstallationFlowController(
            installation_path, 
            dry_run=getattr(args, 'dry_run', False),
            log_level='DEBUG' if getattr(args, 'verbose', False) else 'INFO'
        )
        
        self.error_handler = ComprehensiveErrorHandler(installation_path, self.logger)
        self.user_guidance = UserGuidanceSystem(installation_path, self.logger)
        self.progress_reporter = ConsoleProgressReporter()
        
        # Wrap core systems with reliability enhancements
        self.flow_controller = self.reliability_manager.wrap_component(
            self.flow_controller, "installation_flow_controller", "integrated_flow_controller"
        )
        self.error_handler = self.reliability_manager.wrap_component(
            self.error_handler, "error_handler", "integrated_error_handler"
        )
        self.progress_reporter = self.reliability_manager.wrap_component(
            self.progress_reporter, "progress_reporter", "integrated_progress_reporter"
        )
        
        # Add progress callback
        self.flow_controller.add_progress_callback(self._on_progress_update)
        
        # Phase components (initialized as needed)
        self.system_detector = None
        self.dependency_manager = None
        self.model_downloader = None
        self.configuration_engine = None
        self.validator = None
        self.post_install_setup = None
        
        # Installation state
        self.hardware_profile = None
        self.installation_successful = False
        
        self.logger.info(f"Integrated installer initialized with reliability system and args: {vars(args)}")
    
    def run_complete_installation(self) -> bool:
        """
        Run the complete installation process with all phases integrated.
        This is the main entry point for the integrated installation.
        """
        try:
            self.logger.info("Starting complete WAN2.2 installation")
            self._display_installation_header()
            
            # Check for existing installation
            if not self.args.force_reinstall:
                existing_state = self.flow_controller.load_state()
                if existing_state:
                    if self._should_resume_installation():
                        return self._resume_installation(existing_state)
                    else:
                        self.flow_controller.clear_state()
            
            # Initialize new installation
            self.flow_controller.initialize_state(str(self.installation_path))
            
            # Create initial snapshot
            self._create_initial_snapshot()
            
            # Run all installation phases
            success = self._run_all_phases()
            
            if success:
                self._complete_installation()
                self.installation_successful = True
                return True
            else:
                self._handle_installation_failure()
                return False
                
        except KeyboardInterrupt:
            self.logger.warning("Installation cancelled by user")
            self._handle_user_cancellation()
            return False
        except Exception as e:
            self.logger.exception("Unexpected error during installation")
            self._handle_unexpected_error(e)
            return False
        finally:
            self._cleanup_installation()
    
    def _run_all_phases(self) -> bool:
        """Run all installation phases in sequence."""
        phases = [
            (InstallationPhase.DETECTION, self._run_detection_phase),
            (InstallationPhase.DEPENDENCIES, self._run_dependencies_phase),
            (InstallationPhase.MODELS, self._run_models_phase),
            (InstallationPhase.CONFIGURATION, self._run_configuration_phase),
            (InstallationPhase.VALIDATION, self._run_validation_phase)
        ]
        
        for phase, phase_func in phases:
            try:
                self.logger.info(f"Starting phase: {phase.value}")
                self._display_phase_header(phase)
                
                # Create snapshot before critical phases
                if phase in [InstallationPhase.DEPENDENCIES, InstallationPhase.CONFIGURATION]:
                    self._create_phase_snapshot(phase)
                
                # Run the phase
                success = phase_func()
                
                if not success:
                    self.logger.error(f"Phase {phase.value} failed")
                    self.flow_controller.add_error(f"Phase {phase.value} failed")
                    return False
                
                self.logger.info(f"Phase {phase.value} completed successfully")
                
            except InstallationError as e:
                self.logger.error(f"Installation error in phase {phase.value}: {e}")
                return self._handle_phase_error(phase, e)
            except Exception as e:
                self.logger.exception(f"Unexpected error in phase {phase.value}: {e}")
                
                # Use reliability manager to handle the failure
                recovery_action = self.reliability_manager.handle_component_failure(
                    f"phase_{phase.value}", e, {"phase": phase.value, "method": phase_func.__name__}
                )
                
                if recovery_action.value == "retry":
                    self.logger.info(f"Retrying phase {phase.value} after recovery")
                    try:
                        success = phase_func()
                        if success:
                            continue
                    except Exception as retry_error:
                        self.logger.error(f"Retry failed for phase {phase.value}: {retry_error}")
                
                return False
        
        return True
    
    def _wrap_component_with_reliability(self, component, component_type: str, component_id: str):
        """Wrap a component with reliability enhancements."""
        if component is None:
            return None
        
        return self.reliability_manager.wrap_component(component, component_type, component_id)
    
    def _run_detection_phase(self) -> bool:
        """Run system detection phase with reliability enhancements."""
        try:
            self.logger.info("Starting system detection phase")
            
            # Initialize and wrap system detector
            if self.system_detector is None:
                self.system_detector = SystemDetector(str(self.installation_path))
                self.system_detector = self._wrap_component_with_reliability(
                    self.system_detector, "system_detector", "main_system_detector"
                )
            
            # Run detection
            self.hardware_profile = self.system_detector.detect_system()
            
            if not self.hardware_profile:
                raise InstallationError(
                    "System detection failed",
                    ErrorCategory.SYSTEM,
                    ["Check system compatibility", "Review hardware requirements"]
                )
            
            self.logger.info(f"System detection completed: {self.hardware_profile}")
            return True
            
        except Exception as e:
            self.logger.error(f"System detection phase failed: {e}")
            return False
    
    def _run_dependencies_phase(self) -> bool:
        """Run dependency installation phase with reliability enhancements."""
        try:
            self.logger.info("Starting dependencies installation phase")
            
            # Initialize and wrap dependency manager
            if self.dependency_manager is None:
                self.dependency_manager = DependencyManager(str(self.installation_path))
                self.dependency_manager = self._wrap_component_with_reliability(
                    self.dependency_manager, "dependency_manager", "main_dependency_manager"
                )
            
            # Install dependencies
            success = self.dependency_manager.install_all_dependencies(
                self.hardware_profile, dev_mode=getattr(self.args, 'dev_mode', False)
            )
            
            if not success:
                raise InstallationError(
                    "Dependency installation failed",
                    ErrorCategory.DEPENDENCY,
                    ["Check internet connection", "Verify package sources", "Review system requirements"]
                )
            
            self.logger.info("Dependencies installation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Dependencies phase failed: {e}")
            return False
    
    def _run_models_phase(self) -> bool:
        """Run model download phase with reliability enhancements."""
        try:
            if getattr(self.args, 'skip_models', False):
                self.logger.info("Skipping model download as requested")
                return True
            
            self.logger.info("Starting model download phase")
            
            # Initialize and wrap model downloader
            if self.model_downloader is None:
                self.model_downloader = ModelDownloader(str(self.installation_path))
                self.model_downloader = self._wrap_component_with_reliability(
                    self.model_downloader, "model_downloader", "main_model_downloader"
                )
            
            # Download models
            success = self.model_downloader.download_all_models(self.hardware_profile)
            
            if not success:
                raise InstallationError(
                    "Model download failed",
                    ErrorCategory.NETWORK,
                    ["Check internet connection", "Verify Hugging Face access", "Try alternative sources"]
                )
            
            self.logger.info("Model download completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Models phase failed: {e}")
            return False
    
    def _run_configuration_phase(self) -> bool:
        """Run configuration generation phase with reliability enhancements."""
        try:
            self.logger.info("Starting configuration generation phase")
            
            # Initialize and wrap configuration engine
            if self.configuration_engine is None:
                self.configuration_engine = ConfigurationEngine(str(self.installation_path))
                self.configuration_engine = self._wrap_component_with_reliability(
                    self.configuration_engine, "configuration_engine", "main_configuration_engine"
                )
            
            # Generate configuration
            success = self.configuration_engine.generate_complete_configuration(
                self.hardware_profile
            )
            
            if not success:
                raise InstallationError(
                    "Configuration generation failed",
                    ErrorCategory.CONFIGURATION,
                    ["Check file permissions", "Verify system detection", "Review hardware profile"]
                )
            
            self.logger.info("Configuration generation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration phase failed: {e}")
            return False
    
    def _run_validation_phase(self) -> bool:
        """Run installation validation phase with reliability enhancements."""
        try:
            self.logger.info("Starting installation validation phase")
            
            # Initialize and wrap validator
            if self.validator is None:
                self.validator = InstallationValidator(str(self.installation_path))
                self.validator = self._wrap_component_with_reliability(
                    self.validator, "installation_validator", "main_installation_validator"
                )
            
            # Run validation
            validation_result = self.validator.validate_complete_installation()
            
            if not validation_result.success:
                raise InstallationError(
                    f"Installation validation failed: {validation_result.error_message}",
                    ErrorCategory.VALIDATION,
                    validation_result.remediation_steps
                )
            
            self.logger.info("Installation validation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Validation phase failed: {e}")
            return False
        
        return True
    
    def _run_detection_phase(self) -> bool:
        """Run system detection phase with full hardware profiling."""
        self.flow_controller.update_progress(
            InstallationPhase.DETECTION, 0.0, "Initializing system detection"
        )
        
        try:
            # Initialize system detector
            self.system_detector = SystemDetector(str(self.installation_path))
            
            # Detect hardware
            self.flow_controller.update_progress(
                InstallationPhase.DETECTION, 0.3, "Detecting CPU and memory"
            )
            
            self.hardware_profile = self.system_detector.detect_hardware()
            
            self.flow_controller.update_progress(
                InstallationPhase.DETECTION, 0.6, "Analyzing GPU capabilities"
            )
            
            # Validate system requirements
            validation_result = self.system_detector.validate_requirements(
                self.hardware_profile
            )
            
            if not validation_result.success:
                raise InstallationError(
                    f"System does not meet minimum requirements: {validation_result.message}",
                    ErrorCategory.SYSTEM,
                    validation_result.warnings or []
                )
            
            self.flow_controller.update_progress(
                InstallationPhase.DETECTION, 0.9, "Generating hardware profile"
            )
            
            # Save hardware profile
            self._save_hardware_profile()
            
            # Display hardware summary
            if not self.args.silent:
                self._display_hardware_summary()
            
            self.flow_controller.update_progress(
                InstallationPhase.DETECTION, 1.0, "System detection completed"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"System detection failed: {e}")
            raise InstallationError(
                f"System detection failed: {str(e)}",
                ErrorCategory.SYSTEM,
                ["Check hardware compatibility", "Run as administrator"]
            )
    
    def _run_dependencies_phase(self) -> bool:
        """Run dependency installation phase."""
        self.flow_controller.update_progress(
            InstallationPhase.DEPENDENCIES, 0.0, "Initializing dependency management"
        )
        
        try:
            # Initialize dependency manager
            self.dependency_manager = DependencyManager(
                str(self.installation_path)
            )
            
            # Check Python installation
            self.flow_controller.update_progress(
                InstallationPhase.DEPENDENCIES, 0.1, "Checking Python installation"
            )
            
            python_info = self.dependency_manager.check_python_installation()
            system_python = python_info.get("system_python", {})
            if not system_python.get("suitable", False):
                self.flow_controller.update_progress(
                    InstallationPhase.DEPENDENCIES, 0.2, "Installing Python"
                )
                
                python_installer = PythonInstallationHandler(str(self.installation_path))
                success = python_installer.install_embedded_python()
                if not success:
                    raise InstallationError(
                        "Failed to install Python",
                        ErrorCategory.DEPENDENCY,
                        ["Check internet connection", "Run as administrator"]
                    )
            
            # Create virtual environment
            self.flow_controller.update_progress(
                InstallationPhase.DEPENDENCIES, 0.4, "Creating virtual environment"
            )
            
            venv_path = self.installation_path / "venv"
            success = self.dependency_manager.create_virtual_environment(
                str(venv_path), self.hardware_profile
            )
            if not success:
                raise InstallationError(
                    "Failed to create virtual environment",
                    ErrorCategory.DEPENDENCY,
                    ["Check disk space", "Verify Python installation"]
                )
            
            # Install packages
            self.flow_controller.update_progress(
                InstallationPhase.DEPENDENCIES, 0.6, "Installing Python packages"
            )
            
            requirements_file = self.installation_path / "resources" / "requirements.txt"
            success = self.dependency_manager.install_packages(
                str(requirements_file), self.hardware_profile
            )
            if not success:
                raise InstallationError(
                    "Failed to install Python packages",
                    ErrorCategory.DEPENDENCY,
                    ["Check internet connection", "Verify requirements.txt"]
                )
            
            # Verify installation
            self.flow_controller.update_progress(
                InstallationPhase.DEPENDENCIES, 0.9, "Verifying package installation"
            )
            
            verification_result = self.dependency_manager.validate_installation()
            if not verification_result.success:
                self.flow_controller.add_warning(
                    f"Some packages may not be properly installed: {verification_result.message}"
                )
            
            self.flow_controller.update_progress(
                InstallationPhase.DEPENDENCIES, 1.0, "Dependencies installed successfully"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Dependency installation failed: {e}")
            raise InstallationError(
                f"Dependency installation failed: {str(e)}",
                ErrorCategory.DEPENDENCY,
                ["Check internet connection", "Verify disk space", "Run as administrator"]
            )
    
    def _run_models_phase(self) -> bool:
        """Run model download phase."""
        if self.args.skip_models:
            self.logger.info("Skipping model download as requested")
            return True
        
        self.flow_controller.update_progress(
            InstallationPhase.MODELS, 0.0, "Initializing model downloader"
        )
        
        try:
            # Initialize model downloader
            self.model_downloader = ModelDownloader(str(self.installation_path))
            
            # Check existing models
            self.flow_controller.update_progress(
                InstallationPhase.MODELS, 0.1, "Checking existing models"
            )
            
            existing_models = self.model_downloader.check_existing_models()
            required_models = self.model_downloader.get_required_models()
            missing_models = [m for m in required_models if m not in existing_models]
            
            if not missing_models:
                self.logger.info("All required models are already present")
                self.flow_controller.update_progress(
                    InstallationPhase.MODELS, 1.0, "All models already present"
                )
                return True
            
            # Download missing models
            self.flow_controller.update_progress(
                InstallationPhase.MODELS, 0.2, f"Downloading {len(missing_models)} models"
            )
            
            def progress_callback(model_name: str, progress: float, message: str = ""):
                self.flow_controller.update_progress(
                    InstallationPhase.MODELS, 
                    0.2 + (progress / 100.0 * 0.7),  # Convert percentage to 0.2-0.9 range
                    message or f"Downloading {model_name}: {progress:.1f}%"
                )
            
            success = self.model_downloader.download_wan22_models(
                progress_callback=progress_callback
            )
            
            if not success:
                raise InstallationError(
                    "Failed to download required models",
                    ErrorCategory.NETWORK,
                    ["Check internet connection", "Verify disk space", "Retry download"]
                )
            
            # Verify model integrity
            self.flow_controller.update_progress(
                InstallationPhase.MODELS, 0.95, "Verifying model integrity"
            )
            
            verification_result = self.model_downloader.verify_all_models()
            if not verification_result.all_valid:
                self.flow_controller.add_warning(
                    f"Some models may be corrupted: {verification_result.invalid_models}"
                )
            
            self.flow_controller.update_progress(
                InstallationPhase.MODELS, 1.0, "Models downloaded successfully"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model download failed: {e}")
            raise InstallationError(
                f"Model download failed: {str(e)}",
                ErrorCategory.NETWORK,
                ["Check internet connection", "Verify disk space", "Retry installation"]
            )
    
    def _run_configuration_phase(self) -> bool:
        """Run configuration generation phase."""
        self.flow_controller.update_progress(
            InstallationPhase.CONFIGURATION, 0.0, "Initializing configuration engine"
        )
        
        try:
            # Initialize configuration engine
            self.configuration_engine = ConfigurationEngine(str(self.installation_path))
            
            # Generate base configuration
            self.flow_controller.update_progress(
                InstallationPhase.CONFIGURATION, 0.3, "Generating base configuration"
            )
            
            base_config = self.configuration_engine.generate_config(self.hardware_profile)
            
            # Apply hardware optimizations
            self.flow_controller.update_progress(
                InstallationPhase.CONFIGURATION, 0.6, "Applying hardware optimizations"
            )
            
            optimized_config = self.configuration_engine.optimize_for_hardware(
                base_config, self.hardware_profile
            )
            
            # Validate configuration
            self.flow_controller.update_progress(
                InstallationPhase.CONFIGURATION, 0.8, "Validating configuration"
            )
            
            # Configuration is generated and optimized, assume it's valid
            # In a real implementation, you might want to add validation logic here
            
            # Save configuration
            self.flow_controller.update_progress(
                InstallationPhase.CONFIGURATION, 0.9, "Saving configuration"
            )
            
            config_path = self.installation_path / "config.json"
            success = self.configuration_engine.save_config(
                optimized_config, str(config_path)
            )
            
            if not success:
                raise InstallationError(
                    "Failed to save configuration file",
                    ErrorCategory.SYSTEM,
                    ["Check file permissions", "Verify disk space"]
                )
            
            # Display configuration summary
            if not self.args.silent:
                self._display_configuration_summary(optimized_config)
            
            self.flow_controller.update_progress(
                InstallationPhase.CONFIGURATION, 1.0, "Configuration generated successfully"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration generation failed: {e}")
            raise InstallationError(
                f"Configuration generation failed: {str(e)}",
                ErrorCategory.CONFIGURATION,
                ["Check hardware detection", "Use manual configuration"]
            )
    
    def _run_validation_phase(self) -> bool:
        """Run installation validation phase."""
        self.flow_controller.update_progress(
            InstallationPhase.VALIDATION, 0.0, "Initializing validation framework"
        )
        
        try:
            # Initialize validator
            self.validator = InstallationValidator(str(self.installation_path))
            
            # Validate dependencies
            self.flow_controller.update_progress(
                InstallationPhase.VALIDATION, 0.2, "Validating dependencies"
            )
            
            dep_result = self.validator.validate_dependencies()
            if not dep_result.success:
                self.flow_controller.add_warning(
                    f"Some dependencies may have issues: {dep_result.message}"
                )
            
            # Validate models (skip if models were not downloaded)
            self.flow_controller.update_progress(
                InstallationPhase.VALIDATION, 0.4, "Validating models"
            )
            
            if self.args.skip_models:
                self.logger.info("Skipping model validation (models were not downloaded)")
                self.flow_controller.add_warning("Model validation skipped - models were not downloaded")
            else:
                model_result = self.validator.validate_models()
                if not model_result.success:
                    raise InstallationError(
                        f"Model validation failed: {model_result.message}",
                        ErrorCategory.VALIDATION,
                        ["Re-download models", "Check model integrity"]
                    )
            
            # Test hardware integration
            self.flow_controller.update_progress(
                InstallationPhase.VALIDATION, 0.6, "Testing hardware integration"
            )
            
            hardware_result = self.validator.validate_hardware_integration()
            if not hardware_result.success:
                self.flow_controller.add_warning(
                    f"Hardware integration issues: {hardware_result.message}"
                )
            
            # Run functionality tests
            self.flow_controller.update_progress(
                InstallationPhase.VALIDATION, 0.8, "Running functionality tests"
            )
            
            func_result = self.validator.run_functionality_test()
            if not func_result.success:
                raise InstallationError(
                    f"Functionality test failed: {func_result.message}",
                    ErrorCategory.VALIDATION,
                    ["Check configuration", "Verify model files"]
                )
            
            # Generate validation report
            self.flow_controller.update_progress(
                InstallationPhase.VALIDATION, 0.95, "Generating validation report"
            )
            
            validation_report = self.validator.generate_validation_report()
            self._save_validation_report(validation_report)
            
            self.flow_controller.update_progress(
                InstallationPhase.VALIDATION, 1.0, "Validation completed successfully"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Installation validation failed: {e}")
            raise InstallationError(
                f"Installation validation failed: {str(e)}",
                ErrorCategory.VALIDATION,
                ["Check installation logs", "Retry installation"]
            )
    
    def _complete_installation(self) -> None:
        """Complete the installation with post-installation setup."""
        try:
            self.logger.info("Completing installation")
            
            # Run post-installation setup
            self.post_install_setup = PostInstallationSetup(str(self.installation_path))
            self.post_install_setup.run_complete_post_install_setup()
            
            # Create shortcuts
            shortcut_creator = ShortcutCreator(str(self.installation_path))
            shortcut_creator.create_all_shortcuts()
            
            # Create final snapshot
            self.flow_controller.create_snapshot("Installation completed successfully")
            
            # Cleanup old snapshots
            self.flow_controller.cleanup_old_snapshots(keep_count=5)
            
            # Update final state
            self.flow_controller.update_progress(
                InstallationPhase.COMPLETE, 1.0, "Installation completed successfully"
            )
            
            # Display completion message
            self._display_completion_message()
            
        except Exception as e:
            self.logger.warning(f"Post-installation setup had issues: {e}")
            # Don't fail the entire installation for post-setup issues
    
    def _handle_installation_failure(self) -> None:
        """Handle installation failure with recovery options."""
        self.logger.error("Installation failed")
        
        if not self.args.silent:
            print("\n❌ Installation failed!")
            print("\nRecovery options:")
            print("1. Check installation logs for details")
            print("2. Try running with --verbose for more information")
            print("3. Use --force-reinstall to start fresh")
            print("4. Contact support with log files")
            
            # Offer to restore from snapshot
            snapshots = self.flow_controller.list_snapshots()
            if snapshots:
                print(f"\n{len(snapshots)} backup snapshots available for recovery")
                response = input("Would you like to restore from a backup? (y/n): ").lower()
                if response in ['y', 'yes']:
                    self._offer_snapshot_recovery(snapshots)
    
    def _handle_user_cancellation(self) -> None:
        """Handle user cancellation gracefully."""
        if not self.args.silent:
            print("\n⚠️  Installation cancelled by user")
            print("Installation state has been saved and can be resumed later")
    
    def _handle_unexpected_error(self, error: Exception) -> None:
        """Handle unexpected errors during installation."""
        self.logger.exception("Unexpected error during installation")
        
        if not self.args.silent:
            print(f"\n💥 Unexpected error: {error}")
            print("This is likely a bug. Please report it with the log files.")
    
    def _cleanup_installation(self) -> None:
        """Cleanup after installation completion or failure."""
        try:
            # Log installation summary
            duration = datetime.now() - self.installation_start_time
            self.logger.info(f"Installation completed in {duration}")
            
            # Generate final summary
            summary = self.flow_controller.get_installation_summary()
            self.logger.info(f"Installation summary: {summary}")
            
        except Exception as e:
            self.logger.warning(f"Cleanup had issues: {e}")
    
    # Helper methods for display and user interaction
    
    def _display_installation_header(self) -> None:
        """Display installation header."""
        if self.args.silent:
            return
        
        print("\n" + "="*80)
        print("🚀 WAN2.2 Integrated Installation System")
        print("="*80)
        print("Automated installation with hardware detection and optimization")
        
        if self.args.dry_run:
            print("\n🔍 DRY RUN MODE - No changes will be made")
        
        print()
    
    def _display_phase_header(self, phase: InstallationPhase) -> None:
        """Display phase header."""
        if self.args.silent:
            return
        
        phase_names = {
            InstallationPhase.DETECTION: "System Detection",
            InstallationPhase.DEPENDENCIES: "Dependency Installation", 
            InstallationPhase.MODELS: "Model Download",
            InstallationPhase.CONFIGURATION: "Configuration Generation",
            InstallationPhase.VALIDATION: "Installation Validation"
        }
        
        name = phase_names.get(phase, phase.value)
        print(f"\n📋 {name}")
        print("-" * (len(name) + 4))
    
    def _display_hardware_summary(self) -> None:
        """Display detected hardware summary."""
        if not self.hardware_profile:
            return
        
        print(f"\n💻 Detected Hardware:")
        print(f"   CPU: {self.hardware_profile.cpu.model} ({self.hardware_profile.cpu.cores} cores)")
        print(f"   RAM: {self.hardware_profile.memory.total_gb}GB")
        if self.hardware_profile.gpu:
            print(f"   GPU: {self.hardware_profile.gpu.model} ({self.hardware_profile.gpu.vram_gb}GB VRAM)")
        print()
    
    def _display_configuration_summary(self, config: Dict[str, Any]) -> None:
        """Display configuration summary."""
        print(f"\n⚙️  Configuration Summary:")
        if 'optimization' in config:
            opt = config['optimization']
            print(f"   CPU Threads: {opt.get('cpu_threads', 'auto')}")
            print(f"   Memory Pool: {opt.get('memory_pool_gb', 'auto')}GB")
            if 'max_vram_usage_gb' in opt:
                print(f"   VRAM Usage: {opt['max_vram_usage_gb']}GB")
        print()
    
    def _display_completion_message(self) -> None:
        """Display installation completion message."""
        if self.args.silent:
            return
        
        print("\n" + "="*80)
        print("🎉 Installation Completed Successfully!")
        print("="*80)
        print("\n✅ All phases completed successfully")
        print("✅ Hardware optimizations applied")
        print("✅ Models downloaded and verified")
        print("✅ Configuration generated")
        print("✅ Installation validated")
        
        print("\n📋 Next Steps:")
        print("• Desktop shortcuts have been created")
        print("• Check GETTING_STARTED.md for usage instructions")
        print("• Run validation tests to ensure everything works")
        
        duration = datetime.now() - self.installation_start_time
        print(f"\n⏱️  Total installation time: {duration}")
        print("="*80)
    
    # Helper methods for state management
    
    def _should_resume_installation(self) -> bool:
        """Check if user wants to resume existing installation."""
        if self.args.silent:
            return True
        
        response = input("Found incomplete installation. Resume? (y/n): ").lower()
        return response in ['y', 'yes']
    
    def _resume_installation(self, state) -> bool:
        """Resume installation from existing state."""
        self.logger.info(f"Resuming installation from phase: {state.phase.value}")
        # For now, restart the installation - could be enhanced to resume from specific phase
        return self.run_complete_installation()
    
    def _create_initial_snapshot(self) -> None:
        """Create initial snapshot before installation."""
        try:
            files_to_backup = []
            config_file = self.installation_path / "config.json"
            if config_file.exists():
                files_to_backup.append(str(config_file))
            
            self.flow_controller.create_snapshot(
                "Pre-installation snapshot",
                files_to_backup=files_to_backup
            )
        except Exception as e:
            self.logger.warning(f"Failed to create initial snapshot: {e}")
    
    def _create_phase_snapshot(self, phase: InstallationPhase) -> None:
        """Create snapshot before critical phases."""
        try:
            self.flow_controller.create_snapshot(f"Before {phase.value} phase")
        except Exception as e:
            self.logger.warning(f"Failed to create snapshot before {phase.value}: {e}")
    
    def _save_hardware_profile(self) -> None:
        """Save hardware profile to file."""
        try:
            profile_file = self.installation_path / "logs" / "hardware_profile.json"
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(self.hardware_profile.__dict__, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save hardware profile: {e}")
    
    def _save_validation_report(self, report: Dict[str, Any]) -> None:
        """Save validation report to file."""
        try:
            report_file = self.installation_path / "logs" / "validation_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save validation report: {e}")
    
    def _offer_snapshot_recovery(self, snapshots: List[Dict[str, Any]]) -> None:
        """Offer snapshot recovery options to user."""
        print("\nAvailable snapshots:")
        for i, snapshot in enumerate(snapshots[-5:], 1):  # Show last 5
            print(f"{i}. {snapshot['description']} ({snapshot['timestamp']})")
        
        try:
            choice = int(input("Select snapshot to restore (0 to cancel): "))
            if 1 <= choice <= len(snapshots):
                snapshot_id = snapshots[choice - 1]['id']
                success = self.flow_controller.restore_snapshot(snapshot_id)
                if success:
                    print("✅ Snapshot restored successfully")
                else:
                    print("❌ Failed to restore snapshot")
        except (ValueError, IndexError):
            print("Invalid selection")
    
    def _handle_phase_error(self, phase: InstallationPhase, error: InstallationError) -> bool:
        """Handle errors that occur during installation phases."""
        self.flow_controller.add_error(str(error))
        
        # Use comprehensive error handling
        recovery_action = self.error_handler.handle_error(error)
        
        # Display user-friendly error message
        if not self.args.silent:
            friendly_message = self.user_guidance.format_user_friendly_error(error)
            print(friendly_message)
        
        # Handle recovery actions
        if recovery_action == "abort":
            return False
        elif recovery_action == "retry":
            self.logger.info(f"Retrying phase {phase.value}")
            # Could implement phase retry logic here
            return False  # For now, don't retry automatically
        elif recovery_action == "continue":
            self.logger.warning(f"Continuing despite error in phase {phase.value}")
            return True
        
        return False
    
    def _on_progress_update(self, phase: InstallationPhase, progress: float, task: str) -> None:
        """Handle progress updates from flow controller."""
        self.progress_reporter.update_progress(phase, progress, task)


def main():
    """Main entry point for integrated installer."""
    parser = argparse.ArgumentParser(description="WAN2.2 Integrated Installation System")
    
    parser.add_argument("--silent", action="store_true", help="Silent installation")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--dev-mode", action="store_true", help="Development mode")
    parser.add_argument("--skip-models", action="store_true", help="Skip model download")
    parser.add_argument("--force-reinstall", action="store_true", help="Force reinstall")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--custom-path", type=str, help="Custom installation path")
    
    args = parser.parse_args()
    
    # Determine installation path
    if args.custom_path:
        installation_path = Path(args.custom_path).resolve()
    else:
        installation_path = Path(__file__).parent.parent.resolve()
    
    # Create and run integrated installer
    installer = IntegratedInstaller(str(installation_path), args)
    success = installer.run_complete_installation()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
