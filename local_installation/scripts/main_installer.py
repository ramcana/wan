"""
Main installer orchestrator that coordinates the entire installation process.
This script is called by install.bat and manages all installation phases.
Updated for Task 13.1 - Integration of all components.
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from interfaces import InstallationPhase, InstallationError, ErrorCategory
from base_classes import (
    BaseInstallationComponent, ConsoleProgressReporter, 
    DefaultErrorHandler, InstallationStateManager
)
from error_handler import ComprehensiveErrorHandler, ErrorContext
from user_guidance import UserGuidanceSystem
from installation_flow_controller import InstallationFlowController
from integrated_installer import IntegratedInstaller
from reliability_manager import ReliabilityManager
from pre_installation_validator import PreInstallationValidator


class MainInstaller(BaseInstallationComponent):
    """Main installer that orchestrates the entire installation process."""
    
    def __init__(self, installation_path: str, args: argparse.Namespace):
        super().__init__(installation_path)
        self.args = args
        self.progress_reporter = ConsoleProgressReporter()
        
        # Initialize reliability system first
        self.reliability_manager = ReliabilityManager(installation_path, self.logger)
        
        # Use comprehensive error handler if available, fallback to default
        try:
            self.error_handler = ComprehensiveErrorHandler(installation_path, self.logger)
            self.user_guidance = UserGuidanceSystem(installation_path, self.logger)
        except Exception:
            self.error_handler = DefaultErrorHandler(self.logger)
            self.user_guidance = None
        
        # Wrap error handler with reliability enhancements
        if self.error_handler:
            self.error_handler = self.reliability_manager.wrap_component(
                self.error_handler, "error_handler", "main_error_handler"
            )
        
        self.state_manager = InstallationStateManager(installation_path)
        
        # Initialize flow controller
        self.flow_controller = InstallationFlowController(
            installation_path, 
            dry_run=getattr(args, 'dry_run', False)
        )
        
        # Wrap flow controller with reliability enhancements
        self.flow_controller = self.reliability_manager.wrap_component(
            self.flow_controller, "installation_flow_controller", "main_flow_controller"
        )
        
        # Add progress callback
        self.flow_controller.add_progress_callback(self._on_progress_update)
        
        # Initialize pre-installation validator
        self.pre_validator = PreInstallationValidator(installation_path)
        
        # Setup logging
        self._setup_logging()
        
        # Installation components (will be initialized as needed)
        self.system_detector = None
        self.dependency_manager = None
        self.model_downloader = None
        self.configuration_engine = None
        self.validator = None
    
    def _setup_logging(self) -> None:
        """Setup comprehensive logging for the installation."""
        log_dir = self.installation_path / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        log_level = logging.DEBUG if self.args.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "installation.log", encoding='utf-8'),
                logging.StreamHandler(sys.stdout) if not self.args.silent else logging.NullHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Starting WAN2.2 installation in {self.installation_path}")
        self.logger.info(f"Installation arguments: {vars(self.args)}")
    
    def run_installation(self) -> bool:
        """
        Run the complete installation process using the integrated installer.
        This method now includes pre-installation validation and reliability monitoring.
        """
        try:
            self.logger.info("Starting WAN2.2 installation with reliability system")
            
            # Run pre-installation validation
            if not self._run_pre_installation_validation():
                self.logger.error("Pre-installation validation failed")
                return False
            
            # Create and run the integrated installer with reliability enhancements
            integrated_installer = IntegratedInstaller(self.installation_path, self.args)
            
            # Wrap the integrated installer with reliability enhancements
            integrated_installer = self.reliability_manager.wrap_component(
                integrated_installer, "integrated_installer", "main_integrated_installer"
            )
            
            # Start reliability monitoring
            self.reliability_manager.start_monitoring()
            
            try:
                success = integrated_installer.run_complete_installation()
                
                if success:
                    self.logger.info("Integrated installation completed successfully")
                    self._generate_reliability_report()
                    return True
                else:
                    self.logger.error("Integrated installation failed")
                    self._generate_failure_report()
                    return False
            finally:
                # Stop reliability monitoring
                self.reliability_manager.stop_monitoring()
                
        except Exception as e:
            self.logger.exception("Unexpected error during integrated installation")
            
            # Use enhanced error context for better diagnostics
            enhanced_context = self.error_handler.create_enhanced_error_context(
                component="main_installer",
                method="run_installation",
                retry_count=0
            )
            
            self.progress_reporter.report_error(
                InstallationError(
                    f"Unexpected error: {str(e)}",
                    ErrorCategory.SYSTEM,
                    ["Check installation logs", "Contact support", "Review system requirements"]
                )
            )
            return False
    
    def _prompt_resume_installation(self) -> bool:
        """Prompt user to resume existing installation."""
        if self.args.silent:
            return True  # Auto-resume in silent mode
        
        response = input("Found incomplete installation. Resume? (y/n): ").lower()
        return response in ['y', 'yes']
    
    def _resume_installation(self, state) -> bool:
        """Resume installation from existing state."""
        self.logger.info(f"Resuming installation from phase: {state.phase.value}")
        # Implementation would continue from the last completed phase
        # For now, just restart the installation
        return self.run_installation()
    
    def _run_detection_phase(self) -> bool:
        """Run system detection phase."""
        self.progress_reporter.update_progress(
            InstallationPhase.DETECTION, 0.0, "Detecting system hardware"
        )
        
        # This will be implemented in task 2
        self.logger.info("System detection phase - placeholder implementation")
        
        self.progress_reporter.update_progress(
            InstallationPhase.DETECTION, 1.0, "System detection completed"
        )
        return True
    
    def _run_dependencies_phase(self) -> bool:
        """Run dependency installation phase."""
        self.progress_reporter.update_progress(
            InstallationPhase.DEPENDENCIES, 0.0, "Installing dependencies"
        )
        
        # This will be implemented in task 3
        self.logger.info("Dependencies phase - placeholder implementation")
        
        self.progress_reporter.update_progress(
            InstallationPhase.DEPENDENCIES, 1.0, "Dependencies installed"
        )
        return True
    
    def _run_models_phase(self) -> bool:
        """Run model download phase."""
        if self.args.skip_models:
            self.logger.info("Skipping model download as requested")
            return True
        
        self.progress_reporter.update_progress(
            InstallationPhase.MODELS, 0.0, "Downloading WAN2.2 models"
        )
        
        # This will be implemented in task 4
        self.logger.info("Models phase - placeholder implementation")
        
        self.progress_reporter.update_progress(
            InstallationPhase.MODELS, 1.0, "Models downloaded"
        )
        return True
    
    def _run_configuration_phase(self) -> bool:
        """Run configuration generation phase."""
        self.progress_reporter.update_progress(
            InstallationPhase.CONFIGURATION, 0.0, "Generating configuration"
        )
        
        # This will be implemented in task 5
        self.logger.info("Configuration phase - placeholder implementation")
        
        self.progress_reporter.update_progress(
            InstallationPhase.CONFIGURATION, 1.0, "Configuration generated"
        )
        return True
    
    def _run_validation_phase(self) -> bool:
        """Run installation validation phase."""
        self.progress_reporter.update_progress(
            InstallationPhase.VALIDATION, 0.0, "Validating installation"
        )
        
        # This will be implemented in task 6
        self.logger.info("Validation phase - placeholder implementation")
        
        self.progress_reporter.update_progress(
            InstallationPhase.VALIDATION, 1.0, "Validation completed"
        )
        return True
    
    def _run_pre_installation_validation(self) -> bool:
        """Run comprehensive pre-installation validation."""
        self.logger.info("Running pre-installation validation")
        
        try:
            # Update progress
            self.progress_reporter.update_progress(
                InstallationPhase.DETECTION, 0.0, "Validating system requirements"
            )
            
            # Run system requirements validation
            system_result = self.pre_validator.validate_system_requirements()
            if not system_result.success:
                self.logger.error(f"System requirements validation failed: {system_result.message}")
                self._handle_validation_failure("System Requirements", system_result)
                return False
            
            # Update progress
            self.progress_reporter.update_progress(
                InstallationPhase.DETECTION, 0.25, "Testing network connectivity"
            )
            
            # Run network connectivity validation
            network_result = self.pre_validator.validate_network_connectivity()
            if not network_result.success:
                self.logger.warning(f"Network validation failed: {network_result.message}")
                # Network issues are warnings, not failures
                self._handle_validation_warning("Network Connectivity", network_result)
            
            # Update progress
            self.progress_reporter.update_progress(
                InstallationPhase.DETECTION, 0.5, "Checking file permissions"
            )
            
            # Run permissions validation
            permissions_result = self.pre_validator.validate_permissions()
            if not permissions_result.success:
                self.logger.error(f"Permissions validation failed: {permissions_result.message}")
                self._handle_validation_failure("File Permissions", permissions_result)
                return False
            
            # Update progress
            self.progress_reporter.update_progress(
                InstallationPhase.DETECTION, 0.75, "Detecting installation conflicts"
            )
            
            # Run existing installation validation
            existing_result = self.pre_validator.validate_existing_installation()
            if not existing_result.success:
                self.logger.warning(f"Existing installation detected: {existing_result.message}")
                if not self._handle_existing_installation(existing_result):
                    return False
            
            # Update progress
            self.progress_reporter.update_progress(
                InstallationPhase.DETECTION, 1.0, "Pre-installation validation completed"
            )
            
            # Generate validation report
            validation_report = self.pre_validator.generate_validation_report()
            self._save_validation_report(validation_report)
            
            self.logger.info("Pre-installation validation completed successfully")
            return True
            
        except Exception as e:
            self.logger.exception("Pre-installation validation failed with exception")
            self.progress_reporter.report_error(
                InstallationError(
                    f"Pre-installation validation error: {str(e)}",
                    ErrorCategory.VALIDATION,
                    ["Check system requirements", "Verify permissions", "Contact support"]
                )
            )
            return False
    
    def _handle_validation_failure(self, validation_type: str, result) -> None:
        """Handle validation failure with user guidance."""
        self.logger.error(f"{validation_type} validation failed")
        
        if self.user_guidance:
            self.user_guidance.display_validation_failure(validation_type, result)
        
        self.progress_reporter.report_error(
            InstallationError(
                f"{validation_type} validation failed: {result.message}",
                ErrorCategory.VALIDATION,
                result.remediation_steps if hasattr(result, 'remediation_steps') else []
            )
        )
    
    def _handle_validation_warning(self, validation_type: str, result) -> None:
        """Handle validation warning with user guidance."""
        self.logger.warning(f"{validation_type} validation warning")
        
        if self.user_guidance:
            self.user_guidance.display_validation_warning(validation_type, result)
    
    def _handle_existing_installation(self, result) -> bool:
        """Handle existing installation detection."""
        if self.args.force_reinstall:
            self.logger.info("Force reinstall requested, proceeding with installation")
            return True
        
        if self.args.silent:
            self.logger.info("Silent mode: skipping existing installation")
            return False
        
        # Prompt user for action
        print(f"\nExisting installation detected: {result.message}")
        while True:
            response = input("Continue with installation? (y/n/f for force): ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            elif response in ['f', 'force']:
                self.args.force_reinstall = True
                return True
            else:
                print("Please enter 'y' for yes, 'n' for no, or 'f' for force reinstall.")
    
    def _save_validation_report(self, report) -> None:
        """Save validation report to file."""
        try:
            report_path = Path(self.installation_path) / "logs" / "pre_validation_report.json"
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Validation report saved to {report_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save validation report: {e}")
    
    def _generate_reliability_report(self) -> None:
        """Generate reliability report after successful installation."""
        try:
            self.logger.info("Generating reliability report")
            
            # Get reliability metrics
            metrics = self.reliability_manager.get_reliability_metrics()
            
            # Save metrics to file
            report_path = Path(self.installation_path) / "logs" / "reliability_report.json"
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            self.logger.info(f"Reliability report saved to {report_path}")
            
            # Log summary
            self.logger.info(f"Installation reliability summary:")
            self.logger.info(f"  Total components: {metrics.get('total_components', 0)}")
            self.logger.info(f"  Healthy components: {metrics.get('healthy_components', 0)}")
            self.logger.info(f"  Success rate: {metrics.get('success_rate', 0):.2%}")
            self.logger.info(f"  Recovery attempts: {metrics.get('recovery_attempts', 0)}")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate reliability report: {e}")
    
    def _generate_failure_report(self) -> None:
        """Generate detailed failure report for troubleshooting."""
        try:
            self.logger.info("Generating failure report")
            
            # Get reliability metrics and component health
            metrics = self.reliability_manager.get_reliability_metrics()
            component_health = self.reliability_manager.get_component_health_summary()
            
            failure_report = {
                "timestamp": datetime.now().isoformat(),
                "installation_path": str(self.installation_path),
                "args": vars(self.args),
                "reliability_metrics": metrics,
                "component_health": component_health,
                "recovery_history": self.reliability_manager.get_recovery_history()
            }
            
            # Save failure report
            report_path = Path(self.installation_path) / "logs" / "failure_report.json"
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(failure_report, f, indent=2, default=str)
            
            self.logger.info(f"Failure report saved to {report_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate failure report: {e}")
    
    def _on_progress_update(self, phase: InstallationPhase, progress: float, task: str) -> None:
        """Handle progress updates from the flow controller."""
        self.progress_reporter.update_progress(phase, progress, task)
        
        # Track progress in reliability system
        self.reliability_manager.track_reliability_metrics(
            "installation_progress", f"{phase.value}_{task}", True, 0.0
        )
    
    def _show_completion_message(self) -> None:
        """Show installation completion message."""
        print("\n" + "="*50)
        print("🎉 WAN2.2 Installation Completed Successfully!")
        print("="*50)
        print("\nNext steps:")
        print("• Desktop shortcuts have been created")
        print("• Start menu entries are available")
        print("• Run first-time setup for configuration")
        print("• Check GETTING_STARTED.md for usage instructions")
        print("\nFor support, see the documentation or check logs/installation.log")
        print("="*50)
        
        # Show installation summary
        if self.args.verbose:
            summary = self.flow_controller.get_installation_summary()
            print(f"\nInstallation Summary:")
            print(f"• Installation path: {summary['installation_path']}")
            print(f"• Snapshots created: {summary['snapshots_count']}")
            print(f"• Log files: {len(summary['log_files'])}")
            if summary['current_state']:
                state = summary['current_state']
                print(f"• Final progress: {state['progress']:.1%}")
                if state['warnings_count'] > 0:
                    print(f"• Warnings: {state['warnings_count']}")
        
        # Offer post-installation setup
        self._offer_post_installation_setup()
    
    def _offer_post_installation_setup(self) -> None:
        """Offer to run post-installation setup."""
        if self.args.silent:
            return  # Skip in silent mode
        
        print("\n" + "-"*50)
        print("Post-Installation Setup")
        print("-"*50)
        
        while True:
            response = input("Would you like to run the first-time setup wizard now? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                self._run_post_installation_setup()
                break
            elif response in ['n', 'no']:
                print("\nYou can run the setup wizard later by:")
                print("• Double-clicking 'run_first_setup.bat'")
                print("• Using Start Menu → WAN2.2 → WAN2.2 Configuration")
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")
    
    def _run_post_installation_setup(self) -> None:
        """Run the post-installation setup wizard."""
        try:
            from post_install_setup import PostInstallationSetup
            
            setup = PostInstallationSetup(self.installation_path)
            success = setup.run_complete_post_install_setup()
            
            if success:
                print("\n✓ Post-installation setup completed successfully!")
            else:
                print("\n✗ Post-installation setup encountered some issues.")
                print("You can run it again later using 'run_first_setup.bat'")
                
        except Exception as e:
            self.logger.error(f"Post-installation setup failed: {e}")
            print(f"\n✗ Post-installation setup failed: {e}")
            print("You can run it manually later using 'run_first_setup.bat'")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="WAN2.2 Local Installation System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--silent", 
        action="store_true",
        help="Run installation in silent mode"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dev-mode", 
        action="store_true",
        help="Install development dependencies"
    )
    
    parser.add_argument(
        "--skip-models", 
        action="store_true",
        help="Skip model download (for testing)"
    )
    
    parser.add_argument(
        "--custom-path", 
        type=str,
        help="Specify custom installation path"
    )
    
    parser.add_argument(
        "--force-reinstall", 
        action="store_true",
        help="Force complete reinstallation"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Simulate installation without making changes"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the installer."""
    try:
        args = parse_arguments()
        
        # Determine installation path
        if args.custom_path:
            installation_path = Path(args.custom_path).resolve()
        else:
            installation_path = Path(__file__).parent.parent.resolve()
        
        # Create and run installer
        installer = MainInstaller(str(installation_path), args)
        success = installer.run_installation()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
