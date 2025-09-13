"""
Example usage of the WAN2.2 dependency management system.
Demonstrates how to use the dependency manager in a real installation scenario.
"""

import sys
import logging
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from scripts.setup_dependencies import DependencyManager
from scripts.detect_system import SystemDetector
from scripts.base_classes import ConsoleProgressReporter, DefaultErrorHandler
from scripts.interfaces import InstallationPhase


def setup_logging():
    """Set up logging for the installation process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dependency_installation.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Example of complete dependency management workflow."""
    print("WAN2.2 Dependency Management Example")
    print("=" * 50)
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Set installation path
    installation_path = Path(__file__).parent.parent
    
    try:
        # Step 1: Initialize components
        progress_reporter = ConsoleProgressReporter()
        error_handler = DefaultErrorHandler(logger)
        
        progress_reporter.update_progress(
            InstallationPhase.DETECTION, 0.0, 
            "Initializing dependency management system"
        )
        
        # Step 2: Detect system hardware
        system_detector = SystemDetector(str(installation_path))
        hardware_profile = system_detector.detect_hardware()
        
        logger.info(f"Detected hardware: {hardware_profile.cpu.model}, "
                   f"{hardware_profile.memory.total_gb}GB RAM")
        
        if hardware_profile.gpu:
            logger.info(f"GPU: {hardware_profile.gpu.model}, "
                       f"{hardware_profile.gpu.vram_gb}GB VRAM")
        
        progress_reporter.update_progress(
            InstallationPhase.DETECTION, 1.0, 
            "Hardware detection complete"
        )
        
        # Step 3: Initialize dependency manager
        dep_manager = DependencyManager(str(installation_path), progress_reporter)
        
        progress_reporter.update_progress(
            InstallationPhase.DEPENDENCIES, 0.0, 
            "Starting dependency installation"
        )
        
        # Step 4: Check Python installation
        python_info = dep_manager.check_python_installation()
        logger.info(f"Python check result: {python_info['recommended_action']}")
        
        # Step 5: Create virtual environment
        venv_path = installation_path / "venv"
        logger.info(f"Creating virtual environment at {venv_path}")
        
        venv_success = dep_manager.create_virtual_environment(
            str(venv_path), hardware_profile
        )
        
        if not venv_success:
            raise Exception("Failed to create virtual environment")
        
        progress_reporter.update_progress(
            InstallationPhase.DEPENDENCIES, 0.5, 
            "Virtual environment created"
        )
        
        # Step 6: Install packages (commented out to avoid long installation time)
        # In a real scenario, uncomment this:
        """
        requirements_file = installation_path / "resources" / "requirements.txt"
        if requirements_file.exists():
            logger.info("Installing packages...")
            package_success = dep_manager.install_packages(
                str(requirements_file), hardware_profile
            )
            
            if not package_success:
                raise Exception("Failed to install packages")
        """
        
        progress_reporter.update_progress(
            InstallationPhase.DEPENDENCIES, 0.9, 
            "Package installation complete (skipped in example)"
        )
        
        # Step 7: Validate installation
        validation_result = dep_manager.validate_installation()
        
        if validation_result.success:
            progress_reporter.report_success(
                "Dependency management completed successfully!"
            )
            logger.info(validation_result.message)
            
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    progress_reporter.report_warning(warning)
        else:
            progress_reporter.report_error(
                Exception(validation_result.message)
            )
            return 1
        
        progress_reporter.update_progress(
            InstallationPhase.DEPENDENCIES, 1.0, 
            "Dependency management complete"
        )
        
        # Step 8: Display installation summary
        print("\n" + "=" * 50)
        print("INSTALLATION SUMMARY")
        print("=" * 50)
        
        print(f"Installation Path: {installation_path}")
        print(f"Virtual Environment: {venv_path}")
        print(f"Python: {python_info.get('system_python', {}).get('version', 'Unknown')}")
        print(f"Hardware Profile: {hardware_profile.cpu.model}")
        
        if hardware_profile.gpu:
            print(f"GPU Acceleration: {hardware_profile.gpu.model}")
        else:
            print("GPU Acceleration: Not available")
        
        print("\n✅ Dependency management example completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Dependency management failed: {e}")
        progress_reporter.report_error(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
