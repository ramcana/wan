"""
Dependency Recovery System for WAN2.2 Installation.

This module provides automatic recovery mechanisms for dependency installation failures,
including virtual environment recreation, alternative package sources, version fallback
strategies, and offline package installation capabilities.
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
import platform
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from packaging import version
from packaging.requirements import Requirement
import logging
from datetime import datetime

from interfaces import (
    InstallationError, ErrorCategory, HardwareProfile, ValidationResult
)
from base_classes import BaseInstallationComponent
from setup_dependencies import DependencyManager, PythonInstallationHandler
from package_resolver import PackageInstallationOrchestrator


@dataclass
class RecoveryStrategy:
    """Represents a recovery strategy for dependency failures."""
    name: str
    description: str
    priority: int  # Lower number = higher priority
    applicable_errors: List[str]  # Error patterns this strategy can handle
    success_rate: float  # Historical success rate (0.0 to 1.0)


@dataclass
class PackageSource:
    """Alternative package source configuration."""
    name: str
    index_url: str
    trusted_host: Optional[str]
    description: str
    reliability_score: float  # 0.0 to 1.0


@dataclass
class VersionFallback:
    """Version fallback configuration for packages."""
    package_name: str
    preferred_version: str
    fallback_versions: List[str]
    compatibility_notes: str


class DependencyRecovery(BaseInstallationComponent):
    """
    Handles automatic recovery for dependency installation failures.
    
    This class implements multiple recovery strategies:
    1. Virtual environment recreation
    2. Alternative package source selection
    3. Version fallback strategies
    4. Offline package installation
    """
    
    # Alternative PyPI mirrors and sources
    ALTERNATIVE_SOURCES = [
        PackageSource(
            name="PyPI Official",
            index_url="https://pypi.org/simple/",
            trusted_host="pypi.org",
            description="Official Python Package Index",
            reliability_score=0.95
        ),
        PackageSource(
            name="Alibaba Cloud Mirror",
            index_url="https://mirrors.aliyun.com/pypi/simple/",
            trusted_host="mirrors.aliyun.com",
            description="Alibaba Cloud PyPI Mirror (China)",
            reliability_score=0.90
        ),
        PackageSource(
            name="Tsinghua Mirror",
            index_url="https://pypi.tuna.tsinghua.edu.cn/simple/",
            trusted_host="pypi.tuna.tsinghua.edu.cn",
            description="Tsinghua University PyPI Mirror",
            reliability_score=0.88
        ),
        PackageSource(
            name="Douban Mirror",
            index_url="https://pypi.douban.com/simple/",
            trusted_host="pypi.douban.com",
            description="Douban PyPI Mirror",
            reliability_score=0.85
        ),
        PackageSource(
            name="Microsoft Mirror",
            index_url="https://pkgs.dev.azure.com/dnceng/public/_packaging/dotnet-public-pypi/pypi/simple/",
            trusted_host="pkgs.dev.azure.com",
            description="Microsoft Azure PyPI Mirror",
            reliability_score=0.82
        )
    ]
    
    # Version fallback configurations for critical packages
    VERSION_FALLBACKS = {
        "torch": VersionFallback(
            package_name="torch",
            preferred_version="2.1.0",
            fallback_versions=["2.0.1", "2.0.0", "1.13.1", "1.12.1"],
            compatibility_notes="Older versions may not support latest CUDA"
        ),
        "transformers": VersionFallback(
            package_name="transformers",
            preferred_version="4.35.0",
            fallback_versions=["4.30.0", "4.25.0", "4.20.0"],
            compatibility_notes="Older versions may lack latest model support"
        ),
        "diffusers": VersionFallback(
            package_name="diffusers",
            preferred_version="0.21.0",
            fallback_versions=["0.20.0", "0.19.0", "0.18.0"],
            compatibility_notes="API changes between major versions"
        ),
        "numpy": VersionFallback(
            package_name="numpy",
            preferred_version="1.24.3",
            fallback_versions=["1.23.5", "1.22.4", "1.21.6"],
            compatibility_notes="Compatibility with Python version dependent"
        )
    }
    
    # Recovery strategies in order of preference
    RECOVERY_STRATEGIES = [
        RecoveryStrategy(
            name="retry_with_cache_clear",
            description="Clear pip cache and retry installation",
            priority=1,
            applicable_errors=["cache", "corrupt", "checksum"],
            success_rate=0.75
        ),
        RecoveryStrategy(
            name="alternative_source",
            description="Try alternative package sources/mirrors",
            priority=2,
            applicable_errors=["network", "timeout", "connection", "ssl"],
            success_rate=0.70
        ),
        RecoveryStrategy(
            name="version_fallback",
            description="Use fallback package versions",
            priority=3,
            applicable_errors=["version", "compatibility", "dependency"],
            success_rate=0.65
        ),
        RecoveryStrategy(
            name="venv_recreation",
            description="Recreate virtual environment",
            priority=4,
            applicable_errors=["environment", "permission", "corrupt"],
            success_rate=0.60
        ),
        RecoveryStrategy(
            name="offline_installation",
            description="Use offline package installation",
            priority=5,
            applicable_errors=["network", "firewall", "proxy"],
            success_rate=0.50
        )
    ]
    
    def __init__(self, installation_path: str, dependency_manager: Optional[DependencyManager] = None):
        super().__init__(installation_path)
        self.dependency_manager = dependency_manager or DependencyManager(installation_path)
        self.python_handler = self.dependency_manager.python_handler
        self.package_cache_dir = self.installation_path / "package_cache"
        self.offline_packages_dir = self.installation_path / "offline_packages"
        self.recovery_log = []
        
        # Ensure cache directories exist
        self.ensure_directory(self.package_cache_dir)
        self.ensure_directory(self.offline_packages_dir)
    
    def recover_dependency_failure(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        Main entry point for dependency failure recovery.
        
        Args:
            error: The exception that occurred during dependency installation
            context: Additional context about the failure
            
        Returns:
            bool: True if recovery was successful, False otherwise
        """
        self.logger.info(f"Starting dependency recovery for error: {str(error)}")
        
        # Analyze the error to determine applicable recovery strategies
        applicable_strategies = self._analyze_error_for_strategies(error, context)
        
        if not applicable_strategies:
            self.logger.warning("No applicable recovery strategies found")
            return False
        
        # Sort strategies by priority and success rate
        applicable_strategies.sort(key=lambda s: (s.priority, -s.success_rate))
        
        # Try each strategy until one succeeds
        for strategy in applicable_strategies:
            self.logger.info(f"Attempting recovery strategy: {strategy.name}")
            
            try:
                success = self._execute_recovery_strategy(strategy, error, context)
                
                if success:
                    self.logger.info(f"Recovery successful using strategy: {strategy.name}")
                    self._log_recovery_success(strategy, error, context)
                    return True
                else:
                    self.logger.warning(f"Recovery strategy failed: {strategy.name}")
                    
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy {strategy.name} raised exception: {recovery_error}")
                continue
        
        self.logger.error("All recovery strategies failed")
        return False
    
    def recreate_virtual_environment(self, venv_path: Optional[str] = None, 
                                   hardware_profile: Optional[HardwareProfile] = None) -> bool:
        """
        Recreate virtual environment when create_optimized_virtual_environment fails.
        
        Args:
            venv_path: Path to virtual environment (uses default if None)
            hardware_profile: Hardware profile for optimization
            
        Returns:
            bool: True if recreation was successful
        """
        self.logger.info("Recreating virtual environment...")
        
        try:
            # Determine virtual environment path
            if venv_path:
                venv_dir = Path(venv_path)
            else:
                venv_dir = self.installation_path / "venv"
            
            # Remove existing virtual environment if it exists
            if venv_dir.exists():
                self.logger.info(f"Removing existing virtual environment: {venv_dir}")
                shutil.rmtree(venv_dir, ignore_errors=True)
                
                # Wait a moment for filesystem to catch up
                import time
                time.sleep(1)
            
            # Create new virtual environment with multiple fallback methods
            success = self._create_venv_with_fallbacks(venv_dir, hardware_profile)
            
            if success:
                self.logger.info("Virtual environment recreated successfully")
                return True
            else:
                self.logger.error("Failed to recreate virtual environment")
                return False
                
        except Exception as e:
            self.logger.error(f"Virtual environment recreation failed: {e}")
            return False
    
    def install_with_alternative_sources(self, requirements: List[str], 
                                       max_attempts: int = 3) -> bool:
        """
        Install packages using alternative PyPI mirrors.
        
        Args:
            requirements: List of package requirements
            max_attempts: Maximum number of source attempts
            
        Returns:
            bool: True if installation was successful
        """
        self.logger.info("Attempting installation with alternative sources...")
        
        # Sort sources by reliability score
        sources = sorted(self.ALTERNATIVE_SOURCES, key=lambda s: -s.reliability_score)
        
        for attempt, source in enumerate(sources[:max_attempts], 1):
            self.logger.info(f"Attempt {attempt}: Using {source.name}")
            
            try:
                success = self._install_with_source(requirements, source)
                
                if success:
                    self.logger.info(f"Installation successful with {source.name}")
                    return True
                else:
                    self.logger.warning(f"Installation failed with {source.name}")
                    
            except Exception as e:
                self.logger.warning(f"Source {source.name} failed: {e}")
                continue
        
        self.logger.error("All alternative sources failed")
        return False
    
    def apply_version_fallbacks(self, failed_packages: List[str]) -> bool:
        """
        Apply version fallback strategies for incompatible packages.
        
        Args:
            failed_packages: List of packages that failed to install
            
        Returns:
            bool: True if fallback installation was successful
        """
        self.logger.info(f"Applying version fallbacks for packages: {failed_packages}")
        
        fallback_requirements = []
        
        for package in failed_packages:
            if package in self.VERSION_FALLBACKS:
                fallback = self.VERSION_FALLBACKS[package]
                self.logger.info(f"Using fallback versions for {package}: {fallback.fallback_versions}")
                
                # Try each fallback version
                for fallback_version in fallback.fallback_versions:
                    fallback_requirements.append(f"{package}=={fallback_version}")
                    break  # Use first fallback for now
            else:
                # For packages without specific fallbacks, try without version constraint
                fallback_requirements.append(package)
        
        if not fallback_requirements:
            self.logger.warning("No fallback versions available")
            return False
        
        try:
            # Install with fallback versions
            venv_python = self.python_handler.get_venv_python_executable()
            cmd = [venv_python, "-m", "pip", "install"] + fallback_requirements
            cmd.extend(["--no-cache-dir", "--timeout", "300"])
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info("Version fallback installation successful")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Version fallback installation failed: {e.stderr}")
            return False
    
    def setup_offline_installation(self, requirements: List[str], 
                                 offline_dir: Optional[str] = None) -> bool:
        """
        Set up offline package installation capabilities.
        
        Args:
            requirements: List of package requirements
            offline_dir: Directory to store offline packages
            
        Returns:
            bool: True if offline setup was successful
        """
        if offline_dir:
            offline_path = Path(offline_dir)
        else:
            offline_path = self.offline_packages_dir
        
        self.logger.info(f"Setting up offline installation in: {offline_path}")
        
        try:
            # Ensure offline directory exists
            self.ensure_directory(offline_path)
            
            # Download packages for offline installation
            venv_python = self.python_handler.get_venv_python_executable()
            
            # Download packages without installing
            cmd = [venv_python, "-m", "pip", "download"]
            cmd.extend(requirements)
            cmd.extend(["--dest", str(offline_path)])
            cmd.extend(["--no-cache-dir"])
            
            self.logger.info("Downloading packages for offline installation...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Create requirements file for offline installation
            offline_requirements = offline_path / "requirements.txt"
            with open(offline_requirements, 'w') as f:
                for req in requirements:
                    f.write(f"{req}\n")
            
            self.logger.info("Offline installation setup complete")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Offline setup failed: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Offline setup error: {e}")
            return False
    
    def install_from_offline_cache(self, offline_dir: Optional[str] = None) -> bool:
        """
        Install packages from offline cache.
        
        Args:
            offline_dir: Directory containing offline packages
            
        Returns:
            bool: True if offline installation was successful
        """
        if offline_dir:
            offline_path = Path(offline_dir)
        else:
            offline_path = self.offline_packages_dir
        
        self.logger.info(f"Installing from offline cache: {offline_path}")
        
        if not offline_path.exists():
            self.logger.error("Offline cache directory does not exist")
            return False
        
        try:
            venv_python = self.python_handler.get_venv_python_executable()
            
            # Install from offline directory
            cmd = [venv_python, "-m", "pip", "install"]
            cmd.extend(["--find-links", str(offline_path)])
            cmd.extend(["--no-index"])  # Don't use PyPI
            
            # Add requirements if available
            requirements_file = offline_path / "requirements.txt"
            if requirements_file.exists():
                cmd.extend(["-r", str(requirements_file)])
            else:
                # Install all wheel files in directory
                wheel_files = list(offline_path.glob("*.whl"))
                tar_files = list(offline_path.glob("*.tar.gz"))
                all_packages = wheel_files + tar_files
                
                if all_packages:
                    cmd.extend([str(pkg) for pkg in all_packages])
                else:
                    self.logger.error("No packages found in offline cache")
                    return False
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info("Offline installation successful")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Offline installation failed: {e.stderr}")
            return False
    
    def _analyze_error_for_strategies(self, error: Exception, context: Dict[str, Any]) -> List[RecoveryStrategy]:
        """Analyze error to determine applicable recovery strategies."""
        error_str = str(error).lower()
        applicable_strategies = []
        
        for strategy in self.RECOVERY_STRATEGIES:
            for error_pattern in strategy.applicable_errors:
                if error_pattern in error_str:
                    applicable_strategies.append(strategy)
                    break
        
        # If no specific strategies match, try general strategies
        if not applicable_strategies:
            # Add cache clear and alternative source as general fallbacks
            applicable_strategies.extend([
                s for s in self.RECOVERY_STRATEGIES 
                if s.name in ["retry_with_cache_clear", "alternative_source"]
            ])
        
        return applicable_strategies
    
    def _execute_recovery_strategy(self, strategy: RecoveryStrategy, 
                                 error: Exception, context: Dict[str, Any]) -> bool:
        """Execute a specific recovery strategy."""
        if strategy.name == "retry_with_cache_clear":
            return self._retry_with_cache_clear(context)
        elif strategy.name == "alternative_source":
            return self._try_alternative_sources(context)
        elif strategy.name == "version_fallback":
            return self._apply_version_fallbacks_strategy(context)
        elif strategy.name == "venv_recreation":
            return self._recreate_venv_strategy(context)
        elif strategy.name == "offline_installation":
            return self._try_offline_installation(context)
        else:
            self.logger.warning(f"Unknown recovery strategy: {strategy.name}")
            return False
    
    def _retry_with_cache_clear(self, context: Dict[str, Any]) -> bool:
        """Clear pip cache and retry installation."""
        try:
            venv_python = self.python_handler.get_venv_python_executable()
            
            # Clear pip cache
            subprocess.run([venv_python, "-m", "pip", "cache", "purge"], 
                         check=True, capture_output=True)
            
            # Retry original installation
            requirements = context.get("requirements", [])
            if requirements:
                cmd = [venv_python, "-m", "pip", "install"] + requirements
                cmd.extend(["--no-cache-dir", "--force-reinstall"])
                subprocess.run(cmd, check=True, capture_output=True)
                return True
            
            return False
            
        except subprocess.CalledProcessError:
            return False
    
    def _try_alternative_sources(self, context: Dict[str, Any]) -> bool:
        """Try installation with alternative sources."""
        requirements = context.get("requirements", [])
        if not requirements:
            return False
        
        return self.install_with_alternative_sources(requirements, max_attempts=2)
    
    def _apply_version_fallbacks_strategy(self, context: Dict[str, Any]) -> bool:
        """Apply version fallback strategy."""
        failed_packages = context.get("failed_packages", [])
        if not failed_packages:
            # Extract package names from requirements
            requirements = context.get("requirements", [])
            failed_packages = [req.split("==")[0].split(">=")[0].split("<=")[0] 
                             for req in requirements]
        
        return self.apply_version_fallbacks(failed_packages)
    
    def _recreate_venv_strategy(self, context: Dict[str, Any]) -> bool:
        """Recreate virtual environment strategy."""
        hardware_profile = context.get("hardware_profile")
        return self.recreate_virtual_environment(hardware_profile=hardware_profile)
    
    def _try_offline_installation(self, context: Dict[str, Any]) -> bool:
        """Try offline installation strategy."""
        requirements = context.get("requirements", [])
        if not requirements:
            return False
        
        # Try to set up offline installation first
        try:
            if self.setup_offline_installation(requirements):
                return self.install_from_offline_cache()
        except Exception as e:
            self.logger.warning(f"Offline installation setup failed: {e}")
        
        # If setup fails, try to install from existing offline cache
        try:
            return self.install_from_offline_cache()
        except Exception as e:
            self.logger.warning(f"Offline cache installation failed: {e}")
        
        return False
    
    def _create_venv_with_fallbacks(self, venv_dir: Path, 
                                  hardware_profile: Optional[HardwareProfile]) -> bool:
        """Create virtual environment with multiple fallback methods."""
        
        # Method 1: Use existing python handler
        try:
            result = self.python_handler.create_virtual_environment(
                str(venv_dir), hardware_profile
            )
            if result:
                return True
        except Exception as e:
            self.logger.warning(f"Python handler venv creation failed: {e}")
        
        # Method 2: Direct venv module
        try:
            python_exe = self.python_handler.get_python_executable()
            subprocess.run([python_exe, "-m", "venv", str(venv_dir)], 
                         check=True, capture_output=True)
            self.logger.info("Virtual environment created using direct venv module")
            return True
        except Exception as e:
            self.logger.warning(f"Direct venv creation failed: {e}")
        
        # Method 3: virtualenv package (if available)
        try:
            python_exe = self.python_handler.get_python_executable()
            subprocess.run([python_exe, "-m", "virtualenv", str(venv_dir)], 
                         check=True, capture_output=True)
            self.logger.info("Virtual environment created using virtualenv package")
            return True
        except Exception as e:
            self.logger.warning(f"Virtualenv creation failed: {e}")
        
        # Method 4: System Python fallback
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], 
                         check=True, capture_output=True)
            self.logger.info("Virtual environment created using system Python")
            return True
        except Exception as e:
            self.logger.warning(f"System Python venv creation failed: {e}")
        
        return False
    
    def _install_with_source(self, requirements: List[str], source: PackageSource) -> bool:
        """Install packages using a specific source."""
        try:
            venv_python = self.python_handler.get_venv_python_executable()
            
            cmd = [venv_python, "-m", "pip", "install"]
            cmd.extend(["--index-url", source.index_url])
            
            if source.trusted_host:
                cmd.extend(["--trusted-host", source.trusted_host])
            
            cmd.extend(requirements)
            cmd.extend(["--no-cache-dir", "--timeout", "300"])
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Installation with {source.name} failed: {e.stderr}")
            return False
    
    def _log_recovery_success(self, strategy: RecoveryStrategy, 
                            error: Exception, context: Dict[str, Any]) -> None:
        """Log successful recovery for analytics."""
        recovery_record = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy.name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }
        
        self.recovery_log.append(recovery_record)
        
        # Save to file for persistence
        log_file = self.installation_path / "recovery_log.json"
        try:
            with open(log_file, 'w') as f:
                json.dump(self.recovery_log, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save recovery log: {e}")
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get statistics about recovery attempts and success rates."""
        if not self.recovery_log:
            return {"total_recoveries": 0, "strategies": {}}
        
        stats = {
            "total_recoveries": len(self.recovery_log),
            "strategies": {}
        }
        
        for record in self.recovery_log:
            strategy = record["strategy"]
            if strategy not in stats["strategies"]:
                stats["strategies"][strategy] = {
                    "attempts": 0,
                    "success_rate": 0.0
                }
            stats["strategies"][strategy]["attempts"] += 1
        
        return stats
