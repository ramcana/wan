"""
Environment and system capability detection module.
Extends hardware detection with detailed OS analysis, environment validation,
and performance tier classification.
"""

import os
import platform
import subprocess
import json
import re
import logging
import winreg
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass

from interfaces import (
    HardwareProfile, ValidationResult, InstallationError, ErrorCategory
)
from base_classes import BaseInstallationComponent


@dataclass
class EnvironmentInfo:
    """Extended environment information."""
    windows_version: str
    windows_build: str
    windows_edition: str
    architecture: str
    system_locale: str
    timezone: str
    user_privileges: str
    python_installations: List[Dict[str, str]]
    environment_variables: Dict[str, str]
    installed_software: List[str]
    system_capabilities: Dict[str, bool]


@dataclass
class SystemCapabilities:
    """System capability flags."""
    has_admin_rights: bool
    has_internet_access: bool
    has_sufficient_disk_space: bool
    has_compatible_python: bool
    has_visual_cpp_redist: bool
    has_net_framework: bool
    has_windows_sdk: bool
    supports_long_paths: bool
    has_hyper_v: bool
    has_wsl: bool


class EnvironmentDetector(BaseInstallationComponent):
    """Advanced environment and capability detection."""
    
    def __init__(self, installation_path: str, logger: Optional[logging.Logger] = None):
        super().__init__(installation_path, logger)
        self.minimum_windows_build = 19041  # Windows 10 version 2004
        self.supported_python_versions = ["3.8", "3.9", "3.10", "3.11", "3.12"]
    
    def detect_environment(self) -> EnvironmentInfo:
        """Detect comprehensive environment information."""
        self.logger.info("Starting environment detection...")
        
        try:
            windows_info = self._detect_windows_details()
            architecture = self._detect_architecture()
            locale_info = self._detect_locale_timezone()
            privileges = self._detect_user_privileges()
            python_installs = self._detect_python_installations()
            env_vars = self._get_relevant_environment_variables()
            software = self._detect_installed_software()
            capabilities = self._detect_system_capabilities()
            
            env_info = EnvironmentInfo(
                windows_version=windows_info["version"],
                windows_build=windows_info["build"],
                windows_edition=windows_info["edition"],
                architecture=architecture,
                system_locale=locale_info["locale"],
                timezone=locale_info["timezone"],
                user_privileges=privileges,
                python_installations=python_installs,
                environment_variables=env_vars,
                installed_software=software,
                system_capabilities=capabilities
            )
            
            self.logger.info("Environment detection completed successfully")
            self._log_environment_summary(env_info)
            return env_info
            
        except Exception as e:
            self.logger.error(f"Environment detection failed: {str(e)}")
            raise InstallationError(
                f"Failed to detect system environment: {str(e)}",
                ErrorCategory.SYSTEM,
                ["Check system permissions", "Ensure registry access", "Try running as administrator"]
            )
    
    def _detect_windows_details(self) -> Dict[str, str]:
        """Detect detailed Windows version information."""
        self.logger.debug("Detecting Windows version details...")
        
        try:
            # Method 1: Try registry
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                  r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
                    product_name = winreg.QueryValueEx(key, "ProductName")[0]
                    current_build = winreg.QueryValueEx(key, "CurrentBuild")[0]
                    edition_id = winreg.QueryValueEx(key, "EditionID")[0]
                    
                    return {
                        "version": product_name,
                        "build": current_build,
                        "edition": edition_id
                    }
            except Exception as e:
                self.logger.debug(f"Registry Windows detection failed: {e}")
            
            # Method 2: Try WMI
            cmd = ['wmic', 'os', 'get', 'Caption,BuildNumber,OperatingSystemSKU', '/format:csv']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    data = lines[1].split(',')
                    if len(data) >= 4:
                        return {
                            "version": data[2].strip(),
                            "build": data[1].strip(),
                            "edition": self._get_windows_edition_name(data[3].strip())
                        }
            
            # Fallback
            return {
                "version": platform.platform(),
                "build": platform.version(),
                "edition": "Unknown"
            }
            
        except Exception as e:
            self.logger.warning(f"Windows details detection failed: {e}")
            return {
                "version": "Windows (Unknown)",
                "build": "Unknown",
                "edition": "Unknown"
            }
    
    def _get_windows_edition_name(self, sku: str) -> str:
        """Convert Windows SKU to edition name."""
        sku_map = {
            "1": "Ultimate",
            "4": "Home Basic",
            "48": "Professional",
            "49": "Professional N",
            "101": "Home",
            "100": "Home N",
            "161": "Pro for Workstations",
            "162": "Pro for Workstations N"
        }
        return sku_map.get(sku, f"Edition {sku}")
    
    def _detect_architecture(self) -> str:
        """Detect system architecture with detailed information."""
        try:
            # Get processor architecture
            arch = platform.machine().lower()
            
            # Check for ARM64 emulation
            if "PROCESSOR_ARCHITEW6432" in os.environ:
                native_arch = os.environ["PROCESSOR_ARCHITEW6432"].lower()
                if native_arch == "arm64":
                    return "ARM64"
            
            # Standard architecture mapping
            if arch in ['amd64', 'x86_64']:
                return "x64"
            elif arch in ['i386', 'i686']:
                return "x86"
            elif arch in ['arm64', 'aarch64']:
                return "ARM64"
            else:
                return arch.upper()
                
        except Exception as e:
            self.logger.debug(f"Architecture detection failed: {e}")
            return "x64"  # Default assumption
    
    def _detect_locale_timezone(self) -> Dict[str, str]:
        """Detect system locale and timezone."""
        try:
            # Get locale
            import locale
            system_locale = locale.getdefaultlocale()[0] or "en_US"
            
            # Get timezone
            try:
                cmd = ['powershell', '-Command', 'Get-TimeZone | Select-Object -ExpandProperty Id']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                timezone = result.stdout.strip() if result.returncode == 0 else "Unknown"
            except:
                timezone = "Unknown"
            
            return {
                "locale": system_locale,
                "timezone": timezone
            }
        except Exception as e:
            self.logger.debug(f"Locale/timezone detection failed: {e}")
            return {"locale": "en_US", "timezone": "Unknown"}
    
    def _detect_user_privileges(self) -> str:
        """Detect current user privilege level."""
        try:
            import ctypes
            if ctypes.windll.shell32.IsUserAnAdmin():
                return "Administrator"
            else:
                return "Standard User"
        except Exception as e:
            self.logger.debug(f"Privilege detection failed: {e}")
            return "Unknown"
    
    def _detect_python_installations(self) -> List[Dict[str, str]]:
        """Detect all Python installations on the system."""
        self.logger.debug("Detecting Python installations...")
        
        installations = []
        
        try:
            # Method 1: Check common installation paths
            common_paths = [
                r"C:\Python*",
                r"C:\Program Files\Python*",
                r"C:\Program Files (x86)\Python*",
                os.path.expanduser(r"~\AppData\Local\Programs\Python\Python*"),
                os.path.expanduser(r"~\AppData\Local\Microsoft\WindowsApps\python*.exe")
            ]
            
            import glob
            for path_pattern in common_paths:
                for path in glob.glob(path_pattern):
                    python_exe = os.path.join(path, "python.exe")
                    if os.path.exists(python_exe):
                        version = self._get_python_version(python_exe)
                        if version:
                            installations.append({
                                "path": python_exe,
                                "version": version,
                                "source": "filesystem"
                            })
            
            # Method 2: Check registry
            try:
                registry_pythons = self._get_python_from_registry()
                installations.extend(registry_pythons)
            except Exception as e:
                self.logger.debug(f"Registry Python detection failed: {e}")
            
            # Method 3: Check PATH
            try:
                result = subprocess.run(['where', 'python'], capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        python_path = line.strip()
                        if python_path and os.path.exists(python_path):
                            version = self._get_python_version(python_path)
                            if version:
                                # Check if already found
                                if not any(inst["path"] == python_path for inst in installations):
                                    installations.append({
                                        "path": python_path,
                                        "version": version,
                                        "source": "PATH"
                                    })
            except Exception as e:
                self.logger.debug(f"PATH Python detection failed: {e}")
            
            # Remove duplicates and sort by version
            unique_installations = []
            seen_paths = set()
            
            for install in installations:
                if install["path"] not in seen_paths:
                    seen_paths.add(install["path"])
                    unique_installations.append(install)
            
            # Sort by version (newest first)
            unique_installations.sort(key=lambda x: x["version"], reverse=True)
            
            self.logger.debug(f"Found {len(unique_installations)} Python installations")
            return unique_installations
            
        except Exception as e:
            self.logger.warning(f"Python detection failed: {e}")
            return []
    
    def _get_python_version(self, python_exe: str) -> Optional[str]:
        """Get Python version from executable."""
        try:
            result = subprocess.run([python_exe, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version_match = re.search(r'Python (\d+\.\d+\.\d+)', result.stdout)
                if version_match:
                    return version_match.group(1)
        except Exception:
            pass
        return None
    
    def _get_python_from_registry(self) -> List[Dict[str, str]]:
        """Get Python installations from Windows registry."""
        installations = []
        
        try:
            # Check both 32-bit and 64-bit registry views
            registry_roots = [
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Python\PythonCore"),
                (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Python\PythonCore")
            ]
            
            for root, subkey in registry_roots:
                try:
                    with winreg.OpenKey(root, subkey) as python_key:
                        i = 0
                        while True:
                            try:
                                version = winreg.EnumKey(python_key, i)
                                try:
                                    with winreg.OpenKey(python_key, f"{version}\\InstallPath") as install_key:
                                        install_path = winreg.QueryValue(install_key, "")
                                        python_exe = os.path.join(install_path, "python.exe")
                                        
                                        if os.path.exists(python_exe):
                                            installations.append({
                                                "path": python_exe,
                                                "version": version,
                                                "source": "registry"
                                            })
                                except FileNotFoundError:
                                    pass
                                i += 1
                            except OSError:
                                break
                except FileNotFoundError:
                    pass
        except Exception as e:
            self.logger.debug(f"Registry Python enumeration failed: {e}")
        
        return installations  
  
    def _get_relevant_environment_variables(self) -> Dict[str, str]:
        """Get relevant environment variables for installation."""
        relevant_vars = [
            "PATH", "PYTHONPATH", "CUDA_PATH", "CUDA_HOME", 
            "PROCESSOR_ARCHITECTURE", "NUMBER_OF_PROCESSORS",
            "TEMP", "TMP", "USERPROFILE", "PROGRAMFILES",
            "PROGRAMFILES(X86)", "LOCALAPPDATA", "APPDATA"
        ]
        
        env_vars = {}
        for var in relevant_vars:
            value = os.environ.get(var)
            if value:
                env_vars[var] = value
        
        return env_vars
    
    def _detect_installed_software(self) -> List[str]:
        """Detect relevant installed software."""
        self.logger.debug("Detecting installed software...")
        
        software_list = []
        
        try:
            # Check for Visual C++ Redistributables
            if self._check_visual_cpp_redist():
                software_list.append("Visual C++ Redistributable")
            
            # Check for .NET Framework
            net_versions = self._check_net_framework()
            if net_versions:
                software_list.extend([f".NET Framework {v}" for v in net_versions])
            
            # Check for CUDA Toolkit
            cuda_version = self._check_cuda_toolkit()
            if cuda_version:
                software_list.append(f"CUDA Toolkit {cuda_version}")
            
            # Check for Git
            if self._check_git():
                software_list.append("Git")
            
            # Check for Windows SDK
            sdk_versions = self._check_windows_sdk()
            if sdk_versions:
                software_list.extend([f"Windows SDK {v}" for v in sdk_versions])
            
            # Check for Hyper-V
            if self._check_hyper_v():
                software_list.append("Hyper-V")
            
            # Check for WSL
            if self._check_wsl():
                software_list.append("Windows Subsystem for Linux")
            
        except Exception as e:
            self.logger.debug(f"Software detection failed: {e}")
        
        return software_list
    
    def _check_visual_cpp_redist(self) -> bool:
        """Check for Visual C++ Redistributable."""
        try:
            # Check registry for VC++ redistributables
            registry_paths = [
                r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
                r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x86",
                r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
                r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x86"
            ]
            
            for path in registry_paths:
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                        installed = winreg.QueryValueEx(key, "Installed")[0]
                        if installed == 1:
                            return True
                except FileNotFoundError:
                    continue
            
            return False
        except Exception:
            return False
    
    def _check_net_framework(self) -> List[str]:
        """Check for .NET Framework versions."""
        versions = []
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                              r"SOFTWARE\Microsoft\NET Framework Setup\NDP") as key:
                i = 0
                while True:
                    try:
                        version = winreg.EnumKey(key, i)
                        if version.startswith("v"):
                            versions.append(version)
                        i += 1
                    except OSError:
                        break
        except Exception:
            pass
        
        return versions
    
    def _check_cuda_toolkit(self) -> Optional[str]:
        """Check for CUDA Toolkit installation."""
        try:
            # Check environment variable
            cuda_path = os.environ.get("CUDA_PATH")
            if cuda_path and os.path.exists(cuda_path):
                # Try to get version from nvcc
                nvcc_path = os.path.join(cuda_path, "bin", "nvcc.exe")
                if os.path.exists(nvcc_path):
                    result = subprocess.run([nvcc_path, '--version'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        version_match = re.search(r'release (\d+\.\d+)', result.stdout)
                        if version_match:
                            return version_match.group(1)
                
                return "Unknown Version"
            
            # Check registry
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                  r"SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA") as key:
                    i = 0
                    versions = []
                    while True:
                        try:
                            version = winreg.EnumKey(key, i)
                            versions.append(version)
                            i += 1
                        except OSError:
                            break
                    
                    if versions:
                        return max(versions)  # Return latest version
            except FileNotFoundError:
                pass
            
        except Exception:
            pass
        
        return None
    
    def _check_git(self) -> bool:
        """Check if Git is installed."""
        try:
            result = subprocess.run(['git', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def _check_windows_sdk(self) -> List[str]:
        """Check for Windows SDK versions."""
        versions = []
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                              r"SOFTWARE\WOW6432Node\Microsoft\Microsoft SDKs\Windows") as key:
                i = 0
                while True:
                    try:
                        version = winreg.EnumKey(key, i)
                        versions.append(version)
                        i += 1
                    except OSError:
                        break
        except Exception:
            pass
        
        return versions
    
    def _check_hyper_v(self) -> bool:
        """Check if Hyper-V is enabled."""
        try:
            cmd = ['powershell', '-Command', 
                   'Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All | Select-Object -ExpandProperty State']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.returncode == 0 and "Enabled" in result.stdout
        except Exception:
            return False
    
    def _check_wsl(self) -> bool:
        """Check if WSL is installed."""
        try:
            cmd = ['wsl', '--list']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def _detect_system_capabilities(self) -> Dict[str, bool]:
        """Detect system capabilities and features."""
        self.logger.debug("Detecting system capabilities...")
        
        capabilities = {}
        
        try:
            # Check admin rights
            capabilities["has_admin_rights"] = self._detect_user_privileges() == "Administrator"
            
            # Check internet access
            capabilities["has_internet_access"] = self._check_internet_access()
            
            # Check disk space
            capabilities["has_sufficient_disk_space"] = self._check_disk_space()
            
            # Check compatible Python
            capabilities["has_compatible_python"] = self._check_compatible_python()
            
            # Check Visual C++ Redistributable
            capabilities["has_visual_cpp_redist"] = self._check_visual_cpp_redist()
            
            # Check .NET Framework
            capabilities["has_net_framework"] = bool(self._check_net_framework())
            
            # Check Windows SDK
            capabilities["has_windows_sdk"] = bool(self._check_windows_sdk())
            
            # Check long path support
            capabilities["supports_long_paths"] = self._check_long_path_support()
            
            # Check Hyper-V
            capabilities["has_hyper_v"] = self._check_hyper_v()
            
            # Check WSL
            capabilities["has_wsl"] = self._check_wsl()
            
        except Exception as e:
            self.logger.warning(f"Capability detection failed: {e}")
        
        return capabilities
    
    def _check_internet_access(self) -> bool:
        """Check if system has internet access."""
        try:
            import urllib.request
            urllib.request.urlopen('https://www.google.com', timeout=5)
            return True
        except Exception:
            return False
    
    def _check_disk_space(self, required_gb: int = 50) -> bool:
        """Check if sufficient disk space is available."""
        try:
            import shutil
            free_bytes = shutil.disk_usage(str(self.installation_path)).free
            free_gb = free_bytes / (1024**3)
            return free_gb >= required_gb
        except Exception:
            return False
    
    def _check_compatible_python(self) -> bool:
        """Check if a compatible Python version is available."""
        installations = self._detect_python_installations()
        
        for install in installations:
            version = install["version"]
            # Extract major.minor version
            version_parts = version.split(".")
            if len(version_parts) >= 2:
                major_minor = f"{version_parts[0]}.{version_parts[1]}"
                if major_minor in self.supported_python_versions:
                    return True
        
        return False
    
    def _check_long_path_support(self) -> bool:
        """Check if Windows long path support is enabled."""
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                              r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
                value = winreg.QueryValueEx(key, "LongPathsEnabled")[0]
                return value == 1
        except Exception:
            return False
    
    def validate_system_capabilities(self, hardware_profile: HardwareProfile, 
                                   env_info: EnvironmentInfo) -> ValidationResult:
        """Validate system capabilities against requirements."""
        self.logger.info("Validating system capabilities...")
        
        issues = []
        warnings = []
        recommendations = []
        
        # Check Windows version
        try:
            build_number = int(env_info.windows_build)
            if build_number < self.minimum_windows_build:
                issues.append(f"Windows build {build_number} is too old (minimum: {self.minimum_windows_build})")
        except ValueError:
            warnings.append("Could not determine Windows build number")
        
        # Check architecture
        if env_info.architecture not in ["x64", "ARM64"]:
            issues.append(f"Unsupported architecture: {env_info.architecture}")
        
        # Check Python compatibility
        if not env_info.system_capabilities.get("has_compatible_python", False):
            issues.append("No compatible Python version found")
            recommendations.append("Install Python 3.9 or later")
        
        # Check disk space
        if not env_info.system_capabilities.get("has_sufficient_disk_space", False):
            issues.append("Insufficient disk space")
            recommendations.append("Free up at least 50GB of disk space")
        
        # Check internet access
        if not env_info.system_capabilities.get("has_internet_access", False):
            warnings.append("No internet access detected - offline installation required")
        
        # Check Visual C++ Redistributable
        if not env_info.system_capabilities.get("has_visual_cpp_redist", False):
            warnings.append("Visual C++ Redistributable not found")
            recommendations.append("Install Visual C++ Redistributable 2015-2022")
        
        # Check admin rights for optimal installation
        if not env_info.system_capabilities.get("has_admin_rights", False):
            warnings.append("Not running as administrator - some features may be limited")
            recommendations.append("Consider running as administrator for full functionality")
        
        # Check long path support
        if not env_info.system_capabilities.get("supports_long_paths", False):
            warnings.append("Long path support not enabled")
            recommendations.append("Enable long path support in Windows for better compatibility")
        
        success = len(issues) == 0
        message = "System capabilities validation passed" if success else f"System validation failed: {len(issues)} critical issues"
        
        result = ValidationResult(
            success=success,
            message=message,
            details={
                "issues": issues,
                "recommendations": recommendations,
                "environment_summary": {
                    "windows_version": env_info.windows_version,
                    "architecture": env_info.architecture,
                    "python_installations": len(env_info.python_installations),
                    "capabilities_passed": sum(1 for v in env_info.system_capabilities.values() if v),
                    "capabilities_total": len(env_info.system_capabilities)
                }
            },
            warnings=warnings if warnings else None
        )
        
        if success:
            self.logger.info("System capabilities validation passed")
        else:
            self.logger.error(f"System capabilities validation failed: {issues}")
        
        if warnings:
            self.logger.warning(f"System warnings: {warnings}")
        
        return result
    
    def classify_system_performance_tier(self, hardware_profile: HardwareProfile, 
                                       env_info: EnvironmentInfo) -> str:
        """Classify system into performance tiers with environment considerations."""
        self.logger.info("Classifying system performance tier...")
        
        # Base hardware classification
        hardware_score = 0
        
        # CPU scoring
        if hardware_profile.cpu.cores >= 16:
            hardware_score += 3
        elif hardware_profile.cpu.cores >= 8:
            hardware_score += 2
        elif hardware_profile.cpu.cores >= 4:
            hardware_score += 1
        
        # Memory scoring
        if hardware_profile.memory.total_gb >= 32:
            hardware_score += 3
        elif hardware_profile.memory.total_gb >= 16:
            hardware_score += 2
        elif hardware_profile.memory.total_gb >= 8:
            hardware_score += 1
        
        # GPU scoring
        if hardware_profile.gpu:
            if hardware_profile.gpu.vram_gb >= 12:
                hardware_score += 3
            elif hardware_profile.gpu.vram_gb >= 8:
                hardware_score += 2
            elif hardware_profile.gpu.vram_gb >= 4:
                hardware_score += 1
        
        # Storage scoring
        if hardware_profile.storage.type == "NVMe SSD":
            hardware_score += 2
        elif hardware_profile.storage.type == "SSD":
            hardware_score += 1
        
        # Environment modifiers
        environment_modifier = 0
        
        # Positive modifiers
        if env_info.system_capabilities.get("has_admin_rights", False):
            environment_modifier += 0.5
        
        if env_info.system_capabilities.get("supports_long_paths", False):
            environment_modifier += 0.5
        
        if env_info.system_capabilities.get("has_visual_cpp_redist", False):
            environment_modifier += 0.5
        
        # Negative modifiers
        if not env_info.system_capabilities.get("has_internet_access", False):
            environment_modifier -= 1
        
        if env_info.architecture == "ARM64":
            environment_modifier -= 1  # ARM64 may have compatibility issues
        
        # Calculate final score
        final_score = hardware_score + environment_modifier
        
        # Classify based on score
        if final_score >= 9:
            tier = "enterprise"
        elif final_score >= 7:
            tier = "high_performance"
        elif final_score >= 4:
            tier = "mid_range"
        else:
            tier = "budget"
        
        self.logger.info(f"System classified as: {tier} (score: {final_score:.1f})")
        return tier
    
    def _log_environment_summary(self, env_info: EnvironmentInfo) -> None:
        """Log a summary of detected environment."""
        self.logger.info("=== Environment Detection Summary ===")
        self.logger.info(f"Windows: {env_info.windows_version} (Build {env_info.windows_build})")
        self.logger.info(f"Edition: {env_info.windows_edition}")
        self.logger.info(f"Architecture: {env_info.architecture}")
        self.logger.info(f"Locale: {env_info.system_locale}")
        self.logger.info(f"User Privileges: {env_info.user_privileges}")
        
        self.logger.info(f"Python Installations: {len(env_info.python_installations)}")
        for install in env_info.python_installations[:3]:  # Show first 3
            self.logger.info(f"  - Python {install['version']} ({install['source']})")
        
        self.logger.info(f"Installed Software: {len(env_info.installed_software)} items")
        for software in env_info.installed_software:
            self.logger.info(f"  - {software}")
        
        capabilities_passed = sum(1 for v in env_info.system_capabilities.values() if v)
        capabilities_total = len(env_info.system_capabilities)
        self.logger.info(f"System Capabilities: {capabilities_passed}/{capabilities_total} passed")
        
        for capability, status in env_info.system_capabilities.items():
            status_icon = "✓" if status else "✗"
            self.logger.info(f"  {status_icon} {capability.replace('_', ' ').title()}")
        
        self.logger.info("=== End Environment Summary ===")


def main():
    """Test environment detection functionality."""
    import logging
    from detect_system import SystemDetector
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("=== WAN2.2 Environment Detection Test ===\n")
    
    try:
        # Create detectors
        hardware_detector = SystemDetector(".")
        env_detector = EnvironmentDetector(".")
        
        # Detect hardware first
        print("Detecting hardware...")
        hardware_profile = hardware_detector.detect_hardware()
        
        # Detect environment
        print("\nDetecting environment...")
        env_info = env_detector.detect_environment()
        
        # Validate system capabilities
        print("\n=== SYSTEM VALIDATION ===")
        validation = env_detector.validate_system_capabilities(hardware_profile, env_info)
        print(f"Status: {'PASSED' if validation.success else 'FAILED'}")
        print(f"Message: {validation.message}")
        
        if validation.warnings:
            print("\nWarnings:")
            for warning in validation.warnings:
                print(f"  ⚠️  {warning}")
        
        if not validation.success and validation.details:
            print("\nIssues:")
            for issue in validation.details.get("issues", []):
                print(f"  ❌ {issue}")
            
            print("\nRecommendations:")
            for rec in validation.details.get("recommendations", []):
                print(f"  💡 {rec}")
        
        # Performance tier classification
        print("\n=== PERFORMANCE CLASSIFICATION ===")
        tier = env_detector.classify_system_performance_tier(hardware_profile, env_info)
        print(f"Performance Tier: {tier.upper()}")
        
    except Exception as e:
        print(f"❌ Environment detection failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()