"""
Diagnostic tool for installation problems.

This module provides comprehensive diagnostic capabilities to help identify
and troubleshoot installation issues, including system analysis, dependency
checking, and automated problem detection.
"""

import logging
import platform
import subprocess
import sys
import json
import psutil
import shutil
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import importlib.util

from interfaces import HardwareProfile, ValidationResult, ErrorCategory
from base_classes import BaseInstallationComponent


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""
    name: str
    status: str  # "pass", "fail", "warning", "info"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class SystemDiagnostics:
    """Complete system diagnostic information."""
    timestamp: datetime = field(default_factory=datetime.now)
    system_info: Dict[str, Any] = field(default_factory=dict)
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    software_info: Dict[str, Any] = field(default_factory=dict)
    installation_info: Dict[str, Any] = field(default_factory=dict)
    diagnostic_results: List[DiagnosticResult] = field(default_factory=list)
    overall_status: str = "unknown"


class InstallationDiagnosticTool(BaseInstallationComponent):
    """Comprehensive diagnostic tool for installation problems."""
    
    def __init__(self, installation_path: str, logger: Optional[logging.Logger] = None):
        super().__init__(installation_path, logger)
        self.diagnostics = SystemDiagnostics()
        self.min_requirements = {
            "python_version": (3, 9),
            "memory_gb": 8,
            "disk_space_gb": 50,
            "cpu_cores": 4
        }
    
    def run_full_diagnostics(self) -> SystemDiagnostics:
        """Run complete diagnostic suite."""
        self.logger.info("Starting comprehensive system diagnostics...")
        
        # Reset diagnostics
        self.diagnostics = SystemDiagnostics()
        
        # Run all diagnostic checks
        self._check_system_info()
        self._check_hardware_requirements()
        self._check_python_installation()
        self._check_dependencies()
        self._check_network_connectivity()
        self._check_disk_space()
        self._check_permissions()
        self._check_existing_installation()
        self._check_gpu_support()
        self._check_antivirus_interference()
        
        # Determine overall status
        self._determine_overall_status()
        
        self.logger.info(f"Diagnostics completed. Overall status: {self.diagnostics.overall_status}")
        return self.diagnostics
    
    def _check_system_info(self) -> None:
        """Check basic system information."""
        try:
            system_info = {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "python_version": sys.version,
                "python_executable": sys.executable
            }
            
            self.diagnostics.system_info = system_info
            
            # Check Windows version
            if platform.system() == "Windows":
                version_parts = platform.version().split('.')
                if len(version_parts) >= 2:
                    major, minor = int(version_parts[0]), int(version_parts[1])
                    if major >= 10:
                        status = "pass"
                        message = f"Windows version {platform.release()} is supported"
                    else:
                        status = "warning"
                        message = f"Windows version {platform.release()} may have compatibility issues"
                else:
                    status = "warning"
                    message = "Could not determine Windows version"
            else:
                status = "fail"
                message = f"Unsupported operating system: {platform.system()}"
            
            self.diagnostics.diagnostic_results.append(
                DiagnosticResult(
                    name="Operating System",
                    status=status,
                    message=message,
                    details={"system_info": system_info}
                )
            )
            
        except Exception as e:
            self.diagnostics.diagnostic_results.append(
                DiagnosticResult(
                    name="Operating System",
                    status="fail",
                    message=f"Failed to detect system information: {e}",
                    suggestions=["Check system integrity", "Run as administrator"]
                )
            )
    
    def _check_hardware_requirements(self) -> None:
        """Check hardware requirements."""
        try:
            # Memory check
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            if memory_gb >= self.min_requirements["memory_gb"]:
                memory_status = "pass"
                memory_message = f"Memory: {memory_gb:.1f}GB (sufficient)"
            else:
                memory_status = "fail"
                memory_message = f"Memory: {memory_gb:.1f}GB (minimum {self.min_requirements['memory_gb']}GB required)"
            
            self.diagnostics.diagnostic_results.append(
                DiagnosticResult(
                    name="Memory Requirements",
                    status=memory_status,
                    message=memory_message,
                    details={"total_memory_gb": memory_gb, "available_memory_gb": memory.available / (1024**3)},
                    suggestions=["Consider upgrading system memory"] if memory_status == "fail" else []
                )
            )
            
            # CPU check
            cpu_count = psutil.cpu_count(logical=False)
            logical_cpu_count = psutil.cpu_count(logical=True)
            
            if cpu_count >= self.min_requirements["cpu_cores"]:
                cpu_status = "pass"
                cpu_message = f"CPU: {cpu_count} cores, {logical_cpu_count} threads (sufficient)"
            else:
                cpu_status = "warning"
                cpu_message = f"CPU: {cpu_count} cores (minimum {self.min_requirements['cpu_cores']} recommended)"
            
            self.diagnostics.diagnostic_results.append(
                DiagnosticResult(
                    name="CPU Requirements",
                    status=cpu_status,
                    message=cpu_message,
                    details={"physical_cores": cpu_count, "logical_cores": logical_cpu_count}
                )
            )
            
            self.diagnostics.hardware_info = {
                "memory_gb": memory_gb,
                "cpu_cores": cpu_count,
                "logical_cores": logical_cpu_count
            }
            
        except Exception as e:
            self.diagnostics.diagnostic_results.append(
                DiagnosticResult(
                    name="Hardware Requirements",
                    status="fail",
                    message=f"Failed to check hardware: {e}",
                    suggestions=["Check system hardware", "Update system drivers"]
                )
            )
    
    def _check_python_installation(self) -> None:
        """Check Python installation."""
        try:
            # Check Python version
            python_version = sys.version_info
            min_version = self.min_requirements["python_version"]
            
            if python_version >= min_version:
                status = "pass"
                message = f"Python {python_version.major}.{python_version.minor}.{python_version.micro} (compatible)"
            else:
                status = "fail"
                message = f"Python {python_version.major}.{python_version.minor}.{python_version.micro} (minimum {min_version[0]}.{min_version[1]} required)"
            
            # Check pip availability
            try:
                import pip
                pip_version = pip.__version__
                pip_available = True
            except ImportError:
                pip_version = "Not available"
                pip_available = False
                if status == "pass":
                    status = "warning"
                    message += " (pip not available)"
            
            self.diagnostics.diagnostic_results.append(
                DiagnosticResult(
                    name="Python Installation",
                    status=status,
                    message=message,
                    details={
                        "python_version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                        "python_executable": sys.executable,
                        "pip_version": pip_version,
                        "pip_available": pip_available
                    },
                    suggestions=[
                        "Install Python 3.9 or later",
                        "Ensure pip is installed and accessible"
                    ] if status != "pass" else []
                )
            )
            
        except Exception as e:
            self.diagnostics.diagnostic_results.append(
                DiagnosticResult(
                    name="Python Installation",
                    status="fail",
                    message=f"Failed to check Python: {e}",
                    suggestions=["Install Python", "Check PATH environment variable"]
                )
            )
    
    def _check_dependencies(self) -> None:
        """Check for required dependencies."""
        required_packages = [
            "torch", "torchvision", "transformers", "diffusers",
            "pillow", "numpy", "opencv-python", "psutil"
        ]
        
        missing_packages = []
        installed_packages = {}
        
        for package in required_packages:
            try:
                spec = importlib.util.find_spec(package)
                if spec is not None:
                    # Try to get version
                    try:
                        module = importlib.import_module(package)
                        version = getattr(module, '__version__', 'unknown')
                        installed_packages[package] = version
                    except:
                        installed_packages[package] = 'installed'
                else:
                    missing_packages.append(package)
            except Exception:
                missing_packages.append(package)
        
        if not missing_packages:
            status = "pass"
            message = f"All {len(required_packages)} required packages are installed"
        elif len(missing_packages) < len(required_packages) / 2:
            status = "warning"
            message = f"{len(missing_packages)} packages missing: {', '.join(missing_packages[:3])}{'...' if len(missing_packages) > 3 else ''}"
        else:
            status = "fail"
            message = f"Most required packages are missing ({len(missing_packages)}/{len(required_packages)})"
        
        self.diagnostics.diagnostic_results.append(
            DiagnosticResult(
                name="Dependencies",
                status=status,
                message=message,
                details={
                    "installed_packages": installed_packages,
                    "missing_packages": missing_packages
                },
                suggestions=[
                    "Run the installation script to install missing packages",
                    "Check internet connection for package downloads"
                ] if missing_packages else []
            )
        )
    
    def _check_network_connectivity(self) -> None:
        """Check network connectivity."""
        test_urls = [
            "https://pypi.org",
            "https://huggingface.co",
            "https://github.com"
        ]
        
        connectivity_results = {}
        
        for url in test_urls:
            try:
                import urllib.request
                urllib.request.urlopen(url, timeout=10)
                connectivity_results[url] = "accessible"
            except Exception as e:
                connectivity_results[url] = f"failed: {str(e)}"
        
        accessible_count = sum(1 for result in connectivity_results.values() if result == "accessible")
        
        if accessible_count == len(test_urls):
            status = "pass"
            message = "Network connectivity is good"
        elif accessible_count > 0:
            status = "warning"
            message = f"Partial network connectivity ({accessible_count}/{len(test_urls)} sites accessible)"
        else:
            status = "fail"
            message = "No network connectivity detected"
        
        self.diagnostics.diagnostic_results.append(
            DiagnosticResult(
                name="Network Connectivity",
                status=status,
                message=message,
                details={"connectivity_results": connectivity_results},
                suggestions=[
                    "Check internet connection",
                    "Verify firewall and proxy settings",
                    "Try using a VPN if in a restricted network"
                ] if status != "pass" else []
            )
        )
    
    def _check_disk_space(self) -> None:
        """Check available disk space."""
        try:
            installation_drive = Path(self.installation_path).drive or Path(self.installation_path).anchor
            disk_usage = shutil.disk_usage(installation_drive)
            
            available_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            
            if available_gb >= self.min_requirements["disk_space_gb"]:
                status = "pass"
                message = f"Disk space: {available_gb:.1f}GB available (sufficient)"
            elif available_gb >= self.min_requirements["disk_space_gb"] * 0.7:
                status = "warning"
                message = f"Disk space: {available_gb:.1f}GB available (tight, {self.min_requirements['disk_space_gb']}GB recommended)"
            else:
                status = "fail"
                message = f"Disk space: {available_gb:.1f}GB available (insufficient, {self.min_requirements['disk_space_gb']}GB required)"
            
            self.diagnostics.diagnostic_results.append(
                DiagnosticResult(
                    name="Disk Space",
                    status=status,
                    message=message,
                    details={
                        "available_gb": available_gb,
                        "total_gb": total_gb,
                        "installation_drive": installation_drive
                    },
                    suggestions=[
                        "Free up disk space",
                        "Consider installing on a different drive with more space"
                    ] if status != "pass" else []
                )
            )
            
        except Exception as e:
            self.diagnostics.diagnostic_results.append(
                DiagnosticResult(
                    name="Disk Space",
                    status="fail",
                    message=f"Failed to check disk space: {e}",
                    suggestions=["Check disk accessibility", "Run as administrator"]
                )
            )
    
    def _check_permissions(self) -> None:
        """Check file system permissions."""
        test_file = Path(self.installation_path) / "test_permissions.tmp"
        
        try:
            # Test write permissions
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("permission test")
            test_file.unlink()
            
            status = "pass"
            message = "Write permissions are available"
            suggestions = []
            
        except PermissionError:
            status = "fail"
            message = "Insufficient write permissions"
            suggestions = [
                "Run as administrator",
                "Check folder permissions",
                "Choose a different installation directory"
            ]
        except Exception as e:
            status = "warning"
            message = f"Permission check inconclusive: {e}"
            suggestions = ["Verify installation directory accessibility"]
        
        self.diagnostics.diagnostic_results.append(
            DiagnosticResult(
                name="File Permissions",
                status=status,
                message=message,
                suggestions=suggestions
            )
        )
    
    def _check_existing_installation(self) -> None:
        """Check for existing installation."""
        key_files = [
            "main.py",
            "config.json",
            "models",
            "scripts"
        ]
        
        existing_files = []
        for file_name in key_files:
            file_path = Path(self.installation_path) / file_name
            if file_path.exists():
                existing_files.append(file_name)
        
        if not existing_files:
            status = "info"
            message = "No existing installation detected"
        elif len(existing_files) == len(key_files):
            status = "warning"
            message = "Complete existing installation detected"
            suggestions = ["Consider backing up existing installation", "Use update mode instead of fresh install"]
        else:
            status = "warning"
            message = f"Partial installation detected ({len(existing_files)}/{len(key_files)} components)"
            suggestions = ["Clean up partial installation", "Run fresh installation"]
        
        self.diagnostics.diagnostic_results.append(
            DiagnosticResult(
                name="Existing Installation",
                status=status,
                message=message,
                details={"existing_files": existing_files},
                suggestions=suggestions if status == "warning" else []
            )
        )
    
    def _check_gpu_support(self) -> None:
        """Check GPU support and CUDA availability."""
        try:
            # Try to detect NVIDIA GPU
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10
                )
                
                if result.returncode == 0:
                    gpu_info = result.stdout.strip().split('\n')[0].split(', ')
                    gpu_name = gpu_info[0]
                    gpu_memory = int(gpu_info[1])
                    
                    status = "pass"
                    message = f"NVIDIA GPU detected: {gpu_name} ({gpu_memory}MB VRAM)"
                    
                    # Check CUDA
                    try:
                        import torch
                        if torch.cuda.is_available():
                            cuda_version = torch.version.cuda
                            message += f", CUDA {cuda_version} available"
                        else:
                            status = "warning"
                            message += ", but CUDA not available in PyTorch"
                    except ImportError:
                        status = "warning"
                        message += ", PyTorch not installed"
                    
                else:
                    status = "info"
                    message = "No NVIDIA GPU detected (CPU-only mode)"
                    
            except (subprocess.TimeoutExpired, FileNotFoundError):
                status = "info"
                message = "GPU detection failed (nvidia-smi not found)"
            
            self.diagnostics.diagnostic_results.append(
                DiagnosticResult(
                    name="GPU Support",
                    status=status,
                    message=message,
                    suggestions=[
                        "Install NVIDIA drivers",
                        "Install CUDA toolkit",
                        "Verify GPU compatibility"
                    ] if status == "warning" else []
                )
            )
            
        except Exception as e:
            self.diagnostics.diagnostic_results.append(
                DiagnosticResult(
                    name="GPU Support",
                    status="warning",
                    message=f"GPU check failed: {e}",
                    suggestions=["Check GPU drivers", "Verify hardware installation"]
                )
            )
    
    def _check_antivirus_interference(self) -> None:
        """Check for potential antivirus interference."""
        # This is a heuristic check - we can't directly detect all antivirus software
        potential_issues = []
        
        # Check for Windows Defender
        try:
            result = subprocess.run(
                ["powershell", "-Command", "Get-MpPreference | Select-Object -ExpandProperty DisableRealtimeMonitoring"],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                defender_disabled = result.stdout.strip().lower() == "true"
                if not defender_disabled:
                    potential_issues.append("Windows Defender real-time protection is enabled")
        except:
            pass
        
        # Check for common antivirus processes
        antivirus_processes = [
            "avp.exe", "avgnt.exe", "avguard.exe", "bdagent.exe",
            "mcshield.exe", "nortonsecurity.exe", "avastui.exe"
        ]
        
        running_av = []
        for proc in psutil.process_iter(['name']):
            try:
                if proc.info['name'].lower() in [av.lower() for av in antivirus_processes]:
                    running_av.append(proc.info['name'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if running_av:
            potential_issues.append(f"Antivirus software detected: {', '.join(running_av)}")
        
        if potential_issues:
            status = "warning"
            message = "Potential antivirus interference detected"
            suggestions = [
                "Temporarily disable real-time protection during installation",
                "Add installation directory to antivirus exclusions",
                "Run installation as administrator"
            ]
        else:
            status = "info"
            message = "No obvious antivirus interference detected"
            suggestions = []
        
        self.diagnostics.diagnostic_results.append(
            DiagnosticResult(
                name="Antivirus Interference",
                status=status,
                message=message,
                details={"potential_issues": potential_issues},
                suggestions=suggestions
            )
        )
    
    def _determine_overall_status(self) -> None:
        """Determine overall diagnostic status."""
        fail_count = sum(1 for result in self.diagnostics.diagnostic_results if result.status == "fail")
        warning_count = sum(1 for result in self.diagnostics.diagnostic_results if result.status == "warning")
        
        if fail_count > 0:
            self.diagnostics.overall_status = "fail"
        elif warning_count > 2:
            self.diagnostics.overall_status = "warning"
        else:
            self.diagnostics.overall_status = "pass"
    
    def generate_diagnostic_report(self, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive diagnostic report."""
        if not self.diagnostics.diagnostic_results:
            self.run_full_diagnostics()
        
        report_lines = [
            "WAN2.2 Installation Diagnostic Report",
            "=" * 50,
            f"Generated: {self.diagnostics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Overall Status: {self.diagnostics.overall_status.upper()}",
            ""
        ]
        
        # System summary
        report_lines.extend([
            "System Summary:",
            "-" * 20,
            f"OS: {self.diagnostics.system_info.get('platform', 'Unknown')}",
            f"Python: {self.diagnostics.system_info.get('python_version', 'Unknown').split()[0]}",
            f"Memory: {self.diagnostics.hardware_info.get('memory_gb', 0):.1f}GB",
            f"CPU Cores: {self.diagnostics.hardware_info.get('cpu_cores', 0)}",
            ""
        ])
        
        # Detailed results
        report_lines.append("Detailed Results:")
        report_lines.append("-" * 20)
        
        for result in self.diagnostics.diagnostic_results:
            status_symbol = {
                "pass": "✅",
                "fail": "❌",
                "warning": "⚠️",
                "info": "ℹ️"
            }.get(result.status, "❓")
            
            report_lines.append(f"{status_symbol} {result.name}: {result.message}")
            
            if result.suggestions:
                for suggestion in result.suggestions:
                    report_lines.append(f"   • {suggestion}")
                report_lines.append("")
        
        # Summary recommendations
        all_suggestions = []
        for result in self.diagnostics.diagnostic_results:
            if result.status in ["fail", "warning"]:
                all_suggestions.extend(result.suggestions)
        
        if all_suggestions:
            report_lines.extend([
                "",
                "Priority Actions:",
                "-" * 20
            ])
            
            # Remove duplicates while preserving order
            seen = set()
            unique_suggestions = []
            for suggestion in all_suggestions:
                if suggestion not in seen:
                    seen.add(suggestion)
                    unique_suggestions.append(suggestion)
            
            for i, suggestion in enumerate(unique_suggestions[:10], 1):
                report_lines.append(f"{i}. {suggestion}")
        
        report_text = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            try:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(report_text, encoding='utf-8')
                self.logger.info(f"Diagnostic report saved to: {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to save diagnostic report: {e}")
        
        return report_text
    
    def get_quick_health_check(self) -> Dict[str, Any]:
        """Get a quick health check without full diagnostics."""
        health_check = {
            "timestamp": datetime.now().isoformat(),
            "python_ok": sys.version_info >= self.min_requirements["python_version"],
            "memory_ok": psutil.virtual_memory().total / (1024**3) >= self.min_requirements["memory_gb"],
            "disk_ok": True,  # Will be checked below
            "network_ok": True,  # Will be checked below
            "overall_ok": True
        }
        
        # Quick disk check
        try:
            installation_drive = Path(self.installation_path).drive or Path(self.installation_path).anchor
            disk_usage = shutil.disk_usage(installation_drive)
            available_gb = disk_usage.free / (1024**3)
            health_check["disk_ok"] = available_gb >= self.min_requirements["disk_space_gb"]
        except:
            health_check["disk_ok"] = False
        
        # Quick network check
        try:
            import urllib.request
            urllib.request.urlopen("https://pypi.org", timeout=5)
        except:
            health_check["network_ok"] = False
        
        # Overall status
        health_check["overall_ok"] = all([
            health_check["python_ok"],
            health_check["memory_ok"],
            health_check["disk_ok"],
            health_check["network_ok"]
        ])
        
        return health_check