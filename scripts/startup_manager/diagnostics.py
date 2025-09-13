"""
System diagnostics and troubleshooting features for the startup manager.

This module provides comprehensive system information collection, diagnostic mode,
and log analysis tools to help identify and resolve common startup issues.
"""

import os
import sys
import platform
import subprocess
import psutil
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import socket
import importlib.util

from .logger import get_logger


@dataclass
class SystemInfo:
    """System information data structure."""
    os_name: str
    os_version: str
    os_release: str
    architecture: str
    processor: str
    python_version: str
    python_executable: str
    virtual_env: Optional[str]
    memory_total: int
    memory_available: int
    disk_usage: Dict[str, Any]
    network_interfaces: List[Dict[str, Any]]
    environment_variables: Dict[str, str]
    installed_packages: List[str]


@dataclass
class DiagnosticResult:
    """Diagnostic check result."""
    check_name: str
    status: str  # "pass", "fail", "warning"
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None


@dataclass
class LogAnalysisResult:
    """Log analysis result."""
    total_entries: int
    error_count: int
    warning_count: int
    common_errors: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    suggestions: List[str]


class SystemDiagnostics:
    """
    System diagnostics and information collection.
    
    Provides comprehensive system information gathering for troubleshooting
    and diagnostic purposes.
    """
    
    def __init__(self):
        self.logger = get_logger()
    
    def collect_system_info(self) -> SystemInfo:
        """
        Collect comprehensive system information.
        
        Returns:
            SystemInfo object with all collected information
        """
        self.logger.debug("Collecting system information")
        
        try:
            # Basic OS information
            os_info = self._get_os_info()
            
            # Python information
            python_info = self._get_python_info()
            
            # Hardware information
            hardware_info = self._get_hardware_info()
            
            # Network information
            network_info = self._get_network_info()
            
            # Environment information
            env_info = self._get_environment_info()
            
            # Package information
            package_info = self._get_package_info()
            
            system_info = SystemInfo(
                os_name=os_info["name"],
                os_version=os_info["version"],
                os_release=os_info["release"],
                architecture=os_info["architecture"],
                processor=hardware_info["processor"],
                python_version=python_info["version"],
                python_executable=python_info["executable"],
                virtual_env=python_info["virtual_env"],
                memory_total=hardware_info["memory_total"],
                memory_available=hardware_info["memory_available"],
                disk_usage=hardware_info["disk_usage"],
                network_interfaces=network_info,
                environment_variables=env_info,
                installed_packages=package_info
            )
            
            self.logger.debug("System information collected successfully")
            return system_info
            
        except Exception as e:
            self.logger.error(f"Failed to collect system information: {e}")
            raise
    
    def _get_os_info(self) -> Dict[str, str]:
        """Get operating system information."""
        return {
            "name": platform.system(),
            "version": platform.version(),
            "release": platform.release(),
            "architecture": platform.architecture()[0]
        }
    
    def _get_python_info(self) -> Dict[str, Any]:
        """Get Python environment information."""
        virtual_env = None
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            virtual_env = sys.prefix
        
        return {
            "version": platform.python_version(),
            "executable": sys.executable,
            "virtual_env": virtual_env,
            "path": sys.path[:5]  # First 5 paths to avoid too much data
        }
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "processor": platform.processor() or "Unknown",
            "memory_total": memory.total,
            "memory_available": memory.available,
            "memory_percent": memory.percent,
            "disk_usage": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            }
        }
    
    def _get_network_info(self) -> List[Dict[str, Any]]:
        """Get network interface information."""
        interfaces = []
        
        try:
            for interface, addresses in psutil.net_if_addrs().items():
                interface_info = {
                    "name": interface,
                    "addresses": []
                }
                
                for addr in addresses:
                    if addr.family == socket.AF_INET:  # IPv4
                        interface_info["addresses"].append({
                            "type": "IPv4",
                            "address": addr.address,
                            "netmask": addr.netmask
                        })
                    elif addr.family == socket.AF_INET6:  # IPv6
                        interface_info["addresses"].append({
                            "type": "IPv6",
                            "address": addr.address,
                            "netmask": addr.netmask
                        })
                
                if interface_info["addresses"]:
                    interfaces.append(interface_info)
        
        except Exception as e:
            self.logger.warning(f"Failed to get network information: {e}")
        
        return interfaces
    
    def _get_environment_info(self) -> Dict[str, str]:
        """Get relevant environment variables."""
        relevant_vars = [
            "PATH", "PYTHONPATH", "NODE_PATH", "npm_config_prefix",
            "VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "PIPENV_ACTIVE"
        ]
        
        env_info = {}
        for var in relevant_vars:
            value = os.environ.get(var)
            if value:
                # Truncate very long paths
                if len(value) > 500:
                    value = value[:500] + "..."
                env_info[var] = value
        
        return env_info
    
    def _get_package_info(self) -> List[str]:
        """Get list of installed Python packages."""
        packages = []
        
        try:
            # Try to get pip list
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=freeze"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                packages = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            self.logger.warning("Failed to get package information")
        
        return packages[:50]  # Limit to first 50 packages
    
    def run_diagnostic_checks(self) -> List[DiagnosticResult]:
        """
        Run comprehensive diagnostic checks.
        
        Returns:
            List of diagnostic results
        """
        self.logger.info("Running diagnostic checks")
        
        checks = [
            self._check_python_version,
            self._check_virtual_environment,
            self._check_required_packages,
            self._check_port_availability,
            self._check_disk_space,
            self._check_memory_usage,
            self._check_network_connectivity,
            self._check_file_permissions,
            self._check_node_environment
        ]
        
        results = []
        for check in checks:
            try:
                result = check()
                results.append(result)
                self.logger.debug(f"Diagnostic check '{result.check_name}': {result.status}")
            except Exception as e:
                results.append(DiagnosticResult(
                    check_name=check.__name__,
                    status="fail",
                    message=f"Check failed with error: {str(e)}",
                    suggestions=["Check system logs for more details"]
                ))
        
        self.logger.info(f"Completed {len(results)} diagnostic checks")
        return results
    
    def _check_python_version(self) -> DiagnosticResult:
        """Check Python version compatibility."""
        version = sys.version_info
        
        if version.major == 3 and version.minor >= 8:
            return DiagnosticResult(
                check_name="Python Version",
                status="pass",
                message=f"Python {version.major}.{version.minor}.{version.micro} is supported"
            )
        elif version.major == 3 and version.minor >= 6:
            return DiagnosticResult(
                check_name="Python Version",
                status="warning",
                message=f"Python {version.major}.{version.minor}.{version.micro} may work but 3.8+ is recommended",
                suggestions=["Consider upgrading to Python 3.8 or later"]
            )
        else:
            return DiagnosticResult(
                check_name="Python Version",
                status="fail",
                message=f"Python {version.major}.{version.minor}.{version.micro} is not supported",
                suggestions=["Upgrade to Python 3.8 or later"]
            )
    
    def _check_virtual_environment(self) -> DiagnosticResult:
        """Check if virtual environment is active."""
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            return DiagnosticResult(
                check_name="Virtual Environment",
                status="pass",
                message=f"Virtual environment active: {sys.prefix}"
            )
        else:
            return DiagnosticResult(
                check_name="Virtual Environment",
                status="warning",
                message="No virtual environment detected",
                suggestions=[
                    "Consider using a virtual environment",
                    "Run: python -m venv venv && venv\\Scripts\\activate"
                ]
            )
    
    def _check_required_packages(self) -> DiagnosticResult:
        """Check if required packages are installed."""
        required_packages = [
            "fastapi", "uvicorn", "pydantic", "psutil", "colorama"
        ]
        
        missing_packages = []
        for package in required_packages:
            if importlib.util.find_spec(package) is None:
                missing_packages.append(package)
        
        if not missing_packages:
            return DiagnosticResult(
                check_name="Required Packages",
                status="pass",
                message="All required packages are installed"
            )
        else:
            return DiagnosticResult(
                check_name="Required Packages",
                status="fail",
                message=f"Missing packages: {', '.join(missing_packages)}",
                suggestions=[f"Install missing packages: pip install {' '.join(missing_packages)}"]
            )
    
    def _check_port_availability(self) -> DiagnosticResult:
        """Check if default ports are available."""
        ports_to_check = [8000, 3000]
        occupied_ports = []
        
        for port in ports_to_check:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    result = sock.connect_ex(('localhost', port))
                    if result == 0:
                        occupied_ports.append(port)
            except Exception:
                pass
        
        if not occupied_ports:
            return DiagnosticResult(
                check_name="Port Availability",
                status="pass",
                message="Default ports (8000, 3000) are available"
            )
        else:
            return DiagnosticResult(
                check_name="Port Availability",
                status="warning",
                message=f"Ports in use: {', '.join(map(str, occupied_ports))}",
                suggestions=[
                    "Stop processes using these ports",
                    "Use alternative ports",
                    "Check for zombie processes"
                ]
            )
    
    def _check_disk_space(self) -> DiagnosticResult:
        """Check available disk space."""
        try:
            disk_usage = psutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            percent_used = (disk_usage.used / disk_usage.total) * 100
            
            if free_gb > 5:
                return DiagnosticResult(
                    check_name="Disk Space",
                    status="pass",
                    message=f"Sufficient disk space: {free_gb:.1f} GB free"
                )
            elif free_gb > 1:
                return DiagnosticResult(
                    check_name="Disk Space",
                    status="warning",
                    message=f"Low disk space: {free_gb:.1f} GB free ({percent_used:.1f}% used)",
                    suggestions=["Free up disk space", "Clean temporary files"]
                )
            else:
                return DiagnosticResult(
                    check_name="Disk Space",
                    status="fail",
                    message=f"Very low disk space: {free_gb:.1f} GB free ({percent_used:.1f}% used)",
                    suggestions=["Immediately free up disk space", "Move files to another drive"]
                )
        except Exception as e:
            return DiagnosticResult(
                check_name="Disk Space",
                status="fail",
                message=f"Failed to check disk space: {e}"
            )
    
    def _check_memory_usage(self) -> DiagnosticResult:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            percent_used = memory.percent
            
            if percent_used < 80:
                return DiagnosticResult(
                    check_name="Memory Usage",
                    status="pass",
                    message=f"Memory usage normal: {percent_used:.1f}% used, {available_gb:.1f} GB available"
                )
            elif percent_used < 90:
                return DiagnosticResult(
                    check_name="Memory Usage",
                    status="warning",
                    message=f"High memory usage: {percent_used:.1f}% used, {available_gb:.1f} GB available",
                    suggestions=["Close unnecessary applications", "Consider adding more RAM"]
                )
            else:
                return DiagnosticResult(
                    check_name="Memory Usage",
                    status="fail",
                    message=f"Very high memory usage: {percent_used:.1f}% used, {available_gb:.1f} GB available",
                    suggestions=["Close applications immediately", "Restart system if necessary"]
                )
        except Exception as e:
            return DiagnosticResult(
                check_name="Memory Usage",
                status="fail",
                message=f"Failed to check memory usage: {e}"
            )
    
    def _check_network_connectivity(self) -> DiagnosticResult:
        """Check basic network connectivity."""
        try:
            # Try to connect to a reliable host
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(5)
                result = sock.connect_ex(('8.8.8.8', 53))  # Google DNS
                
            if result == 0:
                return DiagnosticResult(
                    check_name="Network Connectivity",
                    status="pass",
                    message="Network connectivity is working"
                )
            else:
                return DiagnosticResult(
                    check_name="Network Connectivity",
                    status="warning",
                    message="Network connectivity issues detected",
                    suggestions=[
                        "Check internet connection",
                        "Check firewall settings",
                        "Try different DNS servers"
                    ]
                )
        except Exception as e:
            return DiagnosticResult(
                check_name="Network Connectivity",
                status="fail",
                message=f"Network check failed: {e}",
                suggestions=["Check network configuration", "Restart network adapter"]
            )
    
    def _check_file_permissions(self) -> DiagnosticResult:
        """Check file system permissions."""
        try:
            # Test write permissions in current directory
            test_file = Path("test_permissions.tmp")
            test_file.write_text("test")
            test_file.unlink()
            
            return DiagnosticResult(
                check_name="File Permissions",
                status="pass",
                message="File system permissions are working"
            )
        except PermissionError:
            return DiagnosticResult(
                check_name="File Permissions",
                status="fail",
                message="Permission denied when writing to current directory",
                suggestions=[
                    "Run as administrator",
                    "Check folder permissions",
                    "Move to a different directory"
                ]
            )
        except Exception as e:
            return DiagnosticResult(
                check_name="File Permissions",
                status="warning",
                message=f"Permission check failed: {e}"
            )
    
    def _check_node_environment(self) -> DiagnosticResult:
        """Check Node.js environment."""
        try:
            # Check Node.js version
            node_result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if node_result.returncode != 0:
                return DiagnosticResult(
                    check_name="Node.js Environment",
                    status="fail",
                    message="Node.js is not installed or not in PATH",
                    suggestions=[
                        "Install Node.js from https://nodejs.org/",
                        "Add Node.js to system PATH"
                    ]
                )
            
            # Check npm version
            npm_result = subprocess.run(
                ["npm", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if npm_result.returncode != 0:
                return DiagnosticResult(
                    check_name="Node.js Environment",
                    status="warning",
                    message="Node.js is installed but npm is not available",
                    suggestions=["Reinstall Node.js with npm included"]
                )
            
            node_version = node_result.stdout.strip()
            npm_version = npm_result.stdout.strip()
            
            return DiagnosticResult(
                check_name="Node.js Environment",
                status="pass",
                message=f"Node.js {node_version} and npm {npm_version} are available"
            )
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            return DiagnosticResult(
                check_name="Node.js Environment",
                status="fail",
                message="Failed to check Node.js installation",
                suggestions=[
                    "Install Node.js from https://nodejs.org/",
                    "Ensure Node.js is in system PATH"
                ]
            )


class LogAnalyzer:
    """
    Log analysis tools for identifying common issues and patterns.
    
    Analyzes startup logs to identify common problems and provide
    suggestions for resolution.
    """
    
    def __init__(self):
        self.logger = get_logger()
        self.error_patterns = {
            "port_conflict": [
                r"Address already in use",
                r"WinError 10048",
                r"EADDRINUSE"
            ],
            "permission_denied": [
                r"Permission denied",
                r"WinError 5",
                r"Access is denied",
                r"EACCES"
            ],
            "module_not_found": [
                r"ModuleNotFoundError",
                r"ImportError",
                r"No module named"
            ],
            "network_error": [
                r"Connection refused",
                r"Network is unreachable",
                r"Timeout",
                r"WinError 10061"
            ],
            "file_not_found": [
                r"FileNotFoundError",
                r"No such file or directory",
                r"WinError 2"
            ]
        }
    
    def analyze_logs(self, log_dir: Union[str, Path]) -> LogAnalysisResult:
        """
        Analyze log files for common issues and patterns.
        
        Args:
            log_dir: Directory containing log files
            
        Returns:
            LogAnalysisResult with analysis findings
        """
        self.logger.info(f"Analyzing logs in {log_dir}")
        
        log_dir = Path(log_dir)
        if not log_dir.exists():
            return LogAnalysisResult(
                total_entries=0,
                error_count=0,
                warning_count=0,
                common_errors=[],
                performance_metrics={},
                suggestions=["No log files found to analyze"]
            )
        
        # Analyze text logs
        text_logs = list(log_dir.glob("*.log"))
        json_logs = list(log_dir.glob("*.json"))
        
        total_entries = 0
        error_count = 0
        warning_count = 0
        error_patterns_found = {}
        performance_data = []
        
        # Analyze text log files
        for log_file in text_logs:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        total_entries += 1
                        
                        if "ERROR" in line:
                            error_count += 1
                            self._categorize_error(line, error_patterns_found)
                        elif "WARNING" in line:
                            warning_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to analyze {log_file}: {e}")
        
        # Analyze JSON log files
        for log_file in json_logs:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                entry = json.loads(line)
                                total_entries += 1
                                
                                if entry.get("level") == "ERROR":
                                    error_count += 1
                                    self._categorize_error(entry.get("message", ""), error_patterns_found)
                                elif entry.get("level") == "WARNING":
                                    warning_count += 1
                                
                                # Extract performance metrics
                                extra_data = entry.get("extra_data", {})
                                if extra_data.get("metric_type") == "performance":
                                    performance_data.append({
                                        "operation": extra_data.get("operation"),
                                        "duration": extra_data.get("duration")
                                    })
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                self.logger.warning(f"Failed to analyze {log_file}: {e}")
        
        # Process findings
        common_errors = self._process_error_patterns(error_patterns_found)
        performance_metrics = self._process_performance_data(performance_data)
        suggestions = self._generate_suggestions(common_errors, error_count, warning_count)
        
        result = LogAnalysisResult(
            total_entries=total_entries,
            error_count=error_count,
            warning_count=warning_count,
            common_errors=common_errors,
            performance_metrics=performance_metrics,
            suggestions=suggestions
        )
        
        self.logger.info(f"Log analysis complete: {total_entries} entries, {error_count} errors, {warning_count} warnings")
        return result
    
    def _categorize_error(self, message: str, error_patterns_found: Dict[str, int]):
        """Categorize error message by pattern."""
        for category, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    error_patterns_found[category] = error_patterns_found.get(category, 0) + 1
                    break
    
    def _process_error_patterns(self, error_patterns_found: Dict[str, int]) -> List[Dict[str, Any]]:
        """Process error patterns into common errors list."""
        common_errors = []
        
        for category, count in sorted(error_patterns_found.items(), key=lambda x: x[1], reverse=True):
            error_info = {
                "category": category,
                "count": count,
                "description": self._get_error_description(category),
                "solutions": self._get_error_solutions(category)
            }
            common_errors.append(error_info)
        
        return common_errors
    
    def _get_error_description(self, category: str) -> str:
        """Get description for error category."""
        descriptions = {
            "port_conflict": "Port already in use by another process",
            "permission_denied": "Access denied due to insufficient permissions",
            "module_not_found": "Required Python modules are missing",
            "network_error": "Network connectivity or timeout issues",
            "file_not_found": "Required files or directories are missing"
        }
        return descriptions.get(category, f"Unknown error category: {category}")
    
    def _get_error_solutions(self, category: str) -> List[str]:
        """Get solutions for error category."""
        solutions = {
            "port_conflict": [
                "Kill processes using the required ports",
                "Use alternative port numbers",
                "Check for zombie processes",
                "Restart the system if necessary"
            ],
            "permission_denied": [
                "Run as administrator",
                "Check file and folder permissions",
                "Add firewall exceptions",
                "Disable antivirus temporarily"
            ],
            "module_not_found": [
                "Install missing packages with pip",
                "Activate virtual environment",
                "Check Python path configuration",
                "Reinstall requirements.txt"
            ],
            "network_error": [
                "Check internet connection",
                "Configure firewall settings",
                "Try different DNS servers",
                "Check proxy settings"
            ],
            "file_not_found": [
                "Verify file paths are correct",
                "Check working directory",
                "Restore missing files",
                "Check file permissions"
            ]
        }
        return solutions.get(category, ["Check system logs for more details"])
    
    def _process_performance_data(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process performance metrics."""
        if not performance_data:
            return {}
        
        operations = {}
        for metric in performance_data:
            operation = metric.get("operation")
            duration = metric.get("duration")
            
            if operation and duration is not None:
                if operation not in operations:
                    operations[operation] = []
                operations[operation].append(duration)
        
        # Calculate statistics
        metrics = {}
        for operation, durations in operations.items():
            metrics[operation] = {
                "count": len(durations),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations)
            }
        
        return metrics
    
    def _generate_suggestions(self, common_errors: List[Dict[str, Any]], error_count: int, warning_count: int) -> List[str]:
        """Generate suggestions based on analysis."""
        suggestions = []
        
        if error_count == 0 and warning_count == 0:
            suggestions.append("No issues found in logs - system appears to be running normally")
        
        if error_count > 0:
            suggestions.append(f"Found {error_count} errors - review error details for specific solutions")
        
        if warning_count > 0:
            suggestions.append(f"Found {warning_count} warnings - consider addressing to prevent future issues")
        
        # Add specific suggestions based on common errors
        for error in common_errors[:3]:  # Top 3 most common errors
            category = error["category"]
            if category == "port_conflict":
                suggestions.append("Port conflicts detected - consider using automatic port allocation")
            elif category == "permission_denied":
                suggestions.append("Permission issues detected - try running as administrator")
            elif category == "module_not_found":
                suggestions.append("Missing dependencies detected - run pip install -r requirements.txt")
        
        if not suggestions:
            suggestions.append("Review logs for detailed information about system behavior")
        
        return suggestions


class DiagnosticMode:
    """
    Diagnostic mode that captures detailed startup process information.
    
    Provides comprehensive diagnostic information collection during
    startup process for troubleshooting purposes.
    """
    
    def __init__(self):
        self.logger = get_logger()
        self.diagnostics = SystemDiagnostics()
        self.analyzer = LogAnalyzer()
    
    def run_full_diagnostics(self, log_dir: Union[str, Path] = "logs") -> Dict[str, Any]:
        """
        Run comprehensive diagnostics including system info, checks, and log analysis.
        
        Args:
            log_dir: Directory containing log files to analyze
            
        Returns:
            Dictionary containing all diagnostic information
        """
        self.logger.info("Running full diagnostic mode")
        
        diagnostic_data = {
            "timestamp": datetime.now().isoformat(),
            "system_info": None,
            "diagnostic_checks": [],
            "log_analysis": None,
            "summary": {}
        }
        
        try:
            # Collect system information
            self.logger.info("Collecting system information")
            system_info = self.diagnostics.collect_system_info()
            diagnostic_data["system_info"] = asdict(system_info)
            
            # Run diagnostic checks
            self.logger.info("Running diagnostic checks")
            diagnostic_results = self.diagnostics.run_diagnostic_checks()
            diagnostic_data["diagnostic_checks"] = [asdict(result) for result in diagnostic_results]
            
            # Analyze logs
            self.logger.info("Analyzing logs")
            log_analysis = self.analyzer.analyze_logs(log_dir)
            diagnostic_data["log_analysis"] = asdict(log_analysis)
            
            # Generate summary
            diagnostic_data["summary"] = self._generate_summary(diagnostic_results, log_analysis)
            
            self.logger.info("Full diagnostics completed successfully")
            
        except Exception as e:
            self.logger.error(f"Diagnostic mode failed: {e}")
            diagnostic_data["error"] = str(e)
        
        return diagnostic_data
    
    def _generate_summary(self, diagnostic_results: List[DiagnosticResult], log_analysis: LogAnalysisResult) -> Dict[str, Any]:
        """Generate diagnostic summary."""
        passed_checks = sum(1 for result in diagnostic_results if result.status == "pass")
        failed_checks = sum(1 for result in diagnostic_results if result.status == "fail")
        warning_checks = sum(1 for result in diagnostic_results if result.status == "warning")
        
        overall_status = "healthy"
        if failed_checks > 0:
            overall_status = "issues_detected"
        elif warning_checks > 0:
            overall_status = "warnings_present"
        
        return {
            "overall_status": overall_status,
            "diagnostic_checks": {
                "total": len(diagnostic_results),
                "passed": passed_checks,
                "failed": failed_checks,
                "warnings": warning_checks
            },
            "log_analysis": {
                "total_entries": log_analysis.total_entries,
                "error_count": log_analysis.error_count,
                "warning_count": log_analysis.warning_count,
                "common_error_categories": len(log_analysis.common_errors)
            },
            "recommendations": self._get_top_recommendations(diagnostic_results, log_analysis)
        }
    
    def _get_top_recommendations(self, diagnostic_results: List[DiagnosticResult], log_analysis: LogAnalysisResult) -> List[str]:
        """Get top recommendations based on diagnostics."""
        recommendations = []
        
        # Add recommendations from failed checks
        for result in diagnostic_results:
            if result.status == "fail" and result.suggestions:
                recommendations.extend(result.suggestions[:2])  # Top 2 suggestions per check
        
        # Add recommendations from log analysis
        if log_analysis.suggestions:
            recommendations.extend(log_analysis.suggestions[:3])  # Top 3 from log analysis
        
        # Remove duplicates and limit to top 5
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:5]
    
    def save_diagnostic_report(self, diagnostic_data: Dict[str, Any], output_file: Union[str, Path] = None) -> Path:
        """
        Save diagnostic report to file.
        
        Args:
            diagnostic_data: Diagnostic data to save
            output_file: Output file path (optional)
            
        Returns:
            Path to saved report file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"diagnostic_report_{timestamp}.json"
        
        output_path = Path(output_file)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(diagnostic_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Diagnostic report saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to save diagnostic report: {e}")
            raise
