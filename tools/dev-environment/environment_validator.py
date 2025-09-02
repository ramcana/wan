#!/usr/bin/env python3
"""
Development Environment Validator

This module provides comprehensive validation and health checking
for the WAN22 development environment.
"""

import os
import sys
import subprocess
import platform
import json
import shutil
import socket
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

@dataclass
class ValidationResult:
    """Result of a validation check"""
    name: str
    status: str  # 'pass', 'fail', 'warning'
    message: str
    details: Optional[Dict[str, Any]] = None
    fix_suggestion: Optional[str] = None

@dataclass
class EnvironmentHealth:
    """Overall environment health status"""
    overall_status: str  # 'healthy', 'warning', 'critical'
    score: float  # 0-100
    total_checks: int
    passed_checks: int
    failed_checks: int
    warning_checks: int
    results: List[ValidationResult]
    timestamp: str

class EnvironmentValidator:
    """Validates development environment setup and health"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.logger = logging.getLogger(__name__)
        
    def validate_python_environment(self) -> List[ValidationResult]:
        """Validate Python environment"""
        results = []
        
        # Check Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if sys.version_info >= (3, 8):
            results.append(ValidationResult(
                name="Python Version",
                status="pass",
                message=f"Python {python_version} is supported",
                details={"version": python_version}
            ))
        else:
            results.append(ValidationResult(
                name="Python Version",
                status="fail",
                message=f"Python {python_version} is too old",
                details={"version": python_version, "required": "3.8+"},
                fix_suggestion="Upgrade to Python 3.8 or higher"
            ))
        
        # Check pip
        try:
            import pip
            pip_version = pip.__version__
            results.append(ValidationResult(
                name="pip",
                status="pass",
                message=f"pip {pip_version} is available",
                details={"version": pip_version}
            ))
        except ImportError:
            results.append(ValidationResult(
                name="pip",
                status="fail",
                message="pip is not available",
                fix_suggestion="Install pip: python -m ensurepip --upgrade"
            ))
        
        # Check virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        if in_venv:
            results.append(ValidationResult(
                name="Virtual Environment",
                status="pass",
                message="Running in virtual environment",
                details={"prefix": sys.prefix}
            ))
        else:
            results.append(ValidationResult(
                name="Virtual Environment",
                status="warning",
                message="Not running in virtual environment",
                fix_suggestion="Consider using a virtual environment: python -m venv venv"
            ))
        
        # Check backend requirements
        requirements_file = self.project_root / "backend" / "requirements.txt"
        if requirements_file.exists():
            missing_packages = self._check_python_requirements(requirements_file)
            if not missing_packages:
                results.append(ValidationResult(
                    name="Backend Dependencies",
                    status="pass",
                    message="All backend dependencies are installed"
                ))
            else:
                results.append(ValidationResult(
                    name="Backend Dependencies",
                    status="fail",
                    message=f"{len(missing_packages)} packages missing",
                    details={"missing": missing_packages},
                    fix_suggestion="Install missing packages: pip install -r backend/requirements.txt"
                ))
        
        return results
    
    def validate_nodejs_environment(self) -> List[ValidationResult]:
        """Validate Node.js environment"""
        results = []
        
        # Check Node.js
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                node_version = result.stdout.strip().lstrip('v')
                major_version = int(node_version.split('.')[0])
                
                if major_version >= 16:
                    results.append(ValidationResult(
                        name="Node.js Version",
                        status="pass",
                        message=f"Node.js {node_version} is supported",
                        details={"version": node_version}
                    ))
                else:
                    results.append(ValidationResult(
                        name="Node.js Version",
                        status="fail",
                        message=f"Node.js {node_version} is too old",
                        details={"version": node_version, "required": "16+"},
                        fix_suggestion="Upgrade to Node.js 16 or higher"
                    ))
            else:
                results.append(ValidationResult(
                    name="Node.js",
                    status="fail",
                    message="Node.js command failed",
                    fix_suggestion="Install Node.js from https://nodejs.org/"
                ))
        except FileNotFoundError:
            results.append(ValidationResult(
                name="Node.js",
                status="fail",
                message="Node.js is not installed",
                fix_suggestion="Install Node.js from https://nodejs.org/"
            ))
        
        # Check npm
        try:
            result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                npm_version = result.stdout.strip()
                results.append(ValidationResult(
                    name="npm",
                    status="pass",
                    message=f"npm {npm_version} is available",
                    details={"version": npm_version}
                ))
            else:
                results.append(ValidationResult(
                    name="npm",
                    status="fail",
                    message="npm command failed",
                    fix_suggestion="Reinstall Node.js to get npm"
                ))
        except FileNotFoundError:
            results.append(ValidationResult(
                name="npm",
                status="fail",
                message="npm is not installed",
                fix_suggestion="Install Node.js to get npm"
            ))
        
        # Check frontend dependencies
        frontend_dir = self.project_root / "frontend"
        node_modules = frontend_dir / "node_modules"
        package_json = frontend_dir / "package.json"
        
        if package_json.exists():
            if node_modules.exists():
                results.append(ValidationResult(
                    name="Frontend Dependencies",
                    status="pass",
                    message="Frontend dependencies are installed"
                ))
            else:
                results.append(ValidationResult(
                    name="Frontend Dependencies",
                    status="fail",
                    message="Frontend dependencies not installed",
                    fix_suggestion="Install dependencies: cd frontend && npm install"
                ))
        
        return results
    
    def validate_project_structure(self) -> List[ValidationResult]:
        """Validate project structure"""
        results = []
        
        # Check required directories
        required_dirs = [
            "backend",
            "frontend", 
            "docs",
            "config",
            "tests",
            "tools"
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
        
        if not missing_dirs:
            results.append(ValidationResult(
                name="Project Structure",
                status="pass",
                message="All required directories exist"
            ))
        else:
            results.append(ValidationResult(
                name="Project Structure",
                status="warning",
                message=f"Missing directories: {', '.join(missing_dirs)}",
                details={"missing": missing_dirs},
                fix_suggestion="Create missing directories as needed"
            ))
        
        # Check configuration files
        config_files = [
            "backend/requirements.txt",
            "frontend/package.json",
            "config/unified-config.yaml",
            ".pre-commit-config.yaml"
        ]
        
        missing_configs = []
        for config_file in config_files:
            config_path = self.project_root / config_file
            if not config_path.exists():
                missing_configs.append(config_file)
        
        if not missing_configs:
            results.append(ValidationResult(
                name="Configuration Files",
                status="pass",
                message="All configuration files exist"
            ))
        else:
            results.append(ValidationResult(
                name="Configuration Files",
                status="warning",
                message=f"Missing config files: {', '.join(missing_configs)}",
                details={"missing": missing_configs}
            ))
        
        return results
    
    def validate_development_tools(self) -> List[ValidationResult]:
        """Validate development tools"""
        results = []
        
        # Check Git
        try:
            result = subprocess.run(['git', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                git_version = result.stdout.strip().split()[-1]
                results.append(ValidationResult(
                    name="Git",
                    status="pass",
                    message=f"Git {git_version} is available",
                    details={"version": git_version}
                ))
            else:
                results.append(ValidationResult(
                    name="Git",
                    status="fail",
                    message="Git command failed",
                    fix_suggestion="Install Git from https://git-scm.com/"
                ))
        except FileNotFoundError:
            results.append(ValidationResult(
                name="Git",
                status="fail",
                message="Git is not installed",
                fix_suggestion="Install Git from https://git-scm.com/"
            ))
        
        # Check pre-commit hooks
        pre_commit_config = self.project_root / ".pre-commit-config.yaml"
        if pre_commit_config.exists():
            try:
                result = subprocess.run(['pre-commit', '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    results.append(ValidationResult(
                        name="Pre-commit Hooks",
                        status="pass",
                        message="Pre-commit is installed and configured"
                    ))
                else:
                    results.append(ValidationResult(
                        name="Pre-commit Hooks",
                        status="warning",
                        message="Pre-commit config exists but pre-commit not installed",
                        fix_suggestion="Install pre-commit: pip install pre-commit && pre-commit install"
                    ))
            except FileNotFoundError:
                results.append(ValidationResult(
                    name="Pre-commit Hooks",
                    status="warning",
                    message="Pre-commit not installed",
                    fix_suggestion="Install pre-commit: pip install pre-commit && pre-commit install"
                ))
        
        return results
    
    def validate_ports_and_services(self) -> List[ValidationResult]:
        """Validate ports and services"""
        results = []
        
        # Check common ports
        ports_to_check = [
            (3000, "Frontend Development Server"),
            (8000, "Backend API Server"),
            (7860, "Gradio UI Server")
        ]
        
        for port, service_name in ports_to_check:
            if self._is_port_available(port):
                results.append(ValidationResult(
                    name=f"Port {port}",
                    status="pass",
                    message=f"Port {port} is available for {service_name}"
                ))
            else:
                results.append(ValidationResult(
                    name=f"Port {port}",
                    status="warning",
                    message=f"Port {port} is in use",
                    details={"port": port, "service": service_name},
                    fix_suggestion=f"Stop service using port {port} or configure different port"
                ))
        
        return results
    
    def validate_gpu_environment(self) -> List[ValidationResult]:
        """Validate GPU environment (optional)"""
        results = []
        
        try:
            import torch
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                
                results.append(ValidationResult(
                    name="CUDA Support",
                    status="pass",
                    message=f"CUDA {cuda_version} available with {gpu_count} GPU(s)",
                    details={
                        "cuda_version": cuda_version,
                        "gpu_count": gpu_count,
                        "gpu_name": gpu_name
                    }
                ))
            else:
                results.append(ValidationResult(
                    name="CUDA Support",
                    status="warning",
                    message="CUDA not available - CPU mode only",
                    fix_suggestion="Install CUDA-enabled PyTorch for GPU acceleration"
                ))
        except ImportError:
            results.append(ValidationResult(
                name="PyTorch",
                status="warning",
                message="PyTorch not installed",
                fix_suggestion="Install PyTorch: pip install torch"
            ))
        
        return results
    
    def run_full_validation(self) -> EnvironmentHealth:
        """Run complete environment validation"""
        all_results = []
        
        # Run all validation checks
        all_results.extend(self.validate_python_environment())
        all_results.extend(self.validate_nodejs_environment())
        all_results.extend(self.validate_project_structure())
        all_results.extend(self.validate_development_tools())
        all_results.extend(self.validate_ports_and_services())
        all_results.extend(self.validate_gpu_environment())
        
        # Calculate health metrics
        total_checks = len(all_results)
        passed_checks = len([r for r in all_results if r.status == 'pass'])
        failed_checks = len([r for r in all_results if r.status == 'fail'])
        warning_checks = len([r for r in all_results if r.status == 'warning'])
        
        # Calculate score (pass=1, warning=0.5, fail=0)
        score = (passed_checks + warning_checks * 0.5) / total_checks * 100 if total_checks > 0 else 0
        
        # Determine overall status
        if failed_checks == 0 and warning_checks == 0:
            overall_status = "healthy"
        elif failed_checks == 0:
            overall_status = "warning"
        else:
            overall_status = "critical"
        
        return EnvironmentHealth(
            overall_status=overall_status,
            score=score,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warning_checks=warning_checks,
            results=all_results,
            timestamp=datetime.now().isoformat()
        )
    
    def _check_python_requirements(self, requirements_file: Path) -> List[str]:
        """Check which Python packages are missing"""
        missing = []
        
        try:
            with open(requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        package_name = line.split('>=')[0].split('==')[0].split('<')[0]
                        try:
                            import importlib
                            importlib.import_module(package_name.replace('-', '_'))
                        except ImportError:
                            missing.append(package_name)
        except Exception as e:
            self.logger.error(f"Error checking requirements: {e}")
        
        return missing
    
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                return result != 0
        except Exception:
            return True
    
    def export_health_report(self, health: EnvironmentHealth, output_file: Path) -> None:
        """Export health report to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(asdict(health), f, indent=2)
        
        self.logger.info(f"Health report exported to {output_file}")

def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate development environment")
    parser.add_argument('--validate', action='store_true', help='Run full validation')
    parser.add_argument('--python', action='store_true', help='Validate Python environment only')
    parser.add_argument('--nodejs', action='store_true', help='Validate Node.js environment only')
    parser.add_argument('--structure', action='store_true', help='Validate project structure only')
    parser.add_argument('--tools', action='store_true', help='Validate development tools only')
    parser.add_argument('--ports', action='store_true', help='Validate ports and services only')
    parser.add_argument('--gpu', action='store_true', help='Validate GPU environment only')
    parser.add_argument('--export', type=str, help='Export report to JSON file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    validator = EnvironmentValidator()
    
    # Determine what to validate
    if args.validate or not any([args.python, args.nodejs, args.structure, args.tools, args.ports, args.gpu]):
        health = validator.run_full_validation()
        
        print(f"\n🏥 ENVIRONMENT HEALTH REPORT")
        print("=" * 50)
        print(f"Overall Status: {health.overall_status.upper()}")
        print(f"Health Score: {health.score:.1f}/100")
        print(f"Checks: {health.passed_checks} passed, {health.warning_checks} warnings, {health.failed_checks} failed")
        
        # Group results by status
        failed_results = [r for r in health.results if r.status == 'fail']
        warning_results = [r for r in health.results if r.status == 'warning']
        passed_results = [r for r in health.results if r.status == 'pass']
        
        if failed_results:
            print(f"\n❌ FAILED CHECKS ({len(failed_results)}):")
            for result in failed_results:
                print(f"  - {result.name}: {result.message}")
                if result.fix_suggestion:
                    print(f"    Fix: {result.fix_suggestion}")
        
        if warning_results:
            print(f"\n⚠️  WARNING CHECKS ({len(warning_results)}):")
            for result in warning_results:
                print(f"  - {result.name}: {result.message}")
                if result.fix_suggestion:
                    print(f"    Suggestion: {result.fix_suggestion}")
        
        if args.verbose and passed_results:
            print(f"\n✅ PASSED CHECKS ({len(passed_results)}):")
            for result in passed_results:
                print(f"  - {result.name}: {result.message}")
        
        if args.export:
            output_file = Path(args.export)
            validator.export_health_report(health, output_file)
            print(f"\nReport exported to {output_file}")
    
    else:
        # Run specific validations
        results = []
        
        if args.python:
            results.extend(validator.validate_python_environment())
        if args.nodejs:
            results.extend(validator.validate_nodejs_environment())
        if args.structure:
            results.extend(validator.validate_project_structure())
        if args.tools:
            results.extend(validator.validate_development_tools())
        if args.ports:
            results.extend(validator.validate_ports_and_services())
        if args.gpu:
            results.extend(validator.validate_gpu_environment())
        
        # Display results
        for result in results:
            status_icon = {"pass": "✅", "warning": "⚠️", "fail": "❌"}[result.status]
            print(f"{status_icon} {result.name}: {result.message}")
            if result.fix_suggestion:
                print(f"   Fix: {result.fix_suggestion}")

if __name__ == "__main__":
    main()