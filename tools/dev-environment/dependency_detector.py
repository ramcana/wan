#!/usr/bin/env python3
"""
Dependency Detection and Installation Guidance

This module provides automated dependency detection and installation guidance
for the WAN22 development environment.
"""

import os
import sys
import subprocess
import platform
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging

@dataclass
class DependencyInfo:
    """Information about a dependency"""
    name: str
    required_version: Optional[str] = None
    installed_version: Optional[str] = None
    is_installed: bool = False
    installation_command: Optional[str] = None
    installation_url: Optional[str] = None
    notes: Optional[str] = None

@dataclass
class SystemInfo:
    """System information"""
    platform: str
    python_version: str
    node_version: Optional[str] = None
    npm_version: Optional[str] = None
    git_version: Optional[str] = None
    cuda_available: bool = False
    cuda_version: Optional[str] = None

class DependencyDetector:
    """Detects and validates development dependencies"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.logger = logging.getLogger(__name__)
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> SystemInfo:
        """Get system information"""
        system_info = SystemInfo(
            platform=platform.system(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        
        # Check Node.js
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                system_info.node_version = result.stdout.strip().lstrip('v')
        except FileNotFoundError:
            pass
            
        # Check npm
        try:
            result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                system_info.npm_version = result.stdout.strip()
        except FileNotFoundError:
            pass
            
        # Check Git
        try:
            result = subprocess.run(['git', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                system_info.git_version = result.stdout.strip().split()[-1]
        except FileNotFoundError:
            pass
            
        # Check CUDA
        try:
            import torch
            if torch.cuda.is_available():
                system_info.cuda_available = True
                system_info.cuda_version = torch.version.cuda
        except ImportError:
            pass
            
        return system_info
    
    def detect_python_dependencies(self) -> List[DependencyInfo]:
        """Detect Python dependencies"""
        dependencies = []
        
        # Check Python version
        python_dep = DependencyInfo(
            name="Python",
            required_version="3.8+",
            installed_version=self.system_info.python_version,
            is_installed=sys.version_info >= (3, 8),
            installation_url="https://python.org/downloads/",
            notes="Python 3.8 or higher required"
        )
        dependencies.append(python_dep)
        
        # Check pip
        pip_dep = DependencyInfo(
            name="pip",
            is_installed=shutil.which('pip') is not None,
            installation_command="python -m ensurepip --upgrade"
        )
        dependencies.append(pip_dep)
        
        # Check requirements.txt dependencies
        requirements_file = self.project_root / "backend" / "requirements.txt"
        if requirements_file.exists():
            with open(requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        package_name = line.split('>=')[0].split('==')[0].split('<')[0]
                        dep = self._check_python_package(package_name, line)
                        dependencies.append(dep)
        
        return dependencies
    
    def detect_nodejs_dependencies(self) -> List[DependencyInfo]:
        """Detect Node.js dependencies"""
        dependencies = []
        
        # Check Node.js
        node_dep = DependencyInfo(
            name="Node.js",
            required_version="16+",
            installed_version=self.system_info.node_version,
            is_installed=self.system_info.node_version is not None,
            installation_url="https://nodejs.org/",
            notes="Node.js 16 or higher required"
        )
        dependencies.append(node_dep)
        
        # Check npm
        npm_dep = DependencyInfo(
            name="npm",
            installed_version=self.system_info.npm_version,
            is_installed=self.system_info.npm_version is not None,
            notes="Usually comes with Node.js"
        )
        dependencies.append(npm_dep)
        
        # Check package.json dependencies
        package_json = self.project_root / "frontend" / "package.json"
        if package_json.exists():
            try:
                with open(package_json, 'r') as f:
                    package_data = json.load(f)
                    
                # Check if node_modules exists
                node_modules = self.project_root / "frontend" / "node_modules"
                npm_deps_installed = node_modules.exists()
                
                npm_install_dep = DependencyInfo(
                    name="Frontend Dependencies",
                    is_installed=npm_deps_installed,
                    installation_command="cd frontend && npm install",
                    notes=f"Install {len(package_data.get('dependencies', {}))} frontend dependencies"
                )
                dependencies.append(npm_install_dep)
                
            except json.JSONDecodeError:
                self.logger.error("Invalid package.json file")
        
        return dependencies
    
    def detect_system_dependencies(self) -> List[DependencyInfo]:
        """Detect system-level dependencies"""
        dependencies = []
        
        # Check Git
        git_dep = DependencyInfo(
            name="Git",
            installed_version=self.system_info.git_version,
            is_installed=self.system_info.git_version is not None,
            installation_url="https://git-scm.com/downloads",
            notes="Required for version control"
        )
        dependencies.append(git_dep)
        
        # Check CUDA (optional)
        cuda_dep = DependencyInfo(
            name="CUDA",
            installed_version=self.system_info.cuda_version,
            is_installed=self.system_info.cuda_available,
            installation_url="https://developer.nvidia.com/cuda-downloads",
            notes="Optional: Required for GPU acceleration"
        )
        dependencies.append(cuda_dep)
        
        return dependencies
    
    def _check_python_package(self, package_name: str, requirement: str) -> DependencyInfo:
        """Check if a Python package is installed"""
        try:
            import importlib
            importlib.import_module(package_name.replace('-', '_'))
            
            # Try to get version
            try:
                import pkg_resources
                installed_version = pkg_resources.get_distribution(package_name).version
            except:
                installed_version = "unknown"
                
            return DependencyInfo(
                name=package_name,
                required_version=requirement,
                installed_version=installed_version,
                is_installed=True,
                installation_command=f"pip install {requirement}"
            )
        except ImportError:
            return DependencyInfo(
                name=package_name,
                required_version=requirement,
                is_installed=False,
                installation_command=f"pip install {requirement}"
            )
    
    def get_all_dependencies(self) -> Dict[str, List[DependencyInfo]]:
        """Get all dependencies categorized by type"""
        return {
            'system': self.detect_system_dependencies(),
            'python': self.detect_python_dependencies(),
            'nodejs': self.detect_nodejs_dependencies()
        }
    
    def get_missing_dependencies(self) -> Dict[str, List[DependencyInfo]]:
        """Get only missing dependencies"""
        all_deps = self.get_all_dependencies()
        missing = {}
        
        for category, deps in all_deps.items():
            missing_deps = [dep for dep in deps if not dep.is_installed]
            if missing_deps:
                missing[category] = missing_deps
                
        return missing
    
    def generate_installation_guide(self) -> str:
        """Generate installation guide for missing dependencies"""
        missing = self.get_missing_dependencies()
        
        if not missing:
            return "✅ All dependencies are installed!"
        
        guide = ["# Missing Dependencies Installation Guide\n"]
        
        for category, deps in missing.items():
            guide.append(f"## {category.title()} Dependencies\n")
            
            for dep in deps:
                guide.append(f"### {dep.name}")
                if dep.required_version:
                    guide.append(f"Required version: {dep.required_version}")
                
                if dep.installation_command:
                    guide.append(f"```bash\n{dep.installation_command}\n```")
                elif dep.installation_url:
                    guide.append(f"Download from: {dep.installation_url}")
                
                if dep.notes:
                    guide.append(f"Note: {dep.notes}")
                
                guide.append("")
        
        return "\n".join(guide)
    
    def export_dependency_report(self, output_file: Path) -> None:
        """Export dependency report to JSON file"""
        all_deps = self.get_all_dependencies()
        
        report = {
            'system_info': asdict(self.system_info),
            'dependencies': {
                category: [asdict(dep) for dep in deps]
                for category, deps in all_deps.items()
            },
            'missing_count': sum(
                len([dep for dep in deps if not dep.is_installed])
                for deps in all_deps.values()
            )
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Dependency report exported to {output_file}")

def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect and validate development dependencies")
    parser.add_argument('--check', action='store_true', help='Check all dependencies')
    parser.add_argument('--missing', action='store_true', help='Show only missing dependencies')
    parser.add_argument('--guide', action='store_true', help='Generate installation guide')
    parser.add_argument('--export', type=str, help='Export report to JSON file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    detector = DependencyDetector()
    
    if args.check or not any([args.missing, args.guide, args.export]):
        # Show all dependencies
        all_deps = detector.get_all_dependencies()
        
        for category, deps in all_deps.items():
            print(f"\n{category.upper()} DEPENDENCIES:")
            print("-" * 40)
            
            for dep in deps:
                status = "✅" if dep.is_installed else "❌"
                version_info = f" (v{dep.installed_version})" if dep.installed_version else ""
                print(f"{status} {dep.name}{version_info}")
                
                if not dep.is_installed and dep.installation_command:
                    print(f"   Install: {dep.installation_command}")
    
    if args.missing:
        missing = detector.get_missing_dependencies()
        
        if not missing:
            print("✅ All dependencies are installed!")
        else:
            print("❌ Missing dependencies:")
            for category, deps in missing.items():
                print(f"\n{category.upper()}:")
                for dep in deps:
                    print(f"  - {dep.name}")
                    if dep.installation_command:
                        print(f"    Install: {dep.installation_command}")
    
    if args.guide:
        guide = detector.generate_installation_guide()
        print(guide)
    
    if args.export:
        output_file = Path(args.export)
        detector.export_dependency_report(output_file)
        print(f"Report exported to {output_file}")

if __name__ == "__main__":
    main()