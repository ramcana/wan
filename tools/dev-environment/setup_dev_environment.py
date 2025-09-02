#!/usr/bin/env python3
"""
Automated Development Environment Setup

This module provides automated setup for the WAN22 development environment,
including dependency installation, configuration, and validation.
"""

import os
import sys
import subprocess
import platform
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

from dependency_detector import DependencyDetector
from environment_validator import EnvironmentValidator

class DevEnvironmentSetup:
    """Automated development environment setup"""
    
    def __init__(self, project_root: Optional[Path] = None, verbose: bool = False):
        self.project_root = project_root or Path.cwd()
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.dependency_detector = DependencyDetector(self.project_root)
        self.environment_validator = EnvironmentValidator(self.project_root)
        
        # Setup logging
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    def setup_python_environment(self) -> bool:
        """Setup Python environment"""
        self.logger.info("Setting up Python environment...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.logger.error("Python 3.8+ required. Please upgrade Python.")
            return False
        
        # Check if in virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        
        if not in_venv:
            self.logger.warning("Not in virtual environment. Consider creating one:")
            self.logger.warning("  python -m venv venv")
            self.logger.warning("  source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        
        # Install backend dependencies
        requirements_file = self.project_root / "backend" / "requirements.txt"
        if requirements_file.exists():
            self.logger.info("Installing Python dependencies...")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], check=True, capture_output=not self.verbose)
                self.logger.info("✅ Python dependencies installed successfully")
                return True
            except subprocess.CalledProcessError as e:
                self.logger.error(f"❌ Failed to install Python dependencies: {e}")
                return False
        else:
            self.logger.warning("No requirements.txt found in backend/")
            return True
    
    def setup_nodejs_environment(self) -> bool:
        """Setup Node.js environment"""
        self.logger.info("Setting up Node.js environment...")
        
        # Check Node.js
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error("Node.js not found. Please install Node.js 16+ from https://nodejs.org/")
                return False
            
            node_version = result.stdout.strip().lstrip('v')
            major_version = int(node_version.split('.')[0])
            
            if major_version < 16:
                self.logger.error(f"Node.js {node_version} is too old. Please upgrade to Node.js 16+")
                return False
            
            self.logger.info(f"✅ Node.js {node_version} found")
            
        except FileNotFoundError:
            self.logger.error("Node.js not found. Please install Node.js 16+ from https://nodejs.org/")
            return False
        
        # Install frontend dependencies
        frontend_dir = self.project_root / "frontend"
        package_json = frontend_dir / "package.json"
        
        if package_json.exists():
            self.logger.info("Installing frontend dependencies...")
            try:
                subprocess.run([
                    "npm", "install"
                ], cwd=frontend_dir, check=True, capture_output=not self.verbose)
                self.logger.info("✅ Frontend dependencies installed successfully")
                return True
            except subprocess.CalledProcessError as e:
                self.logger.error(f"❌ Failed to install frontend dependencies: {e}")
                return False
        else:
            self.logger.warning("No package.json found in frontend/")
            return True
    
    def setup_development_tools(self) -> bool:
        """Setup development tools"""
        self.logger.info("Setting up development tools...")
        
        success = True
        
        # Check Git
        try:
            result = subprocess.run(['git', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                git_version = result.stdout.strip().split()[-1]
                self.logger.info(f"✅ Git {git_version} found")
            else:
                self.logger.error("Git not working properly")
                success = False
        except FileNotFoundError:
            self.logger.error("Git not found. Please install Git from https://git-scm.com/")
            success = False
        
        # Setup pre-commit hooks
        pre_commit_config = self.project_root / ".pre-commit-config.yaml"
        if pre_commit_config.exists():
            try:
                # Install pre-commit
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "pre-commit"
                ], check=True, capture_output=not self.verbose)
                
                # Install hooks
                subprocess.run([
                    "pre-commit", "install"
                ], cwd=self.project_root, check=True, capture_output=not self.verbose)
                
                self.logger.info("✅ Pre-commit hooks installed")
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"⚠️ Failed to setup pre-commit hooks: {e}")
                success = False
        
        return success
    
    def create_project_structure(self) -> bool:
        """Create missing project directories"""
        self.logger.info("Creating project structure...")
        
        # Required directories
        required_dirs = [
            "backend",
            "frontend",
            "docs",
            "config",
            "tests",
            "tools",
            "scripts",
            "logs"
        ]
        
        created_dirs = []
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(dir_name)
                    self.logger.info(f"Created directory: {dir_name}")
                except Exception as e:
                    self.logger.error(f"Failed to create directory {dir_name}: {e}")
                    return False
        
        if created_dirs:
            self.logger.info(f"✅ Created {len(created_dirs)} directories")
        else:
            self.logger.info("✅ All required directories exist")
        
        return True
    
    def setup_configuration_files(self) -> bool:
        """Setup configuration files"""
        self.logger.info("Setting up configuration files...")
        
        success = True
        
        # Create .env files if they don't exist
        env_files = [
            ("backend/.env", "# Backend environment variables\nDEBUG=true\nLOG_LEVEL=INFO\n"),
            ("frontend/.env", "# Frontend environment variables\nVITE_API_URL=http://localhost:8000\n")
        ]
        
        for env_file, default_content in env_files:
            env_path = self.project_root / env_file
            if not env_path.exists():
                try:
                    env_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(env_path, 'w') as f:
                        f.write(default_content)
                    self.logger.info(f"Created {env_file}")
                except Exception as e:
                    self.logger.error(f"Failed to create {env_file}: {e}")
                    success = False
        
        return success
    
    def validate_setup(self) -> bool:
        """Validate the setup"""
        self.logger.info("Validating environment setup...")
        
        health = self.environment_validator.run_full_validation()
        
        self.logger.info(f"Environment Health Score: {health.score:.1f}/100")
        
        if health.failed_checks > 0:
            self.logger.error(f"❌ {health.failed_checks} critical issues found")
            for result in health.results:
                if result.status == 'fail':
                    self.logger.error(f"  - {result.name}: {result.message}")
                    if result.fix_suggestion:
                        self.logger.error(f"    Fix: {result.fix_suggestion}")
            return False
        
        if health.warning_checks > 0:
            self.logger.warning(f"⚠️ {health.warning_checks} warnings found")
            for result in health.results:
                if result.status == 'warning':
                    self.logger.warning(f"  - {result.name}: {result.message}")
        
        self.logger.info(f"✅ Environment validation completed with {health.passed_checks} checks passed")
        return True
    
    def run_full_setup(self) -> bool:
        """Run complete development environment setup"""
        self.logger.info("🚀 Starting automated development environment setup...")
        self.logger.info(f"Project root: {self.project_root}")
        
        steps = [
            ("Creating project structure", self.create_project_structure),
            ("Setting up Python environment", self.setup_python_environment),
            ("Setting up Node.js environment", self.setup_nodejs_environment),
            ("Setting up development tools", self.setup_development_tools),
            ("Setting up configuration files", self.setup_configuration_files),
            ("Validating setup", self.validate_setup)
        ]
        
        for step_name, step_func in steps:
            self.logger.info(f"\n📋 {step_name}...")
            try:
                if not step_func():
                    self.logger.error(f"❌ Failed: {step_name}")
                    return False
                self.logger.info(f"✅ Completed: {step_name}")
            except Exception as e:
                self.logger.error(f"❌ Error in {step_name}: {e}")
                return False
        
        self.logger.info("\n🎉 Development environment setup completed successfully!")
        self.logger.info("\nNext steps:")
        self.logger.info("1. Activate virtual environment (if not already active)")
        self.logger.info("2. Start development servers:")
        self.logger.info("   - Backend: cd backend && python start_server.py")
        self.logger.info("   - Frontend: cd frontend && npm run dev")
        self.logger.info("3. Open http://localhost:3000 in your browser")
        
        return True
    
    def generate_setup_report(self, output_file: Path) -> None:
        """Generate setup report"""
        # Get dependency info
        all_deps = self.dependency_detector.get_all_dependencies()
        missing_deps = self.dependency_detector.get_missing_dependencies()
        
        # Get environment health
        health = self.environment_validator.run_full_validation()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'system_info': {
                'platform': platform.system(),
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            },
            'dependencies': {
                category: [
                    {
                        'name': dep.name,
                        'installed': dep.is_installed,
                        'version': dep.installed_version
                    }
                    for dep in deps
                ]
                for category, deps in all_deps.items()
            },
            'missing_dependencies': {
                category: [dep.name for dep in deps]
                for category, deps in missing_deps.items()
            },
            'environment_health': {
                'overall_status': health.overall_status,
                'score': health.score,
                'total_checks': health.total_checks,
                'passed_checks': health.passed_checks,
                'failed_checks': health.failed_checks,
                'warning_checks': health.warning_checks
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Setup report saved to {output_file}")

def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated development environment setup")
    parser.add_argument('--python', action='store_true', help='Setup Python environment only')
    parser.add_argument('--nodejs', action='store_true', help='Setup Node.js environment only')
    parser.add_argument('--tools', action='store_true', help='Setup development tools only')
    parser.add_argument('--structure', action='store_true', help='Create project structure only')
    parser.add_argument('--config', action='store_true', help='Setup configuration files only')
    parser.add_argument('--validate', action='store_true', help='Validate setup only')
    parser.add_argument('--report', type=str, help='Generate setup report to file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    setup = DevEnvironmentSetup(verbose=args.verbose)
    
    # Determine what to setup
    if not any([args.python, args.nodejs, args.tools, args.structure, args.config, args.validate]):
        # Run full setup
        success = setup.run_full_setup()
        sys.exit(0 if success else 1)
    
    # Run specific setup steps
    success = True
    
    if args.structure:
        success &= setup.create_project_structure()
    
    if args.python:
        success &= setup.setup_python_environment()
    
    if args.nodejs:
        success &= setup.setup_nodejs_environment()
    
    if args.tools:
        success &= setup.setup_development_tools()
    
    if args.config:
        success &= setup.setup_configuration_files()
    
    if args.validate:
        success &= setup.validate_setup()
    
    if args.report:
        setup.generate_setup_report(Path(args.report))
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()