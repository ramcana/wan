#!/usr/bin/env python3
"""
Setup script for installing pre-commit hooks.
This script installs and configures pre-commit hooks for the project.
"""

import sys
import subprocess
import os
from pathlib import Path


def run_command(cmd, check=True):
    """Run a command and handle errors."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required for pre-commit hooks")
        return False
    print(f"✅ Python version {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True


def install_pre_commit():
    """Install pre-commit package."""
    print("📦 Installing pre-commit...")
    
    # Try to install pre-commit
    success = run_command([sys.executable, "-m", "pip", "install", "pre-commit"])
    
    if not success:
        print("❌ Failed to install pre-commit")
        print("💡 Try running: pip install pre-commit")
        return False
    
    print("✅ pre-commit installed successfully")
    return True


def install_hooks():
    """Install pre-commit hooks."""
    print("🔧 Installing pre-commit hooks...")
    
    # Check if .pre-commit-config.yaml exists
    config_file = Path(".pre-commit-config.yaml")
    if not config_file.exists():
        print("❌ .pre-commit-config.yaml not found")
        return False
    
    # Install hooks
    success = run_command(["pre-commit", "install"])
    
    if not success:
        print("❌ Failed to install pre-commit hooks")
        return False
    
    # Install pre-push hooks
    success = run_command(["pre-commit", "install", "--hook-type", "pre-push"])
    
    if not success:
        print("⚠️  Failed to install pre-push hooks (optional)")
    
    print("✅ Pre-commit hooks installed successfully")
    return True


def make_scripts_executable():
    """Make hook scripts executable on Unix systems."""
    if os.name != 'posix':
        return True
    
    print("🔧 Making hook scripts executable...")
    
    hook_scripts = [
        "tools/health-checker/pre_commit_test_health.py",
        "tools/config-manager/pre_commit_config_validation.py",
        "tools/doc-generator/pre_commit_link_check.py",
        "tools/health-checker/pre_commit_import_validation.py"
    ]
    
    for script in hook_scripts:
        script_path = Path(script)
        if script_path.exists():
            try:
                script_path.chmod(0o755)
                print(f"  Made {script} executable")
            except Exception as e:
                print(f"  ⚠️  Could not make {script} executable: {e}")
    
    return True


def run_initial_check():
    """Run pre-commit on all files to check setup."""
    print("🧪 Running initial pre-commit check...")
    
    # Run pre-commit on all files
    success = run_command(["pre-commit", "run", "--all-files"], check=False)
    
    if success:
        print("✅ Initial pre-commit check passed")
    else:
        print("⚠️  Initial pre-commit check found issues")
        print("💡 This is normal for first setup. Run 'pre-commit run --all-files' to fix auto-fixable issues")
    
    return True


def setup_git_config():
    """Set up git configuration for better pre-commit experience."""
    print("⚙️  Setting up git configuration...")
    
    # Set up git hooks path (optional)
    run_command(["git", "config", "core.hooksPath", ".git/hooks"], check=False)
    
    # Enable pre-commit in git config
    run_command(["git", "config", "pre-commit.enabled", "true"], check=False)
    
    print("✅ Git configuration updated")
    return True


def main():
    """Main setup function."""
    print("🚀 Setting up pre-commit hooks for WAN22 Video Generation System")
    print("=" * 60)
    
    # Check prerequisites
    if not check_python_version():
        return 1
    
    # Install pre-commit
    if not install_pre_commit():
        return 1
    
    # Make scripts executable
    if not make_scripts_executable():
        return 1
    
    # Install hooks
    if not install_hooks():
        return 1
    
    # Set up git config
    if not setup_git_config():
        return 1
    
    # Run initial check
    if not run_initial_check():
        return 1
    
    print("\n" + "=" * 60)
    print("✅ Pre-commit hooks setup completed successfully!")
    print("\n📋 What happens now:")
    print("  • Pre-commit hooks will run automatically on 'git commit'")
    print("  • Hooks will check code quality, tests, config, and documentation")
    print("  • Some issues will be auto-fixed, others will need manual attention")
    print("\n🔧 Useful commands:")
    print("  • Run hooks manually: pre-commit run --all-files")
    print("  • Skip hooks (emergency): git commit --no-verify")
    print("  • Update hooks: pre-commit autoupdate")
    print("  • Uninstall hooks: pre-commit uninstall")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())