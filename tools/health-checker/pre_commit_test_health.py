#!/usr/bin/env python3
"""
Pre-commit hook for test health checking.
Validates that tests are properly organized and functional.
"""

import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any


def run_command(cmd: List[str]) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


def check_test_structure() -> List[str]:
    """Check that test files are properly organized."""
    issues = []
    
    # Check for tests in wrong directories
    root_tests = list(Path("tests").glob("test_*.py"))
    if root_tests:
        issues.append(
            f"Found {len(root_tests)} test files in tests/ root. "
            "Tests should be in unit/, integration/, performance/, or e2e/ subdirectories."
        )
    
    # Check for missing __init__.py files
    test_dirs = ["tests/unit", "tests/integration", "tests/performance", "tests/e2e"]
    for test_dir in test_dirs:
        init_file = Path(test_dir) / "__init__.py"
        if Path(test_dir).exists() and not init_file.exists():
            issues.append(f"Missing __init__.py in {test_dir}")
    
    return issues


def check_test_imports() -> List[str]:
    """Check for import issues in test files."""
    issues = []
    
    test_files = []
    for pattern in ["tests/unit/*.py", "tests/integration/*.py", "tests/performance/*.py", "tests/e2e/*.py"]:
        test_files.extend(Path(".").glob(pattern))
    
    for test_file in test_files:
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for relative imports that might be broken
            if "from .." in content or "import .." in content:
                issues.append(f"{test_file}: Contains relative imports that may be broken")
                
            # Check for missing pytest import
            if "def test_" in content and "import pytest" not in content:
                issues.append(f"{test_file}: Contains test functions but missing pytest import")
                
        except Exception as e:
            issues.append(f"{test_file}: Could not read file - {e}")
    
    return issues


def run_quick_test_check() -> List[str]:
    """Run a quick syntax check on test files."""
    issues = []
    
    # Run python syntax check on test files
    test_files = []
    for pattern in ["tests/unit/*.py", "tests/integration/*.py"]:
        test_files.extend(Path(".").glob(pattern))
    
    for test_file in test_files:
        exit_code, stdout, stderr = run_command([
            sys.executable, "-m", "py_compile", str(test_file)
        ])
        
        if exit_code != 0:
            issues.append(f"{test_file}: Syntax error - {stderr}")
    
    return issues


def check_test_configuration() -> List[str]:
    """Check test configuration files."""
    issues = []
    
    # Check for test config file
    test_config = Path("tests/config/test-config.yaml")
    if not test_config.exists():
        issues.append("Missing test configuration file: tests/config/test-config.yaml")
    
    # Check for pytest configuration
    pytest_configs = [
        Path("tests/config/pytest.ini"),
        Path("pytest.ini"),
        Path("pyproject.toml")
    ]
    
    if not any(config.exists() for config in pytest_configs):
        issues.append("Missing pytest configuration (pytest.ini or pyproject.toml)")
    
    return issues


def main() -> int:
    """Main pre-commit hook function."""
    print("🔍 Running test health check...")
    
    all_issues = []
    
    # Run all checks
    all_issues.extend(check_test_structure())
    all_issues.extend(check_test_imports())
    all_issues.extend(run_quick_test_check())
    all_issues.extend(check_test_configuration())
    
    if all_issues:
        print("❌ Test health check failed:")
        for issue in all_issues:
            print(f"  • {issue}")
        
        print("\n💡 Recommendations:")
        print("  • Move test files to appropriate subdirectories (unit/, integration/, etc.)")
        print("  • Fix import issues in test files")
        print("  • Ensure all test directories have __init__.py files")
        print("  • Add missing test configuration files")
        
        return 1
    
    print("✅ Test health check passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())