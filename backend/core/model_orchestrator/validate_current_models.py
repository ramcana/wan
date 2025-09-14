#!/usr/bin/env python3
"""
Validation script for the current models.toml file.

This script validates the current models.toml file and provides a summary
of the validation results, demonstrating that the validator works correctly.
"""

import os
import sys
from pathlib import Path

# Add the backend directory to the path for imports
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, backend_dir)

from validate_models_toml import ModelsTomlValidator


def main():
    """Main validation function."""
    # Find the project root and models.toml file
    current_dir = Path(__file__).parent
    while current_dir != current_dir.parent:
        if (current_dir / "pyproject.toml").exists() or (current_dir / "setup.py").exists():
            break
        current_dir = current_dir.parent
    
    models_toml_path = current_dir / "config" / "models.toml"
    
    print("🔍 Models.toml Validator - Current File Validation")
    print("=" * 60)
    print(f"Validating: {models_toml_path}")
    print()
    
    if not models_toml_path.exists():
        print("❌ ERROR: models.toml file not found!")
        return 1
    
    # Run validation
    validator = ModelsTomlValidator(str(models_toml_path))
    is_valid, errors, warnings = validator.validate()
    
    # Print detailed results
    print("📋 VALIDATION RESULTS:")
    print("-" * 30)
    
    if errors:
        print("❌ ERRORS FOUND:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        print()
    
    if warnings:
        print("⚠️  WARNINGS FOUND:")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
        print()
    
    # Summary
    if is_valid:
        print("✅ VALIDATION PASSED")
        print("   The models.toml file meets all validation criteria:")
        print("   • Schema version is supported")
        print("   • No duplicate model IDs or file paths")
        print("   • No path traversal vulnerabilities")
        print("   • Windows case sensitivity compatible")
        print("   • All required fields are present")
        print("   • File paths are safe and valid")
        
        if warnings:
            print(f"\n   Note: {len(warnings)} warnings found (non-critical issues)")
    else:
        print("❌ VALIDATION FAILED")
        print(f"   Found {len(errors)} critical errors that must be fixed")
        if warnings:
            print(f"   Also found {len(warnings)} warnings")
    
    print()
    print("🛠️  VALIDATION CRITERIA CHECKED:")
    print("   ✓ Schema version compatibility")
    print("   ✓ Model ID format validation")
    print("   ✓ Duplicate detection (models and file paths)")
    print("   ✓ Path traversal vulnerability detection")
    print("   ✓ Windows reserved name detection")
    print("   ✓ Case collision detection")
    print("   ✓ Required field validation")
    print("   ✓ File size and checksum validation")
    print("   ✓ TOML structure validation")
    
    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())