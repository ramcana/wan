"""Quick validation workflow for fast feedback"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_quick_validation() -> bool:
    """
    Run a fast validation suite for immediate feedback.
    Returns True if all checks pass, False otherwise.
    """
    
    results = {}
    
    # 1. Quick import check
    try:
        from tests.utils.import_fixer import ImportFixer
        fixer = ImportFixer()
        import_issues = fixer.find_critical_import_issues()
        results['imports'] = len(import_issues) == 0
        
        if import_issues:
            print(f"❌ Found {len(import_issues)} critical import issues")
        else:
            print("✅ Import validation passed")
            
    except Exception as e:
        print(f"⚠️ Import validation failed: {e}")
        results['imports'] = False
    
    # 2. Quick syntax check
    try:
        from tools.code_quality.quality_checker import QualityChecker
        checker = QualityChecker()
        syntax_results = checker.run_syntax_check()
        results['syntax'] = not syntax_results.has_errors
        
        if syntax_results.has_errors:
            print(f"❌ Found {len(syntax_results.errors)} syntax errors")
        else:
            print("✅ Syntax validation passed")
            
    except Exception as e:
        print(f"⚠️ Syntax validation failed: {e}")
        results['syntax'] = False
    
    # 3. Quick config validation
    try:
        from tools.config_manager.config_validator import ConfigValidator
        validator = ConfigValidator()
        config_results = validator.validate_critical_configs()
        results['config'] = not config_results.has_critical_errors
        
        if config_results.has_critical_errors:
            print(f"❌ Found {len(config_results.critical_errors)} critical config errors")
        else:
            print("✅ Config validation passed")
            
    except Exception as e:
        print(f"⚠️ Config validation failed: {e}")
        results['config'] = False
    
    # 4. Quick test smoke test
    try:
        from tests.utils.test_execution_engine import TestExecutionEngine
        engine = TestExecutionEngine()
        test_results = engine.run_smoke_tests()
        results['tests'] = test_results.success
        
        if not test_results.success:
            print(f"❌ Smoke tests failed: {test_results.failure_count} failures")
        else:
            print("✅ Smoke tests passed")
            
    except Exception as e:
        print(f"⚠️ Test validation failed: {e}")
        results['tests'] = False
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n📊 Quick Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All quick validations passed!")
        return True
    else:
        print("⚠️ Some validations failed - run specific commands for details")
        return False


def run_pre_commit_validation() -> bool:
    """
    Run validation suitable for pre-commit hooks.
    Fast checks that prevent broken commits.
    """
    
    print("🔍 Running pre-commit validation...")
    
    # Only run the most critical checks
    results = {}
    
    # Critical import check
    try:
        from tests.utils.import_fixer import ImportFixer
        fixer = ImportFixer()
        import_issues = fixer.find_critical_import_issues()
        results['imports'] = len(import_issues) == 0
    except:
        results['imports'] = False
    
    # Basic syntax check
    try:
        import ast
        import glob
        
        syntax_errors = 0
        for py_file in glob.glob("**/*.py", recursive=True):
            if "venv" in py_file or "__pycache__" in py_file:
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
            except SyntaxError:
                syntax_errors += 1
        
        results['syntax'] = syntax_errors == 0
    except:
        results['syntax'] = False
    
    # Quick linting with ruff (if available)
    try:
        import subprocess
        result = subprocess.run(['ruff', 'check', '--select', 'E9,F63,F7,F82'], 
                              capture_output=True, text=True)
        results['linting'] = result.returncode == 0
    except:
        results['linting'] = True  # Skip if ruff not available
    
    passed = sum(results.values())
    total = len(results)
    
    if passed == total:
        print("✅ Pre-commit validation passed!")
        return True
    else:
        print(f"❌ Pre-commit validation failed: {passed}/{total} checks passed")
        return False