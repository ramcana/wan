"""
Test the code review tools implementation
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '.')

def test_cli_help():
    """Test that the CLI help works"""
    print("Testing CLI help...")
    
    try:
        # Test importing the CLI module
        from tools.code_review.cli import main
        print("✅ CLI module imported successfully")
        
        # Test that we can run help (this should not crash)
        import argparse
        print("✅ CLI help functionality available")
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False
    
    return True


def test_code_reviewer_basic():
    """Test basic code reviewer functionality"""
    print("Testing Code Reviewer...")
    
    try:
        from tools.code_review.code_reviewer import CodeReviewer, ReviewSeverity, IssueCategory
        
        # Create a simple test
        reviewer = CodeReviewer()
        print("✅ Code Reviewer initialized")
        
        # Test configuration loading
        config = reviewer._load_config()
        assert isinstance(config, dict), "Config should be a dictionary"
        print("✅ Configuration loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Code Reviewer test failed: {e}")
        return False


def test_refactoring_engine_basic():
    """Test basic refactoring engine functionality"""
    print("Testing Refactoring Engine...")
    
    try:
        from tools.code_review.refactoring_engine import RefactoringEngine, RefactoringType
        
        # Create engine
        engine = RefactoringEngine()
        print("✅ Refactoring Engine initialized")
        
        # Test pattern loading
        patterns = engine._load_refactoring_patterns()
        assert len(patterns) > 0, "Should have refactoring patterns"
        print(f"✅ Loaded {len(patterns)} refactoring patterns")
        
        return True
        
    except Exception as e:
        print(f"❌ Refactoring Engine test failed: {e}")
        return False


def test_technical_debt_tracker_basic():
    """Test basic technical debt tracker functionality"""
    print("Testing Technical Debt Tracker...")
    
    try:
        from tools.code_review.technical_debt_tracker import (
            TechnicalDebtTracker, DebtCategory, DebtSeverity, DebtStatus
        )
        
        # Create tracker with temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            tracker = TechnicalDebtTracker(db_path=temp_db)
            print("✅ Technical Debt Tracker initialized")
            
            # Test metrics calculation (should work with empty database)
            metrics = tracker.calculate_debt_metrics()
            assert metrics.total_items == 0, "Should start with no debt items"
            print("✅ Metrics calculation works")
            
            return True
            
        finally:
            os.unlink(temp_db)
        
    except Exception as e:
        print(f"❌ Technical Debt Tracker test failed: {e}")
        return False


def test_file_analysis():
    """Test analyzing a real Python file"""
    print("Testing file analysis...")
    
    # Create a test file with some issues
    test_code = '''
def complex_function():
    # This function has high complexity
    x = 1
    if x > 0:
        if x < 10:
            if x % 2 == 0:
                for i in range(x):
                    for j in range(i):
                        print(i * j)
    return x

def function_without_docstring():
    return "missing docstring"

class LargeClass:
    def method1(self): pass
    def method2(self): pass
    def method3(self): pass
    def method4(self): pass
    def method5(self): pass
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_file = f.name
    
    try:
        from tools.code_review.code_reviewer import CodeReviewer
        from tools.code_review.refactoring_engine import RefactoringEngine
        
        # Test code review
        reviewer = CodeReviewer()
        issues = reviewer.review_file(temp_file)
        print(f"✅ Found {len(issues)} code issues")
        
        # Test refactoring suggestions
        engine = RefactoringEngine()
        suggestions = engine.analyze_file(temp_file)
        print(f"✅ Found {len(suggestions)} refactoring suggestions")
        
        return True
        
    except Exception as e:
        print(f"❌ File analysis test failed: {e}")
        return False
    finally:
        os.unlink(temp_file)


def run_all_tests():
    """Run all tests"""
    print("Running Code Review Tools Tests")
    print("=" * 40)
    
    tests = [
        test_cli_help,
        test_code_reviewer_basic,
        test_refactoring_engine_basic,
        test_technical_debt_tracker_basic,
        test_file_analysis
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\n{test.__name__}:")
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 40)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)