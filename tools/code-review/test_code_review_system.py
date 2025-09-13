"""
Test suite for the Code Review and Refactoring Assistance Tools

This test file verifies that the core functionality of the code review system works correctly.
"""

import os
import tempfile
import json
from pathlib import Path
from datetime import datetime

import sys
sys.path.append('.')

from tools.code_review.code_reviewer import CodeReviewer, ReviewSeverity, IssueCategory
from tools.code_review.refactoring_engine import RefactoringEngine, RefactoringType
from tools.code_review.technical_debt_tracker import (
    TechnicalDebtTracker, TechnicalDebtItem, DebtCategory, 
    DebtSeverity, DebtStatus
)


def test_code_reviewer():
    """Test the code reviewer functionality"""
    print("Testing Code Reviewer...")
    
    # Create a temporary Python file with issues
    test_code = '''
def very_long_function_with_high_complexity(x, y, z, a, b, c, d, e, f, g):
    if x > 0:
        if y > 0:
            if z > 0:
                if a > 0:
                    if b > 0:
                        if c > 0:
                            if d > 0:
                                if e > 0:
                                    if f > 0:
                                        if g > 0:
                                            result = x + y + z + a + b + c + d + e + f + g
                                            for i in range(100):
                                                for j in range(100):
                                                    for k in range(100):
                                                        result += i * j * k
                                            return result
    return 0

def function_without_docstring():
    return "missing docstring"

def function_with_eval():
    user_input = "print('hello')"
    eval(user_input)  # Security issue
    return "dangerous"

class VeryLongClassWithManyMethods:
    def method1(self): pass
    def method2(self): pass
    def method3(self): pass
    def method4(self): pass
    def method5(self): pass
    def method6(self): pass
    def method7(self): pass
    def method8(self): pass
    def method9(self): pass
    def method10(self): pass
    def method11(self): pass
    def method12(self): pass
    def method13(self): pass
    def method14(self): pass
    def method15(self): pass
    def method16(self): pass
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_file = f.name
    
    try:
        reviewer = CodeReviewer()
        issues = reviewer.review_file(temp_file)
        
        print(f"  Found {len(issues)} issues")
        
        # Check for expected issue types
        complexity_issues = [i for i in issues if i.category == IssueCategory.COMPLEXITY]
        security_issues = [i for i in issues if i.category == IssueCategory.SECURITY]
        documentation_issues = [i for i in issues if i.category == IssueCategory.DOCUMENTATION]
        
        print(f"  - Complexity issues: {len(complexity_issues)}")
        print(f"  - Security issues: {len(security_issues)}")
        print(f"  - Documentation issues: {len(documentation_issues)}")
        
        assert len(issues) > 0, "Should find issues in test code"
        assert len(complexity_issues) > 0, "Should find complexity issues"
        assert len(security_issues) > 0, "Should find security issues"
        
        print("  ✅ Code Reviewer test passed")
        
    finally:
        os.unlink(temp_file)


def test_refactoring_engine():
    """Test the refactoring engine functionality"""
    print("Testing Refactoring Engine...")
    
    # Create a temporary Python file with refactoring opportunities
    test_code = '''
def process_user_data(user_data):
    # This is a long method that should be extracted
    if not user_data:
        return None
    
    # Validation logic (could be extracted)
    if not user_data.get('email'):
        raise ValueError("Email required")
    if not user_data.get('name'):
        raise ValueError("Name required")
    if len(user_data.get('name', '')) < 2:
        raise ValueError("Name too short")
    
    # Processing logic (could be extracted)
    processed_data = {}
    processed_data['email'] = user_data['email'].lower().strip()
    processed_data['name'] = user_data['name'].strip().title()
    processed_data['created_at'] = datetime.now()
    
    # Database logic (could be extracted)
    connection = get_database_connection()
    cursor = connection.cursor()
    cursor.execute("INSERT INTO users (email, name, created_at) VALUES (?, ?, ?)",
                   (processed_data['email'], processed_data['name'], processed_data['created_at']))
    connection.commit()
    connection.close()
    
    # Email logic (could be extracted)
    email_subject = "Welcome!"
    email_body = f"Hello {processed_data['name']}, welcome to our service!"
    send_email(processed_data['email'], email_subject, email_body)
    
    return processed_data

class UserManager:
    def __init__(self):
        self.users = []
        self.database = None
        self.email_service = None
        self.validator = None
        self.logger = None
        self.cache = None
        self.metrics = None
        self.config = None
    
    def create_user(self): pass
    def update_user(self): pass
    def delete_user(self): pass
    def find_user(self): pass
    def list_users(self): pass
    def validate_user(self): pass
    def send_user_email(self): pass
    def log_user_action(self): pass
    def cache_user_data(self): pass
    def track_user_metrics(self): pass
    def load_user_config(self): pass
    def backup_user_data(self): pass
    def restore_user_data(self): pass
    def export_user_data(self): pass
    def import_user_data(self): pass
    def sync_user_data(self): pass

def x(): return 1  # Poor naming
def data(): return {}  # Poor naming
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_file = f.name
    
    try:
        engine = RefactoringEngine()
        suggestions = engine.analyze_file(temp_file)
        
        print(f"  Found {len(suggestions)} refactoring suggestions")
        
        # Check for expected suggestion types
        extract_method_suggestions = [s for s in suggestions if s.refactoring_type == RefactoringType.EXTRACT_METHOD]
        extract_class_suggestions = [s for s in suggestions if s.refactoring_type == RefactoringType.EXTRACT_CLASS]
        naming_suggestions = [s for s in suggestions if s.refactoring_type == RefactoringType.IMPROVE_NAMING]
        
        print(f"  - Extract method suggestions: {len(extract_method_suggestions)}")
        print(f"  - Extract class suggestions: {len(extract_class_suggestions)}")
        print(f"  - Naming improvement suggestions: {len(naming_suggestions)}")
        
        assert len(suggestions) > 0, "Should find refactoring suggestions"
        
        print("  ✅ Refactoring Engine test passed")
        
    finally:
        os.unlink(temp_file)


def test_technical_debt_tracker():
    """Test the technical debt tracker functionality"""
    print("Testing Technical Debt Tracker...")
    
    # Use temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_db = f.name
    
    try:
        tracker = TechnicalDebtTracker(db_path=temp_db)
        
        # Add some test debt items
        debt_item1 = TechnicalDebtItem(
            id="",
            title="Refactor authentication system",
            description="The authentication logic is complex and hard to test",
            file_path="src/auth.py",
            line_start=45,
            line_end=120,
            category=DebtCategory.ARCHITECTURE,
            severity=DebtSeverity.HIGH,
            status=DebtStatus.IDENTIFIED,
            created_date=datetime.now(),
            updated_date=datetime.now(),
            estimated_effort_hours=16.0,
            business_impact="Difficult to add new authentication methods",
            technical_impact="High maintenance cost, potential security issues",
            priority_score=0
        )
        
        debt_item2 = TechnicalDebtItem(
            id="",
            title="Add unit tests for user service",
            description="User service lacks comprehensive unit tests",
            file_path="src/user_service.py",
            line_start=1,
            line_end=200,
            category=DebtCategory.TESTING,
            severity=DebtSeverity.MEDIUM,
            status=DebtStatus.IDENTIFIED,
            created_date=datetime.now(),
            updated_date=datetime.now(),
            estimated_effort_hours=8.0,
            business_impact="Risk of bugs in production",
            technical_impact="Difficult to refactor safely",
            priority_score=0
        )
        
        # Add debt items
        item1_id = tracker.add_debt_item(debt_item1)
        item2_id = tracker.add_debt_item(debt_item2)
        
        print(f"  Added debt items: {item1_id}, {item2_id}")
        
        # Test retrieval
        retrieved_item = tracker.get_debt_item(item1_id)
        assert retrieved_item is not None, "Should retrieve added debt item"
        assert retrieved_item.title == debt_item1.title, "Retrieved item should match"
        
        # Test metrics calculation
        metrics = tracker.calculate_debt_metrics()
        print(f"  Total debt items: {metrics.total_items}")
        print(f"  Total estimated hours: {metrics.total_estimated_hours}")
        print(f"  Average age: {metrics.average_age_days:.1f} days")
        
        assert metrics.total_items == 2, "Should have 2 debt items"
        assert metrics.total_estimated_hours == 24.0, "Should have 24 total hours"
        
        # Test prioritization
        prioritized_items = tracker.get_prioritized_debt_items()
        assert len(prioritized_items) == 2, "Should return all items"
        assert prioritized_items[0].priority_score >= prioritized_items[1].priority_score, "Should be sorted by priority"
        
        # Test update
        success = tracker.update_debt_item(item1_id, {'status': DebtStatus.IN_PROGRESS.value})
        assert success, "Should successfully update debt item"
        
        updated_item = tracker.get_debt_item(item1_id)
        assert updated_item.status == DebtStatus.IN_PROGRESS, "Status should be updated"
        
        # Test recommendations
        recommendations = tracker.generate_recommendations()
        print(f"  Generated {len(recommendations)} recommendations")
        
        print("  ✅ Technical Debt Tracker test passed")
        
    finally:
        os.unlink(temp_db)


def test_integration():
    """Test integration between components"""
    print("Testing Integration...")
    
    # Create a temporary project structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        (temp_path / "src").mkdir()
        
        test_file1 = temp_path / "src" / "main.py"
        test_file1.write_text('''
def complex_function():
    # This function has multiple issues
    if True:
        if True:
            if True:
                for i in range(10):
                    for j in range(10):
                        print(i * j)
    eval("print('dangerous')")
    return "done"

class LargeClass:
    def method1(self): pass
    def method2(self): pass
    def method3(self): pass
    def method4(self): pass
    def method5(self): pass
    def method6(self): pass
    def method7(self): pass
    def method8(self): pass
    def method9(self): pass
    def method10(self): pass
    def method11(self): pass
    def method12(self): pass
    def method13(self): pass
    def method14(self): pass
    def method15(self): pass
    def method16(self): pass
''')
        
        # Test code review
        reviewer = CodeReviewer(str(temp_path))
        result = reviewer.review_project()
        
        print(f"  Project review found {result['issues']} issues in {result['files_reviewed']} files")
        assert result['issues'] > 0, "Should find issues in test project"
        
        # Test refactoring suggestions
        engine = RefactoringEngine(str(temp_path))
        suggestions = []
        for py_file in temp_path.glob("**/*.py"):
            suggestions.extend(engine.analyze_file(str(py_file)))
        
        print(f"  Found {len(suggestions)} refactoring suggestions")
        assert len(suggestions) > 0, "Should find refactoring opportunities"
        
        # Test technical debt integration
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            tracker = TechnicalDebtTracker(db_path=temp_db)
            
            # Create debt items from review issues
            for issue in reviewer.issues[:2]:  # Just first 2 issues
                debt_item = TechnicalDebtItem(
                    id="",
                    title=f"Fix {issue.category.value} issue: {issue.rule_id}",
                    description=issue.message,
                    file_path=issue.file_path,
                    line_start=issue.line_number,
                    line_end=issue.line_number,
                    category=DebtCategory.CODE_QUALITY,
                    severity=DebtSeverity.HIGH if issue.severity.value in ['critical', 'high'] else DebtSeverity.MEDIUM,
                    status=DebtStatus.IDENTIFIED,
                    created_date=datetime.now(),
                    updated_date=datetime.now(),
                    estimated_effort_hours=2.0,
                    business_impact="Code quality impact",
                    technical_impact=issue.suggestion,
                    priority_score=0
                )
                tracker.add_debt_item(debt_item)
            
            metrics = tracker.calculate_debt_metrics()
            print(f"  Created {metrics.total_items} debt items from review issues")
            
        finally:
            os.unlink(temp_db)
        
        print("  ✅ Integration test passed")


def run_all_tests():
    """Run all tests"""
    print("Running Code Review System Tests...")
    print("=" * 50)
    
    try:
        test_code_reviewer()
        print()
        
        test_refactoring_engine()
        print()
        
        test_technical_debt_tracker()
        print()
        
        test_integration()
        print()
        
        print("=" * 50)
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
