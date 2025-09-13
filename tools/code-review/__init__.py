"""
Code Review and Refactoring Assistance Tools

This package provides comprehensive tools for automated code review, refactoring suggestions,
and technical debt management to help development teams maintain high code quality.

Main Components:
- CodeReviewer: Automated code quality analysis
- RefactoringEngine: Intelligent refactoring suggestions  
- TechnicalDebtTracker: Systematic technical debt management
- CLI: Command-line interface for all tools

Example Usage:
    from tools.code_review import CodeReviewer, RefactoringEngine, TechnicalDebtTracker
    
    # Code review
    reviewer = CodeReviewer(project_root=".")
    issues = reviewer.review_file("src/main.py")
    
    # Refactoring suggestions
    engine = RefactoringEngine(project_root=".")
    suggestions = engine.analyze_file("src/main.py")
    
    # Technical debt tracking
    tracker = TechnicalDebtTracker()
    metrics = tracker.calculate_debt_metrics()
"""

from tools..code_reviewer import (
    CodeReviewer,
    CodeIssue,
    ReviewSeverity,
    IssueCategory,
    ComplexityAnalyzer,
    MaintainabilityAnalyzer,
    PerformanceAnalyzer,
    SecurityAnalyzer
)

from tools..refactoring_engine import (
    RefactoringEngine,
    RefactoringType,
    RefactoringPattern,
    RefactoringSuggestion
)

from tools..technical_debt_tracker import (
    TechnicalDebtTracker,
    TechnicalDebtItem,
    DebtCategory,
    DebtSeverity,
    DebtStatus,
    DebtMetrics,
    DebtRecommendation
)

__version__ = "1.0.0"
__author__ = "WAN22 Development Team"
__description__ = "Code Review and Refactoring Assistance Tools"

__all__ = [
    # Code Reviewer
    "CodeReviewer",
    "CodeIssue", 
    "ReviewSeverity",
    "IssueCategory",
    "ComplexityAnalyzer",
    "MaintainabilityAnalyzer", 
    "PerformanceAnalyzer",
    "SecurityAnalyzer",
    
    # Refactoring Engine
    "RefactoringEngine",
    "RefactoringType",
    "RefactoringPattern",
    "RefactoringSuggestion",
    
    # Technical Debt Tracker
    "TechnicalDebtTracker",
    "TechnicalDebtItem",
    "DebtCategory",
    "DebtSeverity", 
    "DebtStatus",
    "DebtMetrics",
    "DebtRecommendation"
]
