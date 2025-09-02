#!/usr/bin/env python3
"""
Test Suite Validation Tests

This module tests that the comprehensive testing suite itself is working correctly.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.comprehensive.validation_framework import TestSuiteValidator
from tests.comprehensive.test_suite_runner import ComprehensiveTestRunner


class TestComprehensiveTestSuite:
    """Tests for the comprehensive test suite"""
    
    def test_validation_framework_works(self):
        """Test that the validation framework can run"""
        validator = TestSuiteValidator()
        report = validator.validate_all()
        
        # Should have run some validation checks
        assert report.total_checks > 0
        assert len(report.validation_results) > 0
        
        # Should have an overall status
        assert report.overall_status in ["PASS", "FAIL"]
    
    def test_test_suite_runner_initialization(self):
        """Test that the test suite runner can be initialized"""
        runner = ComprehensiveTestRunner()
        
        # Should have test suites defined
        assert len(runner.test_suites) > 0
        assert "e2e" in runner.test_suites
        assert "integration" in runner.test_suites
        assert "performance" in runner.test_suites
        assert "acceptance" in runner.test_suites
    
    def test_test_modules_can_be_imported(self):
        """Test that all test modules can be imported"""
        test_modules = [
            "tests.comprehensive.test_e2e_cleanup_quality_suite",
            "tests.integration.test_tool_interactions",
            "tests.performance.test_tool_performance", 
            "tests.acceptance.test_user_acceptance_scenarios"
        ]
        
        for module_name in test_modules:
            try:
                __import__(module_name)
                # If we get here, import succeeded
                assert True
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")
    
    def test_test_files_exist(self):
        """Test that all test files exist"""
        project_root = Path(__file__).parent.parent.parent
        
        test_files = [
            "tests/comprehensive/test_e2e_cleanup_quality_suite.py",
            "tests/integration/test_tool_interactions.py",
            "tests/performance/test_tool_performance.py",
            "tests/acceptance/test_user_acceptance_scenarios.py",
            "tests/comprehensive/test_suite_runner.py",
            "tests/comprehensive/validation_framework.py"
        ]
        
        for test_file in test_files:
            file_path = project_root / test_file
            assert file_path.exists(), f"Test file {test_file} does not exist"
            assert file_path.is_file(), f"Test file {test_file} is not a file"
    
    def test_comprehensive_suite_structure(self):
        """Test that the comprehensive suite has the expected structure"""
        project_root = Path(__file__).parent.parent.parent
        
        # Check directory structure
        expected_dirs = [
            "tests/comprehensive",
            "tests/integration", 
            "tests/performance",
            "tests/acceptance"
        ]
        
        for dir_path in expected_dirs:
            full_path = project_root / dir_path
            assert full_path.exists(), f"Directory {dir_path} does not exist"
            assert full_path.is_dir(), f"Path {dir_path} is not a directory"
    
    def test_requirements_coverage_validation(self):
        """Test that requirements coverage validation works"""
        validator = TestSuiteValidator()
        validator._validate_requirements_coverage()
        
        # Should have added a requirements coverage result
        coverage_results = [r for r in validator.validation_results if "Requirements Coverage" in r.check_name]
        assert len(coverage_results) > 0
        
        coverage_result = coverage_results[0]
        assert "covered" in coverage_result.details
        assert "missing" in coverage_result.details


if __name__ == "__main__":
    pytest.main([__file__, "-v"])