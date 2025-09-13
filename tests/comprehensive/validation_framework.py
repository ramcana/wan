#!/usr/bin/env python3
"""
Validation Framework for Comprehensive Testing Suite

This module provides validation and verification that all test components
are working correctly and meeting requirements.

Requirements covered: 1.1, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6
"""

import pytest
import sys
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from dataclasses import dataclass

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


@dataclass
class ValidationResult:
    """Result of validation check"""
    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any]


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    total_checks: int
    passed_checks: int
    failed_checks: int
    validation_results: List[ValidationResult]
    overall_status: str
    
    def __post_init__(self):
        self.overall_status = "PASS" if self.failed_checks == 0 else "FAIL"


class TestSuiteValidator:
    """Validates comprehensive test suite components"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.test_modules = {
            "e2e": "tests.comprehensive.test_e2e_cleanup_quality_suite",
            "integration": "tests.integration.test_tool_interactions", 
            "performance": "tests.performance.test_tool_performance",
            "acceptance": "tests.acceptance.test_user_acceptance_scenarios"
        }
        self.validation_results = []
    
    def validate_all(self) -> ValidationReport:
        """Run all validation checks"""
        self.validation_results = []
        
        # Module validation
        self._validate_test_modules()
        
        # Test class validation
        self._validate_test_classes()
        
        # Test method validation
        self._validate_test_methods()
        
        # Fixture validation
        self._validate_fixtures()
        
        # Requirements coverage validation
        self._validate_requirements_coverage()
        
        # Performance validation
        self._validate_performance_characteristics()
        
        # Integration validation
        self._validate_integration_points()
        
        # Generate report
        passed = len([r for r in self.validation_results if r.passed])
        failed = len([r for r in self.validation_results if not r.passed])
        
        return ValidationReport(
            total_checks=len(self.validation_results),
            passed_checks=passed,
            failed_checks=failed,
            validation_results=self.validation_results,
            overall_status=""  # Will be set in __post_init__
        )
    
    def _validate_test_modules(self):
        """Validate that all test modules can be imported"""
        for suite_name, module_name in self.test_modules.items():
            try:
                module = importlib.import_module(module_name)
                
                self.validation_results.append(ValidationResult(
                    check_name=f"Module Import - {suite_name}",
                    passed=True,
                    message=f"Successfully imported {module_name}",
                    details={"module": module_name, "file": str(module.__file__)}
                ))
                
            except ImportError as e:
                self.validation_results.append(ValidationResult(
                    check_name=f"Module Import - {suite_name}",
                    passed=False,
                    message=f"Failed to import {module_name}: {str(e)}",
                    details={"module": module_name, "error": str(e)}
                ))
    
    def _validate_test_classes(self):
        """Validate test class structure and naming"""
        expected_patterns = {
            "e2e": ["TestE2E", "E2ETest", "Test.*E2E"],
            "integration": ["TestIntegration", "Integration", "Test.*Integration"],
            "performance": ["TestPerformance", "Performance", "Test.*Performance"],
            "acceptance": ["TestAcceptance", "Acceptance", "Test.*User"]
        }
        
        for suite_name, module_name in self.test_modules.items():
            try:
                module = importlib.import_module(module_name)
                
                # Find test classes
                test_classes = []
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and name.startswith("Test"):
                        test_classes.append(name)
                
                if test_classes:
                    self.validation_results.append(ValidationResult(
                        check_name=f"Test Classes - {suite_name}",
                        passed=True,
                        message=f"Found {len(test_classes)} test classes",
                        details={"classes": test_classes}
                    ))
                else:
                    self.validation_results.append(ValidationResult(
                        check_name=f"Test Classes - {suite_name}",
                        passed=False,
                        message="No test classes found",
                        details={"classes": []}
                    ))
                    
            except Exception as e:
                self.validation_results.append(ValidationResult(
                    check_name=f"Test Classes - {suite_name}",
                    passed=False,
                    message=f"Error analyzing test classes: {str(e)}",
                    details={"error": str(e)}
                ))
    
    def _validate_test_methods(self):
        """Validate test method structure and coverage"""
        for suite_name, module_name in self.test_modules.items():
            try:
                module = importlib.import_module(module_name)
                
                total_test_methods = 0
                test_methods_by_class = {}
                
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and name.startswith("Test"):
                        methods = []
                        for method_name, method_obj in inspect.getmembers(obj):
                            if method_name.startswith("test_") and callable(method_obj):
                                methods.append(method_name)
                                total_test_methods += 1
                        
                        test_methods_by_class[name] = methods
                
                if total_test_methods >= 3:  # Minimum expected test methods
                    self.validation_results.append(ValidationResult(
                        check_name=f"Test Methods - {suite_name}",
                        passed=True,
                        message=f"Found {total_test_methods} test methods",
                        details={"total_methods": total_test_methods, "by_class": test_methods_by_class}
                    ))
                else:
                    self.validation_results.append(ValidationResult(
                        check_name=f"Test Methods - {suite_name}",
                        passed=False,
                        message=f"Insufficient test methods: {total_test_methods} (minimum 3)",
                        details={"total_methods": total_test_methods, "by_class": test_methods_by_class}
                    ))
                    
            except Exception as e:
                self.validation_results.append(ValidationResult(
                    check_name=f"Test Methods - {suite_name}",
                    passed=False,
                    message=f"Error analyzing test methods: {str(e)}",
                    details={"error": str(e)}
                ))
    
    def _validate_fixtures(self):
        """Validate pytest fixtures are properly defined"""
        for suite_name, module_name in self.test_modules.items():
            try:
                module = importlib.import_module(module_name)
                
                fixtures = []
                for name, obj in inspect.getmembers(module):
                    if hasattr(obj, '_pytestfixturefunction'):
                        fixtures.append(name)
                
                self.validation_results.append(ValidationResult(
                    check_name=f"Fixtures - {suite_name}",
                    passed=True,
                    message=f"Found {len(fixtures)} fixtures",
                    details={"fixtures": fixtures}
                ))
                
            except Exception as e:
                self.validation_results.append(ValidationResult(
                    check_name=f"Fixtures - {suite_name}",
                    passed=False,
                    message=f"Error analyzing fixtures: {str(e)}",
                    details={"error": str(e)}
                ))
    
    def _validate_requirements_coverage(self):
        """Validate that tests cover all specified requirements"""
        required_requirements = {
            "1.1", "1.6", "2.6", "3.6", "4.6", "5.6", "6.6"
        }
        
        covered_requirements = set()
        
        for suite_name, module_name in self.test_modules.items():
            try:
                module = importlib.import_module(module_name)
                
                # Check module docstring for requirements
                if module.__doc__:
                    for req in required_requirements:
                        if req in module.__doc__:
                            covered_requirements.add(req)
                
                # Check class and method docstrings
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and name.startswith("Test"):
                        if obj.__doc__:
                            for req in required_requirements:
                                if req in obj.__doc__:
                                    covered_requirements.add(req)
                        
                        # Check method docstrings
                        for method_name, method_obj in inspect.getmembers(obj):
                            if method_name.startswith("test_") and callable(method_obj):
                                if method_obj.__doc__:
                                    for req in required_requirements:
                                        if req in method_obj.__doc__:
                                            covered_requirements.add(req)
                
            except Exception as e:
                pass  # Continue checking other modules
        
        missing_requirements = required_requirements - covered_requirements
        
        if not missing_requirements:
            self.validation_results.append(ValidationResult(
                check_name="Requirements Coverage",
                passed=True,
                message="All requirements are covered by tests",
                details={"covered": list(covered_requirements), "missing": []}
            ))
        else:
            self.validation_results.append(ValidationResult(
                check_name="Requirements Coverage",
                passed=False,
                message=f"Missing coverage for requirements: {missing_requirements}",
                details={"covered": list(covered_requirements), "missing": list(missing_requirements)}
            ))
    
    def _validate_performance_characteristics(self):
        """Validate performance test characteristics"""
        try:
            module = importlib.import_module("tests.performance.test_tool_performance")
            
            # Check for performance-related classes and methods
            performance_indicators = [
                "performance", "benchmark", "timing", "memory", "cpu", 
                "throughput", "scalability", "concurrent"
            ]
            
            found_indicators = []
            
            for name, obj in inspect.getmembers(module):
                name_lower = name.lower()
                if any(indicator in name_lower for indicator in performance_indicators):
                    found_indicators.append(name)
                
                if inspect.isclass(obj) and name.startswith("Test"):
                    for method_name, method_obj in inspect.getmembers(obj):
                        method_name_lower = method_name.lower()
                        if any(indicator in method_name_lower for indicator in performance_indicators):
                            found_indicators.append(f"{name}.{method_name}")
            
            if found_indicators:
                self.validation_results.append(ValidationResult(
                    check_name="Performance Characteristics",
                    passed=True,
                    message=f"Found {len(found_indicators)} performance-related components",
                    details={"indicators": found_indicators}
                ))
            else:
                self.validation_results.append(ValidationResult(
                    check_name="Performance Characteristics",
                    passed=False,
                    message="No performance-related components found",
                    details={"indicators": []}
                ))
                
        except Exception as e:
            self.validation_results.append(ValidationResult(
                check_name="Performance Characteristics",
                passed=False,
                message=f"Error validating performance characteristics: {str(e)}",
                details={"error": str(e)}
            ))
    
    def _validate_integration_points(self):
        """Validate integration test points"""
        try:
            module = importlib.import_module("tests.integration.test_tool_interactions")
            
            # Check for integration-related patterns
            integration_patterns = [
                "workflow", "interaction", "integration", "orchestrat", 
                "data_flow", "state", "concurrent"
            ]
            
            found_patterns = []
            
            for name, obj in inspect.getmembers(module):
                name_lower = name.lower()
                if any(pattern in name_lower for pattern in integration_patterns):
                    found_patterns.append(name)
                
                if inspect.isclass(obj) and name.startswith("Test"):
                    for method_name, method_obj in inspect.getmembers(obj):
                        method_name_lower = method_name.lower()
                        if any(pattern in method_name_lower for pattern in integration_patterns):
                            found_patterns.append(f"{name}.{method_name}")
            
            if found_patterns:
                self.validation_results.append(ValidationResult(
                    check_name="Integration Points",
                    passed=True,
                    message=f"Found {len(found_patterns)} integration-related components",
                    details={"patterns": found_patterns}
                ))
            else:
                self.validation_results.append(ValidationResult(
                    check_name="Integration Points",
                    passed=False,
                    message="No integration-related components found",
                    details={"patterns": []}
                ))
                
        except Exception as e:
            self.validation_results.append(ValidationResult(
                check_name="Integration Points",
                passed=False,
                message=f"Error validating integration points: {str(e)}",
                details={"error": str(e)}
            ))
    
    def print_validation_report(self, report: ValidationReport):
        """Print validation report to console"""
        print(f"\n{'='*80}")
        print("TEST SUITE VALIDATION REPORT")
        print(f"{'='*80}")
        
        print(f"Overall Status: {report.overall_status}")
        print(f"Total Checks: {report.total_checks}")
        print(f"Passed: {report.passed_checks}")
        print(f"Failed: {report.failed_checks}")
        
        print(f"\n{'='*60}")
        print("VALIDATION RESULTS")
        print(f"{'='*60}")
        
        for result in report.validation_results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"{status} {result.check_name}")
            print(f"  {result.message}")
            
            if not result.passed and result.details:
                print(f"  Details: {result.details}")
            
            print()
        
        print(f"{'='*80}")


def validate_comprehensive_test_suite():
    """Main validation function"""
    validator = TestSuiteValidator()
    report = validator.validate_all()
    validator.print_validation_report(report)
    
    return report.overall_status == "PASS"


if __name__ == "__main__":
    success = validate_comprehensive_test_suite()
    sys.exit(0 if success else 1)
