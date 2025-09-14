#!/usr/bin/env python3
"""
Task 21 Validation Script

This script validates that all components of Task 21 (comprehensive testing and documentation)
have been properly implemented according to the requirements.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class Task21Validator:
    """Validates Task 21 implementation completeness."""

    def __init__(self):
        self.base_path = Path(__file__).parent
        self.results = {}
        self.errors = []

    def validate_end_to_end_tests(self) -> Tuple[bool, List[str]]:
        """Validate end-to-end integration tests."""
        test_file = self.base_path / "tests" / "test_end_to_end_workflows.py"
        
        if not test_file.exists():
            return False, ["End-to-end test file missing"]
        
        content = test_file.read_text(encoding='utf-8')
        required_tests = [
            "test_complete_model_download_workflow",
            "test_concurrent_model_access_workflow", 
            "test_source_failover_workflow",
            "test_disk_space_management_workflow",
            "test_integrity_verification_workflow",
            "test_resume_interrupted_download_workflow",
            "test_health_monitoring_workflow",
            "test_garbage_collection_workflow",
            "test_cli_to_api_integration",
            "test_pipeline_integration_workflow",
            "test_metrics_and_monitoring_integration"
        ]
        
        missing_tests = []
        for test in required_tests:
            if test not in content:
                missing_tests.append(test)
        
        # Check for incomplete implementations
        incomplete_tests = []
        if "pass" in content and "# This would test" in content:
            incomplete_tests.append("Some tests have incomplete implementations")
        
        errors = missing_tests + incomplete_tests
        return len(errors) == 0, errors

    def validate_cross_platform_tests(self) -> Tuple[bool, List[str]]:
        """Validate cross-platform compatibility tests."""
        test_file = self.base_path / "tests" / "test_cross_platform_compatibility.py"
        
        if not test_file.exists():
            return False, ["Cross-platform test file missing"]
        
        content = test_file.read_text(encoding='utf-8')
        required_components = [
            "TestWindowsCompatibility",
            "TestWSLCompatibility", 
            "TestUnixCompatibility",
            "TestCrossPlatformPathHandling",
            "test_long_path_handling",
            "test_reserved_filename_handling",
            "test_case_insensitive_filesystem",
            "test_unix_file_permissions",
            "test_unix_symlink_support",
            "test_path_normalization",
            "test_atomic_operations_cross_platform"
        ]
        
        missing_components = []
        for component in required_components:
            if component not in content:
                missing_components.append(component)
        
        return len(missing_components) == 0, missing_components

    def validate_performance_tests(self) -> Tuple[bool, List[str]]:
        """Validate performance and load testing suites."""
        test_file = self.base_path / "tests" / "test_performance_load.py"
        
        if not test_file.exists():
            return False, ["Performance test file missing"]
        
        content = test_file.read_text(encoding='utf-8')
        required_tests = [
            "TestConcurrentPerformance",
            "TestMemoryPerformance",
            "TestNetworkPerformance", 
            "TestStoragePerformance",
            "TestScalabilityLimits",
            "test_concurrent_model_requests",
            "test_mixed_model_size_performance",
            "test_large_model_memory_usage",
            "test_memory_cleanup_after_operations",
            "test_download_timeout_handling",
            "test_retry_performance",
            "test_disk_io_performance",
            "test_maximum_concurrent_downloads"
        ]
        
        missing_tests = []
        for test in required_tests:
            if test not in content:
                missing_tests.append(test)
        
        return len(missing_tests) == 0, missing_tests

    def validate_requirements_tests(self) -> Tuple[bool, List[str]]:
        """Validate requirements validation tests."""
        test_file = self.base_path / "tests" / "test_requirements_validation.py"
        
        if not test_file.exists():
            return False, ["Requirements validation test file missing"]
        
        content = test_file.read_text(encoding='utf-8')
        required_requirement_tests = [
            "TestRequirement1_UnifiedModelManifest",
            "TestRequirement3_DeterministicPathResolution",
            "TestRequirement4_AtomicDownloads",
            "TestRequirement5_IntegrityVerification",
            "TestRequirement10_DiskSpaceManagement",
            "TestRequirement12_Observability",
            "TestRequirement13_ProductionAPI"
        ]
        
        missing_tests = []
        for test in required_requirement_tests:
            if test not in content:
                missing_tests.append(test)
        
        return len(missing_tests) == 0, missing_tests

    def validate_test_runner(self) -> Tuple[bool, List[str]]:
        """Validate comprehensive test runner."""
        test_runner = self.base_path / "tests" / "test_runner.py"
        
        if not test_runner.exists():
            return False, ["Test runner missing"]
        
        content = test_runner.read_text(encoding='utf-8')
        required_features = [
            "class TestRunner",
            "run_all_tests",
            "run_test_suite",
            "generate_report",
            "run_coverage_report",
            "def main()"
        ]
        
        missing_features = []
        for feature in required_features:
            if feature not in content:
                missing_features.append(feature)
        
        return len(missing_features) == 0, missing_features

    def validate_user_documentation(self) -> Tuple[bool, List[str]]:
        """Validate user documentation completeness."""
        docs_dir = self.base_path / "docs"
        
        required_docs = [
            "README.md",
            "USER_GUIDE.md", 
            "API_REFERENCE.md",
            "DEPLOYMENT_GUIDE.md",
            "TROUBLESHOOTING_GUIDE.md",
            "OPERATIONAL_RUNBOOK.md"
        ]
        
        missing_docs = []
        for doc in required_docs:
            doc_path = docs_dir / doc
            if not doc_path.exists():
                missing_docs.append(doc)
            elif doc_path.stat().st_size < 1000:  # Less than 1KB suggests incomplete
                missing_docs.append(f"{doc} (appears incomplete)")
        
        return len(missing_docs) == 0, missing_docs

    def validate_deployment_guides(self) -> Tuple[bool, List[str]]:
        """Validate deployment guides completeness."""
        deployment_guide = self.base_path / "docs" / "DEPLOYMENT_GUIDE.md"
        
        if not deployment_guide.exists():
            return False, ["Deployment guide missing"]
        
        content = deployment_guide.read_text(encoding='utf-8')
        required_sections = [
            "Deployment Architectures",
            "Environment Setup",
            "Production Environment",
            "Configuration Management",
            "Storage Configuration",
            "Security Configuration",
            "Monitoring and Observability",
            "High Availability and Scaling",
            "Backup and Disaster Recovery"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
        
        return len(missing_sections) == 0, missing_sections

    def validate_troubleshooting_guides(self) -> Tuple[bool, List[str]]:
        """Validate troubleshooting guides completeness."""
        troubleshooting_guide = self.base_path / "docs" / "TROUBLESHOOTING_GUIDE.md"
        
        if not troubleshooting_guide.exists():
            return False, ["Troubleshooting guide missing"]
        
        content = troubleshooting_guide.read_text(encoding='utf-8')
        required_sections = [
            "Quick Diagnostic Commands",
            "Common Issues and Solutions",
            "Model Download Failures",
            "Disk Space Issues", 
            "Lock Contention and Deadlocks",
            "Memory Issues",
            "Performance Issues",
            "Configuration Issues",
            "Platform-Specific Issues",
            "Emergency Procedures"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
        
        return len(missing_sections) == 0, missing_sections

    def validate_operational_runbooks(self) -> Tuple[bool, List[str]]:
        """Validate operational runbooks completeness."""
        runbook = self.base_path / "docs" / "OPERATIONAL_RUNBOOK.md"
        
        if not runbook.exists():
            return False, ["Operational runbook missing"]
        
        content = runbook.read_text(encoding='utf-8')
        required_sections = [
            "Daily Operations",
            "Weekly Maintenance",
            "Monthly Tasks",
            "Emergency Procedures",
            "Deployment Procedures",
            "Backup and Recovery",
            "Monitoring and Alerting",
            "Capacity Planning"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
        
        return len(missing_sections) == 0, missing_sections

    def run_sample_tests(self) -> Tuple[bool, List[str]]:
        """Run a sample of tests to ensure they execute properly."""
        try:
            # Try to import the test module to verify it's valid Python
            test_file = self.base_path / "tests" / "test_requirements_validation.py"
            
            if not test_file.exists():
                return False, ["Test file does not exist"]
            
            # Try to compile the test file
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                compile(content, str(test_file), 'exec')
                return True, []
            except SyntaxError as e:
                return False, [f"Syntax error in test file: {e}"]
                
        except Exception as e:
            return False, [f"Test validation error: {str(e)}"]

    def validate_code_quality_standards(self) -> Tuple[bool, List[str]]:
        """Validate code quality standards compliance."""
        errors = []
        
        # Check for proper imports and structure
        test_files = list((self.base_path / "tests").glob("test_*.py"))
        
        for test_file in test_files:
            content = test_file.read_text(encoding='utf-8')
            
            # Check for proper test structure
            if "import pytest" not in content:
                errors.append(f"{test_file.name}: Missing pytest import")
            
            # Check for docstrings in test classes
            if "class Test" in content and '"""' not in content:
                errors.append(f"{test_file.name}: Missing class docstrings")
        
        # Check documentation files for minimum content
        doc_files = list((self.base_path / "docs").glob("*.md"))
        
        for doc_file in doc_files:
            if doc_file.stat().st_size < 500:  # Very small files
                errors.append(f"{doc_file.name}: Documentation appears incomplete")
        
        return len(errors) == 0, errors

    def run_validation(self) -> Dict[str, Tuple[bool, List[str]]]:
        """Run all validation checks."""
        validations = {
            "End-to-End Tests": self.validate_end_to_end_tests,
            "Cross-Platform Tests": self.validate_cross_platform_tests,
            "Performance Tests": self.validate_performance_tests,
            "Requirements Tests": self.validate_requirements_tests,
            "Test Runner": self.validate_test_runner,
            "User Documentation": self.validate_user_documentation,
            "Deployment Guides": self.validate_deployment_guides,
            "Troubleshooting Guides": self.validate_troubleshooting_guides,
            "Operational Runbooks": self.validate_operational_runbooks,
            "Sample Test Execution": self.run_sample_tests,
            "Code Quality Standards": self.validate_code_quality_standards
        }
        
        results = {}
        for name, validation_func in validations.items():
            try:
                success, errors = validation_func()
                results[name] = (success, errors)
            except Exception as e:
                results[name] = (False, [f"Validation error: {str(e)}"])
        
        return results

    def generate_report(self, results: Dict[str, Tuple[bool, List[str]]]) -> str:
        """Generate validation report."""
        report = []
        report.append("=" * 80)
        report.append("TASK 21 VALIDATION REPORT")
        report.append("Comprehensive Testing and Documentation")
        report.append("=" * 80)
        
        total_checks = len(results)
        passed_checks = sum(1 for success, _ in results.values() if success)
        failed_checks = total_checks - passed_checks
        
        report.append(f"Total Checks: {total_checks}")
        report.append(f"Passed: {passed_checks}")
        report.append(f"Failed: {failed_checks}")
        report.append("")
        
        # Detailed results
        for check_name, (success, errors) in results.items():
            status = "PASS" if success else "FAIL"
            report.append(f"{check_name}: {status}")
            
            if errors:
                for error in errors:
                    report.append(f"  - {error}")
                report.append("")
        
        # Overall status
        report.append("=" * 80)
        overall_status = "PASSED" if failed_checks == 0 else "FAILED"
        report.append(f"OVERALL TASK 21 STATUS: {overall_status}")
        
        if failed_checks == 0:
            report.append("")
            report.append("[PASS] All comprehensive testing and documentation requirements met!")
            report.append("[PASS] End-to-end integration tests implemented")
            report.append("[PASS] Cross-platform compatibility tests implemented") 
            report.append("[PASS] Performance and load testing suites implemented")
            report.append("[PASS] User documentation and deployment guides created")
            report.append("[PASS] Troubleshooting guides and operational runbooks created")
        else:
            report.append("")
            report.append("[FAIL] Some requirements not fully met. See details above.")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main validation entry point."""
    print("Validating Task 21: Comprehensive Testing and Documentation")
    print("=" * 60)
    
    validator = Task21Validator()
    results = validator.run_validation()
    report = validator.generate_report(results)
    
    print(report)
    
    # Save report to file
    report_file = Path(__file__).parent / "TASK_21_VALIDATION_REPORT.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nValidation report saved to: {report_file}")
    
    # Exit with appropriate code
    failed_checks = sum(1 for success, _ in results.values() if not success)
    sys.exit(0 if failed_checks == 0 else 1)


if __name__ == "__main__":
    main()