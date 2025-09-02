"""
Test suite health checker
"""

import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

from ..health_models import ComponentHealth, HealthIssue, HealthCategory, Severity, HealthConfig


class TestHealthChecker:
    """Checks the health of the test suite"""
    
    def __init__(self, config: HealthConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def check_health(self) -> ComponentHealth:
        """Check test suite health"""
        issues = []
        metrics = {}
        
        # Check if test directory exists
        if not self.config.test_directory.exists():
            issues.append(HealthIssue(
                severity=Severity.CRITICAL,
                category=HealthCategory.TESTS,
                title="Test Directory Missing",
                description=f"Test directory {self.config.test_directory} does not exist",
                affected_components=["test_suite"],
                remediation_steps=[
                    f"Create test directory: mkdir {self.config.test_directory}",
                    "Add basic test structure and configuration"
                ]
            ))
            return ComponentHealth(
                component_name="test_suite",
                category=HealthCategory.TESTS,
                score=0.0,
                status="critical",
                issues=issues,
                metrics=metrics
            )
        
        # Run test discovery
        test_files = self._discover_tests()
        metrics["total_test_files"] = len(test_files)
        
        if len(test_files) == 0:
            issues.append(HealthIssue(
                severity=Severity.HIGH,
                category=HealthCategory.TESTS,
                title="No Test Files Found",
                description="No test files were discovered in the test directory",
                affected_components=["test_suite"],
                remediation_steps=[
                    "Add test files following naming convention (test_*.py)",
                    "Ensure tests are properly structured"
                ]
            ))
        
        # Check for broken tests
        broken_tests = self._check_broken_tests()
        metrics["broken_tests"] = len(broken_tests)
        
        if broken_tests:
            issues.append(HealthIssue(
                severity=Severity.HIGH,
                category=HealthCategory.TESTS,
                title="Broken Tests Detected",
                description=f"Found {len(broken_tests)} broken test files",
                affected_components=broken_tests,
                remediation_steps=[
                    "Fix syntax errors in test files",
                    "Update import statements",
                    "Remove or fix failing tests"
                ]
            ))
        
        # Run test execution if possible
        test_results = self._run_test_suite()
        if test_results:
            metrics.update(test_results)
            
            # Check pass rate
            pass_rate = test_results.get("pass_rate", 0)
            if pass_rate < 80:
                severity = Severity.CRITICAL if pass_rate < 50 else Severity.HIGH
                issues.append(HealthIssue(
                    severity=severity,
                    category=HealthCategory.TESTS,
                    title="Low Test Pass Rate",
                    description=f"Test pass rate is {pass_rate:.1f}% (target: 80%+)",
                    affected_components=["test_suite"],
                    remediation_steps=[
                        "Fix failing tests",
                        "Update tests for code changes",
                        "Remove obsolete tests"
                    ]
                ))
            
            # Check execution time
            execution_time = test_results.get("execution_time", 0)
            if execution_time > 900:  # 15 minutes
                issues.append(HealthIssue(
                    severity=Severity.MEDIUM,
                    category=HealthCategory.TESTS,
                    title="Slow Test Execution",
                    description=f"Test suite takes {execution_time:.1f}s (target: <900s)",
                    affected_components=["test_suite"],
                    remediation_steps=[
                        "Optimize slow tests",
                        "Use test parallelization",
                        "Mock external dependencies"
                    ]
                ))
        
        # Check test coverage
        coverage_data = self._check_test_coverage()
        if coverage_data:
            metrics.update(coverage_data)
            
            coverage_percent = coverage_data.get("coverage_percent", 0)
            if coverage_percent < 70:
                severity = Severity.HIGH if coverage_percent < 50 else Severity.MEDIUM
                issues.append(HealthIssue(
                    severity=severity,
                    category=HealthCategory.TESTS,
                    title="Low Test Coverage",
                    description=f"Code coverage is {coverage_percent:.1f}% (target: 70%+)",
                    affected_components=["test_suite"],
                    remediation_steps=[
                        "Add tests for uncovered code",
                        "Focus on critical path coverage",
                        "Remove dead code"
                    ]
                ))
        
        # Calculate score
        score = self._calculate_test_score(metrics, issues)
        status = self._determine_status(score)
        
        return ComponentHealth(
            component_name="test_suite",
            category=HealthCategory.TESTS,
            score=score,
            status=status,
            issues=issues,
            metrics=metrics
        )
    
    def _discover_tests(self) -> List[str]:
        """Discover test files"""
        test_files = []
        
        try:
            # Find Python test files
            for pattern in ["test_*.py", "*_test.py"]:
                test_files.extend([
                    str(f.relative_to(self.config.project_root))
                    for f in self.config.test_directory.rglob(pattern)
                ])
        except Exception as e:
            self.logger.warning(f"Failed to discover tests: {e}")
        
        return test_files
    
    def _check_broken_tests(self) -> List[str]:
        """Check for broken test files"""
        broken_tests = []
        
        try:
            # Try to import each test file to check for syntax errors
            for test_file in self._discover_tests():
                try:
                    # Simple syntax check by attempting to compile
                    with open(self.config.project_root / test_file, 'r') as f:
                        compile(f.read(), test_file, 'exec')
                except SyntaxError:
                    broken_tests.append(test_file)
                except Exception:
                    # Other import errors might indicate broken tests
                    broken_tests.append(test_file)
        except Exception as e:
            self.logger.warning(f"Failed to check broken tests: {e}")
        
        return broken_tests
    
    def _run_test_suite(self) -> Dict[str, Any]:
        """Run the test suite and collect metrics"""
        try:
            # Try pytest first
            result = subprocess.run([
                "python", "-m", "pytest", 
                str(self.config.test_directory),
                "--tb=no", "--quiet", "--json-report", "--json-report-file=/tmp/test_report.json"
            ], capture_output=True, text=True, timeout=300, cwd=self.config.project_root)
            
            if result.returncode == 0 or result.returncode == 1:  # 0 = success, 1 = some failures
                # Try to parse JSON report
                try:
                    with open("/tmp/test_report.json", 'r') as f:
                        report = json.load(f)
                    
                    total = report.get("summary", {}).get("total", 0)
                    passed = report.get("summary", {}).get("passed", 0)
                    failed = report.get("summary", {}).get("failed", 0)
                    
                    return {
                        "total_tests": total,
                        "passed_tests": passed,
                        "failed_tests": failed,
                        "pass_rate": (passed / total * 100) if total > 0 else 0,
                        "execution_time": report.get("duration", 0)
                    }
                except:
                    pass
            
            # Fallback: try unittest
            result = subprocess.run([
                "python", "-m", "unittest", "discover", 
                "-s", str(self.config.test_directory), "-v"
            ], capture_output=True, text=True, timeout=300, cwd=self.config.project_root)
            
            if result.returncode is not None:
                # Parse unittest output (basic parsing)
                output = result.stderr + result.stdout
                lines = output.split('\n')
                
                # Look for summary line
                for line in lines:
                    if "Ran" in line and "test" in line:
                        # Extract basic info from unittest output
                        return {
                            "test_runner": "unittest",
                            "execution_successful": result.returncode == 0,
                            "output_lines": len(lines)
                        }
            
        except subprocess.TimeoutExpired:
            return {"error": "Test execution timed out"}
        except Exception as e:
            self.logger.warning(f"Failed to run test suite: {e}")
        
        return {}
    
    def _check_test_coverage(self) -> Dict[str, Any]:
        """Check test coverage"""
        try:
            # Try to run coverage
            result = subprocess.run([
                "python", "-m", "coverage", "run", "-m", "pytest", 
                str(self.config.test_directory)
            ], capture_output=True, text=True, timeout=300, cwd=self.config.project_root)
            
            if result.returncode == 0:
                # Get coverage report
                report_result = subprocess.run([
                    "python", "-m", "coverage", "report", "--format=json"
                ], capture_output=True, text=True, cwd=self.config.project_root)
                
                if report_result.returncode == 0:
                    coverage_data = json.loads(report_result.stdout)
                    return {
                        "coverage_percent": coverage_data.get("totals", {}).get("percent_covered", 0),
                        "lines_covered": coverage_data.get("totals", {}).get("covered_lines", 0),
                        "lines_total": coverage_data.get("totals", {}).get("num_statements", 0)
                    }
        except Exception as e:
            self.logger.warning(f"Failed to check test coverage: {e}")
        
        return {}
    
    def _calculate_test_score(self, metrics: Dict[str, Any], issues: List[HealthIssue]) -> float:
        """Calculate test health score"""
        base_score = 100.0
        
        # Deduct points for issues
        for issue in issues:
            if issue.severity == Severity.CRITICAL:
                base_score -= 30
            elif issue.severity == Severity.HIGH:
                base_score -= 20
            elif issue.severity == Severity.MEDIUM:
                base_score -= 10
            elif issue.severity == Severity.LOW:
                base_score -= 5
        
        # Bonus points for good metrics
        if metrics.get("pass_rate", 0) >= 90:
            base_score += 10
        if metrics.get("coverage_percent", 0) >= 80:
            base_score += 10
        
        return max(0.0, min(100.0, base_score))
    
    def _determine_status(self, score: float) -> str:
        """Determine health status from score"""
        if score >= self.config.warning_threshold:
            return "healthy"
        elif score >= self.config.critical_threshold:
            return "warning"
        else:
            return "critical"