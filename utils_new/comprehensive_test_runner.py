from unittest.mock import Mock, patch
#!/usr/bin/env python3
"""
Comprehensive Test Runner for Wan Model Compatibility System
Orchestrates all testing and validation components
"""

import time
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import logging
from datetime import datetime

# Import testing components
try:
    from smoke_test_runner import SmokeTestRunner, SmokeTestResult
    from integration_test_suite import IntegrationTestSuite, IntegrationTestResult
    from performance_benchmark_suite import PerformanceBenchmarkSuite, BenchmarkSuite
    from test_coverage_validator import TestCoverageValidator, CoverageReport
except ImportError as e:
    logging.warning(f"Some testing components not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestExecutionResult:
    """Result of test execution"""
    test_type: str
    success: bool
    execution_time: float
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class ComprehensiveTestReport:
    """Comprehensive test report"""
    test_session_id: str
    timestamp: str
    total_execution_time: float
    test_results: List[TestExecutionResult] = field(default_factory=list)
    overall_success: bool = False
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

class ComprehensiveTestRunner:
    """
    Comprehensive test runner that orchestrates all testing components
    Provides unified testing interface and reporting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize comprehensive test runner
        
        Args:
            config: Optional configuration for test execution
        """
        self.config = config or self._get_default_config()
        self.session_id = f"test_session_{int(time.time())}"
        self.results_dir = Path("comprehensive_test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize test components
        self.smoke_runner = None
        self.integration_suite = None
        self.benchmark_suite = None
        self.coverage_validator = None
        
        self._initialize_components()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default test configuration"""
        return {
            "test_execution": {
                "run_smoke_tests": True,
                "run_integration_tests": True,
                "run_performance_benchmarks": True,
                "run_coverage_analysis": True,
                "fail_fast": False,
                "timeout_minutes": 60
            },
            "smoke_tests": {
                "test_all_components": True,
                "include_memory_tests": True,
                "include_performance_tests": True
            },
            "integration_tests": {
                "test_end_to_end_workflows": True,
                "test_error_scenarios": True,
                "test_resource_constraints": True,
                "test_concurrent_operations": True
            },
            "performance_benchmarks": {
                "run_comprehensive_suite": True,
                "detect_regressions": True,
                "update_baseline": False
            },
            "coverage_analysis": {
                "analyze_all_modules": True,
                "generate_recommendations": True,
                "minimum_coverage_threshold": 70.0
            },
            "reporting": {
                "generate_html_report": True,
                "generate_json_report": True,
                "generate_summary_report": True,
                "save_artifacts": True
            }
        }
    
    def _initialize_components(self):
        """Initialize test components"""
        try:
            if 'SmokeTestRunner' in globals():
                self.smoke_runner = SmokeTestRunner()
                logger.info("Smoke test runner initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize smoke test runner: {e}")
        
        try:
            if 'IntegrationTestSuite' in globals():
                self.integration_suite = IntegrationTestSuite()
                logger.info("Integration test suite initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize integration test suite: {e}")
        
        try:
            if 'PerformanceBenchmarkSuite' in globals():
                self.benchmark_suite = PerformanceBenchmarkSuite()
                logger.info("Performance benchmark suite initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize performance benchmark suite: {e}")
        
        try:
            if 'TestCoverageValidator' in globals():
                self.coverage_validator = TestCoverageValidator()
                logger.info("Test coverage validator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize coverage validator: {e}")
    
    def run_comprehensive_tests(self) -> ComprehensiveTestReport:
        """
        Run comprehensive test suite
        
        Returns:
            ComprehensiveTestReport with all test results
        """
        logger.info(f"Starting comprehensive test session: {self.session_id}")
        
        start_time = time.time()
        test_results = []
        
        # Run smoke tests
        if self.config["test_execution"]["run_smoke_tests"]:
            logger.info("=" * 60)
            logger.info("RUNNING SMOKE TESTS")
            logger.info("=" * 60)
            smoke_result = self._run_smoke_tests()
            test_results.append(smoke_result)
            
            if not smoke_result.success and self.config["test_execution"]["fail_fast"]:
                logger.error("Smoke tests failed, stopping execution (fail_fast=True)")
                return self._create_report(test_results, time.time() - start_time, False)
        
        # Run integration tests
        if self.config["test_execution"]["run_integration_tests"]:
            logger.info("=" * 60)
            logger.info("RUNNING INTEGRATION TESTS")
            logger.info("=" * 60)
            integration_result = self._run_integration_tests()
            test_results.append(integration_result)
            
            if not integration_result.success and self.config["test_execution"]["fail_fast"]:
                logger.error("Integration tests failed, stopping execution (fail_fast=True)")
                return self._create_report(test_results, time.time() - start_time, False)
        
        # Run performance benchmarks
        if self.config["test_execution"]["run_performance_benchmarks"]:
            logger.info("=" * 60)
            logger.info("RUNNING PERFORMANCE BENCHMARKS")
            logger.info("=" * 60)
            benchmark_result = self._run_performance_benchmarks()
            test_results.append(benchmark_result)
        
        # Run coverage analysis
        if self.config["test_execution"]["run_coverage_analysis"]:
            logger.info("=" * 60)
            logger.info("RUNNING COVERAGE ANALYSIS")
            logger.info("=" * 60)
            coverage_result = self._run_coverage_analysis()
            test_results.append(coverage_result)
        
        # Determine overall success
        overall_success = all(result.success for result in test_results)
        
        # Create comprehensive report
        total_time = time.time() - start_time
        report = self._create_report(test_results, total_time, overall_success)
        
        # Save report
        self._save_comprehensive_report(report)
        
        logger.info(f"Comprehensive test session completed in {total_time:.2f}s")
        logger.info(f"Overall result: {'SUCCESS' if overall_success else 'FAILURE'}")
        
        return report
    
    def _run_smoke_tests(self) -> TestExecutionResult:
        """Run smoke tests"""
        result = TestExecutionResult(
            test_type="smoke_tests",
            success=False,
            execution_time=0.0
        )
        
        start_time = time.time()
        
        try:
            if not self.smoke_runner:
                result.errors.append("Smoke test runner not available")
                return result
            
            # Create mock pipeline for testing
            mock_pipeline = self._create_mock_pipeline()
            
            # Run smoke test
            smoke_result = self.smoke_runner.run_pipeline_smoke_test(mock_pipeline)
            
            # Run memory test
            memory_result = self.smoke_runner.test_memory_usage(mock_pipeline)
            
            # Run performance benchmark
            perf_result = self.smoke_runner.benchmark_generation_speed(mock_pipeline)
            
            # Validate output format
            import numpy as np
            test_output = np.random.rand(8, 256, 256, 3).astype(np.float32)
            format_result = self.smoke_runner.validate_output_format(test_output)
            
            # Aggregate results
            result.details = {
                "smoke_test": {
                    "success": smoke_result.success,
                    "generation_time": smoke_result.generation_time,
                    "memory_peak": smoke_result.memory_peak,
                    "output_shape": smoke_result.output_shape,
                    "errors": smoke_result.errors,
                    "warnings": smoke_result.warnings
                },
                "memory_test": {
                    "peak_memory_mb": memory_result.peak_memory_mb,
                    "memory_increase_mb": memory_result.memory_increase_mb,
                    "memory_leaks_detected": memory_result.memory_leaks_detected,
                    "cleanup_successful": memory_result.cleanup_successful
                },
                "performance_test": {
                    "frames_per_second": perf_result.frames_per_second,
                    "performance_category": perf_result.performance_category,
                    "benchmark_score": perf_result.benchmark_score,
                    "bottlenecks": perf_result.bottlenecks
                },
                "format_validation": {
                    "is_valid": format_result.is_valid,
                    "expected_format": format_result.expected_format,
                    "actual_format": format_result.actual_format,
                    "validation_errors": format_result.validation_errors
                }
            }
            
            # Determine success
            result.success = (
                smoke_result.success and
                not memory_result.memory_leaks_detected and
                format_result.is_valid and
                perf_result.performance_category in ["excellent", "good", "acceptable"]
            )
            
            # Collect errors and warnings
            result.errors.extend(smoke_result.errors)
            result.warnings.extend(smoke_result.warnings)
            
            if memory_result.memory_leaks_detected:
                result.warnings.append("Memory leaks detected during testing")
            
            if not format_result.is_valid:
                result.errors.extend(format_result.validation_errors)
            
            if perf_result.performance_category == "poor":
                result.warnings.append(f"Poor performance: {perf_result.frames_per_second:.2f} FPS")
            
        except Exception as e:
            result.errors.append(f"Smoke tests failed with exception: {str(e)}")
            logger.error(f"Smoke tests error: {e}")
            logger.error(traceback.format_exc())
        
        result.execution_time = time.time() - start_time
        return result
    
    def _run_integration_tests(self) -> TestExecutionResult:
        """Run integration tests"""
        result = TestExecutionResult(
            test_type="integration_tests",
            success=False,
            execution_time=0.0
        )
        
        start_time = time.time()
        
        try:
            if not self.integration_suite:
                result.errors.append("Integration test suite not available")
                return result
            
            # Run integration test suite
            integration_results = self.integration_suite.run_all_tests()
            
            # Aggregate results
            total_tests = len(integration_results)
            passed_tests = sum(1 for r in integration_results if r.success)
            success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            
            result.details = {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": success_rate,
                "test_results": [
                    {
                        "test_name": r.test_name,
                        "success": r.success,
                        "execution_time": r.execution_time,
                        "components_tested": r.components_tested,
                        "error_count": len(r.errors),
                        "warning_count": len(r.warnings),
                        "metrics": r.metrics
                    }
                    for r in integration_results
                ]
            }
            
            # Determine success (require 80% pass rate)
            result.success = success_rate >= 80.0
            
            # Collect errors and warnings
            for test_result in integration_results:
                if not test_result.success:
                    result.errors.extend([f"{test_result.test_name}: {error}" for error in test_result.errors])
                result.warnings.extend([f"{test_result.test_name}: {warning}" for warning in test_result.warnings])
            
        except Exception as e:
            result.errors.append(f"Integration tests failed with exception: {str(e)}")
            logger.error(f"Integration tests error: {e}")
            logger.error(traceback.format_exc())
        
        result.execution_time = time.time() - start_time
        return result
    
    def _run_performance_benchmarks(self) -> TestExecutionResult:
        """Run performance benchmarks"""
        result = TestExecutionResult(
            test_type="performance_benchmarks",
            success=False,
            execution_time=0.0
        )
        
        start_time = time.time()
        
        try:
            if not self.benchmark_suite:
                result.errors.append("Performance benchmark suite not available")
                return result
            
            # Run benchmark suite
            benchmark_results = self.benchmark_suite.run_comprehensive_benchmark()
            
            # Aggregate results
            total_benchmarks = len(benchmark_results.benchmarks)
            successful_benchmarks = sum(1 for b in benchmark_results.benchmarks if b.error_rate < 0.1)
            
            result.details = {
                "total_benchmarks": total_benchmarks,
                "successful_benchmarks": successful_benchmarks,
                "total_execution_time": benchmark_results.total_execution_time,
                "regressions_detected": len(benchmark_results.regressions),
                "summary": benchmark_results.summary,
                "regressions": [
                    {
                        "metric_name": r.metric_name,
                        "regression_percentage": r.regression_percentage,
                        "severity": r.severity
                    }
                    for r in benchmark_results.regressions
                ]
            }
            
            # Determine success (no severe regressions, <20% moderate regressions)
            severe_regressions = [r for r in benchmark_results.regressions if r.severity == "severe"]
            moderate_regressions = [r for r in benchmark_results.regressions if r.severity == "moderate"]
            
            result.success = (
                len(severe_regressions) == 0 and
                len(moderate_regressions) < total_benchmarks * 0.2 and
                successful_benchmarks >= total_benchmarks * 0.8
            )
            
            # Collect warnings for regressions
            for regression in benchmark_results.regressions:
                result.warnings.append(
                    f"Performance regression: {regression.metric_name} - "
                    f"{regression.regression_percentage:.1f}% worse ({regression.severity})"
                )
            
        except Exception as e:
            result.errors.append(f"Performance benchmarks failed with exception: {str(e)}")
            logger.error(f"Performance benchmarks error: {e}")
            logger.error(traceback.format_exc())
        
        result.execution_time = time.time() - start_time
        return result
    
    def _run_coverage_analysis(self) -> TestExecutionResult:
        """Run coverage analysis"""
        result = TestExecutionResult(
            test_type="coverage_analysis",
            success=False,
            execution_time=0.0
        )
        
        start_time = time.time()
        
        try:
            if not self.coverage_validator:
                result.errors.append("Coverage validator not available")
                return result
            
            # Run coverage analysis
            coverage_report = self.coverage_validator.analyze_coverage()
            
            # Aggregate results
            result.details = {
                "overall_coverage_percentage": coverage_report.overall_coverage_percentage,
                "total_modules": coverage_report.total_modules,
                "total_functions": coverage_report.total_functions,
                "tested_functions": coverage_report.tested_functions,
                "coverage_gaps": coverage_report.coverage_gaps,
                "recommendations": coverage_report.recommendations,
                "module_coverage": [
                    {
                        "module_name": m.module_name,
                        "coverage_percentage": m.coverage_percentage,
                        "tested_functions": m.tested_functions,
                        "total_functions": m.total_functions
                    }
                    for m in coverage_report.module_coverage
                ]
            }
            
            # Determine success based on coverage threshold
            threshold = self.config["coverage_analysis"]["minimum_coverage_threshold"]
            result.success = coverage_report.overall_coverage_percentage >= threshold
            
            # Add warnings for low coverage
            if coverage_report.overall_coverage_percentage < threshold:
                result.warnings.append(
                    f"Overall coverage {coverage_report.overall_coverage_percentage:.1f}% "
                    f"is below threshold {threshold}%"
                )
            
            for gap in coverage_report.coverage_gaps:
                result.warnings.append(f"Coverage gap: {gap}")
            
        except Exception as e:
            result.errors.append(f"Coverage analysis failed with exception: {str(e)}")
            logger.error(f"Coverage analysis error: {e}")
            logger.error(traceback.format_exc())
        
        result.execution_time = time.time() - start_time
        return result
    
    def _create_report(self, test_results: List[TestExecutionResult], 
                      total_time: float, overall_success: bool) -> ComprehensiveTestReport:
        """Create comprehensive test report"""
        
        # Generate summary
        summary = {
            "test_types_run": len(test_results),
            "successful_test_types": sum(1 for r in test_results if r.success),
            "total_errors": sum(len(r.errors) for r in test_results),
            "total_warnings": sum(len(r.warnings) for r in test_results),
            "execution_breakdown": {
                r.test_type: r.execution_time for r in test_results
            }
        }
        
        # Generate recommendations
        recommendations = []
        
        for result in test_results:
            if not result.success:
                recommendations.append(f"Address failures in {result.test_type}")
            
            if result.errors:
                recommendations.append(f"Fix {len(result.errors)} errors in {result.test_type}")
        
        # Add specific recommendations based on test results
        for result in test_results:
            if result.test_type == "coverage_analysis" and result.details:
                coverage_recs = result.details.get("recommendations", [])
                recommendations.extend(coverage_recs[:3])  # Add top 3 coverage recommendations
        
        report = ComprehensiveTestReport(
            test_session_id=self.session_id,
            timestamp=datetime.now().isoformat(),
            total_execution_time=total_time,
            test_results=test_results,
            overall_success=overall_success,
            summary=summary,
            recommendations=recommendations
        )
        
        return report
    
    def _save_comprehensive_report(self, report: ComprehensiveTestReport):
        """Save comprehensive test report"""
        
        # Save JSON report
        json_file = self.results_dir / f"{report.test_session_id}_report.json"
        
        report_dict = {
            "test_session_id": report.test_session_id,
            "timestamp": report.timestamp,
            "total_execution_time": report.total_execution_time,
            "overall_success": report.overall_success,
            "summary": report.summary,
            "recommendations": report.recommendations,
            "test_results": [
                {
                    "test_type": r.test_type,
                    "success": r.success,
                    "execution_time": r.execution_time,
                    "error_count": len(r.errors),
                    "warning_count": len(r.warnings),
                    "details": r.details,
                    "errors": r.errors,
                    "warnings": r.warnings
                }
                for r in report.test_results
            ]
        }
        
        with open(json_file, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        # Save summary report
        summary_file = self.results_dir / f"{report.test_session_id}_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE TEST REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Session ID: {report.test_session_id}\n")
            f.write(f"Timestamp: {report.timestamp}\n")
            f.write(f"Total Execution Time: {report.total_execution_time:.2f}s\n")
            f.write(f"Overall Result: {'SUCCESS' if report.overall_success else 'FAILURE'}\n\n")
            
            f.write("TEST RESULTS:\n")
            f.write("-" * 30 + "\n")
            for result in report.test_results:
                status = "PASS" if result.success else "FAIL"
                f.write(f"{status} {result.test_type} ({result.execution_time:.2f}s)\n")
                
                if result.errors:
                    f.write(f"    Errors: {len(result.errors)}\n")
                if result.warnings:
                    f.write(f"    Warnings: {len(result.warnings)}\n")
            
            f.write(f"\nSUMMARY:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Test Types Run: {report.summary['test_types_run']}\n")
            f.write(f"Successful: {report.summary['successful_test_types']}\n")
            f.write(f"Total Errors: {report.summary['total_errors']}\n")
            f.write(f"Total Warnings: {report.summary['total_warnings']}\n")
            
            if report.recommendations:
                f.write(f"\nRECOMMENDATIONS:\n")
                f.write("-" * 30 + "\n")
                for i, rec in enumerate(report.recommendations, 1):
                    f.write(f"{i}. {rec}\n")
        
        logger.info(f"Comprehensive report saved to {json_file} and {summary_file}")
    
    def _create_mock_pipeline(self):
        """Create mock pipeline for testing"""
        class MockPipeline:
            def generate(self, prompt, num_frames=8, height=256, width=256):
                import time
                import numpy as np
                time.sleep(0.1)  # Simulate generation time
                return np.random.rand(num_frames, height, width, 3).astype(np.float32)
            
            def __call__(self, prompt, num_frames=8, height=256, width=256):
                return self.generate(prompt, num_frames, height, width)
        
        return MockPipeline()


def main():
    """Main entry point for comprehensive test runner"""
    
    print("Wan Model Compatibility - Comprehensive Test Runner")
    print("=" * 60)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run comprehensive test suite")
    parser.add_argument("--smoke-only", action="store_true", help="Run only smoke tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--benchmark-only", action="store_true", help="Run only performance benchmarks")
    parser.add_argument("--coverage-only", action="store_true", help="Run only coverage analysis")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in minutes")
    
    args = parser.parse_args()
    
    # Configure test runner based on arguments
    config = {
        "test_execution": {
            "run_smoke_tests": not any([args.integration_only, args.benchmark_only, args.coverage_only]),
            "run_integration_tests": not any([args.smoke_only, args.benchmark_only, args.coverage_only]),
            "run_performance_benchmarks": not any([args.smoke_only, args.integration_only, args.coverage_only]),
            "run_coverage_analysis": not any([args.smoke_only, args.integration_only, args.benchmark_only]),
            "fail_fast": args.fail_fast,
            "timeout_minutes": args.timeout
        }
    }
    
    # Override for specific test type requests
    if args.smoke_only:
        config["test_execution"]["run_smoke_tests"] = True
    if args.integration_only:
        config["test_execution"]["run_integration_tests"] = True
    if args.benchmark_only:
        config["test_execution"]["run_performance_benchmarks"] = True
    if args.coverage_only:
        config["test_execution"]["run_coverage_analysis"] = True
    
    # Run comprehensive tests
    runner = ComprehensiveTestRunner(config)
    report = runner.run_comprehensive_tests()
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)
    print(f"Overall Result: {'SUCCESS' if report.overall_success else 'FAILURE'}")
    print(f"Total Time: {report.total_execution_time:.2f}s")
    print(f"Test Types: {report.summary['test_types_run']}")
    print(f"Successful: {report.summary['successful_test_types']}")
    print(f"Errors: {report.summary['total_errors']}")
    print(f"Warnings: {report.summary['total_warnings']}")
    
    # Exit with appropriate code
    sys.exit(0 if report.overall_success else 1)


if __name__ == "__main__":
    main()