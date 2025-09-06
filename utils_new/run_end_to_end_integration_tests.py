#!/usr/bin/env python3
"""
End-to-End Integration Test Runner
Orchestrates all end-to-end integration tests for Wan Model Compatibility System
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

# Import test suites
try:
    from test_end_to_end_integration import EndToEndIntegrationTestSuite
    from test_wan_model_variants import WanModelVariantsTestSuite
    from test_resource_constraint_simulation import ResourceConstraintSimulationTestSuite
    from test_error_injection_recovery import ErrorInjectionRecoveryTestSuite
    from performance_benchmark_suite import PerformanceBenchmarkSuite
except ImportError as e:
    logging.warning(f"Some test suites not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestSuiteResult:
    """Result of a test suite execution"""
    suite_name: str
    success: bool
    total_tests: int
    passed_tests: int
    execution_time: float
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]

@dataclass
class IntegrationTestReport:
    """Comprehensive integration test report"""
    session_id: str
    timestamp: str
    total_execution_time: float
    suite_results: List[TestSuiteResult]
    overall_success: bool
    summary: Dict[str, Any]
    recommendations: List[str]

class EndToEndIntegrationTestRunner:
    """
    Comprehensive end-to-end integration test runner
    Orchestrates all integration test suites and provides unified reporting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize integration test runner
        
        Args:
            config: Optional configuration for test execution
        """
        self.config = config or self._get_default_config()
        self.session_id = f"e2e_integration_{int(time.time())}"
        self.results_dir = Path("integration_test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize test suites
        self.test_suites = {}
        self._initialize_test_suites()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "test_execution": {
                "run_model_variants": True,
                "run_resource_constraints": True,
                "run_error_injection": True,
                "run_performance_benchmarks": True,
                "run_end_to_end_workflows": True,
                "fail_fast": False,
                "timeout_minutes": 120,
                "parallel_execution": False
            },
            "reporting": {
                "generate_html_report": True,
                "generate_json_report": True,
                "generate_summary_report": True,
                "save_artifacts": True,
                "include_performance_charts": True
            },
            "thresholds": {
                "minimum_success_rate": 80.0,
                "maximum_error_rate": 10.0,
                "performance_regression_threshold": 20.0
            }
        }
    
    def _initialize_test_suites(self):
        """Initialize available test suites"""
        suite_classes = [
            ("model_variants", WanModelVariantsTestSuite),
            ("resource_constraints", ResourceConstraintSimulationTestSuite),
            ("error_injection", ErrorInjectionRecoveryTestSuite),
            ("end_to_end_workflows", EndToEndIntegrationTestSuite),
            ("performance_benchmarks", PerformanceBenchmarkSuite)
        ]
        
        for suite_name, suite_class in suite_classes:
            try:
                if suite_name in globals() or hasattr(sys.modules[__name__], suite_class.__name__):
                    self.test_suites[suite_name] = suite_class()
                    logger.info(f"Initialized test suite: {suite_name}")
                else:
                    logger.warning(f"Test suite not available: {suite_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize test suite {suite_name}: {e}")
    
    def run_all_integration_tests(self) -> IntegrationTestReport:
        """
        Run all integration tests
        
        Returns:
            IntegrationTestReport with comprehensive results
        """
        logger.info(f"Starting comprehensive integration test session: {self.session_id}")
        
        start_time = time.time()
        suite_results = []
        
        # Run each test suite
        test_suites_to_run = [
            ("model_variants", "run_model_variant_tests"),
            ("resource_constraints", "run_resource_constraint_tests"),
            ("error_injection", "run_error_injection_tests"),
            ("performance_benchmarks", "run_performance_benchmark_tests"),
            ("end_to_end_workflows", "run_end_to_end_workflow_tests")
        ]
        
        for suite_name, method_name in test_suites_to_run:
            if self._should_run_suite(suite_name):
                logger.info("=" * 70)
                logger.info(f"RUNNING {suite_name.upper().replace('_', ' ')} TESTS")
                logger.info("=" * 70)
                
                suite_result = self._run_test_suite(suite_name, method_name)
                suite_results.append(suite_result)
                
                if not suite_result.success and self.config["test_execution"]["fail_fast"]:
                    logger.error(f"Test suite {suite_name} failed, stopping execution (fail_fast=True)")
                    break
        
        # Generate comprehensive report
        total_time = time.time() - start_time
        report = self._create_integration_report(suite_results, total_time)
        
        # Save report
        self._save_integration_report(report)
        
        logger.info(f"Integration test session completed in {total_time:.2f}s")
        logger.info(f"Overall result: {'SUCCESS' if report.overall_success else 'FAILURE'}")
        
        return report
    
    def _should_run_suite(self, suite_name: str) -> bool:
        """Check if test suite should be run"""
        config_key = f"run_{suite_name}"
        return self.config["test_execution"].get(config_key, True)
    
    def _run_test_suite(self, suite_name: str, method_name: str) -> TestSuiteResult:
        """Run specific test suite"""
        result = TestSuiteResult(
            suite_name=suite_name,
            success=False,
            total_tests=0,
            passed_tests=0,
            execution_time=0.0,
            errors=[],
            warnings=[],
            details={}
        )
        
        start_time = time.time()
        
        try:
            if suite_name not in self.test_suites:
                result.errors.append(f"Test suite not available: {suite_name}")
                return result
            
            test_suite = self.test_suites[suite_name]
            
            # Run the appropriate test method
            if suite_name == "model_variants":
                test_results = test_suite.test_all_model_variants()
                result = self._process_model_variant_results(test_results, result)
            
            elif suite_name == "resource_constraints":
                test_results = test_suite.test_all_resource_constraints()
                result = self._process_resource_constraint_results(test_results, result)
            
            elif suite_name == "error_injection":
                test_results = test_suite.test_all_error_scenarios()
                result = self._process_error_injection_results(test_results, result)
            
            elif suite_name == "performance_benchmarks":
                test_results = test_suite.run_comprehensive_benchmark()
                result = self._process_performance_benchmark_results(test_results, result)
            
            elif suite_name == "end_to_end_workflows":
                test_results = test_suite.run_all_end_to_end_tests()
                result = self._process_end_to_end_results(test_results, result)
            
            else:
                result.errors.append(f"Unknown test suite: {suite_name}")
        
        except Exception as e:
            result.errors.append(f"Test suite execution failed: {str(e)}")
            logger.error(f"Test suite {suite_name} execution error: {e}")
        
        result.execution_time = time.time() - start_time
        
        # Determine success
        if result.total_tests > 0:
            success_rate = (result.passed_tests / result.total_tests) * 100
            result.success = success_rate >= self.config["thresholds"]["minimum_success_rate"]
        else:
            result.success = len(result.errors) == 0
        
        logger.info(f"Test suite {suite_name} completed: {result.passed_tests}/{result.total_tests} passed")
        
        return result
    
    def _process_model_variant_results(self, test_results: List[Dict[str, Any]], 
                                     result: TestSuiteResult) -> TestSuiteResult:
        """Process model variant test results"""
        result.total_tests = len(test_results)
        result.passed_tests = sum(1 for r in test_results if r.get("success", False))
        
        for test_result in test_results:
            result.errors.extend(test_result.get("errors", []))
            result.warnings.extend(test_result.get("warnings", []))
        
        result.details = {
            "model_variants_tested": [r["model_name"] for r in test_results],
            "success_rate": (result.passed_tests / result.total_tests) * 100 if result.total_tests > 0 else 0,
            "test_results": test_results
        }
        
        return result
    
    def _process_resource_constraint_results(self, test_results: List[Dict[str, Any]], 
                                           result: TestSuiteResult) -> TestSuiteResult:
        """Process resource constraint test results"""
        result.total_tests = len(test_results)
        result.passed_tests = sum(1 for r in test_results if r.get("success", False))
        
        for test_result in test_results:
            result.errors.extend(test_result.get("errors", []))
            result.warnings.extend(test_result.get("warnings", []))
        
        result.details = {
            "constraints_tested": [r["constraint_name"] for r in test_results],
            "optimization_effectiveness": self._calculate_optimization_effectiveness(test_results),
            "success_rate": (result.passed_tests / result.total_tests) * 100 if result.total_tests > 0 else 0,
            "test_results": test_results
        }
        
        return result
    
    def _process_error_injection_results(self, test_results: List[Dict[str, Any]], 
                                       result: TestSuiteResult) -> TestSuiteResult:
        """Process error injection test results"""
        result.total_tests = len(test_results)
        result.passed_tests = sum(1 for r in test_results if r.get("success", False))
        
        for test_result in test_results:
            result.errors.extend(test_result.get("errors", []))
            result.warnings.extend(test_result.get("warnings", []))
        
        # Calculate recovery statistics
        recovery_attempted = sum(1 for r in test_results if r.get("recovery_attempted", False))
        recovery_successful = sum(1 for r in test_results if r.get("recovery_successful", False))
        
        result.details = {
            "error_scenarios_tested": [r["scenario_name"] for r in test_results],
            "recovery_statistics": {
                "recovery_attempted": recovery_attempted,
                "recovery_successful": recovery_successful,
                "recovery_rate": (recovery_successful / recovery_attempted) * 100 if recovery_attempted > 0 else 0
            },
            "severity_breakdown": self._calculate_severity_breakdown(test_results),
            "success_rate": (result.passed_tests / result.total_tests) * 100 if result.total_tests > 0 else 0,
            "test_results": test_results
        }
        
        return result
    
    def _process_performance_benchmark_results(self, test_results: Any, 
                                             result: TestSuiteResult) -> TestSuiteResult:
        """Process performance benchmark test results"""
        if hasattr(test_results, 'benchmarks'):
            result.total_tests = len(test_results.benchmarks)
            result.passed_tests = sum(1 for b in test_results.benchmarks if b.error_rate < 0.1)
            
            result.details = {
                "benchmarks_run": result.total_tests,
                "performance_regressions": len(test_results.regressions) if hasattr(test_results, 'regressions') else 0,
                "total_execution_time": test_results.total_execution_time if hasattr(test_results, 'total_execution_time') else 0,
                "summary": test_results.summary if hasattr(test_results, 'summary') else {},
                "success_rate": (result.passed_tests / result.total_tests) * 100 if result.total_tests > 0 else 0
            }
        else:
            result.errors.append("Invalid performance benchmark results format")
        
        return result
    
    def _process_end_to_end_results(self, test_results: List[Any], 
                                  result: TestSuiteResult) -> TestSuiteResult:
        """Process end-to-end workflow test results"""
        result.total_tests = len(test_results)
        result.passed_tests = sum(1 for r in test_results if getattr(r, 'workflow_success', False))
        
        for test_result in test_results:
            if hasattr(test_result, 'errors'):
                result.errors.extend(test_result.errors)
            if hasattr(test_result, 'warnings'):
                result.warnings.extend(test_result.warnings)
        
        result.details = {
            "workflows_tested": [getattr(r, 'test_name', 'unknown') for r in test_results],
            "component_success_rates": self._calculate_component_success_rates(test_results),
            "performance_metrics": self._aggregate_performance_metrics(test_results),
            "success_rate": (result.passed_tests / result.total_tests) * 100 if result.total_tests > 0 else 0
        }
        
        return result
    
    def _calculate_optimization_effectiveness(self, test_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate optimization effectiveness metrics"""
        effectiveness = {
            "average_vram_reduction": 0.0,
            "average_performance_impact": 0.0,
            "optimization_success_rate": 0.0
        }
        
        try:
            successful_optimizations = []
            for result in test_results:
                if result.get("workflow_completed", False):
                    optimizations = result.get("optimizations_applied", [])
                    if optimizations:
                        successful_optimizations.append(result)
            
            if successful_optimizations:
                effectiveness["optimization_success_rate"] = (len(successful_optimizations) / len(test_results)) * 100
        
        except Exception as e:
            logger.warning(f"Failed to calculate optimization effectiveness: {e}")
        
        return effectiveness
    
    def _calculate_severity_breakdown(self, test_results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate error severity breakdown"""
        breakdown = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for result in test_results:
            severity = result.get("severity", "unknown")
            if severity in breakdown:
                breakdown[severity] += 1
        
        return breakdown
    
    def _calculate_component_success_rates(self, test_results: List[Any]) -> Dict[str, float]:
        """Calculate success rates for different components"""
        success_rates = {
            "model_detection": 0.0,
            "pipeline_loading": 0.0,
            "generation": 0.0,
            "video_encoding": 0.0
        }
        
        try:
            total_tests = len(test_results)
            if total_tests > 0:
                success_rates["model_detection"] = (sum(1 for r in test_results if getattr(r, 'model_detection_success', False)) / total_tests) * 100
                success_rates["pipeline_loading"] = (sum(1 for r in test_results if getattr(r, 'pipeline_loading_success', False)) / total_tests) * 100
                success_rates["generation"] = (sum(1 for r in test_results if getattr(r, 'generation_success', False)) / total_tests) * 100
                success_rates["video_encoding"] = (sum(1 for r in test_results if getattr(r, 'video_encoding_success', False)) / total_tests) * 100
        
        except Exception as e:
            logger.warning(f"Failed to calculate component success rates: {e}")
        
        return success_rates
    
    def _aggregate_performance_metrics(self, test_results: List[Any]) -> Dict[str, float]:
        """Aggregate performance metrics from test results"""
        metrics = {
            "average_total_time": 0.0,
            "average_fps": 0.0,
            "memory_efficiency": 0.0
        }
        
        try:
            valid_results = [r for r in test_results if hasattr(r, 'performance_metrics')]
            if valid_results:
                total_times = [r.total_time for r in valid_results if hasattr(r, 'total_time')]
                if total_times:
                    metrics["average_total_time"] = sum(total_times) / len(total_times)
                
                # Extract FPS from performance metrics
                fps_values = []
                for result in valid_results:
                    perf_metrics = getattr(result, 'performance_metrics', {})
                    if 'fps' in perf_metrics:
                        fps_values.append(perf_metrics['fps'])
                
                if fps_values:
                    metrics["average_fps"] = sum(fps_values) / len(fps_values)
        
        except Exception as e:
            logger.warning(f"Failed to aggregate performance metrics: {e}")
        
        return metrics
    
    def _create_integration_report(self, suite_results: List[TestSuiteResult], 
                                 total_time: float) -> IntegrationTestReport:
        """Create comprehensive integration test report"""
        
        # Calculate overall statistics
        total_tests = sum(r.total_tests for r in suite_results)
        total_passed = sum(r.passed_tests for r in suite_results)
        total_errors = sum(len(r.errors) for r in suite_results)
        total_warnings = sum(len(r.warnings) for r in suite_results)
        
        overall_success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
        overall_success = overall_success_rate >= self.config["thresholds"]["minimum_success_rate"]
        
        # Generate summary
        summary = {
            "total_test_suites": len(suite_results),
            "successful_test_suites": sum(1 for r in suite_results if r.success),
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_tests - total_passed,
            "overall_success_rate": overall_success_rate,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "execution_breakdown": {r.suite_name: r.execution_time for r in suite_results}
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(suite_results, summary)
        
        report = IntegrationTestReport(
            session_id=self.session_id,
            timestamp=datetime.now().isoformat(),
            total_execution_time=total_time,
            suite_results=suite_results,
            overall_success=overall_success,
            summary=summary,
            recommendations=recommendations
        )
        
        return report
    
    def _generate_recommendations(self, suite_results: List[TestSuiteResult], 
                                summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Overall success rate recommendations
        if summary["overall_success_rate"] < 90:
            recommendations.append("Overall success rate is below 90%. Review failed tests and improve system reliability.")
        
        # Suite-specific recommendations
        for result in suite_results:
            if not result.success:
                recommendations.append(f"Address failures in {result.suite_name} test suite")
            
            if len(result.errors) > 5:
                recommendations.append(f"High error count in {result.suite_name}. Investigate root causes.")
        
        # Performance recommendations
        performance_suite = next((r for r in suite_results if r.suite_name == "performance_benchmarks"), None)
        if performance_suite and performance_suite.details.get("performance_regressions", 0) > 0:
            recommendations.append("Performance regressions detected. Review optimization strategies.")
        
        # Resource constraint recommendations
        resource_suite = next((r for r in suite_results if r.suite_name == "resource_constraints"), None)
        if resource_suite:
            effectiveness = resource_suite.details.get("optimization_effectiveness", {})
            if effectiveness.get("optimization_success_rate", 0) < 80:
                recommendations.append("Optimization success rate is low. Improve resource management strategies.")
        
        # Error recovery recommendations
        error_suite = next((r for r in suite_results if r.suite_name == "error_injection"), None)
        if error_suite:
            recovery_stats = error_suite.details.get("recovery_statistics", {})
            if recovery_stats.get("recovery_rate", 0) < 70:
                recommendations.append("Error recovery rate is low. Enhance error handling and recovery mechanisms.")
        
        return recommendations
    
    def _save_integration_report(self, report: IntegrationTestReport):
        """Save comprehensive integration test report"""
        
        # Save JSON report
        json_file = self.results_dir / f"{report.session_id}_report.json"
        
        report_dict = {
            "session_id": report.session_id,
            "timestamp": report.timestamp,
            "total_execution_time": report.total_execution_time,
            "overall_success": report.overall_success,
            "summary": report.summary,
            "recommendations": report.recommendations,
            "suite_results": [
                {
                    "suite_name": r.suite_name,
                    "success": r.success,
                    "total_tests": r.total_tests,
                    "passed_tests": r.passed_tests,
                    "execution_time": r.execution_time,
                    "error_count": len(r.errors),
                    "warning_count": len(r.warnings),
                    "details": r.details,
                    "errors": r.errors,
                    "warnings": r.warnings
                }
                for r in report.suite_results
            ]
        }
        
        with open(json_file, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        # Save summary report
        summary_file = self.results_dir / f"{report.session_id}_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("END-TO-END INTEGRATION TEST REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Session ID: {report.session_id}\n")
            f.write(f"Timestamp: {report.timestamp}\n")
            f.write(f"Total Execution Time: {report.total_execution_time:.2f}s\n")
            f.write(f"Overall Result: {'SUCCESS' if report.overall_success else 'FAILURE'}\n\n")
            
            f.write("TEST SUITE RESULTS:\n")
            f.write("-" * 40 + "\n")
            for result in report.suite_results:
                status = "PASS" if result.success else "FAIL"
                success_rate = (result.passed_tests / result.total_tests) * 100 if result.total_tests > 0 else 0
                f.write(f"{status} {result.suite_name} ({result.passed_tests}/{result.total_tests} - {success_rate:.1f}%)\n")
                f.write(f"    Execution Time: {result.execution_time:.2f}s\n")
                
                if result.errors:
                    f.write(f"    Errors: {len(result.errors)}\n")
                if result.warnings:
                    f.write(f"    Warnings: {len(result.warnings)}\n")
                f.write("\n")
            
            f.write(f"SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Test Suites Run: {report.summary['total_test_suites']}\n")
            f.write(f"Successful Suites: {report.summary['successful_test_suites']}\n")
            f.write(f"Total Tests: {report.summary['total_tests']}\n")
            f.write(f"Tests Passed: {report.summary['total_passed']}\n")
            f.write(f"Tests Failed: {report.summary['total_failed']}\n")
            f.write(f"Success Rate: {report.summary['overall_success_rate']:.1f}%\n")
            f.write(f"Total Errors: {report.summary['total_errors']}\n")
            f.write(f"Total Warnings: {report.summary['total_warnings']}\n\n")
            
            if report.recommendations:
                f.write(f"RECOMMENDATIONS:\n")
                f.write("-" * 40 + "\n")
                for i, rec in enumerate(report.recommendations, 1):
                    f.write(f"{i}. {rec}\n")
        
        logger.info(f"Integration test report saved to {json_file} and {summary_file}")


def main():
    """Main entry point for end-to-end integration test runner"""
    
    print("Wan Model Compatibility - End-to-End Integration Test Runner")
    print("=" * 70)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run comprehensive end-to-end integration tests")
    parser.add_argument("--model-variants", action="store_true", help="Run only model variant tests")
    parser.add_argument("--resource-constraints", action="store_true", help="Run only resource constraint tests")
    parser.add_argument("--error-injection", action="store_true", help="Run only error injection tests")
    parser.add_argument("--performance", action="store_true", help="Run only performance benchmark tests")
    parser.add_argument("--workflows", action="store_true", help="Run only end-to-end workflow tests")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first test suite failure")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in minutes")
    parser.add_argument("--success-threshold", type=float, default=80.0, help="Minimum success rate threshold")
    
    args = parser.parse_args()
    
    # Configure test runner based on arguments
    config = {
        "test_execution": {
            "run_model_variants": not any([args.resource_constraints, args.error_injection, args.performance, args.workflows]) or args.model_variants,
            "run_resource_constraints": not any([args.model_variants, args.error_injection, args.performance, args.workflows]) or args.resource_constraints,
            "run_error_injection": not any([args.model_variants, args.resource_constraints, args.performance, args.workflows]) or args.error_injection,
            "run_performance_benchmarks": not any([args.model_variants, args.resource_constraints, args.error_injection, args.workflows]) or args.performance,
            "run_end_to_end_workflows": not any([args.model_variants, args.resource_constraints, args.error_injection, args.performance]) or args.workflows,
            "fail_fast": args.fail_fast,
            "timeout_minutes": args.timeout
        },
        "thresholds": {
            "minimum_success_rate": args.success_threshold
        }
    }
    
    # Run comprehensive integration tests
    runner = EndToEndIntegrationTestRunner(config)
    report = runner.run_all_integration_tests()
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL INTEGRATION TEST SUMMARY")
    print("=" * 70)
    print(f"Overall Result: {'SUCCESS' if report.overall_success else 'FAILURE'}")
    print(f"Total Time: {report.total_execution_time:.2f}s")
    print(f"Test Suites: {report.summary['total_test_suites']}")
    print(f"Total Tests: {report.summary['total_tests']}")
    print(f"Tests Passed: {report.summary['total_passed']}")
    print(f"Tests Failed: {report.summary['total_failed']}")
    print(f"Success Rate: {report.summary['overall_success_rate']:.1f}%")
    print(f"Errors: {report.summary['total_errors']}")
    print(f"Warnings: {report.summary['total_warnings']}")
    
    # Print test suite breakdown
    print("\nTest Suite Results:")
    print("-" * 50)
    
    for result in report.suite_results:
        status = "PASS" if result.success else "FAIL"
        success_rate = (result.passed_tests / result.total_tests) * 100 if result.total_tests > 0 else 0
        print(f"{status} {result.suite_name} ({result.passed_tests}/{result.total_tests} - {success_rate:.1f}%)")
    
    # Print recommendations
    if report.recommendations:
        print("\nRecommendations:")
        print("-" * 50)
        for i, rec in enumerate(report.recommendations[:5], 1):  # Show top 5 recommendations
            print(f"{i}. {rec}")
    
    # Exit with appropriate code
    sys.exit(0 if report.overall_success else 1)


if __name__ == "__main__":
    main()