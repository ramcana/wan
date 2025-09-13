"""
Comprehensive Reliability System Test Runner

This script orchestrates all reliability system test suites and generates
a comprehensive report covering all testing requirements.

Requirements addressed: 1.1, 1.2, 1.3, 1.4, 1.5
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime
import subprocess

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

# Import test suites
try:
    from test_comprehensive_reliability_suite import run_comprehensive_tests
    from test_failure_injection_suite import run_failure_injection_tests
    from test_performance_impact_suite import run_performance_tests
    from test_error_scenario_suite import run_error_scenario_tests
except ImportError as e:
    print(f"Warning: Could not import some test suites: {e}")
    
    # Create fallback functions
    def run_comprehensive_tests():
        print("Comprehensive tests not available")
        return False
    
    def run_failure_injection_tests():
        print("Failure injection tests not available")
        return False
    
    def run_performance_tests():
        print("Performance tests not available")
        return False
    
    def run_error_scenario_tests():
        print("Error scenario tests not available")
        return False


class ComprehensiveTestRunner:
    """Orchestrates all reliability system test suites."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path("test_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure logging
        self.logger = self._setup_logging()
        
        # Test suite registry
        self.test_suites = {
            'comprehensive': {
                'name': 'Comprehensive Reliability System Tests',
                'runner': run_comprehensive_tests,
                'description': 'Unit and integration tests for all reliability components',
                'requirements': ['1.1', '1.2', '1.3', '1.4', '1.5']
            },
            'failure_injection': {
                'name': 'Failure Injection Tests',
                'runner': run_failure_injection_tests,
                'description': 'Tests for recovery validation through failure injection',
                'requirements': ['1.1', '1.2', '1.3', '1.4']
            },
            'performance_impact': {
                'name': 'Performance Impact Tests',
                'runner': run_performance_tests,
                'description': 'Tests to ensure minimal performance overhead',
                'requirements': ['1.1', '1.2', '1.5']
            },
            'error_scenarios': {
                'name': 'Error Scenario Tests',
                'runner': run_error_scenario_tests,
                'description': 'Tests for specific error conditions from installation logs',
                'requirements': ['1.1', '1.2', '1.3', '1.4']
            }
        }
        
        # Results storage
        self.test_results = {}
        self.overall_results = {
            'start_time': None,
            'end_time': None,
            'total_duration': 0,
            'suites_run': 0,
            'suites_passed': 0,
            'suites_failed': 0,
            'overall_success': False
        }
    
    def _setup_logging(self):
        """Set up comprehensive logging."""
        logger = logging.getLogger('comprehensive_test_runner')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.output_dir / f'test_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def run_single_test_suite(self, suite_name: str) -> dict:
        """Run a single test suite and capture results."""
        if suite_name not in self.test_suites:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        suite_info = self.test_suites[suite_name]
        self.logger.info(f"Starting test suite: {suite_info['name']}")
        
        start_time = time.time()
        
        try:
            # Run the test suite
            success = suite_info['runner']()
            end_time = time.time()
            duration = end_time - start_time
            
            result = {
                'suite_name': suite_name,
                'display_name': suite_info['name'],
                'description': suite_info['description'],
                'requirements': suite_info['requirements'],
                'success': success,
                'duration': duration,
                'start_time': start_time,
                'end_time': end_time,
                'error': None
            }
            
            self.logger.info(f"Test suite '{suite_info['name']}' completed: {'PASSED' if success else 'FAILED'} ({duration:.2f}s)")
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            result = {
                'suite_name': suite_name,
                'display_name': suite_info['name'],
                'description': suite_info['description'],
                'requirements': suite_info['requirements'],
                'success': False,
                'duration': duration,
                'start_time': start_time,
                'end_time': end_time,
                'error': str(e)
            }
            
            self.logger.error(f"Test suite '{suite_info['name']}' failed with error: {e}")
        
        return result
    
    def run_all_test_suites(self) -> dict:
        """Run all test suites and generate comprehensive report."""
        self.logger.info("Starting comprehensive reliability system test run")
        self.overall_results['start_time'] = time.time()
        
        # Run each test suite
        for suite_name in self.test_suites.keys():
            result = self.run_single_test_suite(suite_name)
            self.test_results[suite_name] = result
            
            self.overall_results['suites_run'] += 1
            if result['success']:
                self.overall_results['suites_passed'] += 1
            else:
                self.overall_results['suites_failed'] += 1
        
        # Calculate overall results
        self.overall_results['end_time'] = time.time()
        self.overall_results['total_duration'] = (
            self.overall_results['end_time'] - self.overall_results['start_time']
        )
        self.overall_results['overall_success'] = (
            self.overall_results['suites_failed'] == 0
        )
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report()
        
        self.logger.info(f"Comprehensive test run completed: {self.overall_results['suites_passed']}/{self.overall_results['suites_run']} suites passed")
        
        return report
    
    def _generate_comprehensive_report(self) -> dict:
        """Generate comprehensive test report."""
        report = {
            'test_run_info': {
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': self.overall_results['total_duration'],
                'python_version': sys.version,
                'platform': sys.platform
            },
            'overall_results': self.overall_results,
            'suite_results': self.test_results,
            'requirements_coverage': self._analyze_requirements_coverage(),
            'performance_summary': self._extract_performance_summary(),
            'failure_analysis': self._analyze_failures(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save report to file
        report_file = self.output_dir / f'comprehensive_test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Comprehensive report saved to: {report_file}")
        
        return report
    
    def _analyze_requirements_coverage(self) -> dict:
        """Analyze coverage of requirements across test suites."""
        all_requirements = set()
        covered_requirements = set()
        
        for suite_name, suite_info in self.test_suites.items():
            suite_requirements = set(suite_info['requirements'])
            all_requirements.update(suite_requirements)
            
            if suite_name in self.test_results and self.test_results[suite_name]['success']:
                covered_requirements.update(suite_requirements)
        
        coverage = {
            'total_requirements': len(all_requirements),
            'covered_requirements': len(covered_requirements),
            'coverage_percentage': (len(covered_requirements) / len(all_requirements) * 100) if all_requirements else 0,
            'uncovered_requirements': list(all_requirements - covered_requirements),
            'requirement_details': {
                '1.1': 'Automatic retry with intelligent backoff strategies',
                '1.2': 'Comprehensive error logging with system context',
                '1.3': 'Missing method detection and recovery',
                '1.4': 'Model validation failure recovery',
                '1.5': 'Performance impact minimization'
            }
        }
        
        return coverage
    
    def _extract_performance_summary(self) -> dict:
        """Extract performance summary from test results."""
        performance_summary = {
            'performance_tests_run': 'performance_impact' in self.test_results,
            'performance_acceptable': False,
            'overhead_analysis': 'Not available',
            'memory_usage_analysis': 'Not available',
            'throughput_analysis': 'Not available'
        }
        
        if 'performance_impact' in self.test_results:
            perf_result = self.test_results['performance_impact']
            performance_summary['performance_acceptable'] = perf_result['success']
            
            if perf_result['success']:
                performance_summary['overhead_analysis'] = 'Acceptable overhead levels maintained'
                performance_summary['memory_usage_analysis'] = 'Memory usage within acceptable limits'
                performance_summary['throughput_analysis'] = 'Throughput impact minimized'
            else:
                performance_summary['overhead_analysis'] = 'Performance overhead may be excessive'
                performance_summary['memory_usage_analysis'] = 'Memory usage needs optimization'
                performance_summary['throughput_analysis'] = 'Throughput significantly impacted'
        
        return performance_summary
    
    def _analyze_failures(self) -> dict:
        """Analyze test failures and categorize them."""
        failures = []
        failure_categories = {
            'component_unavailable': 0,
            'test_execution_error': 0,
            'assertion_failure': 0,
            'timeout': 0,
            'other': 0
        }
        
        for suite_name, result in self.test_results.items():
            if not result['success']:
                failure_info = {
                    'suite': suite_name,
                    'error': result.get('error', 'Unknown error'),
                    'duration': result['duration']
                }
                failures.append(failure_info)
                
                # Categorize failure
                error_msg = str(result.get('error', '')).lower()
                if 'not available' in error_msg or 'import' in error_msg:
                    failure_categories['component_unavailable'] += 1
                elif 'timeout' in error_msg:
                    failure_categories['timeout'] += 1
                elif 'assertion' in error_msg:
                    failure_categories['assertion_failure'] += 1
                elif 'execution' in error_msg:
                    failure_categories['test_execution_error'] += 1
                else:
                    failure_categories['other'] += 1
        
        return {
            'total_failures': len(failures),
            'failure_details': failures,
            'failure_categories': failure_categories,
            'most_common_failure': max(failure_categories.items(), key=lambda x: x[1])[0] if failures else None
        }
    
    def _generate_recommendations(self) -> list:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Overall success rate
        success_rate = (self.overall_results['suites_passed'] / self.overall_results['suites_run'] * 100) if self.overall_results['suites_run'] > 0 else 0
        
        if success_rate < 100:
            recommendations.append({
                'category': 'Test Coverage',
                'priority': 'High',
                'recommendation': f'Address failing test suites to improve success rate from {success_rate:.1f}% to 100%'
            })
        
        # Requirements coverage
        requirements_coverage = self._analyze_requirements_coverage()
        if requirements_coverage['coverage_percentage'] < 100:
            recommendations.append({
                'category': 'Requirements Coverage',
                'priority': 'Medium',
                'recommendation': f'Improve requirements coverage from {requirements_coverage["coverage_percentage"]:.1f}% to 100%'
            })
        
        # Performance analysis
        performance_summary = self._extract_performance_summary()
        if not performance_summary['performance_acceptable']:
            recommendations.append({
                'category': 'Performance',
                'priority': 'High',
                'recommendation': 'Optimize reliability system to reduce performance overhead'
            })
        
        # Failure analysis
        failure_analysis = self._analyze_failures()
        if failure_analysis['total_failures'] > 0:
            most_common = failure_analysis['most_common_failure']
            recommendations.append({
                'category': 'Failure Resolution',
                'priority': 'High',
                'recommendation': f'Address {most_common} failures which are the most common issue'
            })
        
        # General recommendations
        if success_rate >= 80:
            recommendations.append({
                'category': 'System Health',
                'priority': 'Low',
                'recommendation': 'Reliability system shows good overall health with minor improvements needed'
            })
        elif success_rate >= 60:
            recommendations.append({
                'category': 'System Health',
                'priority': 'Medium',
                'recommendation': 'Reliability system needs moderate improvements to reach production readiness'
            })
        else:
            recommendations.append({
                'category': 'System Health',
                'priority': 'Critical',
                'recommendation': 'Reliability system requires significant improvements before deployment'
            })
        
        return recommendations
    
    def print_summary_report(self, report: dict):
        """Print a human-readable summary report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE RELIABILITY SYSTEM TEST SUMMARY")
        print("="*80)
        
        # Overall results
        overall = report['overall_results']
        print(f"Test Run Duration: {overall['total_duration']:.2f} seconds")
        print(f"Test Suites Run: {overall['suites_run']}")
        print(f"Test Suites Passed: {overall['suites_passed']}")
        print(f"Test Suites Failed: {overall['suites_failed']}")
        print(f"Overall Success: {'YES' if overall['overall_success'] else 'NO'}")
        
        # Suite details
        print(f"\nTest Suite Details:")
        print("-" * 40)
        for suite_name, result in report['suite_results'].items():
            status = "PASSED" if result['success'] else "FAILED"
            print(f"  {result['display_name']}: {status} ({result['duration']:.2f}s)")
            if not result['success'] and result.get('error'):
                print(f"    Error: {result['error']}")
        
        # Requirements coverage
        coverage = report['requirements_coverage']
        print(f"\nRequirements Coverage:")
        print("-" * 40)
        print(f"  Total Requirements: {coverage['total_requirements']}")
        print(f"  Covered Requirements: {coverage['covered_requirements']}")
        print(f"  Coverage Percentage: {coverage['coverage_percentage']:.1f}%")
        if coverage['uncovered_requirements']:
            print(f"  Uncovered: {', '.join(coverage['uncovered_requirements'])}")
        
        # Performance summary
        perf = report['performance_summary']
        print(f"\nPerformance Summary:")
        print("-" * 40)
        print(f"  Performance Tests Run: {'YES' if perf['performance_tests_run'] else 'NO'}")
        print(f"  Performance Acceptable: {'YES' if perf['performance_acceptable'] else 'NO'}")
        print(f"  Overhead Analysis: {perf['overhead_analysis']}")
        
        # Recommendations
        recommendations = report['recommendations']
        if recommendations:
            print(f"\nRecommendations:")
            print("-" * 40)
            for rec in recommendations:
                print(f"  [{rec['priority']}] {rec['category']}: {rec['recommendation']}")
        
        print("="*80)
    
    def run_specific_suites(self, suite_names: list) -> dict:
        """Run specific test suites only."""
        self.logger.info(f"Running specific test suites: {suite_names}")
        self.overall_results['start_time'] = time.time()
        
        for suite_name in suite_names:
            if suite_name in self.test_suites:
                result = self.run_single_test_suite(suite_name)
                self.test_results[suite_name] = result
                
                self.overall_results['suites_run'] += 1
                if result['success']:
                    self.overall_results['suites_passed'] += 1
                else:
                    self.overall_results['suites_failed'] += 1
            else:
                self.logger.warning(f"Unknown test suite: {suite_name}")
        
        # Calculate overall results
        self.overall_results['end_time'] = time.time()
        self.overall_results['total_duration'] = (
            self.overall_results['end_time'] - self.overall_results['start_time']
        )
        self.overall_results['overall_success'] = (
            self.overall_results['suites_failed'] == 0
        )
        
        # Generate report
        report = self._generate_comprehensive_report()
        return report


def main():
    """Main entry point for comprehensive test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive reliability system tests')
    parser.add_argument('--suites', nargs='+', 
                       choices=['comprehensive', 'failure_injection', 'performance_impact', 'error_scenarios'],
                       help='Specific test suites to run (default: all)')
    parser.add_argument('--output-dir', default='test_results',
                       help='Output directory for test results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = ComprehensiveTestRunner(args.output_dir)
    
    # Run tests
    if args.suites:
        report = runner.run_specific_suites(args.suites)
    else:
        report = runner.run_all_test_suites()
    
    # Print summary
    runner.print_summary_report(report)
    
    # Exit with appropriate code
    success = report['overall_results']['overall_success']
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
