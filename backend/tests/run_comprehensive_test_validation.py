#!/usr/bin/env python3
"""
Comprehensive Test Validation Script

This script runs the complete Enhanced Model Availability testing suite,
including integration tests, stress tests, chaos engineering tests,
performance benchmarks, and user acceptance tests.

It generates a comprehensive validation report with pass/fail status,
performance metrics, and recommendations.

Usage:
    python run_comprehensive_test_validation.py [options]

Options:
    --quick: Run quick validation (reduced test iterations)
    --full: Run full validation suite (default)
    --performance-only: Run only performance tests
    --stress-only: Run only stress tests
    --chaos-only: Run only chaos engineering tests
    --user-acceptance-only: Run only user acceptance tests
    --report-format: json|html|text (default: text)
    --output-file: Output file for report (default: stdout)
"""

import asyncio
import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

# Import test suites
from test_enhanced_model_availability_integration import TestEnhancedModelAvailabilityIntegration
from test_download_stress_testing import DownloadStressTestSuite
from test_chaos_engineering import ChaosEngineeringTestSuite
from test_performance_benchmarks_enhanced import PerformanceBenchmarkSuite
from test_user_acceptance_workflows import UserAcceptanceTestSuite


class ComprehensiveTestValidator:
    """Main test validation orchestrator."""

    def __init__(self, options: Dict[str, Any]):
        self.options = options
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        self.validation_report = {}

    async def run_validation(self):
        """Run comprehensive test validation."""
        print("=" * 80)
        print("ENHANCED MODEL AVAILABILITY - COMPREHENSIVE TEST VALIDATION")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test mode: {self.options.get('mode', 'full')}")
        print("=" * 80)

        self.start_time = time.time()

        try:
            # Run test suites based on options
            if self.options.get('mode') == 'full' or self.options.get('mode') is None:
                await self._run_all_test_suites()
            elif self.options.get('mode') == 'quick':
                await self._run_quick_validation()
            elif self.options.get('mode') == 'performance-only':
                await self._run_performance_tests_only()
            elif self.options.get('mode') == 'stress-only':
                await self._run_stress_tests_only()
            elif self.options.get('mode') == 'chaos-only':
                await self._run_chaos_tests_only()
            elif self.options.get('mode') == 'user-acceptance-only':
                await self._run_user_acceptance_tests_only()

            self.end_time = time.time()

            # Generate validation report
            await self._generate_validation_report()

            # Output report
            await self._output_report()

        except Exception as e:
            print(f"❌ Validation failed with error: {str(e)}")
            traceback.print_exc()
            return False

        return self._determine_overall_validation_result()

    async def _run_all_test_suites(self):
        """Run all test suites."""
        print("\n🚀 Running Complete Test Suite Validation...")

        # 1. Integration Tests
        await self._run_integration_tests()

        # 2. Stress Tests
        await self._run_stress_tests()

        # 3. Chaos Engineering Tests
        await self._run_chaos_tests()

        # 4. Performance Benchmarks
        await self._run_performance_benchmarks()

        # 5. User Acceptance Tests
        await self._run_user_acceptance_tests()

    async def _run_quick_validation(self):
        """Run quick validation with reduced test iterations."""
        print("\n⚡ Running Quick Validation...")

        # Run subset of tests with reduced iterations
        await self._run_integration_tests(quick=True)
        await self._run_performance_benchmarks(quick=True)
        await self._run_user_acceptance_tests(quick=True)

    async def _run_integration_tests(self, quick=False):
        """Run integration tests."""
        print("\n" + "─" * 60)
        print("🔗 INTEGRATION TESTS")
        print("─" * 60)

        try:
            # Create integration test instance
            integration_tests = TestEnhancedModelAvailabilityIntegration()

            # Run key integration tests
            test_methods = [
                ('complete_model_request_workflow', integration_tests.test_complete_model_request_workflow),
                ('model_unavailable_fallback_workflow', integration_tests.test_model_unavailable_fallback_workflow),
                ('health_monitoring_integration', integration_tests.test_health_monitoring_integration),
                ('update_management_integration', integration_tests.test_update_management_integration),
                ('websocket_notification_integration', integration_tests.test_websocket_notification_integration),
                ('error_recovery_escalation', integration_tests.test_error_recovery_escalation),
                ('concurrent_model_operations', integration_tests.test_concurrent_model_operations),
                ('system_startup_integration', integration_tests.test_system_startup_integration)
            ]

            if quick:
                test_methods = test_methods[:4]  # Run only first 4 tests in quick mode

            integration_results = {}
            for test_name, test_method in test_methods:
                print(f"  Running: {test_name}")
                try:
                    # Setup test environment
                    enhanced_system = await integration_tests.enhanced_system()
                    
                    # Run test
                    start_time = time.time()
                    await test_method(enhanced_system)
                    end_time = time.time()

                    integration_results[test_name] = {
                        'status': 'PASSED',
                        'execution_time': end_time - start_time,
                        'error': None
                    }
                    print(f"    ✅ PASSED ({end_time - start_time:.2f}s)")

                except Exception as e:
                    integration_results[test_name] = {
                        'status': 'FAILED',
                        'execution_time': 0,
                        'error': str(e)
                    }
                    print(f"    ❌ FAILED: {str(e)}")

            self.test_results['integration_tests'] = integration_results

        except Exception as e:
            print(f"❌ Integration tests failed to initialize: {str(e)}")
            self.test_results['integration_tests'] = {'error': str(e)}

    async def _run_stress_tests(self):
        """Run stress tests."""
        print("\n" + "─" * 60)
        print("💪 STRESS TESTS")
        print("─" * 60)

        try:
            stress_suite = DownloadStressTestSuite()
            stress_results = await stress_suite.run_comprehensive_stress_tests()
            self.test_results['stress_tests'] = stress_results

        except Exception as e:
            print(f"❌ Stress tests failed: {str(e)}")
            self.test_results['stress_tests'] = {'error': str(e)}

    async def _run_chaos_tests(self):
        """Run chaos engineering tests."""
        print("\n" + "─" * 60)
        print("🌪️  CHAOS ENGINEERING TESTS")
        print("─" * 60)

        try:
            chaos_suite = ChaosEngineeringTestSuite()
            chaos_results = await chaos_suite.run_comprehensive_chaos_tests()
            self.test_results['chaos_tests'] = chaos_results

        except Exception as e:
            print(f"❌ Chaos engineering tests failed: {str(e)}")
            self.test_results['chaos_tests'] = {'error': str(e)}

    async def _run_performance_benchmarks(self, quick=False):
        """Run performance benchmarks."""
        print("\n" + "─" * 60)
        print("📊 PERFORMANCE BENCHMARKS")
        print("─" * 60)

        try:
            benchmark_suite = PerformanceBenchmarkSuite()
            
            if quick:
                # Run only essential benchmarks in quick mode
                benchmarks = await benchmark_suite.benchmark_enhanced_downloader_performance()
                load_results = []
            else:
                benchmark_results = await benchmark_suite.run_comprehensive_performance_benchmarks()
                benchmarks = benchmark_results.get('benchmarks', [])
                load_results = benchmark_results.get('load_results', [])

            self.test_results['performance_benchmarks'] = {
                'benchmarks': benchmarks,
                'load_results': load_results
            }

        except Exception as e:
            print(f"❌ Performance benchmarks failed: {str(e)}")
            self.test_results['performance_benchmarks'] = {'error': str(e)}

    async def _run_user_acceptance_tests(self, quick=False):
        """Run user acceptance tests."""
        print("\n" + "─" * 60)
        print("👥 USER ACCEPTANCE TESTS")
        print("─" * 60)

        try:
            user_suite = UserAcceptanceTestSuite()
            
            if quick:
                # Run only essential user tests in quick mode
                results = []
                results.append(await user_suite.test_new_user_first_model_request_workflow())
                results.append(await user_suite.test_model_unavailable_fallback_workflow())
            else:
                results = await user_suite.run_comprehensive_user_acceptance_tests()

            self.test_results['user_acceptance_tests'] = results

        except Exception as e:
            print(f"❌ User acceptance tests failed: {str(e)}")
            self.test_results['user_acceptance_tests'] = {'error': str(e)}

    async def _run_performance_tests_only(self):
        """Run only performance tests."""
        await self._run_performance_benchmarks()

    async def _run_stress_tests_only(self):
        """Run only stress tests."""
        await self._run_stress_tests()

    async def _run_chaos_tests_only(self):
        """Run only chaos engineering tests."""
        await self._run_chaos_tests()

    async def _run_user_acceptance_tests_only(self):
        """Run only user acceptance tests."""
        await self._run_user_acceptance_tests()

    async def _generate_validation_report(self):
        """Generate comprehensive validation report."""
        total_time = self.end_time - self.start_time if self.end_time and self.start_time else 0

        self.validation_report = {
            'validation_summary': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
                'total_execution_time': total_time,
                'test_mode': self.options.get('mode', 'full'),
                'overall_result': self._determine_overall_validation_result()
            },
            'test_results': self.test_results,
            'performance_summary': self._generate_performance_summary(),
            'reliability_summary': self._generate_reliability_summary(),
            'user_experience_summary': self._generate_user_experience_summary(),
            'recommendations': self._generate_recommendations(),
            'validation_criteria': self._evaluate_validation_criteria()
        }

    def _generate_performance_summary(self):
        """Generate performance summary."""
        performance_data = self.test_results.get('performance_benchmarks', {})
        
        if 'error' in performance_data:
            return {'status': 'ERROR', 'message': performance_data['error']}

        benchmarks = performance_data.get('benchmarks', [])
        load_results = performance_data.get('load_results', [])

        if not benchmarks:
            return {'status': 'NO_DATA', 'message': 'No performance data available'}

        # Calculate performance metrics
        avg_ops_per_sec = sum(b.operations_per_second for b in benchmarks) / len(benchmarks)
        avg_response_time = sum(b.average_time_ms for b in benchmarks) / len(benchmarks)
        total_memory_usage = sum(b.memory_usage_mb for b in benchmarks)

        return {
            'status': 'SUCCESS',
            'average_operations_per_second': avg_ops_per_sec,
            'average_response_time_ms': avg_response_time,
            'total_memory_usage_mb': total_memory_usage,
            'load_test_results': len(load_results),
            'performance_grade': self._calculate_performance_grade(avg_ops_per_sec, avg_response_time)
        }

    def _generate_reliability_summary(self):
        """Generate reliability summary."""
        stress_results = self.test_results.get('stress_tests', {})
        chaos_results = self.test_results.get('chaos_tests', {})

        if 'error' in stress_results or 'error' in chaos_results:
            return {'status': 'ERROR', 'message': 'Reliability tests failed to execute'}

        # Analyze stress test results
        stress_passed = 0
        stress_total = 0
        if isinstance(stress_results, dict) and 'error' not in stress_results:
            for test_name, result in stress_results.items():
                if isinstance(result, dict) and 'status' in result:
                    stress_total += 1
                    if result['status'] == 'PASSED':
                        stress_passed += 1

        # Analyze chaos test results
        chaos_passed = 0
        chaos_total = 0
        if isinstance(chaos_results, dict) and 'error' not in chaos_results:
            for test_name, result in chaos_results.items():
                if isinstance(result, dict) and 'status' in result:
                    chaos_total += 1
                    if result['status'] == 'PASSED':
                        chaos_passed += 1

        stress_success_rate = stress_passed / stress_total if stress_total > 0 else 0
        chaos_success_rate = chaos_passed / chaos_total if chaos_total > 0 else 0
        overall_reliability = (stress_success_rate + chaos_success_rate) / 2

        return {
            'status': 'SUCCESS',
            'stress_test_success_rate': stress_success_rate,
            'chaos_test_success_rate': chaos_success_rate,
            'overall_reliability_score': overall_reliability,
            'reliability_grade': self._calculate_reliability_grade(overall_reliability)
        }

    def _generate_user_experience_summary(self):
        """Generate user experience summary."""
        user_results = self.test_results.get('user_acceptance_tests', [])

        if isinstance(user_results, dict) and 'error' in user_results:
            return {'status': 'ERROR', 'message': user_results['error']}

        if not user_results:
            return {'status': 'NO_DATA', 'message': 'No user acceptance data available'}

        # Calculate user experience metrics
        successful_workflows = sum(1 for r in user_results if hasattr(r, 'success') and r.success)
        total_workflows = len(user_results)
        avg_satisfaction = sum(r.user_satisfaction_score for r in user_results if hasattr(r, 'user_satisfaction_score')) / total_workflows if total_workflows > 0 else 0
        total_issues = sum(len(r.issues_encountered) for r in user_results if hasattr(r, 'issues_encountered'))

        return {
            'status': 'SUCCESS',
            'workflow_success_rate': successful_workflows / total_workflows if total_workflows > 0 else 0,
            'average_user_satisfaction': avg_satisfaction,
            'total_issues_encountered': total_issues,
            'user_experience_grade': self._calculate_user_experience_grade(avg_satisfaction, successful_workflows / total_workflows if total_workflows > 0 else 0)
        }

    def _generate_recommendations(self):
        """Generate recommendations based on test results."""
        recommendations = []

        # Performance recommendations
        performance_summary = self.validation_report.get('performance_summary', {})
        if performance_summary.get('performance_grade', 'F') in ['C', 'D', 'F']:
            recommendations.append({
                'category': 'Performance',
                'priority': 'High',
                'recommendation': 'System performance needs improvement - optimize critical paths and resource usage'
            })

        # Reliability recommendations
        reliability_summary = self.validation_report.get('reliability_summary', {})
        if reliability_summary.get('reliability_grade', 'F') in ['C', 'D', 'F']:
            recommendations.append({
                'category': 'Reliability',
                'priority': 'High',
                'recommendation': 'System reliability needs improvement - enhance error handling and recovery mechanisms'
            })

        # User experience recommendations
        ux_summary = self.validation_report.get('user_experience_summary', {})
        if ux_summary.get('user_experience_grade', 'F') in ['C', 'D', 'F']:
            recommendations.append({
                'category': 'User Experience',
                'priority': 'Medium',
                'recommendation': 'User experience needs improvement - simplify workflows and enhance error messaging'
            })

        # Integration recommendations
        integration_results = self.test_results.get('integration_tests', {})
        if isinstance(integration_results, dict):
            failed_tests = [name for name, result in integration_results.items() 
                          if isinstance(result, dict) and result.get('status') == 'FAILED']
            if failed_tests:
                recommendations.append({
                    'category': 'Integration',
                    'priority': 'High',
                    'recommendation': f'Fix failing integration tests: {", ".join(failed_tests)}'
                })

        if not recommendations:
            recommendations.append({
                'category': 'Overall',
                'priority': 'Low',
                'recommendation': 'System validation passed successfully - maintain current quality standards'
            })

        return recommendations

    def _evaluate_validation_criteria(self):
        """Evaluate validation criteria."""
        criteria = {
            'functional_validation': {
                'integration_test_pass_rate': self._calculate_integration_pass_rate(),
                'required_pass_rate': 0.95,
                'status': 'PASS' if self._calculate_integration_pass_rate() >= 0.95 else 'FAIL'
            },
            'performance_validation': {
                'average_response_time': self._get_average_response_time(),
                'required_response_time': 500.0,  # ms
                'status': 'PASS' if self._get_average_response_time() <= 500.0 else 'FAIL'
            },
            'reliability_validation': {
                'system_stability_score': self._get_system_stability_score(),
                'required_stability_score': 0.8,
                'status': 'PASS' if self._get_system_stability_score() >= 0.8 else 'FAIL'
            },
            'user_experience_validation': {
                'user_satisfaction_score': self._get_user_satisfaction_score(),
                'required_satisfaction_score': 0.8,
                'status': 'PASS' if self._get_user_satisfaction_score() >= 0.8 else 'FAIL'
            }
        }

        # Overall validation status
        all_passed = all(criterion['status'] == 'PASS' for criterion in criteria.values())
        criteria['overall_validation'] = {
            'status': 'PASS' if all_passed else 'FAIL',
            'passed_criteria': sum(1 for c in criteria.values() if c.get('status') == 'PASS'),
            'total_criteria': len(criteria) - 1  # Exclude overall_validation itself
        }

        return criteria

    def _calculate_integration_pass_rate(self):
        """Calculate integration test pass rate."""
        integration_results = self.test_results.get('integration_tests', {})
        if 'error' in integration_results:
            return 0.0

        total_tests = len(integration_results)
        passed_tests = sum(1 for result in integration_results.values() 
                          if isinstance(result, dict) and result.get('status') == 'PASSED')
        
        return passed_tests / total_tests if total_tests > 0 else 0.0

    def _get_average_response_time(self):
        """Get average response time from performance tests."""
        performance_summary = self.validation_report.get('performance_summary', {})
        return performance_summary.get('average_response_time_ms', 1000.0)

    def _get_system_stability_score(self):
        """Get system stability score from reliability tests."""
        reliability_summary = self.validation_report.get('reliability_summary', {})
        return reliability_summary.get('overall_reliability_score', 0.0)

    def _get_user_satisfaction_score(self):
        """Get user satisfaction score from user acceptance tests."""
        ux_summary = self.validation_report.get('user_experience_summary', {})
        return ux_summary.get('average_user_satisfaction', 0.0)

    def _calculate_performance_grade(self, ops_per_sec, response_time):
        """Calculate performance grade."""
        if ops_per_sec > 1000 and response_time < 100:
            return 'A'
        elif ops_per_sec > 500 and response_time < 200:
            return 'B'
        elif ops_per_sec > 200 and response_time < 500:
            return 'C'
        elif ops_per_sec > 100 and response_time < 1000:
            return 'D'
        else:
            return 'F'

    def _calculate_reliability_grade(self, reliability_score):
        """Calculate reliability grade."""
        if reliability_score >= 0.95:
            return 'A'
        elif reliability_score >= 0.85:
            return 'B'
        elif reliability_score >= 0.75:
            return 'C'
        elif reliability_score >= 0.65:
            return 'D'
        else:
            return 'F'

    def _calculate_user_experience_grade(self, satisfaction_score, success_rate):
        """Calculate user experience grade."""
        combined_score = (satisfaction_score + success_rate) / 2
        if combined_score >= 0.9:
            return 'A'
        elif combined_score >= 0.8:
            return 'B'
        elif combined_score >= 0.7:
            return 'C'
        elif combined_score >= 0.6:
            return 'D'
        else:
            return 'F'

    def _determine_overall_validation_result(self):
        """Determine overall validation result."""
        if not self.validation_report:
            return 'UNKNOWN'

        criteria = self.validation_report.get('validation_criteria', {})
        overall_criteria = criteria.get('overall_validation', {})
        
        return overall_criteria.get('status', 'UNKNOWN')

    async def _output_report(self):
        """Output validation report."""
        report_format = self.options.get('report_format', 'text')
        output_file = self.options.get('output_file')

        if report_format == 'json':
            report_content = json.dumps(self.validation_report, indent=2, default=str)
        elif report_format == 'html':
            report_content = self._generate_html_report()
        else:  # text format
            report_content = self._generate_text_report()

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            print(f"\n📄 Validation report saved to: {output_file}")
        else:
            print("\n" + "=" * 80)
            print("VALIDATION REPORT")
            print("=" * 80)
            print(report_content)

    def _generate_text_report(self):
        """Generate text format report."""
        report = []
        
        # Summary
        summary = self.validation_report['validation_summary']
        report.append(f"Validation Result: {summary['overall_result']}")
        report.append(f"Execution Time: {summary['total_execution_time']:.2f} seconds")
        report.append(f"Test Mode: {summary['test_mode']}")
        report.append("")

        # Performance Summary
        perf_summary = self.validation_report.get('performance_summary', {})
        if perf_summary.get('status') == 'SUCCESS':
            report.append("Performance Summary:")
            report.append(f"  - Average Operations/Second: {perf_summary.get('average_operations_per_second', 0):.2f}")
            report.append(f"  - Average Response Time: {perf_summary.get('average_response_time_ms', 0):.2f} ms")
            report.append(f"  - Performance Grade: {perf_summary.get('performance_grade', 'N/A')}")
            report.append("")

        # Reliability Summary
        rel_summary = self.validation_report.get('reliability_summary', {})
        if rel_summary.get('status') == 'SUCCESS':
            report.append("Reliability Summary:")
            report.append(f"  - Overall Reliability Score: {rel_summary.get('overall_reliability_score', 0):.2f}")
            report.append(f"  - Reliability Grade: {rel_summary.get('reliability_grade', 'N/A')}")
            report.append("")

        # User Experience Summary
        ux_summary = self.validation_report.get('user_experience_summary', {})
        if ux_summary.get('status') == 'SUCCESS':
            report.append("User Experience Summary:")
            report.append(f"  - Average User Satisfaction: {ux_summary.get('average_user_satisfaction', 0):.2f}")
            report.append(f"  - Workflow Success Rate: {ux_summary.get('workflow_success_rate', 0):.2f}")
            report.append(f"  - User Experience Grade: {ux_summary.get('user_experience_grade', 'N/A')}")
            report.append("")

        # Validation Criteria
        criteria = self.validation_report.get('validation_criteria', {})
        report.append("Validation Criteria:")
        for criterion_name, criterion_data in criteria.items():
            if criterion_name != 'overall_validation':
                status_icon = "✅" if criterion_data.get('status') == 'PASS' else "❌"
                report.append(f"  {status_icon} {criterion_name}: {criterion_data.get('status', 'UNKNOWN')}")
        report.append("")

        # Recommendations
        recommendations = self.validation_report.get('recommendations', [])
        if recommendations:
            report.append("Recommendations:")
            for rec in recommendations:
                priority_icon = "🔴" if rec['priority'] == 'High' else "🟡" if rec['priority'] == 'Medium' else "🟢"
                report.append(f"  {priority_icon} [{rec['category']}] {rec['recommendation']}")

        return "\n".join(report)

    def _generate_html_report(self):
        """Generate HTML format report."""
        # Basic HTML report template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced Model Availability - Test Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .pass {{ color: green; }}
                .fail {{ color: red; }}
                .grade-A {{ color: green; font-weight: bold; }}
                .grade-B {{ color: blue; font-weight: bold; }}
                .grade-C {{ color: orange; font-weight: bold; }}
                .grade-D {{ color: red; font-weight: bold; }}
                .grade-F {{ color: darkred; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Enhanced Model Availability - Test Validation Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Overall Result: <span class="{'pass' if self.validation_report['validation_summary']['overall_result'] == 'PASS' else 'fail'}">{self.validation_report['validation_summary']['overall_result']}</span></p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <p>Execution Time: {self.validation_report['validation_summary']['total_execution_time']:.2f} seconds</p>
                <p>Test Mode: {self.validation_report['validation_summary']['test_mode']}</p>
            </div>
            
            <!-- Add more sections as needed -->
            
        </body>
        </html>
        """
        return html


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Enhanced Model Availability Test Validation')
    parser.add_argument('--quick', action='store_true', help='Run quick validation')
    parser.add_argument('--full', action='store_true', help='Run full validation (default)')
    parser.add_argument('--performance-only', action='store_true', help='Run only performance tests')
    parser.add_argument('--stress-only', action='store_true', help='Run only stress tests')
    parser.add_argument('--chaos-only', action='store_true', help='Run only chaos engineering tests')
    parser.add_argument('--user-acceptance-only', action='store_true', help='Run only user acceptance tests')
    parser.add_argument('--report-format', choices=['json', 'html', 'text'], default='text', help='Report format')
    parser.add_argument('--output-file', help='Output file for report')

    args = parser.parse_args()

    # Determine test mode
    if args.quick:
        mode = 'quick'
    elif args.performance_only:
        mode = 'performance-only'
    elif args.stress_only:
        mode = 'stress-only'
    elif args.chaos_only:
        mode = 'chaos-only'
    elif args.user_acceptance_only:
        mode = 'user-acceptance-only'
    else:
        mode = 'full'

    options = {
        'mode': mode,
        'report_format': args.report_format,
        'output_file': args.output_file
    }

    # Run validation
    validator = ComprehensiveTestValidator(options)
    success = await validator.run_validation()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
