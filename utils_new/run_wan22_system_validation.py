"""
WAN22 System Validation Test Runner
Comprehensive test runner for all system validation tests
Task 12.2 Implementation
"""

import unittest
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class WAN22ValidationTestRunner:
    """Comprehensive test runner for WAN22 system validation"""
    
    def __init__(self, results_dir: str = "validation_test_results"):
        """Initialize validation test runner"""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Test categories
        self.test_categories = {
            'rtx_4080_optimization': 'RTX 4080 Optimization Tests',
            'threadripper_pro_optimization': 'Threadripper PRO Optimization Tests',
            'edge_case_validation': 'Edge Case Validation Tests',
            'syntax_validation': 'Syntax Validation Tests'
        }
        
        # Test results
        self.test_results = {}
        self.overall_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'error_tests': 0,
            'skipped_tests': 0,
            'success_rate': 0.0,
            'execution_time': 0.0
        }
    
    def run_all_validation_tests(self) -> Dict[str, Any]:
        """Run all validation tests and generate comprehensive report"""
        self.logger.info("Starting WAN22 system validation tests")
        start_time = time.time()
        
        try:
            # Import test modules
            from wan22_system_validation_tests import (
                RTX4080OptimizationValidationTests,
                ThreadripperPROOptimizationValidationTests,
                EdgeCaseValidationTests,
                SyntaxValidationTests
            )
            
            # Run each test category
            test_classes = [
                ('rtx_4080_optimization', RTX4080OptimizationValidationTests),
                ('threadripper_pro_optimization', ThreadripperPROOptimizationValidationTests),
                ('edge_case_validation', EdgeCaseValidationTests),
                ('syntax_validation', SyntaxValidationTests)
            ]
            
            for category, test_class in test_classes:
                self.logger.info(f"Running {self.test_categories[category]}")
                category_results = self._run_test_category(category, test_class)
                self.test_results[category] = category_results
            
            # Calculate overall results
            self._calculate_overall_results()
            
            # Generate comprehensive report
            execution_time = time.time() - start_time
            self.overall_results['execution_time'] = execution_time
            
            report = self._generate_validation_report()
            
            # Save report
            self._save_validation_report(report)
            
            self.logger.info(f"Validation tests completed in {execution_time:.2f}s")
            self.logger.info(f"Success rate: {self.overall_results['success_rate']:.1f}%")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Validation test execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _run_test_category(self, category: str, test_class) -> Dict[str, Any]:
        """Run tests for a specific category"""
        category_results = {
            'category': category,
            'category_name': self.test_categories[category],
            'tests': {},
            'summary': {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'skipped': 0
            },
            'execution_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromTestCase(test_class)
            
            # Run tests with custom result collector
            result_collector = ValidationTestResult()
            runner = unittest.TextTestRunner(
                stream=sys.stdout,
                verbosity=1,
                resultclass=lambda stream, descriptions, verbosity: result_collector
            )
            
            test_result = runner.run(suite)
            
            # Process results
            category_results['tests'] = result_collector.get_test_results()
            category_results['summary'] = {
                'total': test_result.testsRun,
                'passed': test_result.testsRun - len(test_result.failures) - len(test_result.errors) - len(test_result.skipped),
                'failed': len(test_result.failures),
                'errors': len(test_result.errors),
                'skipped': len(test_result.skipped)
            }
            
            category_results['execution_time'] = time.time() - start_time
            
            self.logger.info(f"Category {category}: {category_results['summary']['passed']}/{category_results['summary']['total']} tests passed")
            
        except Exception as e:
            self.logger.error(f"Error running category {category}: {e}")
            category_results['error'] = str(e)
        
        return category_results
    
    def _calculate_overall_results(self):
        """Calculate overall test results"""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        error_tests = 0
        skipped_tests = 0
        
        for category_results in self.test_results.values():
            if 'summary' in category_results:
                summary = category_results['summary']
                total_tests += summary['total']
                passed_tests += summary['passed']
                failed_tests += summary['failed']
                error_tests += summary['errors']
                skipped_tests += summary['skipped']
        
        self.overall_results.update({
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'error_tests': error_tests,
            'skipped_tests': skipped_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0.0
        })
    
    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_type': 'WAN22 System Validation',
            'overall_results': self.overall_results,
            'category_results': self.test_results,
            'recommendations': [],
            'critical_issues': [],
            'warnings': []
        }
        
        # Analyze results and generate recommendations
        self._analyze_results_and_generate_recommendations(report)
        
        return report
    
    def _analyze_results_and_generate_recommendations(self, report: Dict[str, Any]):
        """Analyze test results and generate recommendations"""
        recommendations = []
        critical_issues = []
        warnings = []
        
        # Check overall success rate
        success_rate = self.overall_results['success_rate']
        if success_rate < 90.0:
            critical_issues.append(f"Overall test success rate is low: {success_rate:.1f}%")
            recommendations.append("Review failed tests and address underlying issues")
        elif success_rate < 95.0:
            warnings.append(f"Test success rate could be improved: {success_rate:.1f}%")
        
        # Analyze category-specific results
        for category, results in self.test_results.items():
            if 'summary' in results:
                summary = results['summary']
                category_success_rate = (summary['passed'] / summary['total'] * 100) if summary['total'] > 0 else 0.0
                
                if category_success_rate < 80.0:
                    critical_issues.append(f"{self.test_categories[category]} has low success rate: {category_success_rate:.1f}%")
                
                # Category-specific recommendations
                if category == 'rtx_4080_optimization' and summary['failed'] > 0:
                    recommendations.append("Review RTX 4080 optimization settings and hardware detection")
                
                elif category == 'threadripper_pro_optimization' and summary['failed'] > 0:
                    recommendations.append("Review Threadripper PRO CPU utilization and NUMA optimization")
                
                elif category == 'edge_case_validation' and summary['failed'] > 0:
                    recommendations.append("Improve error handling and fallback mechanisms for edge cases")
                    recommendations.append("Test with actual low-VRAM hardware configurations")
                
                elif category == 'syntax_validation' and summary['failed'] > 0:
                    critical_issues.append("Syntax validation failures indicate code quality issues")
                    recommendations.append("Fix syntax errors in critical system files")
                    recommendations.append("Implement automated syntax checking in CI/CD pipeline")
        
        # Check for specific test failures
        for category, results in self.test_results.items():
            if 'tests' in results:
                for test_name, test_result in results['tests'].items():
                    if test_result['status'] == 'FAIL':
                        if 'syntax' in test_name.lower():
                            critical_issues.append(f"Syntax validation test failed: {test_name}")
                        elif 'vram' in test_name.lower():
                            warnings.append(f"VRAM management test failed: {test_name}")
                        elif 'optimization' in test_name.lower():
                            warnings.append(f"Optimization test failed: {test_name}")
        
        # General recommendations
        if self.overall_results['error_tests'] > 0:
            recommendations.append("Investigate test environment setup and dependencies")
        
        if self.overall_results['execution_time'] > 300:  # 5 minutes
            recommendations.append("Consider optimizing test execution time")
        
        # Add to report
        report['recommendations'] = recommendations
        report['critical_issues'] = critical_issues
        report['warnings'] = warnings
    
    def _save_validation_report(self, report: Dict[str, Any]):
        """Save validation report to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.results_dir / f"wan22_validation_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also save a summary report
        summary_path = self.results_dir / f"wan22_validation_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("WAN22 System Validation Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Timestamp: {report['timestamp']}\n")
            f.write(f"Total Tests: {self.overall_results['total_tests']}\n")
            f.write(f"Passed: {self.overall_results['passed_tests']}\n")
            f.write(f"Failed: {self.overall_results['failed_tests']}\n")
            f.write(f"Errors: {self.overall_results['error_tests']}\n")
            f.write(f"Success Rate: {self.overall_results['success_rate']:.1f}%\n")
            f.write(f"Execution Time: {self.overall_results['execution_time']:.2f}s\n\n")
            
            # Category breakdown
            f.write("Category Breakdown:\n")
            f.write("-" * 20 + "\n")
            for category, results in self.test_results.items():
                if 'summary' in results:
                    summary = results['summary']
                    f.write(f"{self.test_categories[category]}: {summary['passed']}/{summary['total']} passed\n")
            
            f.write("\n")
            
            # Critical issues
            if report['critical_issues']:
                f.write("Critical Issues:\n")
                f.write("-" * 15 + "\n")
                for issue in report['critical_issues']:
                    f.write(f"- {issue}\n")
                f.write("\n")
            
            # Recommendations
            if report['recommendations']:
                f.write("Recommendations:\n")
                f.write("-" * 15 + "\n")
                for rec in report['recommendations']:
                    f.write(f"- {rec}\n")
        
        self.logger.info(f"Validation report saved to: {report_path}")
        self.logger.info(f"Summary report saved to: {summary_path}")

class ValidationTestResult(unittest.TestResult):
    """Custom test result collector for validation tests"""
    
    def __init__(self):
        super().__init__()
        self.test_results = {}
    
    def startTest(self, test):
        super().startTest(test)
        test_name = f"{test.__class__.__name__}.{test._testMethodName}"
        self.test_results[test_name] = {
            'status': 'RUNNING',
            'start_time': time.time(),
            'end_time': None,
            'duration': 0.0,
            'message': '',
            'traceback': ''
        }
    
    def stopTest(self, test):
        super().stopTest(test)
        test_name = f"{test.__class__.__name__}.{test._testMethodName}"
        if test_name in self.test_results:
            self.test_results[test_name]['end_time'] = time.time()
            self.test_results[test_name]['duration'] = (
                self.test_results[test_name]['end_time'] - 
                self.test_results[test_name]['start_time']
            )
            if self.test_results[test_name]['status'] == 'RUNNING':
                self.test_results[test_name]['status'] = 'PASS'
    
    def addSuccess(self, test):
        super().addSuccess(test)
        test_name = f"{test.__class__.__name__}.{test._testMethodName}"
        if test_name in self.test_results:
            self.test_results[test_name]['status'] = 'PASS'
    
    def addError(self, test, err):
        super().addError(test, err)
        test_name = f"{test.__class__.__name__}.{test._testMethodName}"
        if test_name in self.test_results:
            self.test_results[test_name]['status'] = 'ERROR'
            self.test_results[test_name]['message'] = str(err[1])
            self.test_results[test_name]['traceback'] = self._exc_info_to_string(err, test)
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        test_name = f"{test.__class__.__name__}.{test._testMethodName}"
        if test_name in self.test_results:
            self.test_results[test_name]['status'] = 'FAIL'
            self.test_results[test_name]['message'] = str(err[1])
            self.test_results[test_name]['traceback'] = self._exc_info_to_string(err, test)
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        test_name = f"{test.__class__.__name__}.{test._testMethodName}"
        if test_name in self.test_results:
            self.test_results[test_name]['status'] = 'SKIP'
            self.test_results[test_name]['message'] = reason
    
    def get_test_results(self) -> Dict[str, Any]:
        """Get collected test results"""
        return self.test_results

def main():
    """Main function to run validation tests"""
    print("WAN22 System Validation Test Runner")
    print("=" * 40)
    
    # Create test runner
    runner = WAN22ValidationTestRunner()
    
    # Run all validation tests
    report = runner.run_all_validation_tests()
    
    # Print summary
    if 'overall_results' in report:
        results = report['overall_results']
        print(f"\nValidation Results:")
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed_tests']}")
        print(f"Failed: {results['failed_tests']}")
        print(f"Errors: {results['error_tests']}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        print(f"Execution Time: {results['execution_time']:.2f}s")
        
        # Print critical issues
        if report.get('critical_issues'):
            print(f"\nCritical Issues:")
            for issue in report['critical_issues']:
                print(f"  - {issue}")
        
        # Print recommendations
        if report.get('recommendations'):
            print(f"\nRecommendations:")
            for rec in report['recommendations'][:5]:  # Show first 5
                print(f"  - {rec}")
        
        # Exit with appropriate code
        if results['success_rate'] >= 95.0:
            print("\n✅ Validation tests passed successfully!")
            sys.exit(0)
        elif results['success_rate'] >= 80.0:
            print("\n⚠️  Validation tests passed with warnings.")
            sys.exit(0)
        else:
            print("\n❌ Validation tests failed!")
            sys.exit(1)
    else:
        print("\n❌ Validation test execution failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()