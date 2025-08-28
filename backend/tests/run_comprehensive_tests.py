#!/usr/bin/env python3
"""
Comprehensive Test Suite Runner
Executes all comprehensive integration tests and generates detailed reports
"""

import asyncio
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any

class ComprehensiveTestRunner:
    """Runner for comprehensive integration tests"""
    
    def __init__(self):
        self.test_results = {
            'test_suites': {},
            'overall_summary': {},
            'execution_metadata': {}
        }
        self.start_time = time.time()
    
    def run_test_suite(self, test_file: str, suite_name: str) -> Dict[str, Any]:
        """Run a specific test suite and capture results"""
        print(f"\n🧪 Running {suite_name}")
        print("-" * 50)
        
        suite_start = time.time()
        
        try:
            # Run pytest on the specific test file
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                test_file, 
                "-v", 
                "--tb=short",
                "--json-report",
                f"--json-report-file=test_results_{suite_name}.json"
            ], capture_output=True, text=True, cwd=Path(__file__).parent)
            
            suite_duration = time.time() - suite_start
            
            # Parse results
            success = result.returncode == 0
            
            suite_result = {
                'success': success,
                'duration': suite_duration,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            # Try to load JSON report if available
            json_report_path = Path(f"test_results_{suite_name}.json")
            if json_report_path.exists():
                try:
                    with open(json_report_path, 'r') as f:
                        json_report = json.load(f)
                    suite_result['detailed_results'] = json_report
                except Exception as e:
                    suite_result['json_parse_error'] = str(e)
            
            self.test_results['test_suites'][suite_name] = suite_result
            
            if success:
                print(f"✅ {suite_name} completed successfully in {suite_duration:.1f}s")
            else:
                print(f"❌ {suite_name} failed in {suite_duration:.1f}s")
                print(f"Error output: {result.stderr}")
            
            return suite_result
            
        except Exception as e:
            suite_duration = time.time() - suite_start
            error_result = {
                'success': False,
                'duration': suite_duration,
                'error': str(e),
                'exception_type': type(e).__name__
            }
            
            self.test_results['test_suites'][suite_name] = error_result
            print(f"💥 {suite_name} crashed: {str(e)}")
            
            return error_result
    
    def run_all_comprehensive_tests(self):
        """Run all comprehensive test suites"""
        print("🚀 Starting Comprehensive Integration Test Suite")
        print("=" * 60)
        
        # Define test suites to run
        test_suites = [
            {
                'file': 'test_model_integration_comprehensive.py',
                'name': 'model_integration_comprehensive',
                'description': 'Model Integration Bridge Tests'
            },
            {
                'file': 'test_real_generation_pipeline.py',
                'name': 'real_generation_pipeline',
                'description': 'Real Generation Pipeline Tests'
            },
            {
                'file': 'test_end_to_end_comprehensive.py',
                'name': 'end_to_end_comprehensive',
                'description': 'End-to-End Integration Tests'
            },
            {
                'file': 'test_performance_benchmarks.py',
                'name': 'performance_benchmarks',
                'description': 'Performance Benchmark Tests'
            },
            {
                'file': 'test_comprehensive_integration_suite.py',
                'name': 'comprehensive_integration_suite',
                'description': 'Full Integration Suite'
            }
        ]
        
        # Run each test suite
        for suite in test_suites:
            print(f"\n📋 {suite['description']}")
            self.run_test_suite(suite['file'], suite['name'])
        
        # Generate overall summary
        self.generate_summary()
        
        # Save comprehensive report
        self.save_comprehensive_report()
        
        # Print final results
        self.print_final_results()
    
    def generate_summary(self):
        """Generate overall test summary"""
        total_suites = len(self.test_results['test_suites'])
        successful_suites = sum(1 for result in self.test_results['test_suites'].values() 
                               if result['success'])
        failed_suites = total_suites - successful_suites
        
        total_duration = time.time() - self.start_time
        
        # Calculate detailed statistics from JSON reports
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for suite_name, suite_result in self.test_results['test_suites'].items():
            if 'detailed_results' in suite_result:
                json_report = suite_result['detailed_results']
                if 'summary' in json_report:
                    summary = json_report['summary']
                    total_tests += summary.get('total', 0)
                    passed_tests += summary.get('passed', 0)
                    failed_tests += summary.get('failed', 0)
        
        self.test_results['overall_summary'] = {
            'total_suites': total_suites,
            'successful_suites': successful_suites,
            'failed_suites': failed_suites,
            'suite_success_rate': (successful_suites / total_suites * 100) if total_suites > 0 else 0,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'test_success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'total_duration': total_duration
        }
        
        self.test_results['execution_metadata'] = {
            'start_time': self.start_time,
            'end_time': time.time(),
            'python_version': sys.version,
            'platform': sys.platform,
            'working_directory': str(Path.cwd())
        }
    
    def save_comprehensive_report(self):
        """Save comprehensive test report"""
        report_path = Path("comprehensive_test_report.json")
        
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"\n📄 Comprehensive report saved to: {report_path}")
        
        # Also save a human-readable summary
        summary_path = Path("test_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("COMPREHENSIVE TEST SUITE RESULTS\n")
            f.write("=" * 40 + "\n\n")
            
            summary = self.test_results['overall_summary']
            f.write(f"Test Suites: {summary['successful_suites']}/{summary['total_suites']} passed\n")
            f.write(f"Individual Tests: {summary['passed_tests']}/{summary['total_tests']} passed\n")
            f.write(f"Suite Success Rate: {summary['suite_success_rate']:.1f}%\n")
            f.write(f"Test Success Rate: {summary['test_success_rate']:.1f}%\n")
            f.write(f"Total Duration: {summary['total_duration']:.1f} seconds\n\n")
            
            f.write("SUITE DETAILS:\n")
            f.write("-" * 20 + "\n")
            
            for suite_name, suite_result in self.test_results['test_suites'].items():
                status = "✅ PASSED" if suite_result['success'] else "❌ FAILED"
                f.write(f"{suite_name}: {status} ({suite_result['duration']:.1f}s)\n")
        
        print(f"📄 Summary report saved to: {summary_path}")
    
    def print_final_results(self):
        """Print final test results"""
        summary = self.test_results['overall_summary']
        
        print("\n" + "=" * 60)
        print("🏁 COMPREHENSIVE TEST SUITE FINAL RESULTS")
        print("=" * 60)
        
        print(f"📊 Test Suites: {summary['successful_suites']}/{summary['total_suites']} passed")
        print(f"📊 Individual Tests: {summary['passed_tests']}/{summary['total_tests']} passed")
        print(f"📊 Suite Success Rate: {summary['suite_success_rate']:.1f}%")
        print(f"📊 Test Success Rate: {summary['test_success_rate']:.1f}%")
        print(f"⏱️  Total Duration: {summary['total_duration']:.1f} seconds")
        
        print(f"\n📋 Suite Breakdown:")
        for suite_name, suite_result in self.test_results['test_suites'].items():
            status_emoji = "✅" if suite_result['success'] else "❌"
            print(f"  {status_emoji} {suite_name}: {suite_result['duration']:.1f}s")
        
        # Overall assessment
        if summary['suite_success_rate'] >= 80:
            print(f"\n🎉 COMPREHENSIVE TEST SUITE PASSED!")
            print(f"   Success rate of {summary['suite_success_rate']:.1f}% meets the 80% threshold.")
        else:
            print(f"\n💥 COMPREHENSIVE TEST SUITE FAILED!")
            print(f"   Success rate of {summary['suite_success_rate']:.1f}% below 80% threshold.")
        
        return summary['suite_success_rate'] >= 80
    
    def run_specific_test(self, test_name: str):
        """Run a specific test suite"""
        test_mapping = {
            'model': 'test_model_integration_comprehensive.py',
            'pipeline': 'test_real_generation_pipeline.py',
            'e2e': 'test_end_to_end_comprehensive.py',
            'performance': 'test_performance_benchmarks.py',
            'full': 'test_comprehensive_integration_suite.py'
        }
        
        if test_name in test_mapping:
            test_file = test_mapping[test_name]
            print(f"🧪 Running specific test: {test_name}")
            result = self.run_test_suite(test_file, test_name)
            
            if result['success']:
                print(f"✅ {test_name} test completed successfully")
                return True
            else:
                print(f"❌ {test_name} test failed")
                return False
        else:
            print(f"❌ Unknown test name: {test_name}")
            print(f"Available tests: {', '.join(test_mapping.keys())}")
            return False

def main():
    """Main entry point"""
    runner = ComprehensiveTestRunner()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        success = runner.run_specific_test(test_name)
        sys.exit(0 if success else 1)
    else:
        # Run all tests
        success = runner.run_all_comprehensive_tests()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()