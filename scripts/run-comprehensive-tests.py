import pytest
#!/usr/bin/env python3
"""
Comprehensive Test Runner for React Frontend FastAPI Backend
Executes all testing categories and generates detailed reports
"""

import os
import sys
import subprocess
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class ComprehensiveTestRunner:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.frontend_path = project_root / "frontend"
        self.backend_path = project_root / "backend"
        self.results = {}
        self.start_time = None
        self.end_time = None

    def setup_environment(self):
        """Setup test environment"""
        print("🔧 Setting up test environment...")
        
        # Check if required directories exist
        if not self.frontend_path.exists():
            raise FileNotFoundError(f"Frontend directory not found: {self.frontend_path}")
        
        if not self.backend_path.exists():
            raise FileNotFoundError(f"Backend directory not found: {self.backend_path}")
        
        # Install frontend dependencies
        print("📦 Installing frontend dependencies...")
        subprocess.run(
            ["npm", "install"],
            cwd=self.frontend_path,
            check=True,
            capture_output=True
        )
        
        # Install backend dependencies
        print("📦 Installing backend dependencies...")
        subprocess.run(
            ["pip", "install", "-r", "requirements.txt"],
            cwd=self.backend_path,
            check=True,
            capture_output=True
        )
        
        print("✅ Environment setup complete")

    def run_frontend_tests(self) -> Dict[str, Any]:
        """Run frontend tests"""
        print("\n🎯 Running frontend tests...")
        
        frontend_results = {}
        
        # Unit tests
        print("  📋 Running unit tests...")
        try:
            result = subprocess.run(
                ["npm", "run", "test:unit", "--", "--run"],
                cwd=self.frontend_path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )
            
            frontend_results["unit"] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "output": result.stdout,
                "errors": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            frontend_results["unit"] = {
                "status": "timeout",
                "output": "",
                "errors": "Test execution timed out",
                "returncode": -1
            }
        except Exception as e:
            frontend_results["unit"] = {
                "status": "error",
                "output": "",
                "errors": str(e),
                "returncode": -1
            }
        
        # Integration tests
        print("  🔗 Running integration tests...")
        try:
            result = subprocess.run(
                ["npm", "run", "test:integration", "--", "--run"],
                cwd=self.frontend_path,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes
            )
            
            frontend_results["integration"] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "output": result.stdout,
                "errors": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            frontend_results["integration"] = {
                "status": "timeout",
                "output": "",
                "errors": "Test execution timed out",
                "returncode": -1
            }
        except Exception as e:
            frontend_results["integration"] = {
                "status": "error",
                "output": "",
                "errors": str(e),
                "returncode": -1
            }
        
        # E2E tests
        print("  🌐 Running E2E tests...")
        try:
            result = subprocess.run(
                ["npm", "run", "test:e2e", "--", "--run"],
                cwd=self.frontend_path,
                capture_output=True,
                text=True,
                timeout=900  # 15 minutes
            )
            
            frontend_results["e2e"] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "output": result.stdout,
                "errors": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            frontend_results["e2e"] = {
                "status": "timeout",
                "output": "",
                "errors": "Test execution timed out",
                "returncode": -1
            }
        except Exception as e:
            frontend_results["e2e"] = {
                "status": "error",
                "output": "",
                "errors": str(e),
                "returncode": -1
            }
        
        # Performance tests
        print("  ⚡ Running performance tests...")
        try:
            result = subprocess.run(
                ["npm", "run", "test:performance", "--", "--run"],
                cwd=self.frontend_path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )
            
            frontend_results["performance"] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "output": result.stdout,
                "errors": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            frontend_results["performance"] = {
                "status": "timeout",
                "output": "",
                "errors": "Test execution timed out",
                "returncode": -1
            }
        except Exception as e:
            frontend_results["performance"] = {
                "status": "error",
                "output": "",
                "errors": str(e),
                "returncode": -1
            }
        
        return frontend_results

    def run_backend_tests(self) -> Dict[str, Any]:
        """Run backend tests"""
        print("\n🔧 Running backend tests...")
        
        backend_results = {}
        
        # Unit tests
        print("  📋 Running unit tests...")
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
                cwd=self.backend_path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )
            
            backend_results["unit"] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "output": result.stdout,
                "errors": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            backend_results["unit"] = {
                "status": "timeout",
                "output": "",
                "errors": "Test execution timed out",
                "returncode": -1
            }
        except Exception as e:
            backend_results["unit"] = {
                "status": "error",
                "output": "",
                "errors": str(e),
                "returncode": -1
            }
        
        # API tests
        print("  🌐 Running API tests...")
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/test_comprehensive_api.py", "-v"],
                cwd=self.backend_path,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes
            )
            
            backend_results["api"] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "output": result.stdout,
                "errors": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            backend_results["api"] = {
                "status": "timeout",
                "output": "",
                "errors": "Test execution timed out",
                "returncode": -1
            }
        except Exception as e:
            backend_results["api"] = {
                "status": "error",
                "output": "",
                "errors": str(e),
                "returncode": -1
            }
        
        # Performance tests
        print("  ⚡ Running performance tests...")
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/test_performance_monitoring.py", "-v"],
                cwd=self.backend_path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )
            
            backend_results["performance"] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "output": result.stdout,
                "errors": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            backend_results["performance"] = {
                "status": "timeout",
                "output": "",
                "errors": "Test execution timed out",
                "returncode": -1
            }
        except Exception as e:
            backend_results["performance"] = {
                "status": "error",
                "output": "",
                "errors": str(e),
                "returncode": -1
            }
        
        return backend_results

    def run_integration_tests(self) -> Dict[str, Any]:
        """Run full-stack integration tests"""
        print("\n🔗 Running full-stack integration tests...")
        
        integration_results = {}
        
        # Start backend server
        print("  🚀 Starting backend server...")
        backend_process = None
        try:
            backend_process = subprocess.Popen(
                ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
                cwd=self.backend_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(5)
            
            # Start frontend dev server
            print("  🎨 Starting frontend dev server...")
            frontend_process = None
            try:
                frontend_process = subprocess.Popen(
                    ["npm", "run", "dev", "--", "--port", "3000"],
                    cwd=self.frontend_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Wait for frontend to start
                time.sleep(10)
                
                # Run integration tests
                print("  🧪 Running integration tests...")
                result = subprocess.run(
                    ["npm", "run", "test:integration:full", "--", "--run"],
                    cwd=self.frontend_path,
                    capture_output=True,
                    text=True,
                    timeout=900  # 15 minutes
                )
                
                integration_results["full_stack"] = {
                    "status": "passed" if result.returncode == 0 else "failed",
                    "output": result.stdout,
                    "errors": result.stderr,
                    "returncode": result.returncode
                }
                
            finally:
                if frontend_process:
                    frontend_process.terminate()
                    frontend_process.wait()
            
        except Exception as e:
            integration_results["full_stack"] = {
                "status": "error",
                "output": "",
                "errors": str(e),
                "returncode": -1
            }
        finally:
            if backend_process:
                backend_process.terminate()
                backend_process.wait()
        
        return integration_results

    def generate_coverage_report(self) -> Dict[str, Any]:
        """Generate code coverage report"""
        print("\n📊 Generating coverage reports...")
        
        coverage_results = {}
        
        # Frontend coverage
        try:
            result = subprocess.run(
                ["npm", "run", "coverage"],
                cwd=self.frontend_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            coverage_results["frontend"] = {
                "status": "success" if result.returncode == 0 else "failed",
                "output": result.stdout,
                "errors": result.stderr
            }
        except Exception as e:
            coverage_results["frontend"] = {
                "status": "error",
                "output": "",
                "errors": str(e)
            }
        
        # Backend coverage
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=.", "--cov-report=json", "--cov-report=html"],
                cwd=self.backend_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            coverage_results["backend"] = {
                "status": "success" if result.returncode == 0 else "failed",
                "output": result.stdout,
                "errors": result.stderr
            }
        except Exception as e:
            coverage_results["backend"] = {
                "status": "error",
                "output": "",
                "errors": str(e)
            }
        
        return coverage_results

    def run_security_tests(self) -> Dict[str, Any]:
        """Run security tests"""
        print("\n🔒 Running security tests...")
        
        security_results = {}
        
        # Frontend security scan
        try:
            result = subprocess.run(
                ["npm", "audit", "--audit-level", "moderate"],
                cwd=self.frontend_path,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            security_results["frontend_audit"] = {
                "status": "passed" if result.returncode == 0 else "vulnerabilities_found",
                "output": result.stdout,
                "errors": result.stderr,
                "returncode": result.returncode
            }
        except Exception as e:
            security_results["frontend_audit"] = {
                "status": "error",
                "output": "",
                "errors": str(e),
                "returncode": -1
            }
        
        # Backend security scan (using safety)
        try:
            result = subprocess.run(
                ["python", "-m", "safety", "check"],
                cwd=self.backend_path,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            security_results["backend_safety"] = {
                "status": "passed" if result.returncode == 0 else "vulnerabilities_found",
                "output": result.stdout,
                "errors": result.stderr,
                "returncode": result.returncode
            }
        except FileNotFoundError:
            security_results["backend_safety"] = {
                "status": "skipped",
                "output": "",
                "errors": "Safety not installed",
                "returncode": 0
            }
        except Exception as e:
            security_results["backend_safety"] = {
                "status": "error",
                "output": "",
                "errors": str(e),
                "returncode": -1
            }
        
        return security_results

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_duration = (self.end_time - self.start_time) if self.end_time and self.start_time else 0
        
        # Calculate summary statistics
        all_tests = []
        for category, tests in self.results.items():
            if isinstance(tests, dict):
                for test_type, result in tests.items():
                    if isinstance(result, dict) and "status" in result:
                        all_tests.append(result["status"])
        
        passed = sum(1 for status in all_tests if status == "passed")
        failed = sum(1 for status in all_tests if status == "failed")
        errors = sum(1 for status in all_tests if status == "error")
        timeouts = sum(1 for status in all_tests if status == "timeout")
        skipped = sum(1 for status in all_tests if status == "skipped")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "duration": total_duration,
            "summary": {
                "total": len(all_tests),
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "timeouts": timeouts,
                "skipped": skipped,
                "success_rate": (passed / len(all_tests) * 100) if all_tests else 0
            },
            "results": self.results,
            "recommendations": self.generate_recommendations()
        }
        
        return report

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check for failed tests
        failed_categories = []
        for category, tests in self.results.items():
            if isinstance(tests, dict):
                for test_type, result in tests.items():
                    if isinstance(result, dict) and result.get("status") == "failed":
                        failed_categories.append(f"{category}.{test_type}")
        
        if failed_categories:
            recommendations.append(f"Address failing tests in: {', '.join(failed_categories)}")
        
        # Check for timeouts
        timeout_categories = []
        for category, tests in self.results.items():
            if isinstance(tests, dict):
                for test_type, result in tests.items():
                    if isinstance(result, dict) and result.get("status") == "timeout":
                        timeout_categories.append(f"{category}.{test_type}")
        
        if timeout_categories:
            recommendations.append(f"Optimize performance for slow tests: {', '.join(timeout_categories)}")
        
        # Check coverage
        if "coverage" in self.results:
            coverage = self.results["coverage"]
            if isinstance(coverage, dict):
                for component, result in coverage.items():
                    if isinstance(result, dict) and result.get("status") == "failed":
                        recommendations.append(f"Improve test coverage for {component}")
        
        # Check security
        if "security" in self.results:
            security = self.results["security"]
            if isinstance(security, dict):
                for scan_type, result in security.items():
                    if isinstance(result, dict) and result.get("status") == "vulnerabilities_found":
                        recommendations.append(f"Address security vulnerabilities found in {scan_type}")
        
        if not recommendations:
            recommendations.append("All tests passed! Consider adding more comprehensive test coverage.")
        
        return recommendations

    def save_report(self, report: Dict[str, Any], output_file: str):
        """Save report to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Report saved to: {output_path}")

    def print_summary(self, report: Dict[str, Any]):
        """Print test summary to console"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
        print("="*60)
        
        summary = report["summary"]
        print(f"Total Tests: {summary['total']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"🚨 Errors: {summary['errors']}")
        print(f"⏰ Timeouts: {summary['timeouts']}")
        print(f"⏭️  Skipped: {summary['skipped']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        print(f"⏱️  Duration: {report['duration']:.2f} seconds")
        
        print("\n🎯 RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"{i}. {rec}")
        
        print("\n" + "="*60)

    def run_all_tests(self, categories: List[str] = None):
        """Run all test categories"""
        self.start_time = time.time()
        
        try:
            # Setup environment
            self.setup_environment()
            
            # Determine which categories to run
            all_categories = ["frontend", "backend", "integration", "coverage", "security"]
            categories_to_run = categories if categories else all_categories
            
            print(f"\n🚀 Running test categories: {', '.join(categories_to_run)}")
            
            # Run tests
            if "frontend" in categories_to_run:
                self.results["frontend"] = self.run_frontend_tests()
            
            if "backend" in categories_to_run:
                self.results["backend"] = self.run_backend_tests()
            
            if "integration" in categories_to_run:
                self.results["integration"] = self.run_integration_tests()
            
            if "coverage" in categories_to_run:
                self.results["coverage"] = self.generate_coverage_report()
            
            if "security" in categories_to_run:
                self.results["security"] = self.run_security_tests()
            
        except KeyboardInterrupt:
            print("\n⚠️  Test execution interrupted by user")
            self.results["interrupted"] = True
        except Exception as e:
            print(f"\n❌ Test execution failed: {str(e)}")
            self.results["error"] = str(e)
        finally:
            self.end_time = time.time()

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Test Runner")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["frontend", "backend", "integration", "coverage", "security"],
        help="Test categories to run (default: all)"
    )
    parser.add_argument(
        "--output",
        default="test-results/comprehensive-report.json",
        help="Output file for test report"
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    project_root = Path(args.project_root).resolve()
    runner = ComprehensiveTestRunner(project_root)
    
    # Run tests
    runner.run_all_tests(args.categories)
    
    # Generate and save report
    report = runner.generate_report()
    runner.save_report(report, args.output)
    runner.print_summary(report)
    
    # Exit with appropriate code
    success_rate = report["summary"]["success_rate"]
    if success_rate < 80:
        print(f"\n❌ Test suite failed (success rate: {success_rate:.1f}%)")
        sys.exit(1)
    elif success_rate < 95:
        print(f"\n⚠️  Test suite passed with warnings (success rate: {success_rate:.1f}%)")
        sys.exit(0)
    else:
        print(f"\n✅ Test suite passed successfully (success rate: {success_rate:.1f}%)")
        sys.exit(0)

if __name__ == "__main__":
    main()
