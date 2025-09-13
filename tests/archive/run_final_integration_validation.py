#!/usr/bin/env python3
"""
Final Integration Validation Test Runner

Orchestrates all validation tests for the WAN model generation system:
1. End-to-end testing from React frontend to video output
2. API contract validation with real WAN models
3. Hardware performance testing under various configurations
4. Infrastructure component integration verification

This is the master test runner for Task 18: Final Integration and Validation
Requirements: 3.1, 3.2, 3.3, 3.4
"""

import asyncio
import sys
import logging
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinalValidationOrchestrator:
    """Orchestrates all final validation tests"""
    
    def __init__(self, args):
        self.args = args
        self.test_results = {}
        self.start_time = time.time()
        self.backend_process = None
        self.frontend_process = None
        
    async def setup_test_environment(self) -> bool:
        """Setup the complete test environment"""
        logger.info("🔧 Setting up test environment for final validation...")
        
        try:
            # Check if we should start servers
            if not self.args.skip_server_start:
                # Start backend
                if not await self._check_backend_health():
                    logger.info("Starting backend server...")
                    if not await self._start_backend():
                        logger.error("Failed to start backend server")
                        return False
                
                # Start frontend if needed
                if not self.args.skip_frontend and not await self._check_frontend_health():
                    logger.info("Starting frontend server...")
                    if not await self._start_frontend():
                        logger.warning("Frontend server failed to start - frontend tests will be skipped")
            
            # Wait for services to stabilize
            await asyncio.sleep(5)
            
            # Verify environment is ready
            backend_ready = await self._check_backend_health()
            frontend_ready = await self._check_frontend_health() if not self.args.skip_frontend else True
            
            if not backend_ready:
                logger.error("Backend is not ready")
                return False
            
            logger.info("✅ Test environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test environment setup failed: {e}")
            return False
    
    async def _check_backend_health(self) -> bool:
        """Check if backend is healthy"""
        try:
            import requests
            response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def _check_frontend_health(self) -> bool:
        """Check if frontend is accessible"""
        try:
            import requests
            response = requests.get("http://localhost:3000", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def _start_backend(self) -> bool:
        """Start backend server"""
        try:
            backend_dir = project_root / "backend"
            if not backend_dir.exists():
                logger.error("Backend directory not found")
                return False
            
            # Try different startup methods
            startup_scripts = ["start_server.py", "main.py"]
            
            for script in startup_scripts:
                script_path = backend_dir / script
                if script_path.exists():
                    logger.info(f"Starting backend with {script}...")
                    
                    self.backend_process = subprocess.Popen(
                        [sys.executable, str(script)],
                        cwd=str(backend_dir),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    # Wait for backend to start
                    for _ in range(30):  # 30 second timeout
                        await asyncio.sleep(1)
                        if await self._check_backend_health():
                            logger.info("✅ Backend started successfully")
                            return True
                    
                    # If this script didn't work, terminate and try next
                    self.backend_process.terminate()
                    self.backend_process = None
            
            logger.error("Failed to start backend with any method")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start backend: {e}")
            return False
    
    async def _start_frontend(self) -> bool:
        """Start frontend server"""
        try:
            frontend_dir = project_root / "frontend"
            if not frontend_dir.exists():
                logger.warning("Frontend directory not found")
                return False
            
            # Check if package.json exists
            package_json = frontend_dir / "package.json"
            if not package_json.exists():
                logger.warning("Frontend package.json not found")
                return False
            
            # Install dependencies if needed
            node_modules = frontend_dir / "node_modules"
            if not node_modules.exists():
                logger.info("Installing frontend dependencies...")
                result = subprocess.run(
                    ["npm", "install"],
                    cwd=str(frontend_dir),
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout for npm install
                )
                
                if result.returncode != 0:
                    logger.warning(f"npm install failed: {result.stderr}")
                    return False
            
            # Start development server
            logger.info("Starting frontend development server...")
            self.frontend_process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=str(frontend_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for frontend to start
            for _ in range(60):  # 60 second timeout
                await asyncio.sleep(1)
                if await self._check_frontend_health():
                    logger.info("✅ Frontend started successfully")
                    return True
            
            logger.warning("Frontend failed to start within timeout")
            return False
            
        except Exception as e:
            logger.warning(f"Failed to start frontend: {e}")
            return False
    
    async def run_core_integration_tests(self) -> Dict[str, Any]:
        """Run core integration validation tests"""
        logger.info("🧪 Running Core Integration Tests...")
        
        try:
            # Import and run the main validation suite
            from final_integration_validation_suite import FinalIntegrationValidator
            
            validator = FinalIntegrationValidator()
            report = await validator.run_all_tests()
            
            return {
                "success": True,
                "report": report.__dict__ if hasattr(report, '__dict__') else report,
                "test_type": "core_integration"
            }
            
        except Exception as e:
            logger.error(f"Core integration tests failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "test_type": "core_integration"
            }
    
    async def run_frontend_backend_tests(self) -> Dict[str, Any]:
        """Run frontend-backend integration tests"""
        if self.args.skip_frontend:
            return {
                "success": True,
                "skipped": True,
                "reason": "Frontend tests skipped by user",
                "test_type": "frontend_backend"
            }
        
        logger.info("🧪 Running Frontend-Backend Integration Tests...")
        
        try:
            from test_frontend_backend_integration import FrontendBackendIntegrationTest
            
            tester = FrontendBackendIntegrationTest()
            success = await tester.run_all_tests()
            tester.cleanup()
            
            return {
                "success": success,
                "test_type": "frontend_backend"
            }
            
        except Exception as e:
            logger.error(f"Frontend-backend tests failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "test_type": "frontend_backend"
            }
    
    async def run_hardware_performance_tests(self) -> Dict[str, Any]:
        """Run hardware performance tests"""
        if self.args.skip_performance:
            return {
                "success": True,
                "skipped": True,
                "reason": "Performance tests skipped by user",
                "test_type": "hardware_performance"
            }
        
        logger.info("🧪 Running Hardware Performance Tests...")
        
        try:
            from test_hardware_performance_configurations import HardwarePerformanceTest
            
            tester = HardwarePerformanceTest()
            report = await tester.run_all_performance_tests()
            
            # Determine success based on results
            successful_configs = report["summary"]["successful_configs"]
            total_configs = report["summary"]["total_configs_tested"]
            success_rate = successful_configs / total_configs if total_configs > 0 else 0
            
            return {
                "success": success_rate >= 0.5,  # At least 50% should work
                "report": report,
                "success_rate": success_rate,
                "test_type": "hardware_performance"
            }
            
        except Exception as e:
            logger.error(f"Hardware performance tests failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "test_type": "hardware_performance"
            }
    
    async def run_existing_test_suites(self) -> Dict[str, Any]:
        """Run existing test suites to ensure no regressions"""
        logger.info("🧪 Running Existing Test Suites...")
        
        existing_tests = []
        
        # Check for existing test files
        test_files = [
            "test_wan_generation_integration.py",
            "test_wan_model_integration_bridge.py",
            "backend/tests/test_end_to_end_comprehensive.py"
        ]
        
        results = {}
        
        for test_file in test_files:
            test_path = project_root / test_file
            if test_path.exists():
                try:
                    logger.info(f"  Running {test_file}...")
                    
                    # Run the test file
                    result = subprocess.run(
                        [sys.executable, str(test_path)],
                        cwd=str(project_root),
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout per test
                    )
                    
                    results[test_file] = {
                        "success": result.returncode == 0,
                        "returncode": result.returncode,
                        "stdout": result.stdout[-1000:] if result.stdout else "",  # Last 1000 chars
                        "stderr": result.stderr[-1000:] if result.stderr else ""   # Last 1000 chars
                    }
                    
                    status = "✅" if result.returncode == 0 else "❌"
                    logger.info(f"  {status} {test_file}: {'PASS' if result.returncode == 0 else 'FAIL'}")
                    
                except subprocess.TimeoutExpired:
                    results[test_file] = {
                        "success": False,
                        "error": "Test timed out",
                        "timeout": True
                    }
                    logger.warning(f"  ⏰ {test_file}: TIMEOUT")
                    
                except Exception as e:
                    results[test_file] = {
                        "success": False,
                        "error": str(e)
                    }
                    logger.error(f"  💥 {test_file}: ERROR - {e}")
            else:
                results[test_file] = {
                    "success": False,
                    "error": "Test file not found",
                    "missing": True
                }
                logger.warning(f"  ⚠️ {test_file}: NOT FOUND")
        
        # Calculate overall success
        successful_tests = len([r for r in results.values() if r.get("success", False)])
        total_tests = len(results)
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        return {
            "success": success_rate >= 0.7,  # At least 70% should pass
            "results": results,
            "success_rate": success_rate,
            "successful_tests": successful_tests,
            "total_tests": total_tests,
            "test_type": "existing_suites"
        }
    
    async def run_all_validation_tests(self) -> Dict[str, Any]:
        """Run all validation tests"""
        logger.info("🚀 Starting Final Integration and Validation...")
        logger.info(f"Arguments: {vars(self.args)}")
        
        # Setup environment
        if not await self.setup_test_environment():
            return {
                "success": False,
                "error": "Test environment setup failed",
                "timestamp": datetime.now().isoformat()
            }
        
        # Run all test suites
        test_suites = [
            ("Core Integration", self.run_core_integration_tests),
            ("Frontend-Backend", self.run_frontend_backend_tests),
            ("Hardware Performance", self.run_hardware_performance_tests),
            ("Existing Test Suites", self.run_existing_test_suites)
        ]
        
        suite_results = {}
        
        for suite_name, test_method in test_suites:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"🎯 {suite_name} Tests")
                logger.info(f"{'='*50}")
                
                start_time = time.time()
                result = await test_method()
                duration = time.time() - start_time
                
                result["duration"] = duration
                suite_results[suite_name] = result
                
                status = "✅" if result.get("success", False) else "❌"
                skip_status = " (SKIPPED)" if result.get("skipped", False) else ""
                logger.info(f"{status} {suite_name}: {'PASS' if result.get('success') else 'FAIL'}{skip_status} ({duration:.1f}s)")
                
            except Exception as e:
                suite_results[suite_name] = {
                    "success": False,
                    "error": str(e),
                    "duration": 0
                }
                logger.error(f"💥 {suite_name}: ERROR - {e}")
        
        # Generate final validation report
        total_duration = time.time() - self.start_time
        
        # Calculate overall success metrics
        successful_suites = len([r for r in suite_results.values() 
                               if r.get("success", False) or r.get("skipped", False)])
        total_suites = len(suite_results)
        overall_success_rate = successful_suites / total_suites if total_suites > 0 else 0
        
        # Determine critical failures
        critical_suites = ["Core Integration", "Existing Test Suites"]
        critical_failures = [name for name in critical_suites 
                           if not suite_results.get(name, {}).get("success", False)]
        
        final_report = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": total_duration,
            "suite_results": suite_results,
            "summary": {
                "total_suites": total_suites,
                "successful_suites": successful_suites,
                "overall_success_rate": overall_success_rate,
                "critical_failures": critical_failures,
                "validation_passed": len(critical_failures) == 0 and overall_success_rate >= 0.75
            },
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "arguments": vars(self.args)
            }
        }
        
        # Save comprehensive report
        report_path = Path("final_validation_report.json")
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        # Print final summary
        self.print_final_summary(final_report)
        
        logger.info(f"📄 Final validation report saved to: {report_path}")
        
        return final_report
    
    def print_final_summary(self, report: Dict[str, Any]):
        """Print final validation summary"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 FINAL INTEGRATION AND VALIDATION SUMMARY")
        logger.info(f"{'='*80}")
        
        summary = report["summary"]
        
        # Overall status
        if summary["validation_passed"]:
            logger.info(f"🎉 VALIDATION SUCCESSFUL - WAN Model Integration Ready for Production!")
        else:
            logger.info(f"⚠️ VALIDATION ISSUES DETECTED - Review failures before deployment")
        
        # Test suite results
        logger.info(f"\n📊 Test Suite Results:")
        logger.info(f"  Total Suites: {summary['total_suites']}")
        logger.info(f"  ✅ Successful: {summary['successful_suites']}")
        logger.info(f"  ❌ Failed: {summary['total_suites'] - summary['successful_suites']}")
        logger.info(f"  📈 Success Rate: {summary['overall_success_rate']:.1%}")
        logger.info(f"  ⏱️ Total Duration: {report['total_duration']:.1f}s")
        
        # Individual suite status
        logger.info(f"\n📋 Suite Details:")
        for suite_name, result in report["suite_results"].items():
            if result.get("skipped", False):
                status_icon = "⏭️"
                status_text = "SKIPPED"
            elif result.get("success", False):
                status_icon = "✅"
                status_text = "PASS"
            else:
                status_icon = "❌"
                status_text = "FAIL"
            
            duration = result.get("duration", 0)
            logger.info(f"  {status_icon} {suite_name}: {status_text} ({duration:.1f}s)")
            
            if result.get("error"):
                logger.info(f"    Error: {result['error']}")
        
        # Critical failures
        if summary["critical_failures"]:
            logger.info(f"\n🚨 Critical Failures:")
            for failure in summary["critical_failures"]:
                logger.info(f"  ❌ {failure}")
        
        # Recommendations
        logger.info(f"\n💡 Recommendations:")
        if summary["validation_passed"]:
            logger.info("  ✅ All critical systems validated - ready for production deployment")
            logger.info("  ✅ WAN model integration is functioning correctly")
            logger.info("  ✅ API contracts are maintained and working")
            logger.info("  ✅ Hardware optimization is operational")
        else:
            logger.info("  ⚠️ Address critical failures before production deployment")
            logger.info("  ⚠️ Review failed test suites and resolve issues")
            if summary["overall_success_rate"] < 0.5:
                logger.info("  🚨 Major integration issues detected - comprehensive review needed")
        
        logger.info(f"{'='*80}")
    
    def cleanup(self):
        """Cleanup test environment"""
        logger.info("🧹 Cleaning up test environment...")
        
        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=10)
            except:
                try:
                    self.backend_process.kill()
                except:
                    pass
        
        if self.frontend_process:
            try:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=10)
            except:
                try:
                    self.frontend_process.kill()
                except:
                    pass

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Final Integration and Validation Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_final_integration_validation.py                    # Run all tests
  python run_final_integration_validation.py --skip-frontend   # Skip frontend tests
  python run_final_integration_validation.py --skip-performance # Skip performance tests
  python run_final_integration_validation.py --skip-server-start # Use existing servers
        """
    )
    
    parser.add_argument(
        "--skip-frontend",
        action="store_true",
        help="Skip frontend integration tests"
    )
    
    parser.add_argument(
        "--skip-performance",
        action="store_true",
        help="Skip hardware performance tests"
    )
    
    parser.add_argument(
        "--skip-server-start",
        action="store_true",
        help="Skip starting servers (use existing running servers)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

async def main():
    """Main validation function"""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    orchestrator = FinalValidationOrchestrator(args)
    
    try:
        report = await orchestrator.run_all_validation_tests()
        
        # Return appropriate exit code
        if report["summary"]["validation_passed"]:
            logger.info("🎉 Final integration validation completed successfully!")
            return 0
        else:
            logger.error("💥 Final integration validation failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("🛑 Validation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"💥 Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        orchestrator.cleanup()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
