import pytest
#!/usr/bin/env python3
"""
Comprehensive performance validation script for React Frontend FastAPI system.
Runs both backend and frontend performance tests and generates deployment report.
"""

import asyncio
import subprocess
import sys
import json
import time
import os
from pathlib import Path
from typing import Dict, Any, List
import argparse

class PerformanceValidator:
    """Main performance validation orchestrator"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.results = {
            'backend': {},
            'frontend': {},
            'integration': {},
            'deployment_ready': False,
            'timestamp': time.time()
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load performance validation configuration"""
        default_config = {
            'backend': {
                'test_file': 'backend/test_performance_validation.py',
                'timeout': 1800,  # 30 minutes
                'required_tests': [
                    'test_720p_t2v_generation_timing',
                    'test_1080p_generation_timing',
                    'test_api_response_times',
                    'test_establish_baseline_metrics'
                ]
            },
            'frontend': {
                'test_file': 'frontend/src/tests/performance/performance-test-runner.test.ts',
                'build_command': 'npm run build',
                'test_command': 'npm run test:performance',
                'bundle_size_limit': 500 * 1024,  # 500KB
                'fmp_limit': 2000  # 2 seconds
            },
            'integration': {
                'endpoints_to_test': [
                    '/api/v1/health',
                    '/api/v1/system/stats',
                    '/api/v1/queue',
                    '/api/v1/outputs'
                ],
                'response_time_limits': {
                    '/api/v1/health': 1000,
                    '/api/v1/system/stats': 2000,
                    '/api/v1/queue': 1500,
                    '/api/v1/outputs': 3000
                }
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge with default config
                default_config.update(user_config)
        
        return default_config
    
    async def run_backend_performance_tests(self) -> Dict[str, Any]:
        """Run backend performance validation tests"""
        print("🔧 Running backend performance tests...")
        
        backend_config = self.config['backend']
        test_file = backend_config['test_file']
        
        if not os.path.exists(test_file):
            return {
                'status': 'error',
                'message': f'Backend test file not found: {test_file}'
            }
        
        try:
            # Run pytest with performance tests
            cmd = [
                sys.executable, '-m', 'pytest',
                test_file,
                '-v',
                '--tb=short',
                '--json-report',
                '--json-report-file=backend_performance_report.json'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=backend_config['timeout']
            )
            
            # Load test results
            report_file = 'backend_performance_report.json'
            if os.path.exists(report_file):
                with open(report_file, 'r') as f:
                    test_report = json.load(f)
                
                return {
                    'status': 'success' if process.returncode == 0 else 'failed',
                    'return_code': process.returncode,
                    'test_report': test_report,
                    'stdout': stdout.decode(),
                    'stderr': stderr.decode()
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Test report file not generated',
                    'stdout': stdout.decode(),
                    'stderr': stderr.decode()
                }
                
        except asyncio.TimeoutError:
            return {
                'status': 'timeout',
                'message': f'Backend tests timed out after {backend_config["timeout"]} seconds'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Backend test execution failed: {str(e)}'
            }
    
    async def run_frontend_performance_tests(self) -> Dict[str, Any]:
        """Run frontend performance validation tests"""
        print("🎨 Running frontend performance tests...")
        
        frontend_config = self.config['frontend']
        
        try:
            # First, build the frontend
            print("Building frontend...")
            build_process = await asyncio.create_subprocess_exec(
                *frontend_config['build_command'].split(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd='frontend'
            )
            
            build_stdout, build_stderr = await build_process.communicate()
            
            if build_process.returncode != 0:
                return {
                    'status': 'build_failed',
                    'message': 'Frontend build failed',
                    'stdout': build_stdout.decode(),
                    'stderr': build_stderr.decode()
                }
            
            # Check bundle size
            bundle_analysis = self._analyze_bundle_size()
            
            # Run performance tests
            test_process = await asyncio.create_subprocess_exec(
                *frontend_config['test_command'].split(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd='frontend'
            )
            
            test_stdout, test_stderr = await test_process.communicate()
            
            return {
                'status': 'success' if test_process.returncode == 0 else 'failed',
                'return_code': test_process.returncode,
                'bundle_analysis': bundle_analysis,
                'stdout': test_stdout.decode(),
                'stderr': test_stderr.decode()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Frontend test execution failed: {str(e)}'
            }
    
    def _analyze_bundle_size(self) -> Dict[str, Any]:
        """Analyze frontend bundle size"""
        dist_path = Path('frontend/dist')
        
        if not dist_path.exists():
            return {
                'status': 'error',
                'message': 'Frontend dist directory not found'
            }
        
        # Find main JavaScript bundle
        js_files = list(dist_path.glob('**/*.js'))
        main_bundle = None
        
        for js_file in js_files:
            if 'main' in js_file.name or 'index' in js_file.name:
                main_bundle = js_file
                break
        
        if not main_bundle:
            main_bundle = max(js_files, key=lambda f: f.stat().st_size) if js_files else None
        
        if not main_bundle:
            return {
                'status': 'error',
                'message': 'No JavaScript bundle found'
            }
        
        bundle_size = main_bundle.stat().st_size
        estimated_gzipped = int(bundle_size * 0.3)  # Rough gzip estimate
        
        limit = self.config['frontend']['bundle_size_limit']
        
        return {
            'status': 'success',
            'bundle_path': str(main_bundle),
            'size_bytes': bundle_size,
            'size_kb': bundle_size / 1024,
            'estimated_gzipped_bytes': estimated_gzipped,
            'estimated_gzipped_kb': estimated_gzipped / 1024,
            'limit_kb': limit / 1024,
            'passes_budget': estimated_gzipped < limit
        }
    
    async def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration performance tests"""
        print("🔗 Running integration performance tests...")
        
        integration_config = self.config['integration']
        
        try:
            # Test API endpoints response times
            import aiohttp
            
            endpoint_results = {}
            
            async with aiohttp.ClientSession() as session:
                for endpoint in integration_config['endpoints_to_test']:
                    times = []
                    
                    # Test each endpoint 5 times
                    for _ in range(5):
                        start_time = time.time()
                        
                        try:
                            async with session.get(f'http://localhost:8000{endpoint}') as response:
                                await response.text()
                                end_time = time.time()
                                
                                if response.status == 200:
                                    times.append((end_time - start_time) * 1000)  # Convert to ms
                        except Exception as e:
                            print(f"Warning: Failed to test {endpoint}: {e}")
                        
                        # Small delay between requests
                        await asyncio.sleep(0.1)
                    
                    if times:
                        avg_time = sum(times) / len(times)
                        limit = integration_config['response_time_limits'].get(endpoint, 5000)
                        
                        endpoint_results[endpoint] = {
                            'avg_response_time_ms': avg_time,
                            'min_response_time_ms': min(times),
                            'max_response_time_ms': max(times),
                            'limit_ms': limit,
                            'passes_budget': avg_time < limit,
                            'sample_count': len(times)
                        }
            
            return {
                'status': 'success',
                'endpoint_results': endpoint_results
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Integration test execution failed: {str(e)}'
            }
    
    def validate_deployment_readiness(self) -> bool:
        """Validate if system is ready for deployment based on performance results"""
        backend_ready = (
            self.results['backend'].get('status') == 'success' and
            self.results['backend'].get('return_code') == 0
        )
        
        frontend_ready = (
            self.results['frontend'].get('status') == 'success' and
            self.results['frontend'].get('bundle_analysis', {}).get('passes_budget', False)
        )
        
        integration_ready = (
            self.results['integration'].get('status') == 'success' and
            all(
                result.get('passes_budget', False)
                for result in self.results['integration'].get('endpoint_results', {}).values()
            )
        )
        
        return backend_ready and frontend_ready and integration_ready
    
    def generate_deployment_report(self) -> str:
        """Generate comprehensive deployment readiness report"""
        report = []
        report.append("# Performance Validation Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.results['timestamp']))}")
        report.append("")
        
        # Overall status
        deployment_ready = self.validate_deployment_readiness()
        status_emoji = "✅" if deployment_ready else "❌"
        report.append(f"## Overall Status: {status_emoji} {'READY FOR DEPLOYMENT' if deployment_ready else 'NOT READY FOR DEPLOYMENT'}")
        report.append("")
        
        # Backend results
        report.append("## Backend Performance")
        backend = self.results['backend']
        if backend.get('status') == 'success':
            report.append("✅ Backend performance tests passed")
            if 'test_report' in backend:
                test_report = backend['test_report']
                report.append(f"- Tests run: {test_report.get('summary', {}).get('total', 'N/A')}")
                report.append(f"- Tests passed: {test_report.get('summary', {}).get('passed', 'N/A')}")
                report.append(f"- Tests failed: {test_report.get('summary', {}).get('failed', 'N/A')}")
        else:
            report.append(f"❌ Backend performance tests failed: {backend.get('message', 'Unknown error')}")
        report.append("")
        
        # Frontend results
        report.append("## Frontend Performance")
        frontend = self.results['frontend']
        if frontend.get('status') == 'success':
            report.append("✅ Frontend performance tests passed")
            
            bundle = frontend.get('bundle_analysis', {})
            if bundle.get('status') == 'success':
                size_kb = bundle.get('estimated_gzipped_kb', 0)
                limit_kb = bundle.get('limit_kb', 500)
                status = "✅" if bundle.get('passes_budget') else "❌"
                report.append(f"{status} Bundle size: {size_kb:.1f}KB (limit: {limit_kb:.0f}KB)")
        else:
            report.append(f"❌ Frontend performance tests failed: {frontend.get('message', 'Unknown error')}")
        report.append("")
        
        # Integration results
        report.append("## Integration Performance")
        integration = self.results['integration']
        if integration.get('status') == 'success':
            report.append("✅ Integration performance tests passed")
            
            for endpoint, result in integration.get('endpoint_results', {}).items():
                avg_time = result.get('avg_response_time_ms', 0)
                limit = result.get('limit_ms', 0)
                status = "✅" if result.get('passes_budget') else "❌"
                report.append(f"{status} {endpoint}: {avg_time:.0f}ms (limit: {limit}ms)")
        else:
            report.append(f"❌ Integration performance tests failed: {integration.get('message', 'Unknown error')}")
        report.append("")
        
        # Recommendations
        if not deployment_ready:
            report.append("## Recommendations")
            if self.results['backend'].get('status') != 'success':
                report.append("- Fix backend performance issues before deployment")
            if not self.results['frontend'].get('bundle_analysis', {}).get('passes_budget', True):
                report.append("- Optimize frontend bundle size to meet 500KB budget")
            if self.results['integration'].get('status') != 'success':
                report.append("- Resolve API response time issues")
            report.append("")
        
        return "\n".join(report)
    
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete performance validation suite"""
        print("🚀 Starting comprehensive performance validation...")
        
        # Run all test suites
        self.results['backend'] = await self.run_backend_performance_tests()
        self.results['frontend'] = await self.run_frontend_performance_tests()
        self.results['integration'] = await self.run_integration_tests()
        
        # Determine deployment readiness
        self.results['deployment_ready'] = self.validate_deployment_readiness()
        
        return self.results

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Performance validation for React Frontend FastAPI system')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--output', help='Output file for results', default='performance_validation_report.json')
    parser.add_argument('--report', help='Output file for deployment report', default='deployment_readiness_report.md')
    parser.add_argument('--backend-only', action='store_true', help='Run only backend tests')
    parser.add_argument('--frontend-only', action='store_true', help='Run only frontend tests')
    parser.add_argument('--integration-only', action='store_true', help='Run only integration tests')
    
    args = parser.parse_args()
    
    validator = PerformanceValidator(args.config)
    
    try:
        if args.backend_only:
            results = {'backend': await validator.run_backend_performance_tests()}
        elif args.frontend_only:
            results = {'frontend': await validator.run_frontend_performance_tests()}
        elif args.integration_only:
            results = {'integration': await validator.run_integration_tests()}
        else:
            results = await validator.run_full_validation()
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate deployment report
        validator.results = results
        report = validator.generate_deployment_report()
        
        with open(args.report, 'w') as f:
            f.write(report)
        
        print(f"\n📊 Results saved to: {args.output}")
        print(f"📋 Deployment report saved to: {args.report}")
        print("\n" + report)
        
        # Exit with appropriate code
        deployment_ready = validator.validate_deployment_readiness()
        sys.exit(0 if deployment_ready else 1)
        
    except KeyboardInterrupt:
        print("\n⚠ Performance validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Performance validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())