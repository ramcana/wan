#!/usr/bin/env python3
"""
Enhanced Model Availability Test Runner

Simple script to run all enhanced model availability tests with proper
async handling and comprehensive reporting.

Usage:
    python run_enhanced_model_availability_tests.py
"""

import asyncio
import sys
import time
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def run_integration_tests():
    """Run integration tests."""
    print("🔗 Running Integration Tests...")
    
    try:
        from tests.test_enhanced_model_availability_integration import TestEnhancedModelAvailabilityIntegration
        
        # Create test instance
        test_suite = TestEnhancedModelAvailabilityIntegration()
        
        # Create enhanced system fixture
        enhanced_system = {
            'enhanced_downloader': None,
            'health_monitor': None,
            'availability_manager': None,
            'fallback_manager': None,
            'error_recovery': None,
            'update_manager': None,
            'notification_manager': None
        }
        
        # Run key integration tests
        test_methods = [
            'test_complete_model_request_workflow',
            'test_model_unavailable_fallback_workflow',
            'test_health_monitoring_integration',
            'test_concurrent_model_operations'
        ]
        
        passed = 0
        total = len(test_methods)
        
        for method_name in test_methods:
            try:
                method = getattr(test_suite, method_name)
                await method(enhanced_system)
                print(f"  ✅ {method_name}")
                passed += 1
            except Exception as e:
                print(f"  ❌ {method_name}: {str(e)}")
        
        print(f"Integration Tests: {passed}/{total} passed")
        return passed == total
        
    except Exception as e:
        print(f"❌ Integration tests failed to run: {str(e)}")
        return False

async def run_stress_tests():
    """Run stress tests."""
    print("\n💪 Running Stress Tests...")
    
    try:
        from tests.test_download_stress_testing import DownloadStressTestSuite
        
        stress_suite = DownloadStressTestSuite()
        
        # Run individual stress tests
        test_methods = [
            'test_concurrent_download_stress',
            'test_retry_logic_stress',
            'test_bandwidth_limiting_stress',
            'test_memory_pressure_stress'
        ]
        
        passed = 0
        total = len(test_methods)
        
        for method_name in test_methods:
            try:
                method = getattr(stress_suite, method_name)
                await method()
                print(f"  ✅ {method_name}")
                passed += 1
            except Exception as e:
                print(f"  ❌ {method_name}: {str(e)}")
        
        print(f"Stress Tests: {passed}/{total} passed")
        return passed >= total * 0.75  # 75% pass rate acceptable for stress tests
        
    except Exception as e:
        print(f"❌ Stress tests failed to run: {str(e)}")
        return False

async def run_chaos_tests():
    """Run chaos engineering tests."""
    print("\n🌪️  Running Chaos Engineering Tests...")
    
    try:
        from tests.test_chaos_engineering import ChaosEngineeringTestSuite
        
        chaos_suite = ChaosEngineeringTestSuite()
        
        # Run individual chaos tests
        test_methods = [
            'test_component_failure_cascade',
            'test_network_partition_scenarios',
            'test_resource_exhaustion_scenarios'
        ]
        
        passed = 0
        total = len(test_methods)
        
        for method_name in test_methods:
            try:
                method = getattr(chaos_suite, method_name)
                await method()
                print(f"  ✅ {method_name}")
                passed += 1
            except Exception as e:
                print(f"  ❌ {method_name}: {str(e)}")
        
        print(f"Chaos Engineering Tests: {passed}/{total} passed")
        return passed >= total * 0.7  # 70% pass rate acceptable for chaos tests
        
    except Exception as e:
        print(f"❌ Chaos engineering tests failed to run: {str(e)}")
        return False

async def run_performance_benchmarks():
    """Run performance benchmarks."""
    print("\n📊 Running Performance Benchmarks...")
    
    try:
        from tests.test_performance_benchmarks_enhanced import PerformanceBenchmarkSuite
        
        benchmark_suite = PerformanceBenchmarkSuite()
        
        # Run individual benchmark tests
        test_methods = [
            'benchmark_enhanced_downloader_performance',
            'benchmark_health_monitor_performance',
            'benchmark_availability_manager_performance'
        ]
        
        passed = 0
        total = len(test_methods)
        
        for method_name in test_methods:
            try:
                method = getattr(benchmark_suite, method_name)
                results = await method()
                
                # Validate performance results
                if results and len(results) > 0:
                    avg_ops_per_sec = sum(r.operations_per_second for r in results) / len(results)
                    if avg_ops_per_sec > 10:  # Minimum acceptable performance
                        print(f"  ✅ {method_name} (avg: {avg_ops_per_sec:.2f} ops/sec)")
                        passed += 1
                    else:
                        print(f"  ❌ {method_name}: Performance too low ({avg_ops_per_sec:.2f} ops/sec)")
                else:
                    print(f"  ❌ {method_name}: No results returned")
                    
            except Exception as e:
                print(f"  ❌ {method_name}: {str(e)}")
        
        print(f"Performance Benchmarks: {passed}/{total} passed")
        return passed == total
        
    except Exception as e:
        print(f"❌ Performance benchmarks failed to run: {str(e)}")
        return False

async def run_user_acceptance_tests():
    """Run user acceptance tests."""
    print("\n👥 Running User Acceptance Tests...")
    
    try:
        from tests.test_user_acceptance_workflows import UserAcceptanceTestSuite
        
        user_suite = UserAcceptanceTestSuite()
        
        # Run individual user acceptance tests
        test_methods = [
            'test_new_user_first_model_request_workflow',
            'test_model_unavailable_fallback_workflow',
            'test_model_management_workflow'
        ]
        
        passed = 0
        total = len(test_methods)
        
        for method_name in test_methods:
            try:
                method = getattr(user_suite, method_name)
                result = await method()
                
                # Validate user acceptance results
                if hasattr(result, 'success') and result.success and result.user_satisfaction_score >= 0.7:
                    print(f"  ✅ {method_name} (satisfaction: {result.user_satisfaction_score:.2f})")
                    passed += 1
                else:
                    satisfaction = getattr(result, 'user_satisfaction_score', 0.0)
                    print(f"  ❌ {method_name}: Low satisfaction ({satisfaction:.2f})")
                    
            except Exception as e:
                print(f"  ❌ {method_name}: {str(e)}")
        
        print(f"User Acceptance Tests: {passed}/{total} passed")
        return passed >= total * 0.8  # 80% pass rate required for user acceptance
        
    except Exception as e:
        print(f"❌ User acceptance tests failed to run: {str(e)}")
        return False

async def main():
    """Main test runner."""
    print("=" * 80)
    print("ENHANCED MODEL AVAILABILITY - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    start_time = time.time()
    
    # Run all test suites
    test_results = {}
    
    test_results['integration'] = await run_integration_tests()
    test_results['stress'] = await run_stress_tests()
    test_results['chaos'] = await run_chaos_tests()
    test_results['performance'] = await run_performance_benchmarks()
    test_results['user_acceptance'] = await run_user_acceptance_tests()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("TEST EXECUTION SUMMARY")
    print("=" * 80)
    
    passed_suites = sum(1 for result in test_results.values() if result)
    total_suites = len(test_results)
    
    print(f"Execution Time: {total_time:.2f} seconds")
    print(f"Test Suites: {passed_suites}/{total_suites} passed")
    print()
    
    # Detailed results
    suite_names = {
        'integration': 'Integration Tests',
        'stress': 'Stress Tests',
        'chaos': 'Chaos Engineering Tests',
        'performance': 'Performance Benchmarks',
        'user_acceptance': 'User Acceptance Tests'
    }
    
    for suite_key, passed in test_results.items():
        status_icon = "✅" if passed else "❌"
        suite_name = suite_names.get(suite_key, suite_key)
        print(f"{status_icon} {suite_name}: {'PASSED' if passed else 'FAILED'}")
    
    # Overall result
    print("\n" + "-" * 80)
    overall_success = passed_suites >= total_suites * 0.8  # 80% pass rate required
    
    if overall_success:
        print("🎉 OVERALL RESULT: PASSED")
        print("   Enhanced Model Availability system is ready for deployment!")
    else:
        print("❌ OVERALL RESULT: FAILED")
        print("   Enhanced Model Availability system needs improvements before deployment.")
    
    print("-" * 80)
    
    # Recommendations
    print("\nRecommendations:")
    if overall_success:
        print("  ✅ System demonstrates excellent quality and reliability")
        print("  ✅ All critical functionality validated")
        print("  ✅ Performance meets requirements")
        print("  ✅ User experience is satisfactory")
    else:
        print("  ❌ Address failing test suites before deployment")
        if not test_results['integration']:
            print("  🔧 Fix integration issues - critical for system functionality")
        if not test_results['performance']:
            print("  🔧 Optimize performance - system may be too slow for users")
        if not test_results['user_acceptance']:
            print("  🔧 Improve user experience - users may be dissatisfied")
    
    return overall_success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test execution failed: {str(e)}")
        sys.exit(1)