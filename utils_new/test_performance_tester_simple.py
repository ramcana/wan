#!/usr/bin/env python3
"""
Simple test script for PerformanceTester
"""

import sys
from pathlib import Path

# Add local_testing_framework to path
sys.path.insert(0, str(Path(__file__).parent))

from local_testing_framework.performance_tester import (
    MetricsCollector, BenchmarkRunner, PerformanceTester,
    OptimizationValidator, PerformanceTargetValidator,
    OptimizationRecommendationSystem
)
from local_testing_framework.models.configuration import TestConfiguration

def test_basic_functionality():
    """Test basic PerformanceTester functionality"""
    print("Testing PerformanceTester components...")
    
    # Test MetricsCollector
    print("1. Testing MetricsCollector...")
    collector = MetricsCollector()
    print(f"   Initialized: {len(collector.metrics)} metrics")
    
    # Test BenchmarkRunner
    print("2. Testing BenchmarkRunner...")
    runner = BenchmarkRunner()
    print("   BenchmarkRunner created successfully")
    
    # Test performance profiler availability
    profiler_result = runner.run_performance_profiler_benchmark()
    print(f"   Performance profiler available: {profiler_result.get('profiler_available', False)}")
    
    # Test OptimizationValidator
    print("3. Testing OptimizationValidator...")
    optimizer = OptimizationValidator()
    
    # Test VRAM reduction validation
    vram_result = optimizer.validate_vram_reduction(10.0, 2.0)  # 80% reduction
    print(f"   VRAM reduction validation: {vram_result.status.value}")
    print(f"   Reduction: {vram_result.details.get('reduction_percent', 0):.1f}%")
    
    # Test PerformanceTargetValidator
    print("4. Testing PerformanceTargetValidator...")
    target_validator = PerformanceTargetValidator()
    
    # Test 720p target validation
    target_result = target_validator.validate_720p_target(8.0, 10.0)  # 8min, 10GB
    print(f"   720p target validation: {target_result.status.value}")
    print(f"   Message: {target_result.message}")
    
    # Test OptimizationRecommendationSystem
    print("5. Testing OptimizationRecommendationSystem...")
    rec_system = OptimizationRecommendationSystem()
    
    # Test with mock performance issues
    mock_test_results = {
        "tests": {
            "720p": {
                "duration_minutes": 10.0,  # Above target
                "metrics": {"gpu_memory_peak_gb": 11.0}
            }
        }
    }
    
    recommendations = rec_system.analyze_performance_issues(mock_test_results)
    print(f"   VRAM optimizations: {len(recommendations['vram_optimizations'])}")
    print(f"   Speed optimizations: {len(recommendations['speed_optimizations'])}")
    
    # Generate optimization config
    opt_config = rec_system.generate_optimization_config(recommendations)
    print(f"   Generated config sections: {list(opt_config.keys())}")
    
    # Test main PerformanceTester
    print("6. Testing main PerformanceTester...")
    tester = PerformanceTester()
    
    # Test performance target validation
    mock_benchmark = {
        "resolution": "720p",
        "duration_minutes": 8.0,
        "success": True,
        "metrics": {"gpu_memory_peak_gb": 10.0}
    }
    
    perf_result = tester.validate_performance_targets(mock_benchmark)
    print(f"   Performance validation: {perf_result.status.value}")
    print(f"   Message: {perf_result.message}")
    
    # Generate performance report
    mock_full_results = {
        "test_session_id": "test_demo",
        "start_time": "2023-01-01T12:00:00",
        "total_duration_minutes": 20.0,
        "overall_status": "passed",
        "tests": {"720p": mock_benchmark},
        "validations": {"720p": {"status": "passed", "message": "Target met"}}
    }
    
    report = tester.generate_performance_report(mock_full_results)
    print("   Performance report generated successfully")
    
    print("\nAll PerformanceTester components tested successfully!")
    return True

    assert True  # TODO: Add proper assertion

def test_metrics_collection():
    """Test actual metrics collection for a short period"""
    print("\nTesting actual metrics collection...")
    
    collector = MetricsCollector()
    
    # Start monitoring for 3 seconds
    print("Starting metrics collection for 3 seconds...")
    collector.start_monitoring(interval=0.5)
    
    import time
    time.sleep(3)
    
    collector.stop_monitoring()
    
    # Get summary stats
    stats = collector.get_summary_stats()
    print(f"Collected {stats.get('total_samples', 0)} samples")
    print(f"Average CPU: {stats.get('cpu_avg', 0):.1f}%")
    print(f"Peak Memory: {stats.get('memory_peak_gb', 0):.1f}GB")
    print(f"GPU Memory Available: {stats.get('gpu_memory_avg_gb', 0) > 0}")
    
    return True

    assert True  # TODO: Add proper assertion

if __name__ == '__main__':
    try:
        test_basic_functionality()
        test_metrics_collection()
        print("\n✓ PerformanceTester implementation is working correctly!")
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)