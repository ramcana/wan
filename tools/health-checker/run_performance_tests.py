#!/usr/bin/env python3
"""
Performance testing script for health monitoring system.

This script runs various performance tests to validate and optimize
the health monitoring system performance.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import psutil

from health_checker import ProjectHealthChecker
from performance_optimizer import HealthCheckCache, PerformanceProfiler, LightweightHealthChecker
from parallel_executor import ParallelHealthExecutor, HealthCheckTask


class HealthPerformanceTester:
    """Performance tester for health monitoring system."""
    
    def __init__(self):
        self.results = {}
        self.baseline_metrics = {}
    
    def run_all_performance_tests(self) -> Dict:
        """Run comprehensive performance tests."""
        
        print("🚀 Starting health monitoring performance tests...")
        
        # Test 1: Baseline sequential execution
        print("\n📊 Test 1: Baseline sequential execution")
        self.results["baseline"] = self.test_baseline_performance()
        
        # Test 2: Parallel execution
        print("\n📊 Test 2: Parallel execution")
        self.results["parallel"] = self.test_parallel_performance()
        
        # Test 3: Cached execution
        print("\n📊 Test 3: Cached execution")
        self.results["cached"] = self.test_cached_performance()
        
        # Test 4: Lightweight execution
        print("\n📊 Test 4: Lightweight execution")
        self.results["lightweight"] = self.test_lightweight_performance()
        
        # Test 5: Incremental analysis
        print("\n📊 Test 5: Incremental analysis")
        self.results["incremental"] = self.test_incremental_performance()
        
        # Test 6: Resource usage
        print("\n📊 Test 6: Resource usage analysis")
        self.results["resource_usage"] = self.test_resource_usage()
        
        # Generate performance report
        self.generate_performance_report()
        
        return self.results
    
    def test_baseline_performance(self) -> Dict:
        """Test baseline sequential health check performance."""
        
        checker = ProjectHealthChecker()
        
        # Measure execution time and resource usage
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            # Run synchronous health check
            health_score = checker.get_health_score()
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            result = {
                "execution_time": execution_time,
                "memory_usage_mb": memory_delta / (1024 * 1024),
                "health_score": health_score,
                "success": True
            }
            
            print(f"   ✅ Baseline: {execution_time:.2f}s, {memory_delta / (1024 * 1024):.1f}MB")
            return result
            
        except Exception as e:
            print(f"   ❌ Baseline failed: {e}")
            return {
                "execution_time": 0,
                "memory_usage_mb": 0,
                "health_score": 0,
                "success": False,
                "error": str(e)
            }
    
    def test_parallel_performance(self) -> Dict:
        """Test parallel execution performance."""
        
        async def run_parallel_test():
            checker = ProjectHealthChecker()
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            try:
                # Run optimized health check with parallel execution
                report = await checker.run_optimized_health_check(
                    lightweight=False,
                    use_cache=False,
                    parallel=True
                )
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss
                
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                result = {
                    "execution_time": execution_time,
                    "memory_usage_mb": memory_delta / (1024 * 1024),
                    "health_score": report.overall_score,
                    "success": True,
                    "parallel_speedup": 0  # Will be calculated later
                }
                
                print(f"   ✅ Parallel: {execution_time:.2f}s, {memory_delta / (1024 * 1024):.1f}MB")
                return result
                
            except Exception as e:
                print(f"   ❌ Parallel failed: {e}")
                return {
                    "execution_time": 0,
                    "memory_usage_mb": 0,
                    "health_score": 0,
                    "success": False,
                    "error": str(e)
                }
        
        return asyncio.run(run_parallel_test())
    
    def test_cached_performance(self) -> Dict:
        """Test cached execution performance."""
        
        async def run_cached_test():
            checker = ProjectHealthChecker()
            
            # First run to populate cache
            print("   🔄 Populating cache...")
            await checker.run_optimized_health_check(use_cache=True, parallel=False)
            
            # Second run to test cache performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            try:
                report = await checker.run_optimized_health_check(
                    use_cache=True,
                    parallel=False
                )
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss
                
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                result = {
                    "execution_time": execution_time,
                    "memory_usage_mb": memory_delta / (1024 * 1024),
                    "health_score": report.overall_score,
                    "success": True,
                    "cache_speedup": 0  # Will be calculated later
                }
                
                print(f"   ✅ Cached: {execution_time:.2f}s, {memory_delta / (1024 * 1024):.1f}MB")
                return result
                
            except Exception as e:
                print(f"   ❌ Cached failed: {e}")
                return {
                    "execution_time": 0,
                    "memory_usage_mb": 0,
                    "health_score": 0,
                    "success": False,
                    "error": str(e)
                }
        
        return asyncio.run(run_cached_test())
    
    def test_lightweight_performance(self) -> Dict:
        """Test lightweight execution performance."""
        
        async def run_lightweight_test():
            checker = ProjectHealthChecker()
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            try:
                report = await checker.run_optimized_health_check(
                    lightweight=True,
                    use_cache=True,
                    parallel=False
                )
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss
                
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                result = {
                    "execution_time": execution_time,
                    "memory_usage_mb": memory_delta / (1024 * 1024),
                    "health_score": report.overall_score,
                    "success": True,
                    "lightweight_speedup": 0  # Will be calculated later
                }
                
                print(f"   ✅ Lightweight: {execution_time:.2f}s, {memory_delta / (1024 * 1024):.1f}MB")
                return result
                
            except Exception as e:
                print(f"   ❌ Lightweight failed: {e}")
                return {
                    "execution_time": 0,
                    "memory_usage_mb": 0,
                    "health_score": 0,
                    "success": False,
                    "error": str(e)
                }
        
        return asyncio.run(run_lightweight_test())
    
    def test_incremental_performance(self) -> Dict:
        """Test incremental analysis performance."""
        
        from performance_optimizer import IncrementalAnalyzer
        
        analyzer = IncrementalAnalyzer()
        
        start_time = time.time()
        
        try:
            # Test file change detection
            changed_files = analyzer.get_changed_files(["**/*.py", "**/*.yaml", "**/*.json"])
            
            # Test check scheduling
            should_run_tests = analyzer.should_run_check("test_health", ["tests/**/*.py"])
            should_run_config = analyzer.should_run_check("config_health", ["config/**/*"])
            
            analyzer.finalize_analysis()
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            result = {
                "execution_time": execution_time,
                "changed_files_count": len(changed_files),
                "should_run_tests": should_run_tests,
                "should_run_config": should_run_config,
                "success": True
            }
            
            print(f"   ✅ Incremental: {execution_time:.2f}s, {len(changed_files)} changed files")
            return result
            
        except Exception as e:
            print(f"   ❌ Incremental failed: {e}")
            return {
                "execution_time": 0,
                "changed_files_count": 0,
                "success": False,
                "error": str(e)
            }
    
    def test_resource_usage(self) -> Dict:
        """Test resource usage during health checks."""
        
        from performance_optimizer import ResourceMonitor
        
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        try:
            # Run a comprehensive health check while monitoring
            checker = ProjectHealthChecker()
            
            async def monitored_health_check():
                return await checker.run_optimized_health_check(parallel=True)
            
            report = asyncio.run(monitored_health_check())
            
            # Stop monitoring and get stats
            monitor.stop_monitoring()
            stats = monitor.stats
            
            result = {
                "max_cpu_percent": stats["max_cpu"],
                "max_memory_percent": stats["max_memory"],
                "avg_cpu_percent": stats["avg_cpu"],
                "avg_memory_percent": stats["avg_memory"],
                "sample_count": len(stats["samples"]),
                "health_score": report.overall_score,
                "success": True
            }
            
            print(f"   ✅ Resource usage: CPU {stats['max_cpu']:.1f}%, Memory {stats['max_memory']:.1f}%")
            return result
            
        except Exception as e:
            monitor.stop_monitoring()
            print(f"   ❌ Resource monitoring failed: {e}")
            return {
                "max_cpu_percent": 0,
                "max_memory_percent": 0,
                "success": False,
                "error": str(e)
            }
    
    def calculate_performance_improvements(self):
        """Calculate performance improvements from optimizations."""
        
        baseline_time = self.results.get("baseline", {}).get("execution_time", 0)
        
        if baseline_time > 0:
            # Calculate speedups
            parallel_time = self.results.get("parallel", {}).get("execution_time", baseline_time)
            cached_time = self.results.get("cached", {}).get("execution_time", baseline_time)
            lightweight_time = self.results.get("lightweight", {}).get("execution_time", baseline_time)
            
            self.results["parallel"]["parallel_speedup"] = baseline_time / parallel_time if parallel_time > 0 else 0
            self.results["cached"]["cache_speedup"] = baseline_time / cached_time if cached_time > 0 else 0
            self.results["lightweight"]["lightweight_speedup"] = baseline_time / lightweight_time if lightweight_time > 0 else 0
    
    def generate_performance_report(self):
        """Generate comprehensive performance report."""
        
        self.calculate_performance_improvements()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "test_results": self.results,
            "summary": {
                "baseline_time": self.results.get("baseline", {}).get("execution_time", 0),
                "best_time": min([
                    result.get("execution_time", float('inf'))
                    for result in self.results.values()
                    if result.get("success", False)
                ]),
                "parallel_speedup": self.results.get("parallel", {}).get("parallel_speedup", 0),
                "cache_speedup": self.results.get("cached", {}).get("cache_speedup", 0),
                "lightweight_speedup": self.results.get("lightweight", {}).get("lightweight_speedup", 0)
            },
            "recommendations": self.generate_performance_recommendations()
        }
        
        # Save report
        report_file = Path("performance_test_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Performance Test Summary:")
        print(f"   Baseline execution time: {report['summary']['baseline_time']:.2f}s")
        print(f"   Best execution time: {report['summary']['best_time']:.2f}s")
        print(f"   Parallel speedup: {report['summary']['parallel_speedup']:.2f}x")
        print(f"   Cache speedup: {report['summary']['cache_speedup']:.2f}x")
        print(f"   Lightweight speedup: {report['summary']['lightweight_speedup']:.2f}x")
        print(f"\n📋 Report saved to: {report_file}")
        
        return report
    
    def generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        
        recommendations = []
        
        # Check parallel execution effectiveness
        parallel_speedup = self.results.get("parallel", {}).get("parallel_speedup", 0)
        if parallel_speedup < 1.5:
            recommendations.append(
                "Parallel execution shows limited improvement. Consider optimizing individual checks or increasing parallelism."
            )
        elif parallel_speedup > 2.0:
            recommendations.append(
                "Parallel execution is highly effective. Consider using it as the default mode."
            )
        
        # Check cache effectiveness
        cache_speedup = self.results.get("cached", {}).get("cache_speedup", 0)
        if cache_speedup < 2.0:
            recommendations.append(
                "Caching shows limited improvement. Consider increasing cache TTL or improving cache key generation."
            )
        elif cache_speedup > 5.0:
            recommendations.append(
                "Caching is highly effective. Consider more aggressive caching strategies."
            )
        
        # Check lightweight mode effectiveness
        lightweight_speedup = self.results.get("lightweight", {}).get("lightweight_speedup", 0)
        if lightweight_speedup > 10.0:
            recommendations.append(
                "Lightweight mode is very fast. Consider using it for frequent CI checks."
            )
        
        # Check resource usage
        resource_result = self.results.get("resource_usage", {})
        max_cpu = resource_result.get("max_cpu_percent", 0)
        max_memory = resource_result.get("max_memory_percent", 0)
        
        if max_cpu > 80:
            recommendations.append(
                f"High CPU usage detected ({max_cpu:.1f}%). Consider reducing parallelism or optimizing algorithms."
            )
        
        if max_memory > 80:
            recommendations.append(
                f"High memory usage detected ({max_memory:.1f}%). Consider processing data in chunks or clearing intermediate results."
            )
        
        return recommendations


def main():
    """Main function for performance testing."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Health monitoring performance tests")
    parser.add_argument("--test", choices=["all", "baseline", "parallel", "cached", "lightweight", "incremental", "resource"],
                       default="all", help="Specific test to run")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    tester = HealthPerformanceTester()
    
    if args.test == "all":
        results = tester.run_all_performance_tests()
    else:
        # Run specific test
        test_method = getattr(tester, f"test_{args.test}_performance")
        results = {args.test: test_method()}
        tester.results = results
        tester.generate_performance_report()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()