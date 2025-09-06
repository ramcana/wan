#!/usr/bin/env python3
"""
Performance optimizer for health monitoring system.

This module provides performance profiling, caching, and optimization
capabilities for health checks to ensure fast execution in CI/CD environments.
"""

import asyncio
import functools
import hashlib
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import cProfile
import pstats
import psutil


class HealthCheckCache:
    """Caching system for health check results."""
    
    def __init__(self, cache_dir: Path = None, default_ttl: int = 3600):
        self.cache_dir = cache_dir or Path(".health_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.default_ttl = default_ttl
        
        # In-memory cache for frequently accessed items
        self._memory_cache = {}
        self._memory_cache_timestamps = {}
        self.max_memory_items = 100
    
    def _get_cache_key(self, check_name: str, inputs: Dict) -> str:
        """Generate cache key from check name and inputs."""
        # Create deterministic hash from inputs
        input_str = json.dumps(inputs, sort_keys=True)
        input_hash = hashlib.md5(input_str.encode()).hexdigest()
        return f"{check_name}_{input_hash}"
    
    def _get_cache_file(self, cache_key: str) -> Path:
        """Get cache file path for given key."""
        return self.cache_dir / f"{cache_key}.cache"
    
    def _is_cache_valid(self, cache_file: Path, ttl: int) -> bool:
        """Check if cache file is still valid."""
        if not cache_file.exists():
            return False
        
        file_age = time.time() - cache_file.stat().st_mtime
        return file_age < ttl
    
    def get(self, check_name: str, inputs: Dict, ttl: int = None) -> Optional[Any]:
        """Get cached result for health check."""
        if ttl is None:
            ttl = self.default_ttl
        
        cache_key = self._get_cache_key(check_name, inputs)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            timestamp = self._memory_cache_timestamps[cache_key]
            if time.time() - timestamp < ttl:
                return self._memory_cache[cache_key]
            else:
                # Expired, remove from memory cache
                del self._memory_cache[cache_key]
                del self._memory_cache_timestamps[cache_key]
        
        # Check disk cache
        cache_file = self._get_cache_file(cache_key)
        if self._is_cache_valid(cache_file, ttl):
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                
                # Add to memory cache
                self._add_to_memory_cache(cache_key, result)
                return result
            except Exception as e:
                print(f"⚠️ Error reading cache file {cache_file}: {e}")
                # Remove corrupted cache file
                cache_file.unlink(missing_ok=True)
        
        return None
    
    def set(self, check_name: str, inputs: Dict, result: Any):
        """Cache result for health check."""
        cache_key = self._get_cache_key(check_name, inputs)
        
        # Save to disk cache
        cache_file = self._get_cache_file(cache_key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"⚠️ Error writing cache file {cache_file}: {e}")
        
        # Add to memory cache
        self._add_to_memory_cache(cache_key, result)
    
    def _add_to_memory_cache(self, cache_key: str, result: Any):
        """Add result to memory cache with LRU eviction."""
        # Remove oldest items if cache is full
        while len(self._memory_cache) >= self.max_memory_items:
            oldest_key = min(self._memory_cache_timestamps.keys(), 
                           key=lambda k: self._memory_cache_timestamps[k])
            del self._memory_cache[oldest_key]
            del self._memory_cache_timestamps[oldest_key]
        
        self._memory_cache[cache_key] = result
        self._memory_cache_timestamps[cache_key] = time.time()
    
    def invalidate(self, check_name: str = None):
        """Invalidate cache entries."""
        if check_name is None:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            self._memory_cache.clear()
            self._memory_cache_timestamps.clear()
        else:
            # Clear specific check cache
            pattern = f"{check_name}_*.cache"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
            
            # Clear from memory cache
            keys_to_remove = [k for k in self._memory_cache.keys() if k.startswith(f"{check_name}_")]
            for key in keys_to_remove:
                del self._memory_cache[key]
                del self._memory_cache_timestamps[key]
    
    def cleanup_expired(self, ttl: int = None):
        """Clean up expired cache entries."""
        if ttl is None:
            ttl = self.default_ttl
        
        current_time = time.time()
        
        # Clean disk cache
        for cache_file in self.cache_dir.glob("*.cache"):
            if current_time - cache_file.stat().st_mtime > ttl:
                cache_file.unlink()
        
        # Clean memory cache
        expired_keys = [
            key for key, timestamp in self._memory_cache_timestamps.items()
            if current_time - timestamp > ttl
        ]
        for key in expired_keys:
            del self._memory_cache[key]
            del self._memory_cache_timestamps[key]


class PerformanceProfiler:
    """Performance profiler for health checks."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(".health_profiles")
        self.output_dir.mkdir(exist_ok=True)
        self.profiles = {}
        self.timing_data = {}
    
    def profile_function(self, func_name: str):
        """Decorator to profile function execution."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Start profiling
                profiler = cProfile.Profile()
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss
                
                try:
                    profiler.enable()
                    result = func(*args, **kwargs)
                    profiler.disable()
                    
                    # Record timing and memory usage
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss
                    
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    # Store profiling data
                    self.timing_data[func_name] = {
                        "execution_time": execution_time,
                        "memory_delta": memory_delta,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Save profile data
                    profile_file = self.output_dir / f"{func_name}_{int(time.time())}.prof"
                    profiler.dump_stats(str(profile_file))
                    
                    return result
                    
                except Exception as e:
                    profiler.disable()
                    raise e
            
            return wrapper
        return decorator
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary for all profiled functions."""
        summary = {
            "total_functions": len(self.timing_data),
            "functions": {},
            "overall_stats": {
                "total_execution_time": 0,
                "total_memory_usage": 0,
                "slowest_function": None,
                "most_memory_intensive": None
            }
        }
        
        total_time = 0
        total_memory = 0
        slowest_func = None
        slowest_time = 0
        memory_intensive_func = None
        max_memory = 0
        
        for func_name, data in self.timing_data.items():
            execution_time = data["execution_time"]
            memory_delta = data["memory_delta"]
            
            summary["functions"][func_name] = {
                "execution_time": execution_time,
                "memory_delta": memory_delta,
                "memory_delta_mb": memory_delta / (1024 * 1024),
                "timestamp": data["timestamp"]
            }
            
            total_time += execution_time
            total_memory += memory_delta
            
            if execution_time > slowest_time:
                slowest_time = execution_time
                slowest_func = func_name
            
            if memory_delta > max_memory:
                max_memory = memory_delta
                memory_intensive_func = func_name
        
        summary["overall_stats"].update({
            "total_execution_time": total_time,
            "total_memory_usage": total_memory,
            "total_memory_usage_mb": total_memory / (1024 * 1024),
            "slowest_function": slowest_func,
            "slowest_time": slowest_time,
            "most_memory_intensive": memory_intensive_func,
            "max_memory_usage": max_memory,
            "max_memory_usage_mb": max_memory / (1024 * 1024)
        })
        
        return summary
    
    def generate_optimization_recommendations(self) -> List[Dict]:
        """Generate optimization recommendations based on profiling data."""
        recommendations = []
        
        summary = self.get_performance_summary()
        
        # Check for slow functions
        for func_name, data in summary["functions"].items():
            execution_time = data["execution_time"]
            memory_usage_mb = data["memory_delta_mb"]
            
            if execution_time > 10:  # Functions taking more than 10 seconds
                recommendations.append({
                    "type": "performance",
                    "severity": "high" if execution_time > 30 else "medium",
                    "function": func_name,
                    "issue": f"Slow execution time: {execution_time:.2f}s",
                    "suggestions": [
                        "Consider caching results if function is deterministic",
                        "Optimize algorithms or data structures",
                        "Use parallel processing if applicable",
                        "Profile individual operations within the function"
                    ]
                })
            
            if memory_usage_mb > 100:  # Functions using more than 100MB
                recommendations.append({
                    "type": "memory",
                    "severity": "high" if memory_usage_mb > 500 else "medium",
                    "function": func_name,
                    "issue": f"High memory usage: {memory_usage_mb:.1f}MB",
                    "suggestions": [
                        "Use generators instead of lists for large datasets",
                        "Process data in chunks",
                        "Clear intermediate variables",
                        "Consider streaming processing"
                    ]
                })
        
        # Overall recommendations
        total_time = summary["overall_stats"]["total_execution_time"]
        if total_time > 60:  # Total execution time over 1 minute
            recommendations.append({
                "type": "overall",
                "severity": "medium",
                "function": "all",
                "issue": f"Total execution time too high: {total_time:.2f}s",
                "suggestions": [
                    "Enable parallel execution of independent checks",
                    "Implement incremental analysis",
                    "Use lightweight mode for frequent checks",
                    "Cache results more aggressively"
                ]
            })
        
        return recommendations


class IncrementalAnalyzer:
    """Incremental analysis system for large codebases."""
    
    def __init__(self, state_file: Path = None):
        self.state_file = state_file or Path(".health_state.json")
        self.previous_state = self._load_state()
        self.current_state = {}
    
    def _load_state(self) -> Dict:
        """Load previous analysis state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading state file: {e}")
        return {}
    
    def _save_state(self):
        """Save current analysis state."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.current_state, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving state file: {e}")
    
    def get_file_hash(self, file_path: Path) -> str:
        """Get hash of file content."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return ""
    
    def get_changed_files(self, file_patterns: List[str]) -> Set[Path]:
        """Get list of files that have changed since last analysis."""
        changed_files = set()
        
        for pattern in file_patterns:
            for file_path in Path(".").glob(pattern):
                if file_path.is_file():
                    current_hash = self.get_file_hash(file_path)
                    previous_hash = self.previous_state.get("file_hashes", {}).get(str(file_path))
                    
                    if current_hash != previous_hash:
                        changed_files.add(file_path)
                    
                    # Update current state
                    if "file_hashes" not in self.current_state:
                        self.current_state["file_hashes"] = {}
                    self.current_state["file_hashes"][str(file_path)] = current_hash
        
        return changed_files
    
    def should_run_check(self, check_name: str, dependencies: List[str] = None) -> bool:
        """Determine if a check should run based on changes."""
        if dependencies is None:
            dependencies = ["**/*.py", "**/*.yaml", "**/*.json"]
        
        # Always run if no previous state
        if not self.previous_state:
            return True
        
        # Check if any dependencies have changed
        changed_files = self.get_changed_files(dependencies)
        
        # Update check state
        self.current_state.setdefault("check_runs", {})[check_name] = {
            "timestamp": datetime.now().isoformat(),
            "dependencies": dependencies,
            "changed_files": [str(f) for f in changed_files]
        }
        
        return len(changed_files) > 0
    
    def finalize_analysis(self):
        """Finalize analysis and save state."""
        self.current_state["last_analysis"] = datetime.now().isoformat()
        self._save_state()


class LightweightHealthChecker:
    """Lightweight health checker for frequent execution."""
    
    def __init__(self, cache: HealthCheckCache, profiler: PerformanceProfiler):
        self.cache = cache
        self.profiler = profiler
        self.incremental = IncrementalAnalyzer()
    
    @property
    def lightweight_checks(self) -> List[str]:
        """List of checks suitable for lightweight mode."""
        return [
            "syntax_check",
            "import_validation", 
            "config_validation",
            "basic_test_health",
            "critical_file_check"
        ]
    
    def run_lightweight_health_check(self) -> Dict:
        """Run lightweight health check with minimal overhead."""
        start_time = time.time()
        
        results = {
            "mode": "lightweight",
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "summary": {
                "total_checks": 0,
                "passed_checks": 0,
                "failed_checks": 0,
                "skipped_checks": 0
            }
        }
        
        for check_name in self.lightweight_checks:
            # Check if we need to run this check
            if not self.incremental.should_run_check(check_name):
                results["checks"][check_name] = {
                    "status": "skipped",
                    "reason": "no_changes",
                    "cached": True
                }
                results["summary"]["skipped_checks"] += 1
                continue
            
            # Try to get cached result
            cached_result = self.cache.get(check_name, {}, ttl=1800)  # 30 minute cache
            if cached_result:
                results["checks"][check_name] = cached_result
                results["checks"][check_name]["cached"] = True
                if cached_result["status"] == "passed":
                    results["summary"]["passed_checks"] += 1
                else:
                    results["summary"]["failed_checks"] += 1
                continue
            
            # Run the check
            try:
                check_result = self._run_single_lightweight_check(check_name)
                results["checks"][check_name] = check_result
                results["checks"][check_name]["cached"] = False
                
                # Cache the result
                self.cache.set(check_name, {}, check_result)
                
                if check_result["status"] == "passed":
                    results["summary"]["passed_checks"] += 1
                else:
                    results["summary"]["failed_checks"] += 1
                    
            except Exception as e:
                results["checks"][check_name] = {
                    "status": "error",
                    "error": str(e),
                    "cached": False
                }
                results["summary"]["failed_checks"] += 1
            
            results["summary"]["total_checks"] += 1
        
        # Calculate overall health score
        total_checks = results["summary"]["total_checks"]
        if total_checks > 0:
            pass_rate = results["summary"]["passed_checks"] / total_checks
            results["health_score"] = pass_rate * 100
        else:
            results["health_score"] = 100
        
        # Finalize incremental analysis
        self.incremental.finalize_analysis()
        
        results["execution_time"] = time.time() - start_time
        
        return results
    
    def _run_single_lightweight_check(self, check_name: str) -> Dict:
        """Run a single lightweight health check."""
        
        if check_name == "syntax_check":
            return self._check_python_syntax()
        elif check_name == "import_validation":
            return self._check_imports()
        elif check_name == "config_validation":
            return self._check_config_files()
        elif check_name == "basic_test_health":
            return self._check_basic_test_health()
        elif check_name == "critical_file_check":
            return self._check_critical_files()
        else:
            return {"status": "error", "error": f"Unknown check: {check_name}"}
    
    def _check_python_syntax(self) -> Dict:
        """Quick syntax check for Python files."""
        import ast
        
        issues = []
        checked_files = 0
        
        for py_file in Path(".").glob("**/*.py"):
            if "venv" in str(py_file) or "__pycache__" in str(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                ast.parse(content)
                checked_files += 1
            except SyntaxError as e:
                issues.append(f"Syntax error in {py_file}: {e}")
            except Exception as e:
                issues.append(f"Error checking {py_file}: {e}")
        
        return {
            "status": "passed" if not issues else "failed",
            "checked_files": checked_files,
            "issues": issues,
            "description": f"Checked {checked_files} Python files for syntax errors"
        }
    
    def _check_imports(self) -> Dict:
        """Quick import validation."""
        import subprocess
        
        try:
            # Quick import check for main modules
            result = subprocess.run([
                "python", "-c", 
                "import backend.app; import tools.health_checker.health_checker"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return {
                    "status": "passed",
                    "description": "Main modules import successfully"
                }
            else:
                return {
                    "status": "failed",
                    "error": result.stderr,
                    "description": "Import validation failed"
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "description": "Error during import validation"
            }
    
    def _check_config_files(self) -> Dict:
        """Quick configuration file validation."""
        import yaml
        
        config_files = [
            "config/base.yaml",
            "config/unified-config.yaml",
            "backend/config.json"
        ]
        
        issues = []
        checked_files = 0
        
        for config_file in config_files:
            config_path = Path(config_file)
            if not config_path.exists():
                continue
            
            try:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    with open(config_path, 'r') as f:
                        yaml.safe_load(f)
                elif config_file.endswith('.json'):
                    with open(config_path, 'r') as f:
                        json.load(f)
                
                checked_files += 1
            except Exception as e:
                issues.append(f"Invalid config file {config_file}: {e}")
        
        return {
            "status": "passed" if not issues else "failed",
            "checked_files": checked_files,
            "issues": issues,
            "description": f"Validated {checked_files} configuration files"
        }
    
    def _check_basic_test_health(self) -> Dict:
        """Quick test health check."""
        test_files = list(Path("tests").glob("**/*.py")) if Path("tests").exists() else []
        
        if not test_files:
            return {
                "status": "failed",
                "error": "No test files found",
                "description": "Basic test health check"
            }
        
        # Quick check for test file structure
        test_functions = 0
        for test_file in test_files[:10]:  # Check first 10 test files only
            try:
                with open(test_file, 'r') as f:
                    content = f.read()
                    test_functions += content.count("def test_")
            except Exception:
                continue
        
        return {
            "status": "passed" if test_functions > 0 else "failed",
            "test_files": len(test_files),
            "test_functions": test_functions,
            "description": f"Found {len(test_files)} test files with {test_functions} test functions"
        }
    
    def _check_critical_files(self) -> Dict:
        """Check for presence of critical files."""
        critical_files = [
            "README.md",
            "requirements.txt",
            "backend/requirements.txt",
            "config/base.yaml"
        ]
        
        missing_files = []
        for file_path in critical_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        return {
            "status": "passed" if not missing_files else "failed",
            "missing_files": missing_files,
            "description": f"Checked {len(critical_files)} critical files"
        }


def main():
    """Main function for performance optimization."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Health monitoring performance optimization")
    parser.add_argument("--mode", choices=["profile", "lightweight", "cache-cleanup"], 
                       default="profile", help="Operation mode")
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument("--cache-ttl", type=int, default=3600, help="Cache TTL in seconds")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else Path(".")
    
    if args.mode == "profile":
        # Run performance profiling
        profiler = PerformanceProfiler(output_dir / "profiles")
        
        # This would normally be integrated with actual health checks
        print("🔍 Performance profiling mode - integrate with health checker")
        
        # Generate recommendations
        recommendations = profiler.generate_optimization_recommendations()
        
        with open(output_dir / "optimization_recommendations.json", 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        print(f"✅ Generated {len(recommendations)} optimization recommendations")
    
    elif args.mode == "lightweight":
        # Run lightweight health check
        cache = HealthCheckCache(ttl=args.cache_ttl)
        profiler = PerformanceProfiler()
        lightweight_checker = LightweightHealthChecker(cache, profiler)
        
        print("🚀 Running lightweight health check...")
        results = lightweight_checker.run_lightweight_health_check()
        
        with open(output_dir / "lightweight_health_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Lightweight check completed in {results['execution_time']:.2f}s")
        print(f"   Health score: {results['health_score']:.1f}%")
        print(f"   Checks: {results['summary']['passed_checks']} passed, "
              f"{results['summary']['failed_checks']} failed, "
              f"{results['summary']['skipped_checks']} skipped")
    
    elif args.mode == "cache-cleanup":
        # Clean up expired cache entries
        cache = HealthCheckCache(ttl=args.cache_ttl)
        cache.cleanup_expired()
        print("✅ Cache cleanup completed")


if __name__ == "__main__":
    main()