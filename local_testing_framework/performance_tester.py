"""
Performance Tester for Local Testing Framework

Integrates with existing performance_profiler.py to provide comprehensive
performance testing, benchmarking, and VRAM optimization validation.
"""

import json
import os
import platform
import subprocess
import time
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading
import multiprocessing

from .models.test_results import ValidationResult, ValidationStatus, TestResults, TestStatus
from .models.configuration import LocalTestConfiguration, PerformanceTargets


class MetricsCollector:
    """
    Collects CPU, memory, GPU, and VRAM metrics during performance testing
    """
    
    def __init__(self):
        self.metrics = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self, interval: float = 1.0):
        """Start collecting metrics at specified interval"""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop collecting metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                timestamp = datetime.now()
                
                # CPU and Memory metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                # GPU metrics (if available)
                gpu_metrics = self._get_gpu_metrics()
                
                metric = {
                    "timestamp": timestamp.isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    **gpu_metrics
                }
                
                self.metrics.append(metric)
                
            except Exception as e:
                # Continue monitoring even if individual metric collection fails
                pass
            
            time.sleep(interval)
    
    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Get GPU and VRAM metrics using torch if available"""
        gpu_metrics = {
            "gpu_available": False,
            "gpu_memory_used_gb": 0,
            "gpu_memory_total_gb": 0,
            "gpu_memory_percent": 0,
            "gpu_utilization": 0
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                
                # Memory metrics
                memory_used = torch.cuda.memory_allocated(device)
                memory_total = torch.cuda.get_device_properties(device).total_memory
                
                gpu_metrics.update({
                    "gpu_available": True,
                    "gpu_memory_used_gb": memory_used / (1024**3),
                    "gpu_memory_total_gb": memory_total / (1024**3),
                    "gpu_memory_percent": (memory_used / memory_total) * 100 if memory_total > 0 else 0
                })
                
                # Try to get GPU utilization using nvidia-ml-py if available
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_metrics["gpu_utilization"] = utilization.gpu
                except:
                    pass
                    
        except ImportError:
            pass
        
        return gpu_metrics
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from collected metrics"""
        if not self.metrics:
            return {}
        
        # Calculate averages, peaks, etc.
        cpu_values = [m["cpu_percent"] for m in self.metrics]
        memory_values = [m["memory_used_gb"] for m in self.metrics]
        gpu_memory_values = [m["gpu_memory_used_gb"] for m in self.metrics]
        
        return {
            "duration_seconds": len(self.metrics),
            "cpu_avg": sum(cpu_values) / len(cpu_values),
            "cpu_peak": max(cpu_values),
            "memory_avg_gb": sum(memory_values) / len(memory_values),
            "memory_peak_gb": max(memory_values),
            "gpu_memory_avg_gb": sum(gpu_memory_values) / len(gpu_memory_values) if gpu_memory_values else 0,
            "gpu_memory_peak_gb": max(gpu_memory_values) if gpu_memory_values else 0,
            "total_samples": len(self.metrics)
        }


class BenchmarkRunner:
    """
    Automated timing measurements and benchmark execution
    """
    
    def __init__(self, config: Optional[LocalTestConfiguration] = None):
        self.config = config or LocalTestConfiguration()
        self.metrics_collector = MetricsCollector()
    
    def run_video_generation_benchmark(self, resolution: str = "720p", 
                                     prompt: str = "A serene landscape with mountains") -> Dict[str, Any]:
        """
        Run video generation benchmark with timing and resource monitoring
        """
        start_time = datetime.now()
        
        # Start metrics collection
        self.metrics_collector.start_monitoring(interval=1.0)
        
        try:
            # Create sample input for testing
            sample_input = {
                "input": prompt,
                "resolution": resolution,
                "output_path": f"outputs/benchmark_{resolution}_{int(time.time())}.mp4"
            }
            
            # Write sample input file
            input_file = Path("sample_input.json")
            with open(input_file, 'w') as f:
                json.dump(sample_input, f, indent=2)
            
            # Run the main application
            result = subprocess.run([
                "python", "main.py", 
                "--input", str(input_file),
                "--benchmark"
            ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            # Stop metrics collection
            self.metrics_collector.stop_monitoring()
            
            # Get performance metrics
            metrics_summary = self.metrics_collector.get_summary_stats()
            
            # Check if output file was created
            output_created = Path(sample_input["output_path"]).exists()
            
            benchmark_result = {
                "resolution": resolution,
                "prompt": prompt,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "duration_minutes": duration.total_seconds() / 60,
                "success": result.returncode == 0 and output_created,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_created": output_created,
                "output_path": sample_input["output_path"],
                "metrics": metrics_summary,
                "raw_metrics": self.metrics_collector.metrics
            }
            
            # Clean up
            if input_file.exists():
                input_file.unlink()
            
            return benchmark_result
            
        except subprocess.TimeoutExpired:
            self.metrics_collector.stop_monitoring()
            return {
                "resolution": resolution,
                "prompt": prompt,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": 1800,
                "duration_minutes": 30,
                "success": False,
                "error": "Benchmark timed out after 30 minutes",
                "metrics": self.metrics_collector.get_summary_stats(),
                "raw_metrics": self.metrics_collector.metrics
            }
        except Exception as e:
            self.metrics_collector.stop_monitoring()
            return {
                "resolution": resolution,
                "prompt": prompt,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "success": False,
                "error": str(e),
                "metrics": self.metrics_collector.get_summary_stats(),
                "raw_metrics": self.metrics_collector.metrics
            }
    
    def run_performance_profiler_benchmark(self) -> Dict[str, Any]:
        """
        Run benchmark using the existing performance_profiler.py
        """
        try:
            # Run performance profiler in benchmark mode
            result = subprocess.run([
                "python", "performance_profiler.py", "--benchmark"
            ], capture_output=True, text=True, timeout=1800)
            
            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "profiler_available": True
            }
            
        except FileNotFoundError:
            return {
                "success": False,
                "error": "performance_profiler.py not found",
                "profiler_available": False
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Performance profiler timed out",
                "profiler_available": True
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "profiler_available": False
            }


class PerformanceTester:
    """
    Main performance testing orchestrator that integrates with performance_profiler.py
    """
    
    def __init__(self, config: Optional[LocalTestConfiguration] = None):
        self.config = config or LocalTestConfiguration()
        self.benchmark_runner = BenchmarkRunner(config)
        
    def validate_performance_targets(self, benchmark_results: Dict[str, Any]) -> ValidationResult:
        """
        Validate benchmark results against performance targets
        """
        targets = self.config.performance_targets
        resolution = benchmark_results.get("resolution", "unknown")
        duration_minutes = benchmark_results.get("duration_minutes", float('inf'))
        success = benchmark_results.get("success", False)
        
        if not success:
            return ValidationResult(
                component="performance_targets",
                status=ValidationStatus.FAILED,
                message=f"Benchmark failed for {resolution}",
                details=benchmark_results,
                remediation_steps=[
                    "Check error logs for specific failure reasons",
                    "Ensure sufficient system resources are available",
                    "Verify CUDA is working properly",
                    "Check input file format and content"
                ]
            )
        
        # Check resolution-specific targets
        target_time = None
        if resolution == "720p":
            target_time = targets.target_720p_time_minutes
        elif resolution == "1080p":
            target_time = targets.target_1080p_time_minutes
        
        if target_time is None:
            return ValidationResult(
                component="performance_targets",
                status=ValidationStatus.WARNING,
                message=f"No performance target defined for {resolution}",
                details=benchmark_results
            )
        
        # Check if duration meets target
        meets_target = duration_minutes <= target_time
        
        # Check VRAM usage
        metrics = benchmark_results.get("metrics", {})
        vram_peak_gb = metrics.get("gpu_memory_peak_gb", 0)
        vram_within_limit = vram_peak_gb <= targets.max_vram_usage_gb
        
        issues = []
        if not meets_target:
            issues.append(f"Duration {duration_minutes:.1f}min exceeds target {target_time}min")
        if not vram_within_limit:
            issues.append(f"VRAM usage {vram_peak_gb:.1f}GB exceeds limit {targets.max_vram_usage_gb}GB")
        
        if issues:
            return ValidationResult(
                component="performance_targets",
                status=ValidationStatus.FAILED,
                message=f"Performance targets not met for {resolution}: {'; '.join(issues)}",
                details={
                    "resolution": resolution,
                    "duration_minutes": duration_minutes,
                    "target_minutes": target_time,
                    "vram_peak_gb": vram_peak_gb,
                    "vram_limit_gb": targets.max_vram_usage_gb,
                    "benchmark_results": benchmark_results
                },
                remediation_steps=[
                    "Enable VRAM optimizations in config.json",
                    "Use attention slicing: enable_attention_slicing=true",
                    "Enable VAE tiling: enable_vae_tiling=true",
                    "Consider using lower precision: use_fp16=true",
                    "Reduce batch size or sequence length",
                    "Close other GPU-intensive applications"
                ]
            )
        
        return ValidationResult(
            component="performance_targets",
            status=ValidationStatus.PASSED,
            message=f"Performance targets met for {resolution}: {duration_minutes:.1f}min <= {target_time}min, VRAM {vram_peak_gb:.1f}GB <= {targets.max_vram_usage_gb}GB",
            details={
                "resolution": resolution,
                "duration_minutes": duration_minutes,
                "target_minutes": target_time,
                "vram_peak_gb": vram_peak_gb,
                "vram_limit_gb": targets.max_vram_usage_gb,
                "benchmark_results": benchmark_results
            }
        )
    
    def run_comprehensive_performance_test(self) -> Dict[str, Any]:
        """
        Run comprehensive performance testing including multiple resolutions
        """
        test_start = datetime.now()
        results = {
            "test_session_id": f"perf_test_{int(time.time())}",
            "start_time": test_start.isoformat(),
            "tests": {},
            "validations": {},
            "overall_status": TestStatus.ERROR
        }
        
        # Test different resolutions
        resolutions = ["720p", "1080p"]
        
        for resolution in resolutions:
            print(f"Running {resolution} benchmark...")
            
            benchmark_result = self.benchmark_runner.run_video_generation_benchmark(
                resolution=resolution,
                prompt=f"Test video generation at {resolution} resolution"
            )
            
            results["tests"][resolution] = benchmark_result
            
            # Validate against targets
            validation = self.validate_performance_targets(benchmark_result)
            results["validations"][resolution] = validation.to_dict()
        
        # Run performance profiler benchmark if available
        print("Running performance profiler benchmark...")
        profiler_result = self.benchmark_runner.run_performance_profiler_benchmark()
        results["tests"]["profiler"] = profiler_result
        
        # Determine overall status
        validation_statuses = [
            ValidationStatus(v["status"]) for v in results["validations"].values()
        ]
        
        if all(status == ValidationStatus.PASSED for status in validation_statuses):
            results["overall_status"] = TestStatus.PASSED
        elif any(status == ValidationStatus.FAILED for status in validation_statuses):
            results["overall_status"] = TestStatus.FAILED
        else:
            results["overall_status"] = TestStatus.PARTIAL
        
        results["end_time"] = datetime.now().isoformat()
        results["total_duration_minutes"] = (datetime.now() - test_start).total_seconds() / 60
        
        return results
    
    def generate_performance_report(self, test_results: Dict[str, Any]) -> str:
        """
        Generate human-readable performance test report
        """
        report_lines = [
            "Performance Test Report",
            "=" * 25,
            f"Session ID: {test_results.get('test_session_id', 'unknown')}",
            f"Start Time: {test_results.get('start_time', 'unknown')}",
            f"Total Duration: {test_results.get('total_duration_minutes', 0):.1f} minutes",
            f"Overall Status: {test_results.get('overall_status', 'unknown').upper()}",
            "",
            "Test Results:",
            "-" * 15
        ]
        
        # Add individual test results
        tests = test_results.get("tests", {})
        validations = test_results.get("validations", {})
        
        for test_name, test_result in tests.items():
            if test_name == "profiler":
                continue  # Handle separately
                
            success_symbol = "✓" if test_result.get("success", False) else "✗"
            duration = test_result.get("duration_minutes", 0)
            
            report_lines.append(f"{success_symbol} {test_name.upper()}: {duration:.1f} minutes")
            
            # Add validation result
            validation = validations.get(test_name, {})
            validation_status = validation.get("status", "unknown")
            validation_symbol = {
                "passed": "✓",
                "failed": "✗", 
                "warning": "⚠"
            }.get(validation_status, "?")
            
            report_lines.append(f"  {validation_symbol} Target Validation: {validation.get('message', 'No message')}")
            
            # Add metrics summary
            metrics = test_result.get("metrics", {})
            if metrics:
                report_lines.extend([
                    f"  • CPU Peak: {metrics.get('cpu_peak', 0):.1f}%",
                    f"  • Memory Peak: {metrics.get('memory_peak_gb', 0):.1f}GB",
                    f"  • VRAM Peak: {metrics.get('gpu_memory_peak_gb', 0):.1f}GB"
                ])
        
        # Add profiler results
        profiler_result = tests.get("profiler", {})
        if profiler_result:
            success_symbol = "✓" if profiler_result.get("success", False) else "✗"
            report_lines.extend([
                "",
                f"{success_symbol} Performance Profiler: {'Available' if profiler_result.get('profiler_available', False) else 'Not Available'}"
            ])
        
        return "\n".join(report_lines)


class OptimizationValidator:
    """
    VRAM optimization testing using optimize_performance.py
    """
    
    def __init__(self, config: Optional[LocalTestConfiguration] = None):
        self.config = config or LocalTestConfiguration()
    
    def test_vram_optimization(self) -> ValidationResult:
        """
        Test VRAM optimization using optimize_performance.py --test-vram
        """
        try:
            # Run VRAM optimization test
            result = subprocess.run([
                "python", "optimize_performance.py", "--test-vram"
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            if result.returncode == 0:
                return ValidationResult(
                    component="vram_optimization",
                    status=ValidationStatus.PASSED,
                    message="VRAM optimization test passed",
                    details={
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "return_code": result.returncode
                    }
                )
            else:
                return ValidationResult(
                    component="vram_optimization",
                    status=ValidationStatus.FAILED,
                    message="VRAM optimization test failed",
                    details={
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "return_code": result.returncode
                    },
                    remediation_steps=[
                        "Check CUDA availability and drivers",
                        "Ensure sufficient VRAM is available",
                        "Review optimize_performance.py configuration",
                        "Check for memory leaks or fragmentation"
                    ]
                )
                
        except FileNotFoundError:
            return ValidationResult(
                component="vram_optimization",
                status=ValidationStatus.FAILED,
                message="optimize_performance.py not found",
                details={"error": "optimize_performance.py script not available"},
                remediation_steps=[
                    "Ensure optimize_performance.py exists in the project root",
                    "Check file permissions",
                    "Verify script is executable"
                ]
            )
        except subprocess.TimeoutExpired:
            return ValidationResult(
                component="vram_optimization",
                status=ValidationStatus.FAILED,
                message="VRAM optimization test timed out",
                details={"error": "Test exceeded 10 minute timeout"},
                remediation_steps=[
                    "Check for infinite loops or hanging processes",
                    "Ensure adequate system resources",
                    "Review test complexity and reduce if necessary"
                ]
            )
        except Exception as e:
            return ValidationResult(
                component="vram_optimization",
                status=ValidationStatus.FAILED,
                message=f"VRAM optimization test error: {str(e)}",
                details={"error": str(e)},
                remediation_steps=[
                    "Check system stability",
                    "Verify Python environment",
                    "Review error logs for specific issues"
                ]
            )
    
    def validate_vram_reduction(self, baseline_vram: float, optimized_vram: float) -> ValidationResult:
        """
        Validate 80% VRAM reduction using optimize_performance.py --benchmark
        """
        if baseline_vram <= 0:
            return ValidationResult(
                component="vram_reduction",
                status=ValidationStatus.FAILED,
                message="Invalid baseline VRAM measurement",
                details={
                    "baseline_vram_gb": baseline_vram,
                    "optimized_vram_gb": optimized_vram
                },
                remediation_steps=[
                    "Ensure baseline measurement is taken correctly",
                    "Check GPU monitoring tools",
                    "Verify CUDA is working properly"
                ]
            )
        
        reduction_percent = ((baseline_vram - optimized_vram) / baseline_vram) * 100
        target_reduction = self.config.performance_targets.expected_vram_reduction_percent
        
        if reduction_percent >= target_reduction:
            return ValidationResult(
                component="vram_reduction",
                status=ValidationStatus.PASSED,
                message=f"VRAM reduction achieved: {reduction_percent:.1f}% >= {target_reduction}%",
                details={
                    "baseline_vram_gb": baseline_vram,
                    "optimized_vram_gb": optimized_vram,
                    "reduction_percent": reduction_percent,
                    "target_reduction_percent": target_reduction,
                    "reduction_gb": baseline_vram - optimized_vram
                }
            )
        else:
            return ValidationResult(
                component="vram_reduction",
                status=ValidationStatus.FAILED,
                message=f"VRAM reduction insufficient: {reduction_percent:.1f}% < {target_reduction}%",
                details={
                    "baseline_vram_gb": baseline_vram,
                    "optimized_vram_gb": optimized_vram,
                    "reduction_percent": reduction_percent,
                    "target_reduction_percent": target_reduction,
                    "reduction_gb": baseline_vram - optimized_vram
                },
                remediation_steps=[
                    "Enable attention slicing in configuration",
                    "Enable VAE tiling for memory efficiency",
                    "Use mixed precision (fp16) if supported",
                    "Reduce batch size or model complexity",
                    "Clear GPU cache between operations",
                    "Check for memory leaks in the application"
                ]
            )
    
    def run_optimization_benchmark(self) -> Dict[str, Any]:
        """
        Run optimization benchmark using optimize_performance.py --benchmark
        """
        try:
            # Run optimization benchmark
            result = subprocess.run([
                "python", "optimize_performance.py", "--benchmark"
            ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
            
            benchmark_data = {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat()
            }
            
            # Try to parse performance data from output
            if result.returncode == 0:
                benchmark_data.update(self._parse_benchmark_output(result.stdout))
            
            return benchmark_data
            
        except FileNotFoundError:
            return {
                "success": False,
                "error": "optimize_performance.py not found",
                "timestamp": datetime.now().isoformat()
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Optimization benchmark timed out after 30 minutes",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _parse_benchmark_output(self, output: str) -> Dict[str, Any]:
        """
        Parse benchmark output to extract performance metrics
        """
        parsed_data = {}
        
        # Look for common performance indicators in output
        lines = output.split('\n')
        for line in lines:
            line = line.strip().lower()
            
            # Parse VRAM usage
            if 'vram' in line and 'gb' in line:
                try:
                    # Extract number before 'gb'
                    parts = line.split('gb')[0].split()
                    if parts:
                        vram_value = float(parts[-1])
                        if 'baseline' in line or 'before' in line:
                            parsed_data['baseline_vram_gb'] = vram_value
                        elif 'optimized' in line or 'after' in line:
                            parsed_data['optimized_vram_gb'] = vram_value
                except (ValueError, IndexError):
                    pass
            
            # Parse timing information
            if 'time' in line and ('min' in line or 'sec' in line):
                try:
                    if 'min' in line:
                        parts = line.split('min')[0].split()
                        if parts:
                            time_value = float(parts[-1])
                            parsed_data['duration_minutes'] = time_value
                    elif 'sec' in line:
                        parts = line.split('sec')[0].split()
                        if parts:
                            time_value = float(parts[-1])
                            parsed_data['duration_seconds'] = time_value
                except (ValueError, IndexError):
                    pass
        
        return parsed_data


class PerformanceTargetValidator:
    """
    Validates performance against specific targets (720p < 9min, 1080p < 17min, VRAM < 12GB)
    """
    
    def __init__(self, config: Optional[LocalTestConfiguration] = None):
        self.config = config or LocalTestConfiguration()
        self.targets = self.config.performance_targets
    
    def validate_720p_target(self, duration_minutes: float, vram_gb: float) -> ValidationResult:
        """Validate 720p performance target"""
        return self._validate_resolution_target("720p", duration_minutes, vram_gb, 
                                               self.targets.target_720p_time_minutes)
    
    def validate_1080p_target(self, duration_minutes: float, vram_gb: float) -> ValidationResult:
        """Validate 1080p performance target"""
        return self._validate_resolution_target("1080p", duration_minutes, vram_gb,
                                               self.targets.target_1080p_time_minutes)
    
    def _validate_resolution_target(self, resolution: str, duration_minutes: float, 
                                  vram_gb: float, target_minutes: float) -> ValidationResult:
        """Common validation logic for resolution targets"""
        issues = []
        
        # Check timing target
        if duration_minutes > target_minutes:
            issues.append(f"Duration {duration_minutes:.1f}min > {target_minutes}min target")
        
        # Check VRAM target
        if vram_gb > self.targets.max_vram_usage_gb:
            issues.append(f"VRAM {vram_gb:.1f}GB > {self.targets.max_vram_usage_gb}GB limit")
        
        if issues:
            return ValidationResult(
                component=f"{resolution}_performance",
                status=ValidationStatus.FAILED,
                message=f"{resolution} performance targets not met: {'; '.join(issues)}",
                details={
                    "resolution": resolution,
                    "duration_minutes": duration_minutes,
                    "target_minutes": target_minutes,
                    "vram_gb": vram_gb,
                    "vram_limit_gb": self.targets.max_vram_usage_gb,
                    "issues": issues
                },
                remediation_steps=[
                    f"Optimize for {resolution} generation speed",
                    "Enable VRAM optimizations (attention slicing, VAE tiling)",
                    "Use mixed precision (fp16) if supported",
                    "Consider model quantization",
                    "Close other GPU-intensive applications",
                    "Upgrade hardware if consistently failing targets"
                ]
            )
        
        return ValidationResult(
            component=f"{resolution}_performance",
            status=ValidationStatus.PASSED,
            message=f"{resolution} performance targets met: {duration_minutes:.1f}min <= {target_minutes}min, VRAM {vram_gb:.1f}GB <= {self.targets.max_vram_usage_gb}GB",
            details={
                "resolution": resolution,
                "duration_minutes": duration_minutes,
                "target_minutes": target_minutes,
                "vram_gb": vram_gb,
                "vram_limit_gb": self.targets.max_vram_usage_gb
            }
        )
    
    def validate_all_targets(self, test_results: Dict[str, Any]) -> List[ValidationResult]:
        """Validate all performance targets from test results"""
        validations = []
        
        # Validate 720p if available
        if "720p" in test_results:
            result_720p = test_results["720p"]
            duration = result_720p.get("duration_minutes", float('inf'))
            vram = result_720p.get("metrics", {}).get("gpu_memory_peak_gb", 0)
            validations.append(self.validate_720p_target(duration, vram))
        
        # Validate 1080p if available
        if "1080p" in test_results:
            result_1080p = test_results["1080p"]
            duration = result_1080p.get("duration_minutes", float('inf'))
            vram = result_1080p.get("metrics", {}).get("gpu_memory_peak_gb", 0)
            validations.append(self.validate_1080p_target(duration, vram))
        
        return validations


class FrameworkOverheadMonitor:
    """
    Monitors framework overhead with 2% CPU and 100MB RAM limits
    Uses separate monitoring process to avoid skewing benchmarks
    """
    
    def __init__(self, config: Optional[LocalTestConfiguration] = None):
        self.config = config or LocalTestConfiguration()
        self.monitoring_process = None
        self.overhead_data = []
        self.cleanup_interval = 300  # 300 seconds = 5 minutes
        self.last_cleanup = time.time()
        
    def start_overhead_monitoring(self) -> bool:
        """
        Start separate monitoring process for framework overhead
        """
        try:
            # Create monitoring function that runs in separate process
            manager = multiprocessing.Manager()
            self.overhead_data = manager.list()
            
            self.monitoring_process = multiprocessing.Process(
                target=self._monitor_framework_overhead,
                args=(self.overhead_data,),
                daemon=True
            )
            self.monitoring_process.start()
            return True
            
        except Exception as e:
            print(f"Failed to start overhead monitoring: {e}")
            return False
    
    def stop_overhead_monitoring(self) -> Dict[str, Any]:
        """
        Stop monitoring process and return overhead statistics
        """
        if self.monitoring_process and self.monitoring_process.is_alive():
            self.monitoring_process.terminate()
            self.monitoring_process.join(timeout=5)
        
        # Convert shared list to regular list for analysis
        overhead_metrics = list(self.overhead_data) if self.overhead_data else []
        
        return self._analyze_overhead_data(overhead_metrics)
    
    def _monitor_framework_overhead(self, shared_data: list):
        """
        Monitor framework overhead in separate process
        """
        current_pid = os.getpid()
        parent_pid = os.getppid()
        
        while True:
            try:
                timestamp = time.time()
                
                # Monitor current process (framework)
                try:
                    current_process = psutil.Process(current_pid)
                    framework_cpu = current_process.cpu_percent(interval=0.1)
                    framework_memory = current_process.memory_info().rss / (1024**2)  # MB
                except psutil.NoSuchProcess:
                    break
                
                # Monitor parent process if different
                parent_cpu = 0
                parent_memory = 0
                if parent_pid != current_pid:
                    try:
                        parent_process = psutil.Process(parent_pid)
                        parent_cpu = parent_process.cpu_percent(interval=0.1)
                        parent_memory = parent_process.memory_info().rss / (1024**2)  # MB
                    except psutil.NoSuchProcess:
                        pass
                
                # Total framework overhead
                total_cpu = framework_cpu + parent_cpu
                total_memory = framework_memory + parent_memory
                
                overhead_sample = {
                    "timestamp": timestamp,
                    "framework_cpu_percent": framework_cpu,
                    "framework_memory_mb": framework_memory,
                    "parent_cpu_percent": parent_cpu,
                    "parent_memory_mb": parent_memory,
                    "total_cpu_percent": total_cpu,
                    "total_memory_mb": total_memory
                }
                
                shared_data.append(overhead_sample)
                
                # Periodic cleanup to prevent memory buildup
                if timestamp - self.last_cleanup > self.cleanup_interval:
                    self._cleanup_monitoring_data(shared_data)
                    self.last_cleanup = timestamp
                
                time.sleep(1.0)  # Sample every second
                
            except Exception as e:
                # Continue monitoring even if individual samples fail
                time.sleep(1.0)
    
    def _cleanup_monitoring_data(self, shared_data: list):
        """
        Periodic cleanup to prevent memory buildup during long sessions
        """
        try:
            # Keep only last 1000 samples (about 16 minutes at 1 sample/second)
            if len(shared_data) > 1000:
                # Remove older samples, keep recent ones
                recent_samples = shared_data[-1000:]
                shared_data.clear()
                shared_data.extend(recent_samples)
                
            # Also clear GPU cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
                
            # Clean temporary files in outputs directory
            outputs_dir = Path("outputs")
            if outputs_dir.exists():
                current_time = time.time()
                for file_path in outputs_dir.glob("*"):
                    try:
                        # Remove files older than 1 hour
                        if current_time - file_path.stat().st_mtime > 3600:
                            if file_path.is_file():
                                file_path.unlink()
                    except Exception:
                        pass  # Continue cleanup even if individual files fail
                        
        except Exception as e:
            # Don't let cleanup errors stop monitoring
            pass
    
    def _analyze_overhead_data(self, overhead_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze overhead data and validate against limits
        """
        if not overhead_metrics:
            return {
                "samples": 0,
                "cpu_overhead_valid": False,
                "memory_overhead_valid": False,
                "error": "No overhead data collected"
            }
        
        # Extract CPU and memory values
        cpu_values = [sample["total_cpu_percent"] for sample in overhead_metrics]
        memory_values = [sample["total_memory_mb"] for sample in overhead_metrics]
        
        # Calculate statistics
        cpu_avg = sum(cpu_values) / len(cpu_values)
        cpu_peak = max(cpu_values)
        memory_avg = sum(memory_values) / len(memory_values)
        memory_peak = max(memory_values)
        
        # Check against limits (2% CPU, 100MB RAM)
        cpu_limit = 2.0  # 2% CPU
        memory_limit = 100.0  # 100MB RAM
        
        cpu_overhead_valid = cpu_avg <= cpu_limit and cpu_peak <= cpu_limit * 2  # Allow 2x peak
        memory_overhead_valid = memory_avg <= memory_limit and memory_peak <= memory_limit * 2
        
        return {
            "samples": len(overhead_metrics),
            "duration_minutes": (overhead_metrics[-1]["timestamp"] - overhead_metrics[0]["timestamp"]) / 60,
            "cpu_avg_percent": cpu_avg,
            "cpu_peak_percent": cpu_peak,
            "cpu_limit_percent": cpu_limit,
            "cpu_overhead_valid": cpu_overhead_valid,
            "memory_avg_mb": memory_avg,
            "memory_peak_mb": memory_peak,
            "memory_limit_mb": memory_limit,
            "memory_overhead_valid": memory_overhead_valid,
            "overall_overhead_valid": cpu_overhead_valid and memory_overhead_valid,
            "raw_samples": overhead_metrics[-100:] if len(overhead_metrics) > 100 else overhead_metrics  # Last 100 samples
        }
    
    def validate_framework_overhead(self, overhead_analysis: Dict[str, Any]) -> ValidationResult:
        """
        Validate framework overhead against performance limits
        """
        if overhead_analysis.get("samples", 0) == 0:
            return ValidationResult(
                component="framework_overhead",
                status=ValidationStatus.FAILED,
                message="No overhead monitoring data available",
                details=overhead_analysis,
                remediation_steps=[
                    "Ensure monitoring process starts correctly",
                    "Check for process permission issues",
                    "Verify psutil library is installed"
                ]
            )
        
        cpu_valid = overhead_analysis.get("cpu_overhead_valid", False)
        memory_valid = overhead_analysis.get("memory_overhead_valid", False)
        
        issues = []
        if not cpu_valid:
            cpu_avg = overhead_analysis.get("cpu_avg_percent", 0)
            cpu_limit = overhead_analysis.get("cpu_limit_percent", 2)
            issues.append(f"CPU overhead {cpu_avg:.1f}% exceeds {cpu_limit}% limit")
        
        if not memory_valid:
            memory_avg = overhead_analysis.get("memory_avg_mb", 0)
            memory_limit = overhead_analysis.get("memory_limit_mb", 100)
            issues.append(f"Memory overhead {memory_avg:.1f}MB exceeds {memory_limit}MB limit")
        
        if issues:
            return ValidationResult(
                component="framework_overhead",
                status=ValidationStatus.FAILED,
                message=f"Framework overhead limits exceeded: {'; '.join(issues)}",
                details=overhead_analysis,
                remediation_steps=[
                    "Optimize framework code for lower CPU usage",
                    "Reduce memory allocations and improve garbage collection",
                    "Profile framework code to identify bottlenecks",
                    "Consider using more efficient monitoring intervals",
                    "Implement better resource cleanup procedures"
                ]
            )
        
        return ValidationResult(
            component="framework_overhead",
            status=ValidationStatus.PASSED,
            message=f"Framework overhead within limits: CPU {overhead_analysis.get('cpu_avg_percent', 0):.1f}% <= 2%, Memory {overhead_analysis.get('memory_avg_mb', 0):.1f}MB <= 100MB",
            details=overhead_analysis
        )


class OptimizationRecommendationSystem:
    """
    Generates optimization recommendations for attention slicing, VAE tiling, and other optimizations
    """
    
    def __init__(self, config: Optional[LocalTestConfiguration] = None):
        self.config = config or LocalTestConfiguration()
    
    def analyze_performance_issues(self, test_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Analyze performance test results and identify optimization opportunities
        """
        recommendations = {
            "vram_optimizations": [],
            "speed_optimizations": [],
            "stability_optimizations": [],
            "configuration_changes": []
        }
        
        # Analyze VRAM usage issues
        for resolution, result in test_results.get("tests", {}).items():
            if resolution in ["720p", "1080p"]:
                metrics = result.get("metrics", {})
                vram_peak = metrics.get("gpu_memory_peak_gb", 0)
                duration = result.get("duration_minutes", 0)
                
                # VRAM optimization recommendations
                if vram_peak > self.config.performance_targets.max_vram_usage_gb * 0.8:  # 80% of limit
                    recommendations["vram_optimizations"].extend([
                        "Enable attention slicing to reduce VRAM usage",
                        "Enable VAE tiling for memory-efficient VAE operations",
                        "Use mixed precision (fp16) to halve memory requirements",
                        "Consider model quantization (int8) for further reduction"
                    ])
                
                # Speed optimization recommendations
                target_time = (self.config.performance_targets.target_720p_time_minutes 
                             if resolution == "720p" 
                             else self.config.performance_targets.target_1080p_time_minutes)
                
                if duration > target_time * 0.9:  # 90% of target time
                    recommendations["speed_optimizations"].extend([
                        f"Optimize {resolution} generation pipeline",
                        "Enable CUDA optimizations and kernel fusion",
                        "Use optimized attention mechanisms (xformers, flash-attention)",
                        "Consider using faster schedulers (DPM++, Euler)",
                        "Reduce inference steps if quality allows"
                    ])
        
        # Analyze framework overhead
        overhead_data = test_results.get("overhead_analysis", {})
        if not overhead_data.get("overall_overhead_valid", True):
            recommendations["stability_optimizations"].extend([
                "Optimize framework monitoring overhead",
                "Implement more efficient resource tracking",
                "Reduce monitoring frequency during intensive operations",
                "Improve garbage collection and memory cleanup"
            ])
        
        # Configuration recommendations
        recommendations["configuration_changes"].extend([
            "Set enable_attention_slicing=true in config.json",
            "Set enable_vae_tiling=true for memory efficiency",
            "Configure use_fp16=true for mixed precision",
            "Set torch_compile=true for PyTorch 2.0+ optimization",
            "Adjust batch_size based on available VRAM"
        ])
        
        # Remove duplicates
        for category in recommendations:
            recommendations[category] = list(set(recommendations[category]))
        
        return recommendations
    
    def generate_optimization_config(self, recommendations: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Generate optimized configuration based on recommendations
        """
        optimized_config = {
            "optimization": {
                "enable_attention_slicing": True,
                "enable_vae_tiling": True,
                "use_fp16": True,
                "torch_compile": True,
                "enable_xformers": True,
                "memory_efficient_attention": True
            },
            "performance": {
                "batch_size": 1,  # Conservative for memory
                "num_inference_steps": 20,  # Balanced quality/speed
                "guidance_scale": 7.5,
                "scheduler": "DPMSolverMultistepScheduler"  # Fast scheduler
            },
            "system": {
                "gpu_memory_fraction": 0.9,  # Reserve some VRAM
                "allow_tf32": True,  # Faster on Ampere GPUs
                "cudnn_benchmark": True,
                "empty_cache_frequency": 5  # Clear cache every 5 operations
            }
        }
        
        # Adjust based on specific recommendations
        vram_recs = recommendations.get("vram_optimizations", [])
        if any("quantization" in rec.lower() for rec in vram_recs):
            optimized_config["optimization"]["use_int8"] = True
        
        speed_recs = recommendations.get("speed_optimizations", [])
        if any("reduce inference steps" in rec.lower() for rec in speed_recs):
            optimized_config["performance"]["num_inference_steps"] = 15
        
        return optimized_config
    
    def create_detailed_performance_report(self, test_results: Dict[str, Any], 
                                         recommendations: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Create detailed performance report in JSON format
        """
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "framework_version": "1.0.0",
                "test_session_id": test_results.get("test_session_id", "unknown")
            },
            "executive_summary": self._generate_executive_summary(test_results),
            "performance_results": self._format_performance_results(test_results),
            "optimization_analysis": {
                "recommendations": recommendations,
                "priority_actions": self._prioritize_recommendations(recommendations),
                "estimated_improvements": self._estimate_improvements(test_results, recommendations)
            },
            "technical_details": {
                "system_info": self._get_system_info(),
                "test_configuration": self.config.to_dict(),
                "raw_metrics": test_results.get("tests", {})
            },
            "next_steps": self._generate_next_steps(recommendations)
        }
        
        return report
    
    def _generate_executive_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of test results"""
        overall_status = test_results.get("overall_status", "unknown")
        total_duration = test_results.get("total_duration_minutes", 0)
        
        tests = test_results.get("tests", {})
        passed_tests = sum(1 for test in tests.values() if test.get("success", False))
        total_tests = len([t for t in tests.keys() if t != "profiler"])
        
        return {
            "overall_status": overall_status,
            "test_success_rate": f"{passed_tests}/{total_tests}",
            "total_test_duration_minutes": total_duration,
            "key_findings": self._extract_key_findings(test_results),
            "critical_issues": self._identify_critical_issues(test_results)
        }
    
    def _format_performance_results(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Format performance results for report"""
        formatted_results = {}
        
        tests = test_results.get("tests", {})
        validations = test_results.get("validations", {})
        
        for resolution in ["720p", "1080p"]:
            if resolution in tests:
                test_data = tests[resolution]
                validation_data = validations.get(resolution, {})
                
                formatted_results[resolution] = {
                    "duration_minutes": test_data.get("duration_minutes", 0),
                    "success": test_data.get("success", False),
                    "target_met": validation_data.get("status") == "passed",
                    "metrics": test_data.get("metrics", {}),
                    "validation_message": validation_data.get("message", "")
                }
        
        return formatted_results
    
    def _prioritize_recommendations(self, recommendations: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Prioritize recommendations by impact and ease of implementation"""
        priority_actions = []
        
        # High priority: VRAM optimizations (high impact, easy to implement)
        for rec in recommendations.get("vram_optimizations", []):
            if "attention slicing" in rec.lower():
                priority_actions.append({
                    "action": rec,
                    "priority": "high",
                    "impact": "high",
                    "difficulty": "easy",
                    "category": "vram"
                })
        
        # Medium priority: Speed optimizations
        for rec in recommendations.get("speed_optimizations", []):
            priority_actions.append({
                "action": rec,
                "priority": "medium",
                "impact": "medium",
                "difficulty": "medium",
                "category": "speed"
            })
        
        # Low priority: Configuration changes
        for rec in recommendations.get("configuration_changes", []):
            priority_actions.append({
                "action": rec,
                "priority": "low",
                "impact": "low",
                "difficulty": "easy",
                "category": "config"
            })
        
        return priority_actions
    
    def _estimate_improvements(self, test_results: Dict[str, Any], 
                             recommendations: Dict[str, List[str]]) -> Dict[str, Any]:
        """Estimate potential improvements from recommendations"""
        improvements = {
            "vram_reduction_percent": 0,
            "speed_improvement_percent": 0,
            "stability_improvement": "unknown"
        }
        
        # Estimate VRAM reduction
        vram_recs = recommendations.get("vram_optimizations", [])
        if any("attention slicing" in rec.lower() for rec in vram_recs):
            improvements["vram_reduction_percent"] += 30
        if any("vae tiling" in rec.lower() for rec in vram_recs):
            improvements["vram_reduction_percent"] += 20
        if any("fp16" in rec.lower() for rec in vram_recs):
            improvements["vram_reduction_percent"] += 40
        
        # Estimate speed improvement
        speed_recs = recommendations.get("speed_optimizations", [])
        if any("xformers" in rec.lower() for rec in speed_recs):
            improvements["speed_improvement_percent"] += 15
        if any("scheduler" in rec.lower() for rec in speed_recs):
            improvements["speed_improvement_percent"] += 10
        
        # Cap estimates at reasonable maximums
        improvements["vram_reduction_percent"] = min(improvements["vram_reduction_percent"], 80)
        improvements["speed_improvement_percent"] = min(improvements["speed_improvement_percent"], 50)
        
        return improvements
    
    def _extract_key_findings(self, test_results: Dict[str, Any]) -> List[str]:
        """Extract key findings from test results"""
        findings = []
        
        tests = test_results.get("tests", {})
        for resolution, result in tests.items():
            if resolution in ["720p", "1080p"]:
                duration = result.get("duration_minutes", 0)
                success = result.get("success", False)
                
                if success:
                    findings.append(f"{resolution} generation completed in {duration:.1f} minutes")
                else:
                    findings.append(f"{resolution} generation failed")
        
        return findings
    
    def _identify_critical_issues(self, test_results: Dict[str, Any]) -> List[str]:
        """Identify critical issues that need immediate attention"""
        issues = []
        
        validations = test_results.get("validations", {})
        for resolution, validation in validations.items():
            if validation.get("status") == "failed":
                issues.append(f"{resolution} performance targets not met")
        
        return issues
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for the report"""
        system_info = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3)
        }
        
        # Add GPU info if available
        try:
            import torch
            if torch.cuda.is_available():
                system_info["gpu_name"] = torch.cuda.get_device_name(0)
                system_info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            pass
        
        return system_info
    
    def _generate_next_steps(self, recommendations: Dict[str, List[str]]) -> List[str]:
        """Generate actionable next steps"""
        next_steps = [
            "1. Implement high-priority VRAM optimizations first",
            "2. Update configuration with recommended settings",
            "3. Re-run performance tests to validate improvements",
            "4. Monitor system stability with new optimizations",
            "5. Fine-tune settings based on specific use cases"
        ]
        
        return next_steps
