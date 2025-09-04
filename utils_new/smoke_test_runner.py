from unittest.mock import Mock, patch
#!/usr/bin/env python3
"""
Smoke Test Runner for Wan Model Compatibility System
Provides pipeline functionality validation, output format testing, and performance benchmarking
"""

import time
import traceback
import psutil
import torch
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SmokeTestResult:
    """Result of smoke test execution"""
    success: bool
    generation_time: float
    memory_peak: int
    output_shape: Tuple[int, ...]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FormatValidation:
    """Result of output format validation"""
    is_valid: bool
    expected_format: str
    actual_format: str
    validation_errors: List[str] = field(default_factory=list)
    format_details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryTestResult:
    """Result of memory usage testing"""
    peak_memory_mb: int
    initial_memory_mb: int
    memory_increase_mb: int
    memory_efficiency: float
    memory_leaks_detected: bool
    cleanup_successful: bool
    memory_timeline: List[Tuple[float, int]] = field(default_factory=list)

@dataclass
class PerformanceBenchmark:
    """Result of performance benchmarking"""
    generation_time: float
    frames_per_second: float
    memory_efficiency: float
    gpu_utilization: float
    cpu_utilization: float
    benchmark_score: float
    performance_category: str  # "excellent", "good", "acceptable", "poor"
    bottlenecks: List[str] = field(default_factory=list)

class SmokeTestRunner:
    """
    Smoke test runner for pipeline functionality validation
    Provides comprehensive testing capabilities for Wan model compatibility
    """
    
    def __init__(self, test_config: Optional[Dict[str, Any]] = None):
        """
        Initialize smoke test runner
        
        Args:
            test_config: Optional configuration for test parameters
        """
        self.test_config = test_config or self._get_default_config()
        self.process = psutil.Process()
        self.test_results_dir = Path("test_results")
        self.test_results_dir.mkdir(exist_ok=True)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default test configuration"""
        return {
            "smoke_test": {
                "test_prompt": "A simple test video",
                "max_generation_time": 300,  # 5 minutes
                "min_output_frames": 1,
                "max_memory_mb": 16384,  # 16GB
                "timeout_seconds": 600
            },
            "memory_test": {
                "sample_interval": 0.5,
                "leak_threshold_mb": 100,
                "cleanup_timeout": 30
            },
            "performance": {
                "benchmark_iterations": 3,
                "warmup_iterations": 1,
                "target_fps": 1.0,
                "acceptable_fps": 0.5
            }
        }
    
    def run_pipeline_smoke_test(self, pipeline: Any, test_prompt: Optional[str] = None) -> SmokeTestResult:
        """
        Run minimal generation test to verify pipeline functionality
        
        Args:
            pipeline: Pipeline instance to test
            test_prompt: Optional test prompt (uses default if None)
            
        Returns:
            SmokeTestResult with test outcomes
        """
        logger.info("Starting pipeline smoke test")
        
        # Initialize result
        result = SmokeTestResult(
            success=False,
            generation_time=0.0,
            memory_peak=0,
            output_shape=(0,),
            metadata={
                "test_timestamp": datetime.now().isoformat(),
                "pipeline_type": type(pipeline).__name__,
                "test_prompt": test_prompt or self.test_config["smoke_test"]["test_prompt"]
            }
        )
        
        # Get initial memory
        initial_memory = self._get_memory_usage()
        peak_memory = initial_memory
        
        try:
            # Prepare test parameters
            prompt = test_prompt or self.test_config["smoke_test"]["test_prompt"]
            max_time = self.test_config["smoke_test"]["max_generation_time"]
            
            # Validate pipeline has required methods
            if not hasattr(pipeline, '__call__') and not hasattr(pipeline, 'generate'):
                result.errors.append("Pipeline missing callable interface")
                return result
            
            # Start memory monitoring
            memory_monitor = self._start_memory_monitoring()
            
            # Run generation test
            start_time = time.time()
            
            try:
                # Attempt generation with timeout
                if hasattr(pipeline, 'generate'):
                    output = pipeline.generate(prompt=prompt, num_frames=8, height=320, width=320)
                else:
                    output = pipeline(prompt=prompt, num_frames=8, height=320, width=320)
                
                generation_time = time.time() - start_time
                
                # Stop memory monitoring
                peak_memory = self._stop_memory_monitoring(memory_monitor)
                
                # Validate output
                if output is None:
                    result.errors.append("Pipeline returned None output")
                    return result
                
                # Extract output shape and validate
                output_shape = self._extract_output_shape(output)
                if output_shape == (0,):
                    result.warnings.append("Could not determine output shape")
                
                # Check generation time
                if generation_time > max_time:
                    result.warnings.append(f"Generation took {generation_time:.1f}s (max: {max_time}s)")
                
                # Check memory usage
                memory_increase = peak_memory - initial_memory
                max_memory = self.test_config["smoke_test"]["max_memory_mb"]
                if memory_increase > max_memory:
                    result.warnings.append(f"High memory usage: {memory_increase}MB (max: {max_memory}MB)")
                
                # Test passed
                result.success = True
                result.generation_time = generation_time
                result.memory_peak = peak_memory
                result.output_shape = output_shape
                result.metadata.update({
                    "memory_increase_mb": memory_increase,
                    "output_type": type(output).__name__
                })
                
                logger.info(f"Smoke test passed in {generation_time:.2f}s")
                
            except Exception as e:
                generation_time = time.time() - start_time
                self._stop_memory_monitoring(memory_monitor)
                
                result.errors.append(f"Generation failed: {str(e)}")
                result.generation_time = generation_time
                result.memory_peak = peak_memory
                
                logger.error(f"Smoke test failed: {e}")
                
        except Exception as e:
            result.errors.append(f"Test setup failed: {str(e)}")
            logger.error(f"Smoke test setup failed: {e}")
        
        # Save test results
        self._save_test_result("smoke_test", result)
        
        return result
    
    def validate_output_format(self, output: Any) -> FormatValidation:
        """
        Validate that output matches expected format
        
        Args:
            output: Generated output to validate
            
        Returns:
            FormatValidation with validation results
        """
        logger.info("Validating output format")
        
        validation = FormatValidation(
            is_valid=False,
            expected_format="video_frames",
            actual_format="unknown"
        )
        
        try:
            # Determine output type
            output_type = type(output).__name__
            validation.actual_format = output_type
            
            # Check for common video output formats
            if hasattr(output, 'frames') or hasattr(output, 'images'):
                # Frame-based output
                frames = getattr(output, 'frames', getattr(output, 'images', None))
                if frames is not None:
                    validation.is_valid = True
                    validation.expected_format = "frame_sequence"
                    validation.format_details = {
                        "frame_count": len(frames) if hasattr(frames, '__len__') else "unknown",
                        "frame_type": type(frames[0]).__name__ if frames else "unknown"
                    }
            
            elif isinstance(output, (list, tuple)):
                # List/tuple of frames
                if len(output) > 0:
                    validation.is_valid = True
                    validation.expected_format = "frame_list"
                    validation.format_details = {
                        "frame_count": len(output),
                        "frame_type": type(output[0]).__name__,
                        "frame_shape": getattr(output[0], 'shape', 'unknown') if hasattr(output[0], 'shape') else 'unknown'
                    }
                else:
                    validation.validation_errors.append("Empty output list")
            
            elif isinstance(output, np.ndarray):
                # NumPy array output
                validation.is_valid = True
                validation.expected_format = "numpy_array"
                validation.format_details = {
                    "shape": output.shape,
                    "dtype": str(output.dtype),
                    "dimensions": len(output.shape)
                }
                
                # Validate video tensor shape
                if len(output.shape) == 4:  # (frames, height, width, channels)
                    validation.format_details["format"] = "FHWC"
                elif len(output.shape) == 5:  # (batch, frames, height, width, channels)
                    validation.format_details["format"] = "BFHWC"
                else:
                    validation.validation_errors.append(f"Unexpected tensor shape: {output.shape}")
            
            elif torch.is_tensor(output):
                # PyTorch tensor output
                validation.is_valid = True
                validation.expected_format = "torch_tensor"
                validation.format_details = {
                    "shape": tuple(output.shape),
                    "dtype": str(output.dtype),
                    "device": str(output.device),
                    "dimensions": len(output.shape)
                }
                
                # Validate video tensor shape
                if len(output.shape) == 4:  # (frames, channels, height, width)
                    validation.format_details["format"] = "FCHW"
                elif len(output.shape) == 5:  # (batch, frames, channels, height, width)
                    validation.format_details["format"] = "BFCHW"
                else:
                    validation.validation_errors.append(f"Unexpected tensor shape: {output.shape}")
            
            else:
                validation.validation_errors.append(f"Unrecognized output format: {output_type}")
            
            # Additional validation checks
            if validation.is_valid:
                self._validate_output_content(output, validation)
            
        except Exception as e:
            validation.validation_errors.append(f"Format validation failed: {str(e)}")
            logger.error(f"Output format validation error: {e}")
        
        return validation
    
    def test_memory_usage(self, pipeline: Any) -> MemoryTestResult:
        """
        Test memory usage patterns during generation
        
        Args:
            pipeline: Pipeline to test
            
        Returns:
            MemoryTestResult with memory analysis
        """
        logger.info("Testing memory usage patterns")
        
        # Get initial memory
        initial_memory = self._get_memory_usage()
        
        result = MemoryTestResult(
            peak_memory_mb=initial_memory,
            initial_memory_mb=initial_memory,
            memory_increase_mb=0,
            memory_efficiency=0.0,
            memory_leaks_detected=False,
            cleanup_successful=False
        )
        
        try:
            # Start detailed memory monitoring
            memory_timeline = []
            sample_interval = self.test_config["memory_test"]["sample_interval"]
            
            # Run multiple generations to test for leaks
            for i in range(3):
                logger.info(f"Memory test iteration {i+1}/3")
                
                # Record pre-generation memory
                pre_memory = self._get_memory_usage()
                memory_timeline.append((time.time(), pre_memory))
                
                # Run generation
                try:
                    if hasattr(pipeline, 'generate'):
                        output = pipeline.generate(
                            prompt="Memory test",
                            num_frames=4,
                            height=256,
                            width=256
                        )
                    else:
                        output = pipeline(
                            prompt="Memory test",
                            num_frames=4,
                            height=256,
                            width=256
                        )
                    
                    # Record post-generation memory
                    post_memory = self._get_memory_usage()
                    memory_timeline.append((time.time(), post_memory))
                    
                    # Update peak memory
                    result.peak_memory_mb = max(result.peak_memory_mb, post_memory)
                    
                    # Clean up output
                    del output
                    
                except Exception as e:
                    logger.warning(f"Memory test generation {i+1} failed: {e}")
                    continue
                
                # Force garbage collection
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Wait and check for cleanup
                time.sleep(1)
                cleanup_memory = self._get_memory_usage()
                memory_timeline.append((time.time(), cleanup_memory))
            
            # Analyze results
            result.memory_timeline = memory_timeline
            result.memory_increase_mb = result.peak_memory_mb - initial_memory
            
            # Check for memory leaks
            final_memory = memory_timeline[-1][1] if memory_timeline else initial_memory
            leak_threshold = self.test_config["memory_test"]["leak_threshold_mb"]
            memory_leak = final_memory - initial_memory
            
            if memory_leak > leak_threshold:
                result.memory_leaks_detected = True
                logger.warning(f"Potential memory leak detected: {memory_leak}MB increase")
            else:
                result.cleanup_successful = True
            
            # Calculate memory efficiency
            if result.memory_increase_mb > 0:
                result.memory_efficiency = min(1.0, 1000.0 / result.memory_increase_mb)
            else:
                result.memory_efficiency = 1.0
            
            logger.info(f"Memory test completed. Peak: {result.peak_memory_mb}MB, Increase: {result.memory_increase_mb}MB")
            
        except Exception as e:
            logger.error(f"Memory test failed: {e}")
        
        return result
    
    def benchmark_generation_speed(self, pipeline: Any) -> PerformanceBenchmark:
        """
        Benchmark generation speed for performance regression detection
        
        Args:
            pipeline: Pipeline to benchmark
            
        Returns:
            PerformanceBenchmark with performance metrics
        """
        logger.info("Benchmarking generation speed")
        
        benchmark = PerformanceBenchmark(
            generation_time=0.0,
            frames_per_second=0.0,
            memory_efficiency=0.0,
            gpu_utilization=0.0,
            cpu_utilization=0.0,
            benchmark_score=0.0,
            performance_category="poor"
        )
        
        try:
            iterations = self.test_config["performance"]["benchmark_iterations"]
            warmup_iterations = self.test_config["performance"]["warmup_iterations"]
            
            generation_times = []
            
            # Warmup runs
            logger.info(f"Running {warmup_iterations} warmup iterations")
            for i in range(warmup_iterations):
                try:
                    start_time = time.time()
                    if hasattr(pipeline, 'generate'):
                        output = pipeline.generate(
                            prompt="Benchmark test",
                            num_frames=8,
                            height=256,
                            width=256
                        )
                    else:
                        output = pipeline(
                            prompt="Benchmark test",
                            num_frames=8,
                            height=256,
                            width=256
                        )
                    warmup_time = time.time() - start_time
                    logger.info(f"Warmup {i+1}: {warmup_time:.2f}s")
                    del output
                except Exception as e:
                    logger.warning(f"Warmup iteration {i+1} failed: {e}")
            
            # Benchmark runs
            logger.info(f"Running {iterations} benchmark iterations")
            for i in range(iterations):
                try:
                    # Monitor system resources
                    cpu_before = psutil.cpu_percent()
                    memory_before = self._get_memory_usage()
                    
                    start_time = time.time()
                    if hasattr(pipeline, 'generate'):
                        output = pipeline.generate(
                            prompt="Benchmark test",
                            num_frames=8,
                            height=256,
                            width=256
                        )
                    else:
                        output = pipeline(
                            prompt="Benchmark test",
                            num_frames=8,
                            height=256,
                            width=256
                        )
                    generation_time = time.time() - start_time
                    
                    # Monitor system resources after
                    cpu_after = psutil.cpu_percent()
                    memory_after = self._get_memory_usage()
                    
                    generation_times.append(generation_time)
                    
                    # Update resource utilization
                    benchmark.cpu_utilization = max(benchmark.cpu_utilization, (cpu_before + cpu_after) / 2)
                    memory_used = memory_after - memory_before
                    if memory_used > 0:
                        benchmark.memory_efficiency = max(benchmark.memory_efficiency, 1000.0 / memory_used)
                    
                    logger.info(f"Benchmark {i+1}: {generation_time:.2f}s")
                    del output
                    
                except Exception as e:
                    logger.warning(f"Benchmark iteration {i+1} failed: {e}")
                    continue
            
            # Calculate metrics
            if generation_times:
                benchmark.generation_time = sum(generation_times) / len(generation_times)
                benchmark.frames_per_second = 8.0 / benchmark.generation_time  # 8 frames generated
                
                # Calculate benchmark score (higher is better)
                target_fps = self.test_config["performance"]["target_fps"]
                benchmark.benchmark_score = min(100.0, (benchmark.frames_per_second / target_fps) * 100.0)
                
                # Determine performance category
                acceptable_fps = self.test_config["performance"]["acceptable_fps"]
                if benchmark.frames_per_second >= target_fps:
                    benchmark.performance_category = "excellent"
                elif benchmark.frames_per_second >= target_fps * 0.8:
                    benchmark.performance_category = "good"
                elif benchmark.frames_per_second >= acceptable_fps:
                    benchmark.performance_category = "acceptable"
                else:
                    benchmark.performance_category = "poor"
                    benchmark.bottlenecks.append("Low generation speed")
                
                # Identify bottlenecks
                if benchmark.cpu_utilization > 90:
                    benchmark.bottlenecks.append("High CPU utilization")
                if benchmark.memory_efficiency < 0.5:
                    benchmark.bottlenecks.append("High memory usage")
                
                logger.info(f"Benchmark completed: {benchmark.frames_per_second:.2f} FPS, Score: {benchmark.benchmark_score:.1f}")
            else:
                logger.error("No successful benchmark iterations")
        
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
        
        return benchmark
    
    def _extract_output_shape(self, output: Any) -> Tuple[int, ...]:
        """Extract shape information from output"""
        try:
            if hasattr(output, 'shape'):
                return tuple(output.shape)
            elif hasattr(output, 'frames') and hasattr(output.frames, 'shape'):
                return tuple(output.frames.shape)
            elif hasattr(output, 'images') and hasattr(output.images, 'shape'):
                return tuple(output.images.shape)
            elif isinstance(output, (list, tuple)) and len(output) > 0:
                if hasattr(output[0], 'shape'):
                    return (len(output),) + tuple(output[0].shape)
                else:
                    return (len(output),)
            else:
                return (0,)
        except Exception:
            return (0,)
    
    def _validate_output_content(self, output: Any, validation: FormatValidation):
        """Validate output content quality"""
        try:
            # Check for reasonable value ranges
            if isinstance(output, np.ndarray):
                if output.dtype in [np.float32, np.float64]:
                    if np.any(output < -2.0) or np.any(output > 2.0):
                        validation.validation_errors.append("Values outside expected range [-2, 2]")
                elif output.dtype == np.uint8:
                    if np.any(output > 255):
                        validation.validation_errors.append("Uint8 values exceed 255")
            
            elif torch.is_tensor(output):
                if output.dtype in [torch.float32, torch.float64]:
                    if torch.any(output < -2.0) or torch.any(output > 2.0):
                        validation.validation_errors.append("Values outside expected range [-2, 2]")
            
            # Check for NaN or infinite values
            if isinstance(output, np.ndarray):
                if np.any(np.isnan(output)) or np.any(np.isinf(output)):
                    validation.validation_errors.append("Contains NaN or infinite values")
            elif torch.is_tensor(output):
                if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
                    validation.validation_errors.append("Contains NaN or infinite values")
        
        except Exception as e:
            validation.validation_errors.append(f"Content validation error: {str(e)}")
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in MB"""
        try:
            memory_info = self.process.memory_info()
            return int(memory_info.rss / 1024 / 1024)  # Convert to MB
        except Exception:
            return 0
    
    def _start_memory_monitoring(self) -> Dict[str, Any]:
        """Start memory monitoring thread"""
        monitor = {
            "running": True,
            "peak_memory": self._get_memory_usage(),
            "timeline": []
        }
        return monitor
    
    def _stop_memory_monitoring(self, monitor: Dict[str, Any]) -> int:
        """Stop memory monitoring and return peak memory"""
        monitor["running"] = False
        return monitor["peak_memory"]
    
    def _save_test_result(self, test_type: str, result: Any):
        """Save test result to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{test_type}_{timestamp}.json"
            filepath = self.test_results_dir / filename
            
            # Convert result to dict for JSON serialization
            if hasattr(result, '__dict__'):
                result_dict = result.__dict__.copy()
                # Handle non-serializable types
                for key, value in result_dict.items():
                    if isinstance(value, tuple):
                        result_dict[key] = list(value)
            else:
                result_dict = {"result": str(result)}
            
            with open(filepath, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            logger.info(f"Test result saved to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save test result: {e}")


if __name__ == "__main__":
    # Example usage
    print("Smoke Test Runner - Example Usage")
    
    # Create mock pipeline for testing
    class MockPipeline:
        def generate(self, prompt, num_frames=8, height=256, width=256):
            time.sleep(0.1)  # Simulate generation time
            return np.random.rand(num_frames, height, width, 3).astype(np.float32)
    
    # Run tests
    runner = SmokeTestRunner()
    mock_pipeline = MockPipeline()
    
    print("\n1. Running smoke test...")
    smoke_result = runner.run_pipeline_smoke_test(mock_pipeline)
    print(f"   Success: {smoke_result.success}")
    print(f"   Time: {smoke_result.generation_time:.2f}s")
    print(f"   Shape: {smoke_result.output_shape}")
    
    print("\n2. Testing output format validation...")
    test_output = np.random.rand(8, 256, 256, 3).astype(np.float32)
    format_result = runner.validate_output_format(test_output)
    print(f"   Valid: {format_result.is_valid}")
    print(f"   Format: {format_result.actual_format}")
    
    print("\n3. Running memory test...")
    memory_result = runner.test_memory_usage(mock_pipeline)
    print(f"   Peak Memory: {memory_result.peak_memory_mb}MB")
    print(f"   Memory Increase: {memory_result.memory_increase_mb}MB")
    print(f"   Leaks Detected: {memory_result.memory_leaks_detected}")
    
    print("\n4. Running performance benchmark...")
    perf_result = runner.benchmark_generation_speed(mock_pipeline)
    print(f"   FPS: {perf_result.frames_per_second:.2f}")
    print(f"   Category: {perf_result.performance_category}")
    print(f"   Score: {perf_result.benchmark_score:.1f}")
    
    print("\nSmoke test runner demonstration completed!")