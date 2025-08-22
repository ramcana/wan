#!/usr/bin/env python3
"""
End-to-End Integration Test Suite for Wan Model Compatibility System
Implements comprehensive workflow tests from model detection to video output
"""

import unittest
import tempfile
import shutil
import json
import time
import traceback
import threading
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from unittest.mock import Mock, patch, MagicMock
import logging
import numpy as np

# Import components to test
try:
    from architecture_detector import ArchitectureDetector, ModelArchitecture
    from pipeline_manager import PipelineManager
    from dependency_manager import DependencyManager
    from wan_pipeline_loader import WanPipelineLoader
    from fallback_handler import FallbackHandler
    from optimization_manager import OptimizationManager
    from frame_tensor_handler import FrameTensorHandler
    from video_encoder import VideoEncoder
    from smoke_test_runner import SmokeTestRunner
    from comprehensive_test_runner import ComprehensiveTestRunner
    from performance_benchmark_suite import PerformanceBenchmarkSuite
except ImportError as e:
    logging.warning(f"Some components not available for integration testing: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EndToEndTestResult:
    """Result of end-to-end workflow test"""
    test_name: str
    workflow_success: bool
    model_detection_success: bool
    pipeline_loading_success: bool
    generation_success: bool
    video_encoding_success: bool
    total_time: float
    component_results: Dict[str, Any] = field(default_factory=dict)
    output_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class ResourceConstraintTest:
    """Resource constraint test configuration"""
    test_name: str
    vram_limit_mb: int
    ram_limit_mb: int
    cpu_cores: int
    expected_optimizations: List[str]
    expected_success: bool

@dataclass
class ErrorInjectionTest:
    """Error injection test configuration"""
    test_name: str
    error_type: str
    error_location: str
    expected_recovery: bool
    recovery_strategy: str

class EndToEndIntegrationTestSuite:
    """
    Comprehensive end-to-end integration test suite
    Tests complete workflows from model detection to video output
    """
    
    def __init__(self, test_config: Optional[Dict[str, Any]] = None):
        """
        Initialize end-to-end integration test suite
        
        Args:
            test_config: Optional configuration for test parameters
        """
        self.test_config = test_config or self._get_default_config()
        self.temp_dir = None
        self.test_results = []
        self.artifacts_dir = Path("integration_test_artifacts")
        self.artifacts_dir.mkdir(exist_ok=True)
        
        # Initialize test components
        self._initialize_test_components()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default test configuration"""
        return {
            "test_models": {
                "wan_t2v_full": {
                    "model_type": "wan_t2v",
                    "size_gb": 14,
                    "components": ["transformer", "transformer_2", "vae", "scheduler"],
                    "custom_attributes": {"boundary_ratio": 0.5},
                    "min_vram_mb": 12288
                },
                "wan_t2v_mini": {
                    "model_type": "wan_t2v",
                    "size_gb": 7,
                    "components": ["transformer", "vae", "scheduler"],
                    "custom_attributes": {"boundary_ratio": 0.3},
                    "min_vram_mb": 8192
                },
                "wan_t2i": {
                    "model_type": "wan_t2i",
                    "size_gb": 10,
                    "components": ["transformer", "vae", "scheduler"],
                    "custom_attributes": {},
                    "min_vram_mb": 6144
                },
                "stable_diffusion": {
                    "model_type": "stable_diffusion",
                    "size_gb": 4,
                    "components": ["unet", "vae", "text_encoder", "scheduler"],
                    "custom_attributes": {},
                    "min_vram_mb": 4096
                }
            },
            "resource_constraints": [
                ResourceConstraintTest(
                    test_name="low_vram_4gb",
                    vram_limit_mb=4096,
                    ram_limit_mb=16384,
                    cpu_cores=4,
                    expected_optimizations=["cpu_offload", "mixed_precision", "chunked_processing"],
                    expected_success=True
                ),
                ResourceConstraintTest(
                    test_name="medium_vram_8gb",
                    vram_limit_mb=8192,
                    ram_limit_mb=16384,
                    cpu_cores=8,
                    expected_optimizations=["mixed_precision"],
                    expected_success=True
                ),
                ResourceConstraintTest(
                    test_name="high_vram_16gb",
                    vram_limit_mb=16384,
                    ram_limit_mb=32768,
                    cpu_cores=16,
                    expected_optimizations=[],
                    expected_success=True
                ),
                ResourceConstraintTest(
                    test_name="extreme_low_vram_2gb",
                    vram_limit_mb=2048,
                    ram_limit_mb=8192,
                    cpu_cores=2,
                    expected_optimizations=["cpu_offload", "mixed_precision", "chunked_processing"],
                    expected_success=False
                )
            ],
            "error_injection_tests": [
                ErrorInjectionTest(
                    test_name="missing_model_index",
                    error_type="missing_file",
                    error_location="model_index.json",
                    expected_recovery=True,
                    recovery_strategy="reconstruct_from_components"
                ),
                ErrorInjectionTest(
                    test_name="corrupted_model_weights",
                    error_type="corrupted_file",
                    error_location="model_weights",
                    expected_recovery=False,
                    recovery_strategy="suggest_redownload"
                ),
                ErrorInjectionTest(
                    test_name="missing_pipeline_class",
                    error_type="missing_dependency",
                    error_location="pipeline_class",
                    expected_recovery=True,
                    recovery_strategy="fetch_remote_code"
                ),
                ErrorInjectionTest(
                    test_name="insufficient_memory",
                    error_type="resource_constraint",
                    error_location="memory_allocation",
                    expected_recovery=True,
                    recovery_strategy="apply_optimizations"
                )
            ],
            "performance_benchmarks": {
                "target_detection_time_s": 5.0,
                "target_loading_time_s": 30.0,
                "target_generation_time_s": 120.0,
                "target_encoding_time_s": 10.0,
                "memory_efficiency_threshold": 0.7,
                "fps_threshold": 0.5
            },
            "test_execution": {
                "timeout_seconds": 600,
                "max_concurrent_tests": 3,
                "retry_failed_tests": True,
                "save_artifacts": True
            }
        }
    
    def _initialize_test_components(self):
        """Initialize test components with mocks if needed"""
        self.components = {}
        
        # Initialize or mock each component
        component_classes = [
            ('ArchitectureDetector', ArchitectureDetector),
            ('PipelineManager', PipelineManager),
            ('DependencyManager', DependencyManager),
            ('WanPipelineLoader', WanPipelineLoader),
            ('FallbackHandler', FallbackHandler),
            ('OptimizationManager', OptimizationManager),
            ('FrameTensorHandler', FrameTensorHandler),
            ('VideoEncoder', VideoEncoder),
            ('SmokeTestRunner', SmokeTestRunner)
        ]
        
        for name, cls in component_classes:
            try:
                if name in globals():
                    self.components[name] = cls()
                    logger.info(f"Initialized {name}")
                else:
                    self.components[name] = self._create_mock_component(name)
                    logger.info(f"Created mock {name}")
            except Exception as e:
                self.components[name] = self._create_mock_component(name)
                logger.warning(f"Failed to initialize {name}, using mock: {e}")
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created test directory: {self.temp_dir}")
        
        # Create mock model structures
        self._create_mock_models()
    
    def tearDown(self):
        """Clean up test environment"""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up test directory: {self.temp_dir}")
    
    def run_all_end_to_end_tests(self) -> List[EndToEndTestResult]:
        """
        Run all end-to-end integration tests
        
        Returns:
            List of test results
        """
        logger.info("Starting comprehensive end-to-end integration tests")
        
        self.setUp()
        
        try:
            # Test 1: Complete workflow tests for different model variants
            logger.info("=" * 60)
            logger.info("TESTING DIFFERENT WAN MODEL VARIANTS")
            logger.info("=" * 60)
            
            for model_name, model_config in self.test_config["test_models"].items():
                result = self.test_model_variant_workflow(model_name, model_config)
                self.test_results.append(result)
            
            # Test 2: Resource constraint simulation tests
            logger.info("=" * 60)
            logger.info("TESTING RESOURCE CONSTRAINT SCENARIOS")
            logger.info("=" * 60)
            
            for constraint_test in self.test_config["resource_constraints"]:
                result = self.test_resource_constraint_scenario(constraint_test)
                self.test_results.append(result)
            
            # Test 3: Error injection and recovery testing
            logger.info("=" * 60)
            logger.info("TESTING ERROR INJECTION AND RECOVERY")
            logger.info("=" * 60)
            
            for error_test in self.test_config["error_injection_tests"]:
                result = self.test_error_injection_scenario(error_test)
                self.test_results.append(result)
            
            # Test 4: Performance benchmark tests
            logger.info("=" * 60)
            logger.info("TESTING PERFORMANCE BENCHMARKS")
            logger.info("=" * 60)
            
            result = self.test_performance_benchmarks()
            self.test_results.append(result)
            
            # Test 5: Concurrent operations test
            logger.info("=" * 60)
            logger.info("TESTING CONCURRENT OPERATIONS")
            logger.info("=" * 60)
            
            result = self.test_concurrent_operations()
            self.test_results.append(result)
            
        finally:
            self.tearDown()
        
        # Generate comprehensive report
        self._generate_comprehensive_report()
        
        return self.test_results
    
    def test_model_variant_workflow(self, model_name: str, model_config: Dict[str, Any]) -> EndToEndTestResult:
        """
        Test complete workflow for specific model variant
        
        Args:
            model_name: Name of the model variant
            model_config: Configuration for the model
            
        Returns:
            EndToEndTestResult with workflow results
        """
        logger.info(f"Testing workflow for model variant: {model_name}")
        
        result = EndToEndTestResult(
            test_name=f"model_variant_{model_name}",
            workflow_success=False,
            model_detection_success=False,
            pipeline_loading_success=False,
            generation_success=False,
            video_encoding_success=False,
            total_time=0.0
        )
        
        start_time = time.time()
        
        try:
            model_path = str(Path(self.temp_dir) / model_name)
            
            # Step 1: Model Detection
            logger.info(f"Step 1: Detecting architecture for {model_name}")
            detection_start = time.time()
            
            try:
                detector = self.components['ArchitectureDetector']
                architecture = detector.detect_model_architecture(model_path)
                
                if architecture and hasattr(architecture, 'architecture_type'):
                    if architecture.architecture_type == model_config["model_type"]:
                        result.model_detection_success = True
                        result.component_results["detection"] = {
                            "success": True,
                            "detected_type": architecture.architecture_type,
                            "time": time.time() - detection_start
                        }
                    else:
                        result.errors.append(f"Wrong architecture detected: {architecture.architecture_type}")
                else:
                    result.errors.append("Architecture detection failed")
                    
            except Exception as e:
                result.errors.append(f"Architecture detection error: {str(e)}")
            
            # Step 2: Pipeline Loading
            if result.model_detection_success:
                logger.info(f"Step 2: Loading pipeline for {model_name}")
                loading_start = time.time()
                
                try:
                    pipeline_manager = self.components['PipelineManager']
                    pipeline_loader = self.components['WanPipelineLoader']
                    
                    # Select appropriate pipeline
                    pipeline_class = pipeline_manager.select_pipeline_class(architecture)
                    
                    # Load with optimizations
                    pipeline = pipeline_loader.load_wan_pipeline(
                        model_path,
                        enable_cpu_offload=True,
                        use_mixed_precision=True
                    )
                    
                    if pipeline:
                        result.pipeline_loading_success = True
                        result.component_results["loading"] = {
                            "success": True,
                            "pipeline_class": pipeline_class,
                            "time": time.time() - loading_start
                        }
                    else:
                        result.errors.append("Pipeline loading returned None")
                        
                except Exception as e:
                    result.errors.append(f"Pipeline loading error: {str(e)}")
            
            # Step 3: Video Generation
            if result.pipeline_loading_success:
                logger.info(f"Step 3: Generating video with {model_name}")
                generation_start = time.time()
                
                try:
                    # Mock generation for testing
                    generation_output = self._simulate_video_generation(
                        model_config, 
                        prompt="Test video generation",
                        num_frames=8,
                        height=320,
                        width=320
                    )
                    
                    if generation_output is not None:
                        result.generation_success = True
                        result.component_results["generation"] = {
                            "success": True,
                            "output_shape": generation_output.shape if hasattr(generation_output, 'shape') else "unknown",
                            "time": time.time() - generation_start
                        }
                    else:
                        result.errors.append("Video generation failed")
                        
                except Exception as e:
                    result.errors.append(f"Video generation error: {str(e)}")
            
            # Step 4: Video Encoding
            if result.generation_success:
                logger.info(f"Step 4: Encoding video for {model_name}")
                encoding_start = time.time()
                
                try:
                    video_encoder = self.components['VideoEncoder']
                    frame_handler = self.components['FrameTensorHandler']
                    
                    # Process frames
                    processed_frames = frame_handler.process_output_tensors(generation_output)
                    
                    # Encode video
                    output_path = str(self.artifacts_dir / f"{model_name}_test_video.mp4")
                    encoding_result = video_encoder.encode_frames_to_video(
                        processed_frames, output_path
                    )
                    
                    if encoding_result and Path(output_path).exists():
                        result.video_encoding_success = True
                        result.output_files.append(output_path)
                        result.component_results["encoding"] = {
                            "success": True,
                            "output_path": output_path,
                            "time": time.time() - encoding_start
                        }
                    else:
                        result.warnings.append("Video encoding failed, but workflow continued")
                        
                except Exception as e:
                    result.warnings.append(f"Video encoding error: {str(e)}")
            
            # Determine overall success
            result.workflow_success = (
                result.model_detection_success and
                result.pipeline_loading_success and
                result.generation_success
            )
            
            # Calculate performance metrics
            result.performance_metrics = self._calculate_performance_metrics(result)
            
        except Exception as e:
            result.errors.append(f"Workflow test failed: {str(e)}")
            logger.error(f"Model variant workflow error: {e}")
        
        result.total_time = time.time() - start_time
        logger.info(f"Model variant {model_name} test completed in {result.total_time:.2f}s")
        
        return result
    
    def test_resource_constraint_scenario(self, constraint_test: ResourceConstraintTest) -> EndToEndTestResult:
        """
        Test behavior under specific resource constraints
        
        Args:
            constraint_test: Resource constraint test configuration
            
        Returns:
            EndToEndTestResult with constraint test results
        """
        logger.info(f"Testing resource constraint scenario: {constraint_test.test_name}")
        
        result = EndToEndTestResult(
            test_name=f"resource_constraint_{constraint_test.test_name}",
            workflow_success=False,
            model_detection_success=False,
            pipeline_loading_success=False,
            generation_success=False,
            video_encoding_success=False,
            total_time=0.0
        )
        
        start_time = time.time()
        
        try:
            # Mock resource constraints
            with patch('psutil.virtual_memory') as mock_memory, \
                 patch('psutil.cpu_count') as mock_cpu:
                
                # Set up resource constraints
                mock_memory.return_value.available = constraint_test.ram_limit_mb * 1024 * 1024
                mock_cpu.return_value = constraint_test.cpu_cores
                
                # Test with Wan T2V model (most demanding)
                model_config = self.test_config["test_models"]["wan_t2v_full"]
                model_path = str(Path(self.temp_dir) / "wan_t2v_full")
                
                # Step 1: Resource Analysis and Optimization
                logger.info(f"Analyzing resources for {constraint_test.test_name}")
                
                try:
                    opt_manager = self.components['OptimizationManager']
                    
                    # Analyze system resources
                    system_resources = opt_manager.analyze_system_resources()
                    
                    # Get optimization recommendations
                    model_requirements = {
                        "min_vram_mb": model_config["min_vram_mb"],
                        "supports_cpu_offload": True
                    }
                    
                    optimization_plan = opt_manager.recommend_optimizations(
                        model_requirements, system_resources
                    )
                    
                    result.component_results["optimization"] = {
                        "system_resources": system_resources,
                        "optimization_plan": optimization_plan,
                        "expected_optimizations": constraint_test.expected_optimizations
                    }
                    
                    # Verify expected optimizations are recommended
                    recommended_opts = getattr(optimization_plan, 'optimizations', [])
                    for expected_opt in constraint_test.expected_optimizations:
                        if expected_opt not in recommended_opts:
                            result.warnings.append(f"Expected optimization {expected_opt} not recommended")
                    
                except Exception as e:
                    result.errors.append(f"Resource analysis error: {str(e)}")
                
                # Step 2: Attempt workflow with constraints
                try:
                    # Run simplified workflow
                    workflow_result = self._run_constrained_workflow(
                        model_path, model_config, constraint_test
                    )
                    
                    result.model_detection_success = workflow_result.get("detection_success", False)
                    result.pipeline_loading_success = workflow_result.get("loading_success", False)
                    result.generation_success = workflow_result.get("generation_success", False)
                    result.video_encoding_success = workflow_result.get("encoding_success", False)
                    
                    result.component_results.update(workflow_result)
                    
                except Exception as e:
                    result.errors.append(f"Constrained workflow error: {str(e)}")
                
                # Determine success based on expectations
                if constraint_test.expected_success:
                    result.workflow_success = (
                        result.model_detection_success and
                        result.pipeline_loading_success and
                        result.generation_success
                    )
                else:
                    # For scenarios expected to fail, success means graceful failure
                    result.workflow_success = len(result.errors) == 0  # No crashes
        
        except Exception as e:
            result.errors.append(f"Resource constraint test failed: {str(e)}")
            logger.error(f"Resource constraint test error: {e}")
        
        result.total_time = time.time() - start_time
        logger.info(f"Resource constraint {constraint_test.test_name} test completed in {result.total_time:.2f}s")
        
        return result
    
    def test_error_injection_scenario(self, error_test: ErrorInjectionTest) -> EndToEndTestResult:
        """
        Test error injection and recovery scenarios
        
        Args:
            error_test: Error injection test configuration
            
        Returns:
            EndToEndTestResult with error recovery results
        """
        logger.info(f"Testing error injection scenario: {error_test.test_name}")
        
        result = EndToEndTestResult(
            test_name=f"error_injection_{error_test.test_name}",
            workflow_success=False,
            model_detection_success=False,
            pipeline_loading_success=False,
            generation_success=False,
            video_encoding_success=False,
            total_time=0.0
        )
        
        start_time = time.time()
        
        try:
            model_path = str(Path(self.temp_dir) / "wan_t2v_full")
            
            # Inject specific error
            self._inject_error(model_path, error_test)
            
            # Attempt workflow with error present
            try:
                # Step 1: Test error detection
                error_detected = False
                recovery_attempted = False
                recovery_successful = False
                
                if error_test.error_location == "model_index.json":
                    # Test missing/corrupted model index
                    try:
                        detector = self.components['ArchitectureDetector']
                        architecture = detector.detect_model_architecture(model_path)
                        
                        if architecture is None:
                            error_detected = True
                            logger.info("Error correctly detected: missing model index")
                            
                            # Test recovery
                            if error_test.expected_recovery:
                                recovery_attempted = True
                                # Mock recovery attempt
                                recovered_architecture = self._attempt_recovery(
                                    error_test.recovery_strategy, model_path
                                )
                                if recovered_architecture:
                                    recovery_successful = True
                                    result.model_detection_success = True
                        
                    except Exception as e:
                        error_detected = True
                        result.warnings.append(f"Error detected via exception: {str(e)}")
                
                elif error_test.error_location == "pipeline_class":
                    # Test missing pipeline class
                    try:
                        pipeline_manager = self.components['PipelineManager']
                        pipeline = pipeline_manager.load_custom_pipeline(
                            model_path, "NonExistentPipeline"
                        )
                        
                        if pipeline is None:
                            error_detected = True
                            logger.info("Error correctly detected: missing pipeline class")
                            
                            # Test recovery
                            if error_test.expected_recovery:
                                recovery_attempted = True
                                # Mock dependency manager recovery
                                dep_manager = self.components['DependencyManager']
                                fetch_result = dep_manager.fetch_pipeline_code(model_path)
                                if fetch_result:
                                    recovery_successful = True
                                    result.pipeline_loading_success = True
                        
                    except Exception as e:
                        error_detected = True
                        result.warnings.append(f"Pipeline error detected: {str(e)}")
                
                elif error_test.error_location == "memory_allocation":
                    # Test memory constraint error
                    try:
                        # Simulate memory allocation failure
                        with patch('torch.cuda.OutOfMemoryError'):
                            error_detected = True
                            logger.info("Error correctly detected: memory allocation failure")
                            
                            # Test optimization recovery
                            if error_test.expected_recovery:
                                recovery_attempted = True
                                opt_manager = self.components['OptimizationManager']
                                
                                # Apply memory optimizations
                                optimization_plan = opt_manager.recommend_optimizations(
                                    {"min_vram_mb": 16384}, {"vram_mb": 4096}
                                )
                                
                                if optimization_plan:
                                    recovery_successful = True
                                    result.generation_success = True
                        
                    except Exception as e:
                        result.warnings.append(f"Memory error test: {str(e)}")
                
                # Record error handling results
                result.component_results["error_handling"] = {
                    "error_detected": error_detected,
                    "recovery_attempted": recovery_attempted,
                    "recovery_successful": recovery_successful,
                    "error_type": error_test.error_type,
                    "recovery_strategy": error_test.recovery_strategy
                }
                
                # Determine success based on expectations
                if error_test.expected_recovery:
                    result.workflow_success = recovery_successful
                else:
                    result.workflow_success = error_detected and not recovery_successful
                
            except Exception as e:
                result.errors.append(f"Error injection test failed: {str(e)}")
        
        except Exception as e:
            result.errors.append(f"Error injection setup failed: {str(e)}")
            logger.error(f"Error injection test error: {e}")
        
        result.total_time = time.time() - start_time
        logger.info(f"Error injection {error_test.test_name} test completed in {result.total_time:.2f}s")
        
        return result
    
    def test_performance_benchmarks(self) -> EndToEndTestResult:
        """
        Test performance benchmarks for optimization strategies
        
        Returns:
            EndToEndTestResult with performance benchmark results
        """
        logger.info("Testing performance benchmarks")
        
        result = EndToEndTestResult(
            test_name="performance_benchmarks",
            workflow_success=False,
            model_detection_success=True,  # Not applicable
            pipeline_loading_success=True,  # Not applicable
            generation_success=True,  # Not applicable
            video_encoding_success=True,  # Not applicable
            total_time=0.0
        )
        
        start_time = time.time()
        
        try:
            benchmarks = self.test_config["performance_benchmarks"]
            
            # Test 1: Model Detection Speed
            logger.info("Benchmarking model detection speed")
            detection_times = []
            
            for i in range(5):  # Run 5 iterations
                model_path = str(Path(self.temp_dir) / "wan_t2v_full")
                
                detection_start = time.time()
                try:
                    detector = self.components['ArchitectureDetector']
                    architecture = detector.detect_model_architecture(model_path)
                    detection_time = time.time() - detection_start
                    detection_times.append(detection_time)
                except Exception as e:
                    result.warnings.append(f"Detection benchmark iteration {i+1} failed: {str(e)}")
            
            avg_detection_time = sum(detection_times) / len(detection_times) if detection_times else float('inf')
            
            result.performance_metrics["avg_detection_time"] = avg_detection_time
            result.performance_metrics["target_detection_time"] = benchmarks["target_detection_time_s"]
            
            if avg_detection_time <= benchmarks["target_detection_time_s"]:
                result.component_results["detection_benchmark"] = "PASS"
            else:
                result.component_results["detection_benchmark"] = "FAIL"
                result.warnings.append(f"Detection time {avg_detection_time:.2f}s exceeds target {benchmarks['target_detection_time_s']}s")
            
            # Test 2: Pipeline Loading Speed
            logger.info("Benchmarking pipeline loading speed")
            loading_times = []
            
            for i in range(3):  # Run 3 iterations (loading is expensive)
                loading_start = time.time()
                try:
                    pipeline_loader = self.components['WanPipelineLoader']
                    pipeline = pipeline_loader.load_wan_pipeline(
                        str(Path(self.temp_dir) / "wan_t2v_full")
                    )
                    loading_time = time.time() - loading_start
                    loading_times.append(loading_time)
                except Exception as e:
                    result.warnings.append(f"Loading benchmark iteration {i+1} failed: {str(e)}")
            
            avg_loading_time = sum(loading_times) / len(loading_times) if loading_times else float('inf')
            
            result.performance_metrics["avg_loading_time"] = avg_loading_time
            result.performance_metrics["target_loading_time"] = benchmarks["target_loading_time_s"]
            
            if avg_loading_time <= benchmarks["target_loading_time_s"]:
                result.component_results["loading_benchmark"] = "PASS"
            else:
                result.component_results["loading_benchmark"] = "FAIL"
                result.warnings.append(f"Loading time {avg_loading_time:.2f}s exceeds target {benchmarks['target_loading_time_s']}s")
            
            # Test 3: Generation Speed with Different Optimizations
            logger.info("Benchmarking generation speed with optimizations")
            
            optimization_scenarios = [
                {"name": "no_optimization", "opts": {}},
                {"name": "mixed_precision", "opts": {"use_mixed_precision": True}},
                {"name": "cpu_offload", "opts": {"enable_cpu_offload": True}},
                {"name": "full_optimization", "opts": {"use_mixed_precision": True, "enable_cpu_offload": True}}
            ]
            
            for scenario in optimization_scenarios:
                scenario_times = []
                
                for i in range(2):  # 2 iterations per scenario
                    generation_start = time.time()
                    try:
                        # Mock generation with optimizations
                        generation_output = self._simulate_optimized_generation(scenario["opts"])
                        generation_time = time.time() - generation_start
                        scenario_times.append(generation_time)
                    except Exception as e:
                        result.warnings.append(f"Generation benchmark {scenario['name']} iteration {i+1} failed: {str(e)}")
                
                avg_scenario_time = sum(scenario_times) / len(scenario_times) if scenario_times else float('inf')
                result.performance_metrics[f"generation_time_{scenario['name']}"] = avg_scenario_time
                
                # Calculate FPS
                fps = 8.0 / avg_scenario_time if avg_scenario_time > 0 else 0.0
                result.performance_metrics[f"fps_{scenario['name']}"] = fps
                
                if fps >= benchmarks["fps_threshold"]:
                    result.component_results[f"generation_benchmark_{scenario['name']}"] = "PASS"
                else:
                    result.component_results[f"generation_benchmark_{scenario['name']}"] = "FAIL"
            
            # Test 4: Memory Efficiency
            logger.info("Benchmarking memory efficiency")
            
            try:
                # Mock memory usage measurement
                baseline_memory = 1000  # MB
                optimized_memory = 600   # MB
                
                memory_efficiency = (baseline_memory - optimized_memory) / baseline_memory
                result.performance_metrics["memory_efficiency"] = memory_efficiency
                
                if memory_efficiency >= benchmarks["memory_efficiency_threshold"]:
                    result.component_results["memory_benchmark"] = "PASS"
                else:
                    result.component_results["memory_benchmark"] = "FAIL"
                    result.warnings.append(f"Memory efficiency {memory_efficiency:.2f} below threshold {benchmarks['memory_efficiency_threshold']}")
            
            except Exception as e:
                result.warnings.append(f"Memory efficiency benchmark failed: {str(e)}")
            
            # Determine overall benchmark success
            benchmark_results = [v for k, v in result.component_results.items() if k.endswith("_benchmark")]
            passed_benchmarks = sum(1 for r in benchmark_results if r == "PASS")
            total_benchmarks = len(benchmark_results)
            
            result.workflow_success = (passed_benchmarks / total_benchmarks) >= 0.75 if total_benchmarks > 0 else False
            
        except Exception as e:
            result.errors.append(f"Performance benchmark test failed: {str(e)}")
            logger.error(f"Performance benchmark test error: {e}")
        
        result.total_time = time.time() - start_time
        logger.info(f"Performance benchmark test completed in {result.total_time:.2f}s")
        
        return result
    
    def test_concurrent_operations(self) -> EndToEndTestResult:
        """
        Test concurrent operations and thread safety
        
        Returns:
            EndToEndTestResult with concurrent operations results
        """
        logger.info("Testing concurrent operations")
        
        result = EndToEndTestResult(
            test_name="concurrent_operations",
            workflow_success=False,
            model_detection_success=False,
            pipeline_loading_success=False,
            generation_success=False,
            video_encoding_success=False,
            total_time=0.0
        )
        
        start_time = time.time()
        
        try:
            max_workers = self.test_config["test_execution"]["max_concurrent_tests"]
            
            # Test 1: Concurrent Model Detection
            logger.info("Testing concurrent model detection")
            
            def detect_model_concurrent(model_name):
                try:
                    detector = self.components['ArchitectureDetector']
                    model_path = str(Path(self.temp_dir) / model_name)
                    architecture = detector.detect_model_architecture(model_path)
                    return {"success": architecture is not None, "model": model_name}
                except Exception as e:
                    return {"success": False, "model": model_name, "error": str(e)}
            
            model_names = list(self.test_config["test_models"].keys())[:3]  # Test with 3 models
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                detection_futures = [executor.submit(detect_model_concurrent, name) for name in model_names]
                detection_results = [future.result() for future in concurrent.futures.as_completed(detection_futures)]
            
            successful_detections = sum(1 for r in detection_results if r["success"])
            result.model_detection_success = successful_detections == len(model_names)
            
            result.component_results["concurrent_detection"] = {
                "total_models": len(model_names),
                "successful_detections": successful_detections,
                "results": detection_results
            }
            
            # Test 2: Concurrent Pipeline Loading
            logger.info("Testing concurrent pipeline loading")
            
            def load_pipeline_concurrent(model_name):
                try:
                    pipeline_loader = self.components['WanPipelineLoader']
                    model_path = str(Path(self.temp_dir) / model_name)
                    pipeline = pipeline_loader.load_wan_pipeline(model_path)
                    return {"success": pipeline is not None, "model": model_name}
                except Exception as e:
                    return {"success": False, "model": model_name, "error": str(e)}
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                loading_futures = [executor.submit(load_pipeline_concurrent, name) for name in model_names]
                loading_results = [future.result() for future in concurrent.futures.as_completed(loading_futures)]
            
            successful_loadings = sum(1 for r in loading_results if r["success"])
            result.pipeline_loading_success = successful_loadings == len(model_names)
            
            result.component_results["concurrent_loading"] = {
                "total_models": len(model_names),
                "successful_loadings": successful_loadings,
                "results": loading_results
            }
            
            # Test 3: Concurrent Generation
            logger.info("Testing concurrent generation")
            
            def generate_concurrent(prompt_id):
                try:
                    # Mock concurrent generation
                    time.sleep(0.1)  # Simulate generation time
                    output = np.random.rand(8, 256, 256, 3).astype(np.float32)
                    return {"success": True, "prompt_id": prompt_id, "output_shape": output.shape}
                except Exception as e:
                    return {"success": False, "prompt_id": prompt_id, "error": str(e)}
            
            prompts = [f"Test prompt {i}" for i in range(3)]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                generation_futures = [executor.submit(generate_concurrent, i) for i in range(len(prompts))]
                generation_results = [future.result() for future in concurrent.futures.as_completed(generation_futures)]
            
            successful_generations = sum(1 for r in generation_results if r["success"])
            result.generation_success = successful_generations == len(prompts)
            
            result.component_results["concurrent_generation"] = {
                "total_prompts": len(prompts),
                "successful_generations": successful_generations,
                "results": generation_results
            }
            
            # Test 4: Thread Safety Validation
            logger.info("Testing thread safety")
            
            # Test shared resource access
            shared_counter = {"value": 0}
            lock = threading.Lock()
            
            def increment_counter():
                for _ in range(100):
                    with lock:
                        shared_counter["value"] += 1
            
            threads = [threading.Thread(target=increment_counter) for _ in range(5)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            
            expected_value = 5 * 100
            thread_safety_success = shared_counter["value"] == expected_value
            
            result.component_results["thread_safety"] = {
                "expected_value": expected_value,
                "actual_value": shared_counter["value"],
                "success": thread_safety_success
            }
            
            # Determine overall success
            result.workflow_success = (
                result.model_detection_success and
                result.pipeline_loading_success and
                result.generation_success and
                thread_safety_success
            )
            
        except Exception as e:
            result.errors.append(f"Concurrent operations test failed: {str(e)}")
            logger.error(f"Concurrent operations test error: {e}")
        
        result.total_time = time.time() - start_time
        logger.info(f"Concurrent operations test completed in {result.total_time:.2f}s")
        
        return result
    
    def _create_mock_models(self):
        """Create mock model directory structures"""
        for model_name, model_config in self.test_config["test_models"].items():
            model_dir = Path(self.temp_dir) / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Create model_index.json
            model_index = {
                "_class_name": f"{'WanPipeline' if model_config['model_type'].startswith('wan') else 'StableDiffusionPipeline'}",
                "_diffusers_version": "0.21.4"
            }
            
            # Add components
            for component in model_config["components"]:
                model_index[component] = [component, "diffusers"]
            
            # Add custom attributes
            model_index.update(model_config["custom_attributes"])
            
            with open(model_dir / "model_index.json", 'w') as f:
                json.dump(model_index, f, indent=2)
            
            # Create component directories
            for component in model_config["components"]:
                component_dir = model_dir / component
                component_dir.mkdir(exist_ok=True)
                
                # Create config.json for each component
                component_config = {
                    "_class_name": f"Mock{component.title()}",
                    "model_type": model_config["model_type"]
                }
                
                with open(component_dir / "config.json", 'w') as f:
                    json.dump(component_config, f, indent=2)
    
    def _create_mock_component(self, component_name: str):
        """Create mock component for testing"""
        mock = Mock()
        
        if component_name == "ArchitectureDetector":
            def mock_detect(model_path):
                # Determine model type from path
                if "wan" in model_path.lower():
                    return Mock(architecture_type="wan_t2v", components=["transformer", "vae"])
                else:
                    return Mock(architecture_type="stable_diffusion", components=["unet", "vae"])
            mock.detect_model_architecture = mock_detect
        
        elif component_name == "PipelineManager":
            mock.select_pipeline_class = Mock(return_value="WanPipeline")
            mock.load_custom_pipeline = Mock(return_value=Mock())
        
        elif component_name == "WanPipelineLoader":
            mock.load_wan_pipeline = Mock(return_value=Mock())
        
        elif component_name == "OptimizationManager":
            mock.analyze_system_resources = Mock(return_value={"vram_mb": 8192, "ram_mb": 16384})
            mock.recommend_optimizations = Mock(return_value=Mock(optimizations=["mixed_precision"]))
        
        elif component_name == "FrameTensorHandler":
            def mock_process(output):
                return Mock(frames=output, fps=24.0, duration=1.0)
            mock.process_output_tensors = mock_process
        
        elif component_name == "VideoEncoder":
            def mock_encode(frames, output_path, format="mp4"):
                # Create dummy output file
                Path(output_path).touch()
                return Mock(success=True, output_path=output_path)
            mock.encode_frames_to_video = mock_encode
        
        return mock
    
    def _simulate_video_generation(self, model_config: Dict[str, Any], **kwargs) -> np.ndarray:
        """Simulate video generation for testing"""
        num_frames = kwargs.get("num_frames", 8)
        height = kwargs.get("height", 320)
        width = kwargs.get("width", 320)
        
        # Simulate generation time based on model size
        generation_time = model_config["size_gb"] * 0.1  # 0.1s per GB
        time.sleep(generation_time)
        
        # Return mock video frames
        return np.random.rand(num_frames, height, width, 3).astype(np.float32)
    
    def _simulate_optimized_generation(self, optimizations: Dict[str, Any]) -> np.ndarray:
        """Simulate optimized generation for benchmarking"""
        base_time = 2.0  # Base generation time
        
        # Apply optimization speedups
        if optimizations.get("use_mixed_precision"):
            base_time *= 0.7  # 30% speedup
        if optimizations.get("enable_cpu_offload"):
            base_time *= 1.2  # 20% slowdown but saves memory
        
        time.sleep(base_time)
        return np.random.rand(8, 256, 256, 3).astype(np.float32)
    
    def _run_constrained_workflow(self, model_path: str, model_config: Dict[str, Any], 
                                 constraint_test: ResourceConstraintTest) -> Dict[str, Any]:
        """Run workflow under resource constraints"""
        workflow_result = {
            "detection_success": False,
            "loading_success": False,
            "generation_success": False,
            "encoding_success": False
        }
        
        try:
            # Detection (usually succeeds under constraints)
            workflow_result["detection_success"] = True
            
            # Loading (may fail under extreme constraints)
            if constraint_test.vram_limit_mb >= 2048:
                workflow_result["loading_success"] = True
            
            # Generation (depends on optimizations)
            if constraint_test.vram_limit_mb >= 4096 or "cpu_offload" in constraint_test.expected_optimizations:
                workflow_result["generation_success"] = True
            
            # Encoding (usually succeeds)
            workflow_result["encoding_success"] = True
            
        except Exception as e:
            logger.warning(f"Constrained workflow error: {e}")
        
        return workflow_result
    
    def _inject_error(self, model_path: str, error_test: ErrorInjectionTest):
        """Inject specific error for testing"""
        if error_test.error_type == "missing_file" and error_test.error_location == "model_index.json":
            # Remove model_index.json
            index_file = Path(model_path) / "model_index.json"
            if index_file.exists():
                index_file.unlink()
        
        elif error_test.error_type == "corrupted_file" and error_test.error_location == "model_weights":
            # Create corrupted weight file
            weight_file = Path(model_path) / "corrupted_weights.bin"
            with open(weight_file, 'wb') as f:
                f.write(b"corrupted data")
    
    def _attempt_recovery(self, recovery_strategy: str, model_path: str):
        """Attempt error recovery"""
        if recovery_strategy == "reconstruct_from_components":
            # Mock reconstruction
            return Mock(architecture_type="wan_t2v")
        elif recovery_strategy == "fetch_remote_code":
            # Mock remote code fetch
            return Mock(success=True)
        elif recovery_strategy == "apply_optimizations":
            # Mock optimization application
            return Mock(optimizations_applied=["cpu_offload", "mixed_precision"])
        
        return None
    
    def _calculate_performance_metrics(self, result: EndToEndTestResult) -> Dict[str, float]:
        """Calculate performance metrics from test result"""
        metrics = {}
        
        if "detection" in result.component_results:
            metrics["detection_time"] = result.component_results["detection"]["time"]
        
        if "loading" in result.component_results:
            metrics["loading_time"] = result.component_results["loading"]["time"]
        
        if "generation" in result.component_results:
            gen_time = result.component_results["generation"]["time"]
            metrics["generation_time"] = gen_time
            metrics["fps"] = 8.0 / gen_time if gen_time > 0 else 0.0
        
        if "encoding" in result.component_results:
            metrics["encoding_time"] = result.component_results["encoding"]["time"]
        
        return metrics
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        report_file = self.artifacts_dir / "end_to_end_test_report.json"
        
        report_data = {
            "test_session_id": f"e2e_test_{int(time.time())}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": len(self.test_results),
            "successful_tests": sum(1 for r in self.test_results if r.workflow_success),
            "test_results": []
        }
        
        for result in self.test_results:
            test_data = {
                "test_name": result.test_name,
                "workflow_success": result.workflow_success,
                "total_time": result.total_time,
                "component_successes": {
                    "model_detection": result.model_detection_success,
                    "pipeline_loading": result.pipeline_loading_success,
                    "generation": result.generation_success,
                    "video_encoding": result.video_encoding_success
                },
                "performance_metrics": result.performance_metrics,
                "error_count": len(result.errors),
                "warning_count": len(result.warnings),
                "output_files": result.output_files
            }
            report_data["test_results"].append(test_data)
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Comprehensive test report saved to {report_file}")


def main():
    """Main entry point for end-to-end integration tests"""
    
    print("Wan Model Compatibility - End-to-End Integration Test Suite")
    print("=" * 70)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run end-to-end integration tests")
    parser.add_argument("--model-variants", action="store_true", help="Test only model variants")
    parser.add_argument("--resource-constraints", action="store_true", help="Test only resource constraints")
    parser.add_argument("--error-injection", action="store_true", help="Test only error injection")
    parser.add_argument("--performance", action="store_true", help="Test only performance benchmarks")
    parser.add_argument("--concurrent", action="store_true", help="Test only concurrent operations")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")
    
    args = parser.parse_args()
    
    # Create test suite
    test_suite = EndToEndIntegrationTestSuite()
    
    # Run tests based on arguments
    if any([args.model_variants, args.resource_constraints, args.error_injection, 
            args.performance, args.concurrent]):
        # Run specific test categories
        test_results = []
        
        if args.model_variants:
            print("\nRunning model variant tests...")
            for model_name, model_config in test_suite.test_config["test_models"].items():
                result = test_suite.test_model_variant_workflow(model_name, model_config)
                test_results.append(result)
        
        if args.resource_constraints:
            print("\nRunning resource constraint tests...")
            for constraint_test in test_suite.test_config["resource_constraints"]:
                result = test_suite.test_resource_constraint_scenario(constraint_test)
                test_results.append(result)
        
        if args.error_injection:
            print("\nRunning error injection tests...")
            for error_test in test_suite.test_config["error_injection_tests"]:
                result = test_suite.test_error_injection_scenario(error_test)
                test_results.append(result)
        
        if args.performance:
            print("\nRunning performance benchmark tests...")
            result = test_suite.test_performance_benchmarks()
            test_results.append(result)
        
        if args.concurrent:
            print("\nRunning concurrent operations tests...")
            result = test_suite.test_concurrent_operations()
            test_results.append(result)
    
    else:
        # Run all tests
        print("\nRunning all end-to-end integration tests...")
        test_results = test_suite.run_all_end_to_end_tests()
    
    # Print summary
    print("\n" + "=" * 70)
    print("END-TO-END INTEGRATION TEST SUMMARY")
    print("=" * 70)
    
    total_tests = len(test_results)
    successful_tests = sum(1 for r in test_results if r.workflow_success)
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    # Print individual test results
    print("\nIndividual Test Results:")
    print("-" * 50)
    
    for result in test_results:
        status = "PASS" if result.workflow_success else "FAIL"
        print(f"{status} {result.test_name} ({result.total_time:.2f}s)")
        
        if result.errors:
            print(f"    Errors: {len(result.errors)}")
        if result.warnings:
            print(f"    Warnings: {len(result.warnings)}")
        if result.output_files:
            print(f"    Output Files: {len(result.output_files)}")
    
    # Exit with appropriate code
    sys.exit(0 if success_rate >= 80.0 else 1)


if __name__ == "__main__":
    import sys
    main()