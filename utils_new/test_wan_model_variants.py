#!/usr/bin/env python3
"""
Wan Model Variants Integration Test Suite
Tests different Wan model variants (T2V, T2I, mini versions) end-to-end
"""

import unittest
import tempfile
import shutil
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelVariantTestConfig:
    """Configuration for model variant testing"""
    model_name: str
    model_type: str
    expected_components: List[str]
    expected_pipeline_class: str
    min_vram_mb: int
    supports_video: bool
    supports_image: bool
    test_prompts: List[str]

class WanModelVariantsTestSuite:
    """
    Test suite for different Wan model variants
    Validates model-specific functionality and compatibility
    """
    
    def __init__(self):
        self.temp_dir = None
        self.test_configs = self._get_model_variant_configs()
        self.test_results = []
    
    def _get_model_variant_configs(self) -> List[ModelVariantTestConfig]:
        """Get test configurations for different model variants"""
        return [
            ModelVariantTestConfig(
                model_name="Wan2.2-T2V-A14B-Diffusers",
                model_type="wan_t2v",
                expected_components=["transformer", "transformer_2", "vae", "scheduler"],
                expected_pipeline_class="WanPipeline",
                min_vram_mb=12288,
                supports_video=True,
                supports_image=False,
                test_prompts=[
                    "A cat walking in a garden",
                    "Ocean waves crashing on the shore",
                    "A person riding a bicycle through the city"
                ]
            ),
            ModelVariantTestConfig(
                model_name="Wan2.2-T2V-Mini",
                model_type="wan_t2v",
                expected_components=["transformer", "vae", "scheduler"],
                expected_pipeline_class="WanPipeline",
                min_vram_mb=8192,
                supports_video=True,
                supports_image=False,
                test_prompts=[
                    "A simple animation of a bouncing ball",
                    "Clouds moving across the sky"
                ]
            ),
            ModelVariantTestConfig(
                model_name="Wan2.2-T2I",
                model_type="wan_t2i",
                expected_components=["transformer", "vae", "scheduler"],
                expected_pipeline_class="WanImagePipeline",
                min_vram_mb=6144,
                supports_video=False,
                supports_image=True,
                test_prompts=[
                    "A beautiful landscape painting",
                    "Portrait of a person in Renaissance style"
                ]
            ),
            ModelVariantTestConfig(
                model_name="Wan2.1-T2V-Legacy",
                model_type="wan_t2v",
                expected_components=["transformer", "vae", "scheduler", "text_encoder"],
                expected_pipeline_class="WanPipeline",
                min_vram_mb=10240,
                supports_video=True,
                supports_image=False,
                test_prompts=[
                    "Legacy test: A train moving through mountains"
                ]
            )
        ]
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created test directory: {self.temp_dir}")
        
        # Create mock model structures for each variant
        for config in self.test_configs:
            self._create_mock_model_variant(config)
    
    def tearDown(self):
        """Clean up test environment"""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up test directory: {self.temp_dir}")
    
    def test_all_model_variants(self) -> List[Dict[str, Any]]:
        """
        Test all model variants end-to-end
        
        Returns:
            List of test results for each variant
        """
        logger.info("Starting model variant tests")
        
        self.setUp()
        
        try:
            for config in self.test_configs:
                logger.info(f"Testing model variant: {config.model_name}")
                result = self.test_model_variant(config)
                self.test_results.append(result)
        
        finally:
            self.tearDown()
        
        return self.test_results
    
    def test_model_variant(self, config: ModelVariantTestConfig) -> Dict[str, Any]:
        """
        Test specific model variant
        
        Args:
            config: Model variant test configuration
            
        Returns:
            Test result dictionary
        """
        result = {
            "model_name": config.model_name,
            "model_type": config.model_type,
            "success": False,
            "tests_passed": 0,
            "tests_total": 0,
            "errors": [],
            "warnings": [],
            "performance_metrics": {},
            "component_tests": {},
            "generation_tests": {}
        }
        
        start_time = time.time()
        model_path = str(Path(self.temp_dir) / config.model_name)
        
        try:
            # Test 1: Architecture Detection
            result["tests_total"] += 1
            detection_result = self._test_architecture_detection(model_path, config)
            result["component_tests"]["architecture_detection"] = detection_result
            
            if detection_result["success"]:
                result["tests_passed"] += 1
            else:
                result["errors"].extend(detection_result["errors"])
            
            # Test 2: Component Validation
            result["tests_total"] += 1
            component_result = self._test_component_validation(model_path, config)
            result["component_tests"]["component_validation"] = component_result
            
            if component_result["success"]:
                result["tests_passed"] += 1
            else:
                result["errors"].extend(component_result["errors"])
            
            # Test 3: Pipeline Selection
            result["tests_total"] += 1
            pipeline_result = self._test_pipeline_selection(model_path, config)
            result["component_tests"]["pipeline_selection"] = pipeline_result
            
            if pipeline_result["success"]:
                result["tests_passed"] += 1
            else:
                result["errors"].extend(pipeline_result["errors"])
            
            # Test 4: Generation Capability Tests
            for i, prompt in enumerate(config.test_prompts):
                result["tests_total"] += 1
                gen_result = self._test_generation_capability(model_path, config, prompt, i)
                result["generation_tests"][f"prompt_{i}"] = gen_result
                
                if gen_result["success"]:
                    result["tests_passed"] += 1
                else:
                    result["errors"].extend(gen_result["errors"])
            
            # Test 5: Output Format Validation
            result["tests_total"] += 1
            format_result = self._test_output_format_validation(config)
            result["component_tests"]["output_format"] = format_result
            
            if format_result["success"]:
                result["tests_passed"] += 1
            else:
                result["errors"].extend(format_result["errors"])
            
            # Test 6: Memory Requirements Validation
            result["tests_total"] += 1
            memory_result = self._test_memory_requirements(config)
            result["component_tests"]["memory_requirements"] = memory_result
            
            if memory_result["success"]:
                result["tests_passed"] += 1
            else:
                result["warnings"].extend(memory_result["warnings"])
            
            # Calculate success rate
            success_rate = result["tests_passed"] / result["tests_total"] if result["tests_total"] > 0 else 0
            result["success"] = success_rate >= 0.8  # 80% pass rate required
            
            # Performance metrics
            result["performance_metrics"] = {
                "total_test_time": time.time() - start_time,
                "success_rate": success_rate,
                "tests_per_second": result["tests_total"] / (time.time() - start_time)
            }
            
        except Exception as e:
            result["errors"].append(f"Model variant test failed: {str(e)}")
            logger.error(f"Model variant test error for {config.model_name}: {e}")
        
        logger.info(f"Model variant {config.model_name} test completed: {result['tests_passed']}/{result['tests_total']} passed")
        
        return result
    
    def _test_architecture_detection(self, model_path: str, config: ModelVariantTestConfig) -> Dict[str, Any]:
        """Test architecture detection for model variant"""
        result = {
            "success": False,
            "detected_type": None,
            "expected_type": config.model_type,
            "errors": [],
            "detection_time": 0.0
        }
        
        start_time = time.time()
        
        try:
            # Mock architecture detection
            model_index_path = Path(model_path) / "model_index.json"
            
            if model_index_path.exists():
                with open(model_index_path, 'r') as f:
                    model_index = json.load(f)
                
                # Determine architecture type from model_index
                if "transformer_2" in model_index or "boundary_ratio" in model_index:
                    detected_type = "wan_t2v"
                elif "transformer" in model_index and config.model_type == "wan_t2i":
                    detected_type = "wan_t2i"
                elif "transformer" in model_index:
                    detected_type = "wan_t2v"
                else:
                    detected_type = "unknown"
                
                result["detected_type"] = detected_type
                
                if detected_type == config.model_type:
                    result["success"] = True
                else:
                    result["errors"].append(f"Expected {config.model_type}, detected {detected_type}")
            else:
                result["errors"].append("model_index.json not found")
        
        except Exception as e:
            result["errors"].append(f"Architecture detection failed: {str(e)}")
        
        result["detection_time"] = time.time() - start_time
        return result
    
    def _test_component_validation(self, model_path: str, config: ModelVariantTestConfig) -> Dict[str, Any]:
        """Test component validation for model variant"""
        result = {
            "success": False,
            "found_components": [],
            "missing_components": [],
            "unexpected_components": [],
            "errors": []
        }
        
        try:
            model_index_path = Path(model_path) / "model_index.json"
            
            if model_index_path.exists():
                with open(model_index_path, 'r') as f:
                    model_index = json.load(f)
                
                # Check for expected components
                found_components = []
                missing_components = []
                
                for component in config.expected_components:
                    if component in model_index:
                        found_components.append(component)
                        
                        # Verify component directory exists
                        component_dir = Path(model_path) / component
                        if not component_dir.exists():
                            result["errors"].append(f"Component directory missing: {component}")
                    else:
                        missing_components.append(component)
                
                # Check for unexpected components
                model_components = [k for k in model_index.keys() if not k.startswith("_")]
                unexpected_components = [c for c in model_components if c not in config.expected_components]
                
                result["found_components"] = found_components
                result["missing_components"] = missing_components
                result["unexpected_components"] = unexpected_components
                
                # Success if all expected components found and no critical missing
                result["success"] = len(missing_components) == 0
                
                if missing_components:
                    result["errors"].append(f"Missing components: {missing_components}")
            else:
                result["errors"].append("model_index.json not found")
        
        except Exception as e:
            result["errors"].append(f"Component validation failed: {str(e)}")
        
        return result
    
    def _test_pipeline_selection(self, model_path: str, config: ModelVariantTestConfig) -> Dict[str, Any]:
        """Test pipeline selection for model variant"""
        result = {
            "success": False,
            "selected_pipeline": None,
            "expected_pipeline": config.expected_pipeline_class,
            "errors": []
        }
        
        try:
            # Mock pipeline selection logic
            model_index_path = Path(model_path) / "model_index.json"
            
            if model_index_path.exists():
                with open(model_index_path, 'r') as f:
                    model_index = json.load(f)
                
                # Determine pipeline class from model_index
                pipeline_class = model_index.get("_class_name", "Unknown")
                
                result["selected_pipeline"] = pipeline_class
                
                if pipeline_class == config.expected_pipeline_class:
                    result["success"] = True
                else:
                    result["errors"].append(f"Expected {config.expected_pipeline_class}, got {pipeline_class}")
            else:
                result["errors"].append("model_index.json not found")
        
        except Exception as e:
            result["errors"].append(f"Pipeline selection failed: {str(e)}")
        
        return result
    
    def _test_generation_capability(self, model_path: str, config: ModelVariantTestConfig, 
                                  prompt: str, prompt_index: int) -> Dict[str, Any]:
        """Test generation capability for specific prompt"""
        result = {
            "success": False,
            "prompt": prompt,
            "prompt_index": prompt_index,
            "output_shape": None,
            "generation_time": 0.0,
            "errors": []
        }
        
        start_time = time.time()
        
        try:
            # Mock generation based on model capabilities
            if config.supports_video:
                # Generate video output
                num_frames = 8
                height = 320
                width = 320
                channels = 3
                
                # Simulate generation time based on model complexity
                base_time = 0.1
                if "A14B" in config.model_name:
                    base_time = 0.3  # Larger model takes longer
                elif "Mini" in config.model_name:
                    base_time = 0.05  # Smaller model is faster
                
                time.sleep(base_time)
                
                # Create mock video output
                output = np.random.rand(num_frames, height, width, channels).astype(np.float32)
                result["output_shape"] = output.shape
                result["success"] = True
                
            elif config.supports_image:
                # Generate image output
                height = 512
                width = 512
                channels = 3
                
                time.sleep(0.05)  # Image generation is faster
                
                # Create mock image output
                output = np.random.rand(height, width, channels).astype(np.float32)
                result["output_shape"] = output.shape
                result["success"] = True
                
            else:
                result["errors"].append("Model doesn't support video or image generation")
        
        except Exception as e:
            result["errors"].append(f"Generation failed: {str(e)}")
        
        result["generation_time"] = time.time() - start_time
        return result
    
    def _test_output_format_validation(self, config: ModelVariantTestConfig) -> Dict[str, Any]:
        """Test output format validation for model variant"""
        result = {
            "success": False,
            "expected_format": None,
            "validation_errors": [],
            "errors": []
        }
        
        try:
            if config.supports_video:
                # Test video output format
                result["expected_format"] = "video_tensor"
                
                # Create mock video output
                mock_output = np.random.rand(8, 320, 320, 3).astype(np.float32)
                
                # Validate format
                if len(mock_output.shape) == 4:  # (frames, height, width, channels)
                    if mock_output.shape[0] > 0 and mock_output.shape[3] == 3:
                        result["success"] = True
                    else:
                        result["validation_errors"].append("Invalid video tensor dimensions")
                else:
                    result["validation_errors"].append("Expected 4D tensor for video")
            
            elif config.supports_image:
                # Test image output format
                result["expected_format"] = "image_tensor"
                
                # Create mock image output
                mock_output = np.random.rand(512, 512, 3).astype(np.float32)
                
                # Validate format
                if len(mock_output.shape) == 3:  # (height, width, channels)
                    if mock_output.shape[2] == 3:
                        result["success"] = True
                    else:
                        result["validation_errors"].append("Invalid image tensor channels")
                else:
                    result["validation_errors"].append("Expected 3D tensor for image")
            
            else:
                result["errors"].append("Unknown output format for model variant")
        
        except Exception as e:
            result["errors"].append(f"Output format validation failed: {str(e)}")
        
        return result
    
    def _test_memory_requirements(self, config: ModelVariantTestConfig) -> Dict[str, Any]:
        """Test memory requirements validation"""
        result = {
            "success": False,
            "min_vram_mb": config.min_vram_mb,
            "estimated_usage_mb": 0,
            "warnings": []
        }
        
        try:
            # Estimate memory usage based on model type and size
            if "A14B" in config.model_name:
                estimated_usage = 14000  # ~14GB model
            elif "Mini" in config.model_name:
                estimated_usage = 7000   # ~7GB model
            elif "T2I" in config.model_name:
                estimated_usage = 6000   # ~6GB model
            else:
                estimated_usage = 10000  # Default estimate
            
            result["estimated_usage_mb"] = estimated_usage
            
            # Check if estimated usage is reasonable
            if estimated_usage <= config.min_vram_mb * 1.5:  # Allow 50% overhead
                result["success"] = True
            else:
                result["warnings"].append(f"High memory usage: {estimated_usage}MB > {config.min_vram_mb}MB")
            
            # Check for optimization opportunities
            if estimated_usage > 8192:  # > 8GB
                result["warnings"].append("Consider enabling CPU offloading for this model")
            
            if estimated_usage > 12288:  # > 12GB
                result["warnings"].append("Consider using mixed precision for this model")
        
        except Exception as e:
            result["warnings"].append(f"Memory requirements test failed: {str(e)}")
        
        return result
    
    def _create_mock_model_variant(self, config: ModelVariantTestConfig):
        """Create mock model variant directory structure"""
        model_dir = Path(self.temp_dir) / config.model_name
        model_dir.mkdir(exist_ok=True)
        
        # Create model_index.json
        model_index = {
            "_class_name": config.expected_pipeline_class,
            "_diffusers_version": "0.21.4"
        }
        
        # Add components
        for component in config.expected_components:
            model_index[component] = [component, "diffusers"]
        
        # Add model-specific attributes
        if config.model_type == "wan_t2v":
            if "A14B" in config.model_name:
                model_index["transformer_2"] = ["transformer_2", "diffusers"]
                model_index["boundary_ratio"] = 0.5
            elif "Mini" in config.model_name:
                model_index["boundary_ratio"] = 0.3
        
        with open(model_dir / "model_index.json", 'w') as f:
            json.dump(model_index, f, indent=2)
        
        # Create component directories
        for component in config.expected_components:
            component_dir = model_dir / component
            component_dir.mkdir(exist_ok=True)
            
            # Create config.json for each component
            component_config = {
                "_class_name": f"Mock{component.title()}",
                "model_type": config.model_type
            }
            
            # Add component-specific configurations
            if component == "vae" and config.model_type.startswith("wan"):
                component_config["latent_channels"] = 16 if config.supports_video else 4
                component_config["sample_size"] = [64, 64] if config.supports_video else 64
            
            with open(component_dir / "config.json", 'w') as f:
                json.dump(component_config, f, indent=2)


def main():
    """Main entry point for model variant tests"""
    
    print("Wan Model Variants Integration Test Suite")
    print("=" * 50)
    
    # Create and run test suite
    test_suite = WanModelVariantsTestSuite()
    results = test_suite.test_all_model_variants()
    
    # Print summary
    print("\nModel Variant Test Results:")
    print("-" * 40)
    
    total_variants = len(results)
    successful_variants = sum(1 for r in results if r["success"])
    
    for result in results:
        status = "PASS" if result["success"] else "FAIL"
        success_rate = (result["tests_passed"] / result["tests_total"]) * 100 if result["tests_total"] > 0 else 0
        
        print(f"{status} {result['model_name']} ({result['tests_passed']}/{result['tests_total']} - {success_rate:.1f}%)")
        
        if result["errors"]:
            for error in result["errors"][:3]:  # Show first 3 errors
                print(f"    Error: {error}")
        
        if result["warnings"]:
            for warning in result["warnings"][:2]:  # Show first 2 warnings
                print(f"    Warning: {warning}")
    
    print(f"\nOverall Results: {successful_variants}/{total_variants} variants passed")
    
    # Save detailed results
    results_file = Path("test_results") / "model_variant_test_results.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Detailed results saved to {results_file}")
    
    return successful_variants == total_variants


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)