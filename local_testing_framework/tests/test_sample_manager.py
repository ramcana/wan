"""
Unit tests for SampleManager components
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ..sample_manager import (
    SampleManager, ConfigurationTemplateGenerator, EdgeCaseGenerator
)
from ..models.test_results import ValidationStatus
from ..models.configuration import TestConfiguration


class TestSampleManager(unittest.TestCase):
    """Test cases for SampleManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = SampleManager()
        # Override output directory for testing
        self.manager.output_dir = Path(self.temp_dir) / "test_samples"
        self.manager.output_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_realistic_video_prompts(self):
        """Test realistic video prompt generation"""
        prompts = self.manager.generate_realistic_video_prompts(count=5)
        
        self.assertEqual(len(prompts), 5)
        
        for prompt in prompts:
            self.assertIn("id", prompt)
            self.assertIn("category", prompt)
            self.assertIn("base_prompt", prompt)
            self.assertIn("full_prompt", prompt)
            self.assertIn("style_modifiers", prompt)
            self.assertIn("camera_angle", prompt)
            self.assertIn("estimated_complexity", prompt)
            self.assertIn("recommended_resolution", prompt)
            self.assertIn("expected_duration_range", prompt)
            
            # Check that full prompt contains base prompt
            self.assertIn(prompt["base_prompt"], prompt["full_prompt"])
            
            # Check complexity is valid
            self.assertIn(prompt["estimated_complexity"], ["low", "medium", "high"])
            
            # Check recommended resolution is valid
            self.assertIn(prompt["recommended_resolution"], ["720p", "1080p"])
    
    def test_create_sample_input_json(self):
        """Test sample input JSON creation"""
        prompt_data = {
            "id": "test_001",
            "category": "nature",
            "full_prompt": "A beautiful mountain landscape",
            "estimated_complexity": "medium"
        }
        
        sample_input = self.manager.create_sample_input_json(
            prompt_data, 
            resolution="720p",
            output_path="outputs/test.mp4"
        )
        
        # Check required fields
        self.assertEqual(sample_input["input"], "A beautiful mountain landscape")
        self.assertEqual(sample_input["resolution"], "720p")
        self.assertEqual(sample_input["output_path"], "outputs/test.mp4")
        
        # Check metadata
        self.assertIn("metadata", sample_input)
        self.assertEqual(sample_input["metadata"]["prompt_id"], "test_001")
        self.assertEqual(sample_input["metadata"]["category"], "nature")
        self.assertEqual(sample_input["metadata"]["complexity"], "medium")
    
    def test_create_sample_input_json_auto_output_path(self):
        """Test sample input JSON creation with automatic output path"""
        prompt_data = {
            "id": "test_002",
            "category": "urban",
            "full_prompt": "A busy city street"
        }
        
        sample_input = self.manager.create_sample_input_json(prompt_data, "1080p")
        
        # Check that output path was generated
        self.assertTrue(sample_input["output_path"].startswith("outputs/"))
        self.assertTrue(sample_input["output_path"].endswith(".mp4"))
        self.assertIn("test_002", sample_input["output_path"])
        self.assertIn("1080p", sample_input["output_path"])
    
    def test_generate_sample_input_files(self):
        """Test generation of multiple sample input files"""
        files = self.manager.generate_sample_input_files(
            count=2, 
            resolutions=["720p", "1080p"]
        )
        
        # Should generate 2 prompts × 2 resolutions = 4 files
        self.assertEqual(len(files), 4)
        
        # Check that all files exist
        for file_path in files:
            self.assertTrue(file_path.exists())
            
            # Check file content
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self.assertIn("input", data)
            self.assertIn("resolution", data)
            self.assertIn("output_path", data)
            self.assertIn("metadata", data)
    
    def test_estimate_prompt_complexity(self):
        """Test prompt complexity estimation"""
        # Test high complexity
        high_prompt = "A detailed and complex scene with intricate multiple dynamic elements"
        complexity = self.manager._estimate_prompt_complexity(high_prompt)
        self.assertEqual(complexity, "high")
        
        # Test low complexity
        low_prompt = "Simple basic scene"
        complexity = self.manager._estimate_prompt_complexity(low_prompt)
        self.assertEqual(complexity, "low")
        
        # Test medium complexity
        medium_prompt = "A normal landscape scene"
        complexity = self.manager._estimate_prompt_complexity(medium_prompt)
        self.assertEqual(complexity, "medium")
    
    def test_recommend_resolution(self):
        """Test resolution recommendation"""
        # Nature should recommend 1080p
        self.assertEqual(self.manager._recommend_resolution("nature"), "1080p")
        
        # Abstract should recommend 720p
        self.assertEqual(self.manager._recommend_resolution("abstract"), "720p")
        
        # Unknown category should default to 720p
        self.assertEqual(self.manager._recommend_resolution("unknown"), "720p")
    
    def test_estimate_generation_time(self):
        """Test generation time estimation"""
        time_estimate = self.manager._estimate_generation_time("nature", "A simple scene")
        
        self.assertIn("720p_minutes", time_estimate)
        self.assertIn("1080p_minutes", time_estimate)
        self.assertIn("complexity_factor", time_estimate)
        
        # 1080p should take longer than 720p
        self.assertGreater(time_estimate["1080p_minutes"], time_estimate["720p_minutes"])


class TestConfigurationTemplateGenerator(unittest.TestCase):
    """Test cases for ConfigurationTemplateGenerator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ConfigurationTemplateGenerator()
        # Override template directory for testing
        self.generator.template_dir = Path(self.temp_dir) / "config_templates"
        self.generator.template_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_config_json_template_balanced(self):
        """Test balanced configuration template generation"""
        config = self.generator.generate_config_json_template("balanced")
        
        # Check required sections
        required_sections = ["system", "directories", "optimization", "performance"]
        for section in required_sections:
            self.assertIn(section, config)
        
        # Check optimization settings for balanced preset
        optimization = config["optimization"]
        self.assertTrue(optimization["enable_attention_slicing"])
        self.assertTrue(optimization["enable_vae_tiling"])
        self.assertTrue(optimization["use_fp16"])
        self.assertFalse(optimization["torch_compile"])  # Balanced doesn't use torch_compile
    
    def test_generate_config_json_template_performance(self):
        """Test performance configuration template generation"""
        config = self.generator.generate_config_json_template("performance")
        
        optimization = config["optimization"]
        self.assertTrue(optimization["enable_attention_slicing"])
        self.assertTrue(optimization["enable_vae_tiling"])
        self.assertTrue(optimization["use_fp16"])
        self.assertTrue(optimization["torch_compile"])  # Performance uses torch_compile
        self.assertTrue(optimization["enable_xformers"])
    
    def test_generate_config_json_template_quality(self):
        """Test quality configuration template generation"""
        config = self.generator.generate_config_json_template("quality")
        
        optimization = config["optimization"]
        self.assertFalse(optimization["enable_attention_slicing"])  # Quality disables optimizations
        self.assertFalse(optimization["enable_vae_tiling"])
        self.assertFalse(optimization["use_fp16"])
        self.assertFalse(optimization["torch_compile"])
    
    def test_generate_env_template_with_examples(self):
        """Test environment template generation with examples"""
        env_template = self.generator.generate_env_template(include_examples=True)
        
        # Check required variables
        self.assertIn("HF_TOKEN", env_template)
        self.assertIn("CUDA_VISIBLE_DEVICES", env_template)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF", env_template)
        
        # Check that examples are included
        self.assertEqual(env_template["HF_TOKEN"], "hf_your_token_here")
        self.assertEqual(env_template["CUDA_VISIBLE_DEVICES"], "0")
        
        # Check that comments are included
        comment_keys = [key for key in env_template.keys() if key.startswith("#")]
        self.assertGreater(len(comment_keys), 0)
    
    def test_generate_env_template_without_examples(self):
        """Test environment template generation without examples"""
        env_template = self.generator.generate_env_template(include_examples=False)
        
        # Check required variables
        self.assertIn("HF_TOKEN", env_template)
        self.assertIn("CUDA_VISIBLE_DEVICES", env_template)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF", env_template)
        
        # Check that values are empty
        self.assertEqual(env_template["HF_TOKEN"], "")
        self.assertEqual(env_template["CUDA_VISIBLE_DEVICES"], "0")
        
        # Should have fewer keys (no comments)
        self.assertEqual(len(env_template), 3)
    
    def test_save_config_template(self):
        """Test saving configuration template to file"""
        config = {"test": "data"}
        file_path = self.generator.save_config_template(config, "test_config.json")
        
        self.assertTrue(file_path.exists())
        
        # Check file content
        with open(file_path, 'r') as f:
            loaded_config = json.load(f)
        
        self.assertEqual(loaded_config, config)
    
    def test_save_env_template(self):
        """Test saving environment template to file"""
        env_vars = {
            "VAR1": "value1",
            "# Comment": "",
            "VAR2": "value2"
        }
        
        file_path = self.generator.save_env_template(env_vars, "test_env.txt")
        
        self.assertTrue(file_path.exists())
        
        # Check file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        self.assertIn("VAR1=value1", content)
        self.assertIn("VAR2=value2", content)
        self.assertIn("# Comment", content)
    
    def test_validate_config_template_success(self):
        """Test successful configuration template validation"""
        valid_config = {
            "system": {"device": "cuda", "gpu_memory_fraction": 0.9},
            "directories": {"models": "models/", "outputs": "outputs/", "cache": "cache/", "logs": "logs/"},
            "optimization": {"enable_attention_slicing": True},
            "performance": {"batch_size": 1}
        }
        
        result = self.generator.validate_config_template(valid_config)
        
        self.assertEqual(result.status, ValidationStatus.PASSED)
        self.assertEqual(result.component, "config_template")
    
    def test_validate_config_template_missing_sections(self):
        """Test configuration template validation with missing sections"""
        invalid_config = {
            "system": {"device": "cuda"}
            # Missing directories, optimization, performance
        }
        
        result = self.generator.validate_config_template(invalid_config)
        
        self.assertEqual(result.status, ValidationStatus.FAILED)
        self.assertIn("Missing required sections", result.message)
        self.assertTrue(len(result.remediation_steps) > 0)
    
    def test_validate_config_template_validation_issues(self):
        """Test configuration template validation with field issues"""
        config_with_issues = {
            "system": {"gpu_memory_fraction": 1.5},  # Invalid value > 1.0
            "directories": {"models": "models/"},  # Missing required directories
            "optimization": {},
            "performance": {}
        }
        
        result = self.generator.validate_config_template(config_with_issues)
        
        self.assertEqual(result.status, ValidationStatus.WARNING)
        self.assertIn("validation issues", result.message)


class TestEdgeCaseGenerator(unittest.TestCase):
    """Test cases for EdgeCaseGenerator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = EdgeCaseGenerator()
        # Override edge case directory for testing
        self.generator.edge_case_dir = Path(self.temp_dir) / "edge_cases"
        self.generator.edge_case_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_edge_case_prompts(self):
        """Test edge case prompt generation"""
        edge_cases = self.generator.generate_edge_case_prompts()
        
        self.assertGreater(len(edge_cases), 0)
        
        # Check for different categories
        categories = {case["category"] for case in edge_cases}
        expected_categories = {"empty", "minimal", "length", "special_characters", "unicode", "security"}
        
        # Should have at least some of the expected categories
        self.assertTrue(len(categories.intersection(expected_categories)) > 0)
        
        # Check structure
        for case in edge_cases:
            self.assertIn("id", case)
            self.assertIn("category", case)
            self.assertIn("prompt", case)
            self.assertIn("expected_behavior", case)
            self.assertIn("test_type", case)
    
    def test_create_invalid_input_samples(self):
        """Test invalid input sample creation"""
        invalid_samples = self.generator.create_invalid_input_samples()
        
        self.assertGreater(len(invalid_samples), 0)
        
        # Check for different error types
        error_types = {sample["error_type"] for sample in invalid_samples}
        expected_types = {"json_syntax_error", "missing_required_field", "invalid_field_value"}
        
        self.assertTrue(len(error_types.intersection(expected_types)) > 0)
        
        # Check structure
        for sample in invalid_samples:
            self.assertIn("id", sample)
            self.assertIn("content", sample)
            self.assertIn("error_type", sample)
            self.assertIn("expected_behavior", sample)
    
    def test_create_multi_resolution_test_suite(self):
        """Test multi-resolution test suite creation"""
        test_suite = self.generator.create_multi_resolution_test_suite()
        
        # Check that all resolution categories are present
        self.assertIn("720p", test_suite)
        self.assertIn("1080p", test_suite)
        self.assertIn("unsupported", test_suite)
        
        # Check that each category has tests
        self.assertGreater(len(test_suite["720p"]), 0)
        self.assertGreater(len(test_suite["1080p"]), 0)
        self.assertGreater(len(test_suite["unsupported"]), 0)
        
        # Check 720p test structure
        for test in test_suite["720p"]:
            self.assertEqual(test["resolution"], "720p")
            self.assertIn("expected_duration_minutes", test)
            self.assertIn("complexity", test)
        
        # Check unsupported resolution tests
        for test in test_suite["unsupported"]:
            self.assertIn("expected_behavior", test)
            self.assertEqual(test["test_type"], "unsupported_resolution")
    
    def test_create_stress_test_scenarios(self):
        """Test stress test scenario creation"""
        stress_tests = self.generator.create_stress_test_scenarios()
        
        self.assertGreater(len(stress_tests), 0)
        
        # Check for different stress test types
        test_types = {test["type"] for test in stress_tests}
        expected_types = {"concurrent_load", "memory_stress", "endurance", "rapid_requests"}
        
        self.assertTrue(len(test_types.intersection(expected_types)) > 0)
        
        # Check structure
        for test in stress_tests:
            self.assertIn("id", test)
            self.assertIn("type", test)
            self.assertIn("description", test)
            self.assertIn("parameters", test)
            self.assertIn("expected_behavior", test)
            self.assertIn("success_criteria", test)
    
    def test_save_edge_case_samples(self):
        """Test saving edge case samples to files"""
        saved_files = self.generator.save_edge_case_samples()
        
        self.assertGreater(len(saved_files), 0)
        
        # Check that all files exist
        for file_path in saved_files:
            self.assertTrue(file_path.exists())
        
        # Check that different types of files were created
        filenames = [f.name for f in saved_files]
        
        # Should have edge case files
        edge_case_files = [f for f in filenames if f.startswith("edge_case_")]
        self.assertGreater(len(edge_case_files), 0)
        
        # Should have invalid sample files
        invalid_files = [f for f in filenames if f.startswith("invalid_")]
        self.assertGreater(len(invalid_files), 0)
        
        # Should have resolution test files
        resolution_files = [f for f in filenames if f.startswith("resolution_tests_")]
        self.assertGreater(len(resolution_files), 0)
        
        # Should have stress test file
        stress_files = [f for f in filenames if f == "stress_test_scenarios.json"]
        self.assertEqual(len(stress_files), 1)


if __name__ == '__main__':
    unittest.main()