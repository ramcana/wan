#!/usr/bin/env python3
"""
Integration tests for LoRA application in generation pipeline
Tests the implementation of task 7: Implement LoRA application in generation pipeline
"""

import unittest
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import torch
import numpy as np

# Import the modules to test
from utils import (
    VideoGenerationEngine, 
    GenerationTask,
    get_generation_engine,
    get_lora_manager,
    generate_video,
    generate_t2v_video,
    generate_i2v_video,
    generate_ti2v_video
)


class TestLoRAGenerationPipeline(unittest.TestCase):
    """Test LoRA application in the generation pipeline"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "models"
        self.loras_dir = Path(self.temp_dir) / "loras"
        self.outputs_dir = Path(self.temp_dir) / "outputs"
        
        for dir_path in [self.models_dir, self.loras_dir, self.outputs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create test config
        self.config = {
            "directories": {
                "models_directory": str(self.models_dir),
                "loras_directory": str(self.loras_dir),
                "outputs_directory": str(self.outputs_dir)
            },
            "optimization": {
                "max_vram_usage_gb": 12
            },
            "lora_max_file_size_mb": 2048
        }
        
        # Create test config file
        self.config_path = Path(self.temp_dir) / "test_config.json"
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)
        
        # Create mock LoRA files
        self.create_mock_lora_files()
        
        # Initialize generation engine with test config
        self.engine = VideoGenerationEngine(str(self.config_path))
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_lora_files(self):
        """Create mock LoRA files for testing"""
        # Create mock LoRA weights
        mock_lora_weights = {
            "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora_down.weight": torch.randn(64, 320),
            "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora_up.weight": torch.randn(320, 64),
            "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.lora_down.weight": torch.randn(64, 320),
            "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.lora_up.weight": torch.randn(320, 64),
        }
        
        # Save test LoRA files
        test_loras = ["anime_style", "detail_enhancer", "cinematic_lighting"]
        for lora_name in test_loras:
            lora_path = self.loras_dir / f"{lora_name}.safetensors"
            # For testing, we'll just create empty files since we'll mock the loading
            lora_path.touch()
    
    def create_test_image(self, width=512, height=512):
        """Create a test PIL Image"""
        # Create a simple test image
        image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        return Image.fromarray(image_array)
    
    @patch('utils.VideoGenerationEngine._get_pipeline')
    @patch('utils.get_lora_manager')
    def test_t2v_generation_with_loras(self, mock_get_lora_manager, mock_get_pipeline):
        """Test T2V generation with LoRA application"""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value = Mock(frames=[[Mock() for _ in range(16)]])
        mock_get_pipeline.return_value = mock_pipeline
        
        # Mock LoRA manager
        mock_lora_manager = Mock()
        mock_lora_manager.apply_lora.return_value = mock_pipeline
        mock_lora_manager.estimate_memory_impact.return_value = {
            "total_memory_mb": 500.0,
            "vram_impact_mb": 400.0,
            "recommendations": []
        }
        mock_lora_manager.get_fallback_prompt_enhancement.return_value = "enhanced prompt"
        mock_get_lora_manager.return_value = mock_lora_manager
        
        # Test LoRA selections
        selected_loras = {
            "anime_style": 0.8,
            "detail_enhancer": 0.6
        }
        
        # Generate video with LoRAs
        result = self.engine.generate_t2v(
            prompt="A beautiful landscape",
            resolution="1280x720",
            num_inference_steps=20,
            selected_loras=selected_loras
        )
        
        # Verify results
        self.assertIn("frames", result)
        self.assertIn("metadata", result)
        
        metadata = result["metadata"]
        self.assertEqual(metadata["model_type"], "t2v-A14B")
        self.assertIn("lora_info", metadata)
        self.assertEqual(metadata["lora_info"]["selected_loras"], selected_loras)
        self.assertIn("timing", metadata)
        self.assertIn("lora_load_time_seconds", metadata["timing"])
        
        # Verify LoRA manager was called
        self.assertTrue(mock_lora_manager.apply_lora.called)
        self.assertEqual(mock_lora_manager.apply_lora.call_count, 2)  # Two LoRAs

        assert True  # TODO: Add proper assertion
    
    @patch('utils.VideoGenerationEngine._get_pipeline')
    @patch('utils.get_lora_manager')
    def test_i2v_generation_with_loras(self, mock_get_lora_manager, mock_get_pipeline):
        """Test I2V generation with LoRA application"""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value = Mock(frames=[[Mock() for _ in range(16)]])
        mock_get_pipeline.return_value = mock_pipeline
        
        # Mock LoRA manager
        mock_lora_manager = Mock()
        mock_lora_manager.apply_lora.return_value = mock_pipeline
        mock_lora_manager.estimate_memory_impact.return_value = {
            "total_memory_mb": 300.0,
            "vram_impact_mb": 240.0,
            "recommendations": []
        }
        mock_get_lora_manager.return_value = mock_lora_manager
        
        # Create test image
        test_image = self.create_test_image()
        
        # Test LoRA selections
        selected_loras = {
            "cinematic_lighting": 1.0
        }
        
        # Generate video with LoRAs
        result = self.engine.generate_i2v(
            prompt="Animate this image",
            image=test_image,
            resolution="1280x720",
            num_inference_steps=20,
            selected_loras=selected_loras
        )
        
        # Verify results
        self.assertIn("frames", result)
        self.assertIn("metadata", result)
        
        metadata = result["metadata"]
        self.assertEqual(metadata["model_type"], "i2v-A14B")
        self.assertIn("lora_info", metadata)
        self.assertEqual(metadata["lora_info"]["selected_loras"], selected_loras)
        
        # Verify LoRA manager was called
        self.assertTrue(mock_lora_manager.apply_lora.called)

        assert True  # TODO: Add proper assertion
    
    @patch('utils.VideoGenerationEngine._get_pipeline')
    @patch('utils.get_lora_manager')
    def test_ti2v_generation_with_loras(self, mock_get_lora_manager, mock_get_pipeline):
        """Test TI2V generation with LoRA application"""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value = Mock(frames=[[Mock() for _ in range(16)]])
        mock_get_pipeline.return_value = mock_pipeline
        
        # Mock LoRA manager
        mock_lora_manager = Mock()
        mock_lora_manager.apply_lora.return_value = mock_pipeline
        mock_lora_manager.estimate_memory_impact.return_value = {
            "total_memory_mb": 800.0,
            "vram_impact_mb": 640.0,
            "recommendations": ["High memory usage detected"]
        }
        mock_get_lora_manager.return_value = mock_lora_manager
        
        # Create test image
        test_image = self.create_test_image()
        
        # Test multiple LoRA selections
        selected_loras = {
            "anime_style": 0.7,
            "detail_enhancer": 0.9,
            "cinematic_lighting": 0.5
        }
        
        # Generate video with LoRAs
        result = self.engine.generate_ti2v(
            prompt="Create cinematic video",
            image=test_image,
            resolution="1280x720",
            num_inference_steps=20,
            selected_loras=selected_loras
        )
        
        # Verify results
        self.assertIn("frames", result)
        self.assertIn("metadata", result)
        
        metadata = result["metadata"]
        self.assertEqual(metadata["model_type"], "ti2v-5B")
        self.assertIn("lora_info", metadata)
        self.assertEqual(metadata["lora_info"]["selected_loras"], selected_loras)
        self.assertEqual(metadata["lora_info"]["successful_applications"], 3)
        
        # Verify LoRA manager was called for each LoRA
        self.assertEqual(mock_lora_manager.apply_lora.call_count, 3)

        assert True  # TODO: Add proper assertion
    
    @patch('utils.VideoGenerationEngine._get_pipeline')
    @patch('utils.get_lora_manager')
    def test_lora_loading_failure_fallback(self, mock_get_lora_manager, mock_get_pipeline):
        """Test fallback mechanism when LoRA loading fails"""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value = Mock(frames=[[Mock() for _ in range(16)]])
        mock_get_pipeline.return_value = mock_pipeline
        
        # Mock LoRA manager with failure
        mock_lora_manager = Mock()
        mock_lora_manager.apply_lora.side_effect = Exception("LoRA loading failed")
        mock_lora_manager.get_fallback_prompt_enhancement.return_value = "high quality, detailed"
        mock_get_lora_manager.return_value = mock_lora_manager
        
        # Test LoRA selections
        selected_loras = {
            "failing_lora": 1.0
        }
        
        # Generate video with failing LoRA
        result = self.engine.generate_t2v(
            prompt="Test prompt",
            resolution="1280x720",
            num_inference_steps=20,
            selected_loras=selected_loras
        )
        
        # Verify results
        self.assertIn("frames", result)
        self.assertIn("metadata", result)
        
        metadata = result["metadata"]
        self.assertIn("lora_info", metadata)
        
        # Check that fallback was applied
        lora_metadata = metadata["lora_info"]["lora_metadata"]
        self.assertIn("fallback_enhancements", lora_metadata)
        self.assertTrue(lora_metadata.get("loading_failed", False))
        
        # Verify fallback enhancement was called
        self.assertTrue(mock_lora_manager.get_fallback_prompt_enhancement.called)

        assert True  # TODO: Add proper assertion
    
    @patch('utils.VideoGenerationEngine._get_pipeline')
    @patch('utils.get_lora_manager')
    def test_lora_performance_metrics(self, mock_get_lora_manager, mock_get_pipeline):
        """Test LoRA performance metrics tracking"""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value = Mock(frames=[[Mock() for _ in range(16)]])
        mock_get_pipeline.return_value = mock_pipeline
        
        # Mock LoRA manager with timing simulation
        mock_lora_manager = Mock()
        
        def mock_apply_lora(pipeline, lora_name, strength):
            time.sleep(0.1)  # Simulate load time
            return pipeline
        
        mock_lora_manager.apply_lora.side_effect = mock_apply_lora
        mock_lora_manager.estimate_memory_impact.return_value = {
            "total_memory_mb": 600.0,
            "vram_impact_mb": 480.0,
            "recommendations": []
        }
        mock_get_lora_manager.return_value = mock_lora_manager
        
        # Test LoRA selections
        selected_loras = {
            "lora1": 0.8,
            "lora2": 0.6
        }
        
        # Generate video and measure time
        start_time = time.time()
        result = self.engine.generate_t2v(
            prompt="Test prompt",
            resolution="1280x720",
            num_inference_steps=20,
            selected_loras=selected_loras
        )
        total_time = time.time() - start_time
        
        # Verify timing metrics
        metadata = result["metadata"]
        timing = metadata["timing"]
        
        self.assertIn("total_time_seconds", timing)
        self.assertIn("lora_load_time_seconds", timing)
        self.assertIn("generation_time_seconds", timing)
        
        # LoRA load time should be > 0 due to our mock delay
        self.assertGreater(timing["lora_load_time_seconds"], 0)
        
        # Total time should be reasonable
        self.assertLess(timing["total_time_seconds"], total_time + 1)  # Allow some margin

        assert True  # TODO: Add proper assertion
    
    @patch('utils.VideoGenerationEngine._get_pipeline')
    @patch('utils.get_lora_manager')
    def test_multiple_lora_blending(self, mock_get_lora_manager, mock_get_pipeline):
        """Test multiple LoRA blending with individual strength values"""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value = Mock(frames=[[Mock() for _ in range(16)]])
        mock_get_pipeline.return_value = mock_pipeline
        
        # Mock LoRA manager
        mock_lora_manager = Mock()
        mock_lora_manager.apply_lora.return_value = mock_pipeline
        mock_lora_manager.estimate_memory_impact.return_value = {
            "total_memory_mb": 1200.0,
            "vram_impact_mb": 960.0,
            "recommendations": []
        }
        mock_get_lora_manager.return_value = mock_lora_manager
        
        # Test maximum LoRA selections (5 LoRAs as per requirements)
        selected_loras = {
            "style_lora": 1.0,
            "quality_lora": 0.8,
            "lighting_lora": 0.6,
            "detail_lora": 0.4,
            "color_lora": 0.2
        }
        
        # Generate video with multiple LoRAs
        result = self.engine.generate_t2v(
            prompt="Complex scene",
            resolution="1280x720",
            num_inference_steps=20,
            selected_loras=selected_loras
        )
        
        # Verify all LoRAs were processed
        metadata = result["metadata"]
        lora_info = metadata["lora_info"]
        
        self.assertEqual(len(lora_info["selected_loras"]), 5)
        self.assertEqual(lora_info["successful_applications"], 5)
        
        # Verify each LoRA was applied with correct strength
        for lora_name, expected_strength in selected_loras.items():
            # Check that apply_lora was called with correct parameters
            found_call = False
            for call in mock_lora_manager.apply_lora.call_args_list:
                if call[0][1] == lora_name and call[0][2] == expected_strength:
                    found_call = True
                    break
            self.assertTrue(found_call, f"LoRA {lora_name} not applied with strength {expected_strength}")

        assert True  # TODO: Add proper assertion
    
    def test_generation_task_lora_integration(self):
        """Test GenerationTask integration with LoRA selections"""
        # Create a generation task with LoRA selections
        task = GenerationTask(
            model_type="t2v-A14B",
            prompt="Test prompt",
            resolution="1280x720",
            steps=20,
            selected_loras={"test_lora": 0.8, "another_lora": 0.6}
        )
        
        # Test LoRA validation
        is_valid, errors = task.validate_lora_selections()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Test adding LoRA selection
        success = task.add_lora_selection("new_lora", 1.0)
        self.assertTrue(success)
        self.assertIn("new_lora", task.selected_loras)
        self.assertEqual(task.selected_loras["new_lora"], 1.0)
        
        # Test removing LoRA selection
        success = task.remove_lora_selection("test_lora")
        self.assertTrue(success)
        self.assertNotIn("test_lora", task.selected_loras)
        
        # Test LoRA summary
        summary = task.get_lora_summary()
        self.assertIn("selected_count", summary)
        self.assertIn("selected_loras", summary)
        self.assertIn("is_valid", summary)
        
        # Test updating LoRA metrics
        task.update_lora_metrics(500.0, 2.5, {"applied": ["another_lora", "new_lora"]})
        self.assertEqual(task.lora_memory_usage, 500.0)
        self.assertEqual(task.lora_load_time, 2.5)
        self.assertIn("applied", task.lora_metadata)

        assert True  # TODO: Add proper assertion
    
    @patch('utils.get_generation_engine')
    def test_convenience_functions_with_loras(self, mock_get_engine):
        """Test convenience functions pass through LoRA parameters"""
        # Mock engine
        mock_engine = Mock()
        mock_engine.generate_video.return_value = {"frames": [], "metadata": {}}
        mock_engine.generate_t2v.return_value = {"frames": [], "metadata": {}}
        mock_engine.generate_i2v.return_value = {"frames": [], "metadata": {}}
        mock_engine.generate_ti2v.return_value = {"frames": [], "metadata": {}}
        mock_get_engine.return_value = mock_engine
        
        # Test LoRA selections
        selected_loras = {"test_lora": 0.8}
        
        # Test generate_video function
        generate_video(
            model_type="t2v",
            prompt="Test",
            selected_loras=selected_loras
        )
        
        # Verify LoRA parameter was passed through
        call_kwargs = mock_engine.generate_video.call_args[1]
        self.assertIn("selected_loras", call_kwargs)
        self.assertEqual(call_kwargs["selected_loras"], selected_loras)
        
        # Test generate_t2v_video function
        generate_t2v_video(
            prompt="Test T2V",
            selected_loras=selected_loras
        )
        
        call_kwargs = mock_engine.generate_t2v.call_args[1]
        self.assertIn("selected_loras", call_kwargs)
        self.assertEqual(call_kwargs["selected_loras"], selected_loras)


        assert True  # TODO: Add proper assertion

class TestLoRAGenerationErrorHandling(unittest.TestCase):
    """Test error handling in LoRA generation pipeline"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "directories": {
                "models_directory": str(Path(self.temp_dir) / "models"),
                "loras_directory": str(Path(self.temp_dir) / "loras"),
                "outputs_directory": str(Path(self.temp_dir) / "outputs")
            }
        }
        
        # Create directories
        for dir_path in self.config["directories"].values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('utils.VideoGenerationEngine._get_pipeline')
    @patch('utils.get_lora_manager')
    def test_vram_error_handling(self, mock_get_lora_manager, mock_get_pipeline):
        """Test VRAM error handling during LoRA application"""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_get_pipeline.return_value = mock_pipeline
        
        # Mock LoRA manager with VRAM error
        mock_lora_manager = Mock()
        mock_lora_manager.apply_lora.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")
        mock_lora_manager.get_fallback_prompt_enhancement.return_value = "fallback enhancement"
        mock_get_lora_manager.return_value = mock_lora_manager
        
        engine = VideoGenerationEngine()
        
        # Test that VRAM error is handled gracefully
        selected_loras = {"memory_heavy_lora": 1.0}
        
        # Should not raise exception, should use fallback
        result = engine.generate_t2v(
            prompt="Test prompt",
            resolution="1280x720",
            num_inference_steps=20,
            selected_loras=selected_loras
        )
        
        # Verify fallback was used
        metadata = result["metadata"]
        self.assertIn("lora_info", metadata)
        lora_metadata = metadata["lora_info"]["lora_metadata"]
        self.assertTrue(lora_metadata.get("loading_failed", False))

        assert True  # TODO: Add proper assertion
    
    def test_invalid_lora_selections(self):
        """Test validation of invalid LoRA selections"""
        # Test invalid strength values
        task = GenerationTask(
            model_type="t2v-A14B",
            prompt="Test",
            selected_loras={"invalid_lora": 3.0}  # Strength > 2.0
        )
        
        is_valid, errors = task.validate_lora_selections()
        self.assertFalse(is_valid)
        self.assertTrue(any("strength" in error.lower() for error in errors))
        
        # Test too many LoRAs
        task = GenerationTask(
            model_type="t2v-A14B",
            prompt="Test",
            selected_loras={f"lora_{i}": 1.0 for i in range(6)}  # 6 LoRAs > max 5
        )
        
        is_valid, errors = task.validate_lora_selections()
        self.assertFalse(is_valid)
        self.assertTrue(any("too many" in error.lower() for error in errors))


        assert True  # TODO: Add proper assertion

if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)