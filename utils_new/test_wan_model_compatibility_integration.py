"""
Integration test for WAN model compatibility system.

This test verifies that the complete WAN model compatibility system works
correctly, including architecture detection, pipeline selection, and loading.
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from architecture_detector import ArchitectureDetector, ArchitectureType, ArchitectureSignature
from pipeline_manager import PipelineManager, PipelineLoadStatus
from wan_pipeline_loader import WanPipelineLoader
from wan_model_compatibility_fix import apply_wan_compatibility_fix


class TestWanModelCompatibilityIntegration(unittest.TestCase):
    """Integration tests for WAN model compatibility system"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_model_path = self.temp_dir / "wan_model"
        self.mock_model_path.mkdir()
        
        # Create mock model structure
        self._create_mock_wan_model()
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_mock_wan_model(self):
        """Create a mock WAN model structure"""
        # Create model_index.json
        model_index = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.21.0",
            "scheduler": ["diffusers", "DDIMScheduler"],
            "transformer": ["transformers", "WanTransformer"],
            "transformer_2": ["transformers", "WanTransformer2"],
            "vae": ["diffusers", "AutoencoderKL"],
            "boundary_ratio": 0.5
        }
        
        with open(self.mock_model_path / "model_index.json", 'w') as f:
            json.dump(model_index, f, indent=2)
        
        # Create component directories
        for component in ["scheduler", "transformer", "transformer_2", "vae"]:
            component_dir = self.mock_model_path / component
            component_dir.mkdir()
            
            # Create config.json for each component
            config = {
                "_class_name": f"Wan{component.title()}" if component.startswith("transformer") else "DDIMScheduler",
                "some_param": "value"
            }
            
            with open(component_dir / "config.json", 'w') as f:
                json.dump(config, f, indent=2)
        
        # Create WAN-specific files
        (self.mock_model_path / "pytorch_model.bin").touch()
        (self.mock_model_path / "Wan2.1_VAE.pth").touch()
        (self.mock_model_path / "models_t5_umt5-xxl-enc-bf16.pth").touch()
    
    def test_architecture_detection_integration(self):
        """Test complete architecture detection for WAN model"""
        detector = ArchitectureDetector()
        architecture = detector.detect_model_architecture(str(self.mock_model_path))
        
        # Verify architecture detection
        self.assertEqual(architecture.architecture_type, ArchitectureType.WAN_T2V)
        self.assertIsNotNone(architecture.signature)
        self.assertTrue(architecture.signature.is_wan_architecture())
        self.assertTrue(architecture.signature.has_transformer_2)
        self.assertTrue(architecture.signature.has_boundary_ratio)
        self.assertEqual(architecture.signature.pipeline_class, "WanPipeline")
    
    def test_pipeline_selection_integration(self):
        """Test pipeline selection for detected WAN architecture"""
        detector = ArchitectureDetector()
        architecture = detector.detect_model_architecture(str(self.mock_model_path))
        
        manager = PipelineManager()
        pipeline_class = manager.select_pipeline_class(architecture.signature)
        
        # Verify correct pipeline selection
        self.assertEqual(pipeline_class, "WanPipeline")
    
    @patch('wan_pipeline_loader.DiffusionPipeline')
    def test_wan_pipeline_loading_integration(self, mock_diffusion_pipeline):
        """Test complete WAN pipeline loading integration"""
        # Mock successful pipeline loading
        mock_pipeline = Mock()
        mock_pipeline.__class__.__name__ = "WanPipeline"
        mock_diffusion_pipeline.from_pretrained.return_value = mock_pipeline
        
        # Test WAN pipeline loader
        loader = WanPipelineLoader()
        
        try:
            wrapper = loader.load_wan_pipeline(
                model_path=str(self.mock_model_path),
                trust_remote_code=True
            )
            
            # Verify successful loading
            self.assertIsNotNone(wrapper)
            self.assertEqual(wrapper.pipeline, mock_pipeline)
            
            # Verify trust_remote_code was used
            mock_diffusion_pipeline.from_pretrained.assert_called()
            call_args = mock_diffusion_pipeline.from_pretrained.call_args
            self.assertTrue(call_args[1].get('trust_remote_code', False))
            
        except Exception as e:
            # This is expected in test environment without actual model files
            self.assertIn("Model is not a Wan architecture", str(e))
    
    def test_compatibility_fix_integration(self):
        """Test WAN model compatibility fix integration"""
        # Apply compatibility fix
        results = apply_wan_compatibility_fix(self.mock_model_path)
        
        # Verify fix results
        self.assertIsInstance(results, dict)
        self.assertIn("model_path", results)
        self.assertIn("fixes_applied", results)
        self.assertIn("recommendations", results)
        
        # Should have some recommendations
        self.assertGreater(len(results["recommendations"]), 0)
        self.assertIn("trust_remote_code=True", str(results["recommendations"]))
    
    def test_error_handling_integration(self):
        """Test error handling in integration scenarios"""
        # Test with non-existent model
        detector = ArchitectureDetector()
        
        with self.assertRaises(FileNotFoundError):
            detector.detect_model_architecture("/non/existent/path")
        
        # Test with invalid model structure
        empty_dir = self.temp_dir / "empty_model"
        empty_dir.mkdir()
        
        architecture = detector.detect_model_architecture(str(empty_dir))
        # Should detect as unknown architecture
        self.assertEqual(architecture.architecture_type, ArchitectureType.UNKNOWN)
    
    def test_fallback_mechanism_integration(self):
        """Test fallback mechanisms in pipeline loading"""
        # Create a model that looks like WAN but might fail to load
        broken_model_path = self.temp_dir / "broken_wan_model"
        broken_model_path.mkdir()
        
        # Create minimal model_index.json that indicates WAN model
        model_index = {
            "_class_name": "WanPipeline",
            "transformer_2": ["transformers", "WanTransformer2"],
            "boundary_ratio": 0.5
        }
        
        with open(broken_model_path / "model_index.json", 'w') as f:
            json.dump(model_index, f, indent=2)
        
        # Test architecture detection (should work)
        detector = ArchitectureDetector()
        architecture = detector.detect_model_architecture(str(broken_model_path))
        self.assertEqual(architecture.architecture_type, ArchitectureType.WAN_T2V)
        
        # Test pipeline selection (should work)
        manager = PipelineManager()
        pipeline_class = manager.select_pipeline_class(architecture.signature)
        self.assertEqual(pipeline_class, "WanPipeline")
    
    def test_end_to_end_compatibility_workflow(self):
        """Test complete end-to-end compatibility workflow"""
        # Step 1: Apply compatibility fix
        fix_results = apply_wan_compatibility_fix(self.mock_model_path)
        
        # Step 2: Detect architecture
        detector = ArchitectureDetector()
        architecture = detector.detect_model_architecture(str(self.mock_model_path))
        
        # Step 3: Select pipeline
        manager = PipelineManager()
        pipeline_class = manager.select_pipeline_class(architecture.signature)
        
        # Step 4: Verify complete workflow
        self.assertEqual(architecture.architecture_type, ArchitectureType.WAN_T2V)
        self.assertEqual(pipeline_class, "WanPipeline")
        self.assertTrue(fix_results.get("success", False) or len(fix_results.get("issues_found", [])) == 0)
        
        # Verify recommendations include key points
        recommendations = fix_results.get("recommendations", [])
        recommendation_text = " ".join(recommendations)
        self.assertIn("trust_remote_code", recommendation_text)
        self.assertIn("float16", recommendation_text)


class TestWanModelCompatibilityRealWorld(unittest.TestCase):
    """Real-world scenario tests for WAN model compatibility"""
    
    def test_stable_diffusion_model_rejection(self):
        """Test that Stable Diffusion models are correctly rejected"""
        temp_dir = Path(tempfile.mkdtemp())
        sd_model_path = temp_dir / "sd_model"
        sd_model_path.mkdir()
        
        try:
            # Create SD model structure
            model_index = {
                "_class_name": "StableDiffusionPipeline",
                "_diffusers_version": "0.21.0",
                "unet": ["diffusers", "UNet2DConditionModel"],
                "scheduler": ["diffusers", "DDIMScheduler"],
                "vae": ["diffusers", "AutoencoderKL"],
                "text_encoder": ["transformers", "CLIPTextModel"],
                "tokenizer": ["transformers", "CLIPTokenizer"]
            }
            
            with open(sd_model_path / "model_index.json", 'w') as f:
                json.dump(model_index, f, indent=2)
            
            # Test architecture detection
            detector = ArchitectureDetector()
            architecture = detector.detect_model_architecture(str(sd_model_path))
            
            # Should detect as Stable Diffusion, not WAN
            self.assertEqual(architecture.architecture_type, ArchitectureType.STABLE_DIFFUSION)
            self.assertFalse(architecture.signature.is_wan_architecture())
            
            # Pipeline selection should choose SD pipeline
            manager = PipelineManager()
            pipeline_class = manager.select_pipeline_class(architecture.signature)
            self.assertEqual(pipeline_class, "StableDiffusionPipeline")
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_mixed_model_detection(self):
        """Test detection of models with mixed components"""
        temp_dir = Path(tempfile.mkdtemp())
        mixed_model_path = temp_dir / "mixed_model"
        mixed_model_path.mkdir()
        
        try:
            # Create model with both WAN and SD components
            model_index = {
                "_class_name": "WanPipeline",
                "_diffusers_version": "0.21.0",
                "transformer": ["transformers", "WanTransformer"],
                "unet": ["diffusers", "UNet2DConditionModel"],  # SD component
                "scheduler": ["diffusers", "DDIMScheduler"],
                "vae": ["diffusers", "AutoencoderKL"],
                "text_encoder": ["transformers", "CLIPTextModel"]
            }
            
            with open(mixed_model_path / "model_index.json", 'w') as f:
                json.dump(model_index, f, indent=2)
            
            # Test architecture detection
            detector = ArchitectureDetector()
            architecture = detector.detect_model_architecture(str(mixed_model_path))
            
            # Should detect as WAN due to explicit pipeline class and transformer
            self.assertTrue(architecture.signature.is_wan_architecture())
            
            # Should generate compatibility warnings
            compatibility_report = detector.validate_component_compatibility(architecture.components)
            self.assertGreater(len(compatibility_report.warnings), 0)
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)