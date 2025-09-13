"""
Phase 1 WAN Models Test Suite
Comprehensive testing for T2V, I2V, and TI2V functionality
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import json
from fastapi.testclient import TestClient
from PIL import Image
import io

# Import the FastAPI app
import sys
sys.path.append(str(Path(__file__).parent.parent))

from backend.app import app
from backend.api.enhanced_generation import ModelDetectionService, PromptEnhancementService

client = TestClient(app)

class TestWANModelsPhase1:
    """Test suite for Phase 1 WAN models functionality"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.test_prompt = "A beautiful sunset over mountains"
        self.test_image_data = self._create_test_image()
    
    def _create_test_image(self) -> bytes:
        """Create a test image for upload tests"""
        img = Image.new('RGB', (512, 512), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        return img_bytes.getvalue()
    
    # Model Detection Tests
    def test_model_detection_t2v(self):
        """Test T2V model detection for text-only inputs"""
        detected = ModelDetectionService.detect_model_type(
            prompt="A cat walking in the garden",
            has_image=False,
            has_end_image=False
        )
        assert detected == "T2V-A14B"
    
    def test_model_detection_i2v(self):
        """Test I2V model detection for image-only inputs"""
        detected = ModelDetectionService.detect_model_type(
            prompt="Make this image move naturally",
            has_image=True,
            has_end_image=False
        )
        assert detected == "I2V-A14B"
    
    def test_model_detection_ti2v_single_image(self):
        """Test TI2V model detection for text+image with transformation keywords"""
        detected = ModelDetectionService.detect_model_type(
            prompt="Transform this image into a magical scene",
            has_image=True,
            has_end_image=False
        )
        assert detected == "TI2V-5B"
    
    def test_model_detection_ti2v_interpolation(self):
        """Test TI2V model detection for interpolation (both images)"""
        detected = ModelDetectionService.detect_model_type(
            prompt="Smooth transition between images",
            has_image=True,
            has_end_image=True
        )
        assert detected == "TI2V-5B"
    
    def test_model_requirements(self):
        """Test model requirements retrieval"""
        t2v_req = ModelDetectionService.get_model_requirements("T2V-A14B")
        assert t2v_req["requires_image"] == False
        assert t2v_req["estimated_vram_gb"] == 8.0
        
        i2v_req = ModelDetectionService.get_model_requirements("I2V-A14B")
        assert i2v_req["requires_image"] == True
        assert i2v_req["supports_end_image"] == True
        
        ti2v_req = ModelDetectionService.get_model_requirements("TI2V-5B")
        assert ti2v_req["requires_image"] == True
        assert ti2v_req["estimated_vram_gb"] == 6.0
    
    # Prompt Enhancement Tests
    def test_prompt_enhancement_t2v(self):
        """Test prompt enhancement for T2V model"""
        enhanced = PromptEnhancementService.enhance_prompt(
            "A cat walking",
            "T2V-A14B",
            {"enhance_quality": True, "enhance_technical": True}
        )
        assert "cinematic composition" in enhanced
        assert "smooth camera movement" in enhanced
        assert "high quality, detailed" in enhanced
        assert "HD quality" in enhanced
    
    def test_prompt_enhancement_i2v(self):
        """Test prompt enhancement for I2V model"""
        enhanced = PromptEnhancementService.enhance_prompt(
            "Make it move",
            "I2V-A14B",
            {"enhance_quality": True}
        )
        assert "natural animation" in enhanced
        assert "high quality, detailed" in enhanced
    
    def test_prompt_enhancement_ti2v(self):
        """Test prompt enhancement for TI2V model"""
        enhanced = PromptEnhancementService.enhance_prompt(
            "Change the scene",
            "TI2V-5B",
            {"enhance_quality": True}
        )
        assert "smooth transformation" in enhanced
    
    def test_prompt_enhancement_no_duplicates(self):
        """Test that enhancement doesn't add duplicate terms"""
        prompt_with_quality = "A high quality, detailed cinematic scene"
        enhanced = PromptEnhancementService.enhance_prompt(
            prompt_with_quality,
            "T2V-A14B",
            {"enhance_quality": True, "enhance_technical": True}
        )
        # Should not add duplicate quality terms
        assert enhanced.count("high quality") == 1
        assert enhanced.count("cinematic") == 1
    
    # API Endpoint Tests
    def test_model_detection_endpoint(self):
        """Test the model detection API endpoint"""
        response = client.get(
            "/api/v1/generation/models/detect",
            params={
                "prompt": "A beautiful landscape",
                "has_image": False,
                "has_end_image": False
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["detected_model_type"] == "T2V-A14B"
        assert "explanation" in data
        assert "requirements" in data
    
    def test_prompt_enhancement_endpoint(self):
        """Test the prompt enhancement API endpoint"""
        response = client.post(
            "/api/v1/generation/prompt/enhance",
            data={
                "prompt": "A cat",
                "model_type": "T2V-A14B",
                "enhance_quality": True,
                "enhance_technical": True
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["original_prompt"] == "A cat"
        assert len(data["enhanced_prompt"]) > len(data["original_prompt"])
        assert "enhancements_applied" in data
    
    def test_capabilities_endpoint(self):
        """Test the capabilities API endpoint"""
        response = client.get("/api/v1/generation/capabilities")
        assert response.status_code == 200
        data = response.json()
        
        expected_models = ["T2V-A14B", "I2V-A14B", "TI2V-5B"]
        assert all(model in data["supported_models"] for model in expected_models)
        assert "features" in data
        assert data["features"]["auto_model_detection"] == True
        assert data["features"]["prompt_enhancement"] == True
    
    @patch('backend.services.generation_service.GenerationService')
    def test_enhanced_generation_t2v(self, mock_service):
        """Test enhanced generation endpoint for T2V"""
        # Mock the generation service
        mock_service_instance = Mock()
        mock_service_instance.submit_generation_task = AsyncMock(return_value=True)
        mock_service_instance.get_queue_status = AsyncMock(return_value={"tasks": []})
        
        with patch('backend.app.app.state.generation_service', mock_service_instance):
            response = client.post(
                "/api/v1/generation/submit",
                data={
                    "prompt": self.test_prompt,
                    "model_type": "T2V-A14B",
                    "resolution": "1280x720",
                    "steps": 50,
                    "enable_prompt_enhancement": True,
                    "enable_optimization": True
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "task_id" in data
        assert data["detected_model_type"] == "T2V-A14B"
        assert "applied_optimizations" in data
    
    @patch('backend.services.generation_service.GenerationService')
    def test_enhanced_generation_i2v_with_image(self, mock_service):
        """Test enhanced generation endpoint for I2V with image upload"""
        mock_service_instance = Mock()
        mock_service_instance.submit_generation_task = AsyncMock(return_value=True)
        mock_service_instance.get_queue_status = AsyncMock(return_value={"tasks": []})
        
        with patch('backend.app.app.state.generation_service', mock_service_instance):
            response = client.post(
                "/api/v1/generation/submit",
                data={
                    "prompt": "Animate this image",
                    "model_type": "I2V-A14B",
                    "resolution": "1280x720",
                    "steps": 50
                },
                files={
                    "image": ("test.jpg", self.test_image_data, "image/jpeg")
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["detected_model_type"] == "I2V-A14B"
    
    def test_enhanced_generation_i2v_missing_image(self):
        """Test I2V generation fails without required image"""
        response = client.post(
            "/api/v1/generation/submit",
            data={
                "prompt": "Animate this image",
                "model_type": "I2V-A14B",
                "resolution": "1280x720",
                "steps": 50
            }
        )
        
        assert response.status_code == 422
        assert "requires a start image" in response.json()["detail"]
    
    @patch('backend.services.generation_service.GenerationService')
    def test_enhanced_generation_auto_detection(self, mock_service):
        """Test auto-detection in enhanced generation"""
        mock_service_instance = Mock()
        mock_service_instance.submit_generation_task = AsyncMock(return_value=True)
        mock_service_instance.get_queue_status = AsyncMock(return_value={"tasks": []})
        
        with patch('backend.app.app.state.generation_service', mock_service_instance):
            # Test auto-detection for text-only (should detect T2V)
            response = client.post(
                "/api/v1/generation/submit",
                data={
                    "prompt": "A beautiful landscape scene",
                    # No model_type specified - should auto-detect
                    "resolution": "1280x720",
                    "steps": 50
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["detected_model_type"] == "T2V-A14B"
    
    def test_enhanced_generation_validation(self):
        """Test input validation in enhanced generation"""
        # Test empty prompt
        response = client.post(
            "/api/v1/generation/submit",
            data={
                "prompt": "",
                "model_type": "T2V-A14B"
            }
        )
        assert response.status_code == 422
        
        # Test invalid steps
        response = client.post(
            "/api/v1/generation/submit",
            data={
                "prompt": self.test_prompt,
                "model_type": "T2V-A14B",
                "steps": 150  # Invalid: > 100
            }
        )
        assert response.status_code == 422
        
        # Test invalid resolution
        response = client.post(
            "/api/v1/generation/submit",
            data={
                "prompt": self.test_prompt,
                "model_type": "T2V-A14B",
                "resolution": "invalid_resolution"
            }
        )
        assert response.status_code == 422
    
    def test_image_upload_validation(self):
        """Test image upload validation"""
        # Test oversized image (simulate)
        large_image_data = b"x" * (11 * 1024 * 1024)  # 11MB
        
        response = client.post(
            "/api/v1/generation/submit",
            data={
                "prompt": "Animate this",
                "model_type": "I2V-A14B"
            },
            files={
                "image": ("large.jpg", large_image_data, "image/jpeg")
            }
        )
        
        assert response.status_code == 422
        assert "too large" in response.json()["detail"]

class TestWANModelsCLI:
    """Test suite for WAN CLI commands"""
    
    def test_cli_import(self):
        """Test that WAN CLI module can be imported"""
        from cli.commands.wan import app as wan_app
        assert wan_app is not None
    
    @patch('cli.commands.wan.console')
    def test_list_models_command(self, mock_console):
        """Test the wan models command"""
        from cli.commands.wan import list_models
        
        # Test basic listing
        list_models(detailed=False, status_only=False)
        mock_console.print.assert_called()
        
        # Test status only
        list_models(detailed=False, status_only=True)
        mock_console.print.assert_called()
        
        # Test detailed listing
        list_models(detailed=True, status_only=False)
        mock_console.print.assert_called()

class TestWANModelsIntegration:
    """Integration tests for WAN models Phase 1"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_t2v_flow(self):
        """Test complete T2V generation flow"""
        # This would test the full pipeline from API to model execution
        # For now, we'll test the API integration
        
        response = client.get("/api/v1/generation/capabilities")
        assert response.status_code == 200
        
        capabilities = response.json()
        assert "T2V-A14B" in capabilities["supported_models"]
    
    def test_model_switching_logic(self):
        """Test seamless model switching based on inputs"""
        # Test different input combinations
        test_cases = [
            {
                "prompt": "A cat walking",
                "has_image": False,
                "has_end_image": False,
                "expected": "T2V-A14B"
            },
            {
                "prompt": "Animate this photo",
                "has_image": True,
                "has_end_image": False,
                "expected": "I2V-A14B"
            },
            {
                "prompt": "Transform this image into something magical",
                "has_image": True,
                "has_end_image": False,
                "expected": "TI2V-5B"
            },
            {
                "prompt": "Smooth transition",
                "has_image": True,
                "has_end_image": True,
                "expected": "TI2V-5B"
            }
        ]
        
        for case in test_cases:
            detected = ModelDetectionService.detect_model_type(
                case["prompt"],
                case["has_image"],
                case["has_end_image"]
            )
            assert detected == case["expected"], f"Failed for case: {case}"
    
    def test_optimization_application(self):
        """Test that optimizations are properly applied"""
        # Test prompt enhancement
        original = "A cat"
        enhanced = PromptEnhancementService.enhance_prompt(
            original, "T2V-A14B", {"enhance_quality": True}
        )
        assert len(enhanced) > len(original)
        
        # Test model requirements
        for model in ["T2V-A14B", "I2V-A14B", "TI2V-5B"]:
            req = ModelDetectionService.get_model_requirements(model)
            assert "estimated_vram_gb" in req
            assert "estimated_time_per_frame" in req

# Performance and Load Tests
class TestWANModelsPerformance:
    """Performance tests for WAN models"""
    
    def test_model_detection_performance(self):
        """Test model detection performance"""
        import time
        
        start_time = time.time()
        for _ in range(100):
            ModelDetectionService.detect_model_type(
                "A test prompt for performance testing",
                has_image=False,
                has_end_image=False
            )
        end_time = time.time()
        
        # Should complete 100 detections in under 1 second
        assert (end_time - start_time) < 1.0
    
    def test_prompt_enhancement_performance(self):
        """Test prompt enhancement performance"""
        import time
        
        start_time = time.time()
        for _ in range(50):
            PromptEnhancementService.enhance_prompt(
                "A test prompt for performance testing",
                "T2V-A14B",
                {"enhance_quality": True, "enhance_technical": True}
            )
        end_time = time.time()
        
        # Should complete 50 enhancements in under 1 second
        assert (end_time - start_time) < 1.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
