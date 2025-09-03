"""
Test prompt enhancement API endpoints
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from main import backend.app as app

client = TestClient(app)

class TestPromptEnhancementAPI:
    """Test prompt enhancement API endpoints"""
    
    def test_enhance_prompt_basic(self):
        """Test basic prompt enhancement"""
        with patch('core.system_integration.get_system_integration') as mock_get_integration:
            # Mock the system integration
            mock_integration = AsyncMock()
            mock_enhancer = Mock()
            
            # Mock enhancer methods
            mock_enhancer.validate_prompt.return_value = (True, "Valid prompt")
            mock_enhancer.detect_vace_aesthetics.return_value = False
            mock_enhancer.detect_style_category.return_value = "cinematic"
            mock_enhancer.enhance_comprehensive.return_value = "A beautiful sunset, cinematic lighting, high quality"
            mock_enhancer.get_enhancement_preview.return_value = {
                "suggested_enhancements": ["Cinematic Style", "Quality Keywords"]
            }
            
            mock_integration.get_prompt_enhancer.return_value = mock_enhancer
            mock_get_integration.return_value = mock_integration
            
            # Test request
            response = client.post("/api/v1/prompt/enhance", json={
                "prompt": "A beautiful sunset",
                "apply_cinematic": True,
                "apply_style": True
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["original_prompt"] == "A beautiful sunset"
            assert data["enhanced_prompt"] == "A beautiful sunset, cinematic lighting, high quality"
            assert data["enhancements_applied"] == ["Cinematic Style", "Quality Keywords"]
            assert data["detected_style"] == "cinematic"
            assert data["vace_detected"] == False
            assert data["character_count"]["original"] == 17
            assert data["character_count"]["enhanced"] == 55
            assert data["character_count"]["difference"] == 38
    
    def test_enhance_prompt_with_vace(self):
        """Test prompt enhancement with VACE detection"""
        with patch('core.system_integration.get_system_integration') as mock_get_integration:
            mock_integration = AsyncMock()
            mock_enhancer = Mock()
            
            mock_enhancer.validate_prompt.return_value = (True, "Valid prompt")
            mock_enhancer.detect_vace_aesthetics.return_value = True
            mock_enhancer.detect_style_category.return_value = "artistic"
            mock_enhancer.enhance_comprehensive.return_value = "VACE aesthetic art, dreamy atmosphere, ethereal beauty"
            mock_enhancer.get_enhancement_preview.return_value = {
                "suggested_enhancements": ["VACE Aesthetic", "Artistic Style"]
            }
            
            mock_integration.get_prompt_enhancer.return_value = mock_enhancer
            mock_get_integration.return_value = mock_integration
            
            response = client.post("/api/v1/prompt/enhance", json={
                "prompt": "VACE aesthetic art",
                "apply_vace": True
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["vace_detected"] == True
            assert data["detected_style"] == "artistic"
            assert "VACE Aesthetic" in data["enhancements_applied"]
    
    def test_enhance_prompt_invalid(self):
        """Test prompt enhancement with invalid prompt"""
        with patch('core.system_integration.get_system_integration') as mock_get_integration:
            mock_integration = AsyncMock()
            mock_enhancer = Mock()
            
            mock_enhancer.validate_prompt.return_value = (False, "Prompt too short")
            
            mock_integration.get_prompt_enhancer.return_value = mock_enhancer
            mock_get_integration.return_value = mock_integration
            
            response = client.post("/api/v1/prompt/enhance", json={
                "prompt": "Hi"
            })
            
            assert response.status_code == 400
            assert "Invalid prompt" in response.json()["detail"]
    
    def test_preview_prompt_enhancement(self):
        """Test prompt enhancement preview"""
        with patch('core.system_integration.get_system_integration') as mock_get_integration:
            mock_integration = AsyncMock()
            mock_enhancer = Mock()
            
            mock_enhancer.get_enhancement_preview.return_value = {
                "enhanced_prompt": "A beautiful sunset, cinematic lighting",
                "suggested_enhancements": ["Cinematic Style"],
                "detected_style": "cinematic",
                "vace_detected": False,
                "quality_score": 0.85
            }
            
            mock_integration.get_prompt_enhancer.return_value = mock_enhancer
            mock_get_integration.return_value = mock_integration
            
            response = client.post("/api/v1/prompt/preview", json={
                "prompt": "A beautiful sunset"
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["original_prompt"] == "A beautiful sunset"
            assert data["preview_enhanced"] == "A beautiful sunset, cinematic lighting"
            assert data["suggested_enhancements"] == ["Cinematic Style"]
            assert data["detected_style"] == "cinematic"
            assert data["vace_detected"] == False
            assert data["quality_score"] == 0.85
    
    def test_validate_prompt(self):
        """Test prompt validation"""
        with patch('core.system_integration.get_system_integration') as mock_get_integration:
            mock_integration = AsyncMock()
            mock_enhancer = Mock()
            
            mock_enhancer.validate_prompt.return_value = (True, "Valid prompt")
            mock_enhancer.detect_vace_aesthetics.return_value = False
            mock_enhancer.detect_style_category.return_value = "cinematic"
            
            mock_integration.get_prompt_enhancer.return_value = mock_enhancer
            mock_get_integration.return_value = mock_integration
            
            response = client.post("/api/v1/prompt/validate", json={
                "prompt": "A beautiful sunset over mountains"
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["is_valid"] == True
            assert data["message"] == "Valid prompt"
            assert data["character_count"] == 33
            assert "cinematic style detected" in data["suggestions"][0].lower()
    
    def test_validate_prompt_invalid(self):
        """Test validation of invalid prompt"""
        with patch('core.system_integration.get_system_integration') as mock_get_integration:
            mock_integration = AsyncMock()
            mock_enhancer = Mock()
            
            mock_enhancer.validate_prompt.return_value = (False, "Prompt too short")
            
            mock_integration.get_prompt_enhancer.return_value = mock_enhancer
            mock_get_integration.return_value = mock_integration
            
            response = client.post("/api/v1/prompt/validate", json={
                "prompt": "Hi"
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["is_valid"] == False
            assert data["message"] == "Prompt too short"
            assert data["character_count"] == 2
            assert "Add more descriptive details" in data["suggestions"]
    
    def test_get_available_styles(self):
        """Test getting available style categories"""
        response = client.get("/api/v1/prompt/styles")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "styles" in data
        assert "total_count" in data
        assert data["total_count"] == 6
        
        # Check that expected styles are present
        style_names = [style["name"] for style in data["styles"]]
        assert "cinematic" in style_names
        assert "artistic" in style_names
        assert "photographic" in style_names
        assert "fantasy" in style_names
        assert "sci-fi" in style_names
        assert "vace" in style_names
        
        # Check style structure
        first_style = data["styles"][0]
        assert "name" in first_style
        assert "display_name" in first_style
        assert "description" in first_style
    
    def test_enhance_prompt_no_enhancer(self):
        """Test prompt enhancement when enhancer is not available"""
        with patch('core.system_integration.get_system_integration') as mock_get_integration:
            mock_integration = AsyncMock()
            mock_integration.get_prompt_enhancer.return_value = None
            mock_get_integration.return_value = mock_integration
            
            response = client.post("/api/v1/prompt/enhance", json={
                "prompt": "A beautiful sunset"
            })
            
            assert response.status_code == 500
            assert "Prompt enhancement system not available" in response.json()["detail"]
    
    def test_enhance_prompt_system_error(self):
        """Test prompt enhancement with system error"""
        with patch('core.system_integration.get_system_integration') as mock_get_integration:
            mock_integration = AsyncMock()
            mock_enhancer = Mock()
            
            mock_enhancer.validate_prompt.side_effect = Exception("System error")
            
            mock_integration.get_prompt_enhancer.return_value = mock_enhancer
            mock_get_integration.return_value = mock_integration
            
            response = client.post("/api/v1/prompt/enhance", json={
                "prompt": "A beautiful sunset"
            })
            
            assert response.status_code == 500
            assert "Failed to enhance prompt" in response.json()["detail"]
    
    def test_enhance_prompt_validation_error(self):
        """Test prompt enhancement with validation errors"""
        # Test empty prompt
        response = client.post("/api/v1/prompt/enhance", json={
            "prompt": ""
        })
        assert response.status_code == 422
        
        # Test prompt too long
        long_prompt = "A" * 501
        response = client.post("/api/v1/prompt/enhance", json={
            "prompt": long_prompt
        })
        assert response.status_code == 422

if __name__ == "__main__":
    pytest.main([__file__, "-v"])