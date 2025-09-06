"""
Test LoRA parameter validation logic
"""

import pytest
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class MockGenerationParams:
    """Mock generation parameters for testing"""
    prompt: str
    model_type: str
    lora_path: Optional[str] = None
    lora_strength: float = 1.0


class MockLoRAValidator:
    """Mock LoRA validator for testing validation logic"""
    
    def __init__(self):
        self.lora_manager = None
    
    def _validate_lora_params(self, params: MockGenerationParams) -> dict:
        """Validate LoRA parameters - extracted logic from RealGenerationPipeline"""
        errors = []
        warnings = []
        
        # Validate LoRA strength
        if params.lora_strength < 0.0 or params.lora_strength > 2.0:
            errors.append("LoRA strength must be between 0.0 and 2.0")
        
        # Validate LoRA path if provided
        if params.lora_path:
            lora_path = Path(params.lora_path)
            
            # Check if it's an absolute path or relative to loras directory
            if not lora_path.is_absolute():
                # Try relative to loras directory
                if self.lora_manager:
                    loras_dir = Path(self.lora_manager.loras_directory)
                    lora_path = loras_dir / params.lora_path
                else:
                    # Fallback to project loras directory
                    project_root = Path(__file__).parent.parent.parent
                    lora_path = project_root / "loras" / params.lora_path
            
            # Check if LoRA file exists
            if not lora_path.exists():
                # Try with common extensions if no extension provided
                if not lora_path.suffix:
                    extensions = ['.safetensors', '.pt', '.pth', '.bin']
                    found = False
                    for ext in extensions:
                        if (lora_path.parent / f"{lora_path.name}{ext}").exists():
                            found = True
                            break
                    
                    if not found:
                        errors.append(f"LoRA file not found: {params.lora_path}")
                else:
                    errors.append(f"LoRA file not found: {params.lora_path}")
            else:
                # Validate file extension
                valid_extensions = ['.safetensors', '.pt', '.pth', '.bin']
                if lora_path.suffix.lower() not in valid_extensions:
                    errors.append(f"Invalid LoRA file format. Supported: {', '.join(valid_extensions)}")
                
                # Check file size (warn if very large)
                try:
                    file_size_mb = lora_path.stat().st_size / (1024 * 1024)
                    if file_size_mb > 500:
                        warnings.append(f"Large LoRA file ({file_size_mb:.1f}MB) may slow loading")
                except Exception:
                    pass
        
        # Warn if LoRA is specified but manager is not available
        if params.lora_path and not self.lora_manager:
            warnings.append("LoRA specified but LoRA manager not available - will use fallback prompt enhancement")
        
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}
    
    def _get_basic_lora_fallback(self, base_prompt: str, lora_name: str) -> str:
        """Basic LoRA fallback prompt enhancement - extracted from RealGenerationPipeline"""
        # Simple enhancement based on common LoRA types
        lora_lower = lora_name.lower()
        
        if "anime" in lora_lower:
            enhancement = "anime style, detailed anime art"
        elif "realistic" in lora_lower or "photo" in lora_lower:
            enhancement = "photorealistic, highly detailed"
        elif "art" in lora_lower or "paint" in lora_lower:
            enhancement = "artistic style, detailed artwork"
        elif "detail" in lora_lower or "quality" in lora_lower:
            enhancement = "extremely detailed, high quality"
        else:
            enhancement = "enhanced style, high quality"
        
        # Combine with base prompt
        if base_prompt.strip():
            return f"{base_prompt}, {enhancement}"
        else:
            return enhancement


class TestLoRAValidation:
    """Test LoRA validation logic without external dependencies"""
    
    def test_valid_lora_strength(self):
        """Test valid LoRA strength values"""
        validator = MockLoRAValidator()
        
        valid_strengths = [0.0, 0.5, 1.0, 1.5, 2.0]
        
        for strength in valid_strengths:
            params = MockGenerationParams(
                prompt="test prompt",
                model_type="t2v-A14B",
                lora_strength=strength
            )
            
            result = validator._validate_lora_params(params)
            strength_errors = [e for e in result["errors"] if "strength must be between" in e]
            assert len(strength_errors) == 0, f"Valid strength {strength} should not produce errors"
    
    def test_invalid_lora_strength(self):
        """Test invalid LoRA strength values"""
        validator = MockLoRAValidator()
        
        invalid_strengths = [-0.1, -1.0, 2.1, 3.0, 10.0]
        
        for strength in invalid_strengths:
            params = MockGenerationParams(
                prompt="test prompt",
                model_type="t2v-A14B",
                lora_strength=strength
            )
            
            result = validator._validate_lora_params(params)
            assert result["valid"] == False
            strength_errors = [e for e in result["errors"] if "strength must be between" in e]
            assert len(strength_errors) > 0, f"Invalid strength {strength} should produce errors"
    
    def test_lora_file_extension_validation(self):
        """Test LoRA file extension validation"""
        validator = MockLoRAValidator()
        
        # Test valid extensions
        valid_extensions = ['.safetensors', '.pt', '.pth', '.bin']
        
        for ext in valid_extensions:
            # Create a temporary file for testing
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                params = MockGenerationParams(
                    prompt="test prompt",
                    model_type="t2v-A14B",
                    lora_path=tmp_path,
                    lora_strength=1.0
                )
                
                result = validator._validate_lora_params(params)
                extension_errors = [e for e in result["errors"] if "Invalid LoRA file format" in e]
                assert len(extension_errors) == 0, f"Valid extension {ext} should not produce format errors"
            
            finally:
                # Clean up
                Path(tmp_path).unlink(missing_ok=True)
        
        # Test invalid extensions
        invalid_extensions = ['.txt', '.json', '.pkl', '.ckpt']
        
        for ext in invalid_extensions:
            # Create a temporary file for testing
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                params = MockGenerationParams(
                    prompt="test prompt",
                    model_type="t2v-A14B",
                    lora_path=tmp_path,
                    lora_strength=1.0
                )
                
                result = validator._validate_lora_params(params)
                extension_errors = [e for e in result["errors"] if "Invalid LoRA file format" in e]
                assert len(extension_errors) > 0, f"Invalid extension {ext} should produce format errors"
            
            finally:
                # Clean up
                Path(tmp_path).unlink(missing_ok=True)
    
    def test_missing_lora_file(self):
        """Test validation with missing LoRA file"""
        validator = MockLoRAValidator()
        
        params = MockGenerationParams(
            prompt="test prompt",
            model_type="t2v-A14B",
            lora_path="/nonexistent/path/missing_lora.safetensors",
            lora_strength=1.0
        )
        
        result = validator._validate_lora_params(params)
        assert result["valid"] == False
        file_errors = [e for e in result["errors"] if "LoRA file not found" in e]
        assert len(file_errors) > 0
    
    def test_lora_fallback_enhancement(self):
        """Test LoRA fallback prompt enhancement"""
        validator = MockLoRAValidator()
        
        test_cases = [
            ("a beautiful scene", "anime_style", "anime style"),
            ("portrait", "realistic_photo", "photorealistic"),
            ("landscape", "detail_enhancer", "detailed"),
            ("character", "art_nouveau", "artistic"),
            ("scene", "unknown_lora", "enhanced style")
        ]
        
        for base_prompt, lora_name, expected_keyword in test_cases:
            enhanced = validator._get_basic_lora_fallback(base_prompt, lora_name)
            
            # Check that original prompt is preserved
            if base_prompt.strip():
                assert base_prompt in enhanced
            
            # Check that enhancement keyword is present
            assert expected_keyword.lower() in enhanced.lower()
    
    def test_empty_prompt_fallback(self):
        """Test LoRA fallback with empty prompt"""
        validator = MockLoRAValidator()
        
        enhanced = validator._get_basic_lora_fallback("", "anime_style")
        assert "anime style" in enhanced.lower()
        assert enhanced.strip() != ""  # Should not be empty
    
    def test_no_lora_path_validation(self):
        """Test validation when no LoRA path is specified"""
        validator = MockLoRAValidator()
        
        params = MockGenerationParams(
            prompt="test prompt",
            model_type="t2v-A14B",
            lora_path=None,
            lora_strength=1.0
        )
        
        result = validator._validate_lora_params(params)
        # Should be valid when no LoRA is specified
        assert result["valid"] == True
        assert len(result["errors"]) == 0
    
    def test_lora_manager_warning(self):
        """Test warning when LoRA is specified but manager is not available"""
        validator = MockLoRAValidator()
        validator.lora_manager = None  # No manager available
        
        # Create a temporary LoRA file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            params = MockGenerationParams(
                prompt="test prompt",
                model_type="t2v-A14B",
                lora_path=tmp_path,
                lora_strength=1.0
            )
            
            result = validator._validate_lora_params(params)
            
            # Should have warning about manager not available
            manager_warnings = [w for w in result["warnings"] if "LoRA manager not available" in w]
            assert len(manager_warnings) > 0
        
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])