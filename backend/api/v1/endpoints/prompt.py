"""
Prompt enhancement API endpoints
Updated version - git operations working correctly
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

from backend.core.system_integration import get_system_integration, SystemIntegration

logger = logging.getLogger(__name__)

router = APIRouter()

class PromptEnhanceRequest(BaseModel):
    """Request model for prompt enhancement"""
    prompt: str = Field(..., min_length=1, max_length=500, description="Original prompt to enhance")
    apply_vace: Optional[bool] = Field(default=None, description="Apply VACE aesthetic enhancements")
    apply_cinematic: Optional[bool] = Field(default=True, description="Apply cinematic style enhancements")
    apply_style: Optional[bool] = Field(default=True, description="Apply style-specific enhancements")

class PromptEnhanceResponse(BaseModel):
    """Response model for prompt enhancement"""
    original_prompt: str
    enhanced_prompt: str
    enhancements_applied: List[str]
    character_count: Dict[str, int]
    detected_style: Optional[str] = None
    vace_detected: bool = False

class PromptPreviewRequest(BaseModel):
    """Request model for prompt enhancement preview"""
    prompt: str = Field(..., min_length=1, max_length=500, description="Prompt to preview enhancements for")

class PromptPreviewResponse(BaseModel):
    """Response model for prompt enhancement preview"""
    original_prompt: str
    preview_enhanced: str
    suggested_enhancements: List[str]
    detected_style: Optional[str] = None
    vace_detected: bool = False
    character_count: Dict[str, int]
    quality_score: Optional[float] = None

class PromptValidationRequest(BaseModel):
    """Request model for prompt validation"""
    prompt: str = Field(..., max_length=1000, description="Prompt to validate")

class PromptValidationResponse(BaseModel):
    """Response model for prompt validation"""
    is_valid: bool
    message: str
    character_count: int
    suggestions: Optional[List[str]] = None

@router.post("/prompt/enhance", response_model=PromptEnhanceResponse)
async def enhance_prompt(
    request: PromptEnhanceRequest,
    integration: SystemIntegration = Depends(get_system_integration)
):
    """
    Enhance a prompt with quality keywords and style improvements
    Requirement 5.1: Provide syntax highlighting and auto-suggestions
    Requirement 5.2: Display enhanced prompts with diff highlighting
    """
    try:
        # Get the prompt enhancer from the existing system
        enhancer = await integration.get_prompt_enhancer()
        
        if not enhancer:
            raise HTTPException(
                status_code=500, 
                detail="Prompt enhancement system not available"
            )
        
        original_prompt = request.prompt.strip()
        
        # Validate prompt first
        is_valid, validation_message = enhancer.validate_prompt(original_prompt)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid prompt: {validation_message}"
            )
        
        # Detect VACE aesthetics and style
        vace_detected = enhancer.detect_vace_aesthetics(original_prompt)
        detected_style = enhancer.detect_style_category(original_prompt)
        
        # Apply enhancements
        enhanced_prompt = enhancer.enhance_comprehensive(
            original_prompt,
            enable_vace=request.apply_vace if request.apply_vace is not None else vace_detected,
            enable_cinematic=request.apply_cinematic,
            enable_style=request.apply_style
        )
        
        # Get enhancement preview to see what was applied
        preview = enhancer.get_enhancement_preview(original_prompt)
        enhancements_applied = preview.get("suggested_enhancements", [])
        
        return PromptEnhanceResponse(
            original_prompt=original_prompt,
            enhanced_prompt=enhanced_prompt,
            enhancements_applied=enhancements_applied,
            character_count={
                "original": len(original_prompt),
                "enhanced": len(enhanced_prompt),
                "difference": len(enhanced_prompt) - len(original_prompt)
            },
            detected_style=detected_style,
            vace_detected=vace_detected
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enhancing prompt: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to enhance prompt: {str(e)}"
        )

@router.post("/prompt/preview", response_model=PromptPreviewResponse)
async def preview_prompt_enhancement(
    request: PromptPreviewRequest,
    integration: SystemIntegration = Depends(get_system_integration)
):
    """
    Preview how a prompt would be enhanced without applying changes
    Requirement 5.4: Allow users to accept, reject, or modify suggestions
    """
    try:
        # Get the prompt enhancer from the existing system
        enhancer = await integration.get_prompt_enhancer()
        
        if not enhancer:
            raise HTTPException(
                status_code=500, 
                detail="Prompt enhancement system not available"
            )
        
        original_prompt = request.prompt.strip()
        
        # Get enhancement preview
        preview = enhancer.get_enhancement_preview(original_prompt)
        
        return PromptPreviewResponse(
            original_prompt=original_prompt,
            preview_enhanced=preview.get("enhanced_prompt", original_prompt),
            suggested_enhancements=preview.get("suggested_enhancements", []),
            detected_style=preview.get("detected_style"),
            vace_detected=preview.get("vace_detected", False),
            character_count={
                "original": len(original_prompt),
                "preview": len(preview.get("enhanced_prompt", original_prompt)),
                "difference": len(preview.get("enhanced_prompt", original_prompt)) - len(original_prompt)
            },
            quality_score=preview.get("quality_score")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error previewing prompt enhancement: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to preview prompt enhancement: {str(e)}"
        )

@router.post("/prompt/validate", response_model=PromptValidationResponse)
async def validate_prompt(
    request: PromptValidationRequest,
    integration: SystemIntegration = Depends(get_system_integration)
):
    """
    Validate a prompt for length and content requirements
    """
    try:
        # Get the prompt enhancer from the existing system
        enhancer = await integration.get_prompt_enhancer()
        
        if not enhancer:
            raise HTTPException(
                status_code=500, 
                detail="Prompt enhancement system not available"
            )
        
        prompt = request.prompt.strip()
        is_valid, message = enhancer.validate_prompt(prompt)
        
        suggestions = []
        if not is_valid:
            if len(prompt) < 3:
                suggestions.append("Add more descriptive details to your prompt")
            elif len(prompt) > 500:
                suggestions.append("Shorten your prompt to under 500 characters")
                suggestions.append("Focus on the most important visual elements")
            
            if not prompt.strip():
                suggestions.append("Enter a prompt describing what you want to generate")
        else:
            # Provide enhancement suggestions for valid prompts
            if len(prompt) < 50:
                suggestions.append("Consider adding more details for better results")
            
            vace_detected = enhancer.detect_vace_aesthetics(prompt)
            if vace_detected:
                suggestions.append("VACE aesthetic style detected - enhancement available")
            
            style = enhancer.detect_style_category(prompt)
            if style:
                suggestions.append(f"{style.title()} style detected - specific enhancements available")
        
        return PromptValidationResponse(
            is_valid=is_valid,
            message=message,
            character_count=len(prompt),
            suggestions=suggestions if suggestions else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating prompt: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to validate prompt: {str(e)}"
        )

@router.get("/prompt/styles")
async def get_available_styles():
    """
    Get list of available style categories for prompt enhancement
    """
    try:
        # Return the available style categories
        styles = [
            {
                "name": "cinematic",
                "display_name": "Cinematic",
                "description": "Professional film and movie aesthetics"
            },
            {
                "name": "artistic",
                "display_name": "Artistic",
                "description": "Fine art and creative styles"
            },
            {
                "name": "photographic",
                "display_name": "Photographic",
                "description": "Realistic photography styles"
            },
            {
                "name": "fantasy",
                "display_name": "Fantasy",
                "description": "Fantasy and magical themes"
            },
            {
                "name": "sci-fi",
                "display_name": "Sci-Fi",
                "description": "Science fiction and futuristic themes"
            },
            {
                "name": "vace",
                "display_name": "VACE Aesthetic",
                "description": "VACE aesthetic style enhancements"
            }
        ]
        
        return {
            "styles": styles,
            "total_count": len(styles)
        }
        
    except Exception as e:
        logger.error(f"Error getting available styles: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get available styles: {str(e)}"
        )
