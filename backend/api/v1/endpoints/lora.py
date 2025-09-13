"""
LoRA management API endpoints
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query
from typing import List, Optional, Dict, Any
import os
import shutil
import uuid
from pathlib import Path
import logging
from datetime import datetime

from backend.core.system_integration import get_system_integration
from backend.schemas.schemas import LoRAInfo, LoRAUploadResponse, LoRAListResponse, LoRAStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Supported LoRA file extensions
SUPPORTED_LORA_EXTENSIONS = {'.safetensors', '.pt', '.pth', '.bin'}

# Maximum LoRA file size (500MB)
MAX_LORA_SIZE = 500 * 1024 * 1024

@router.get("/list", response_model=LoRAListResponse)
async def list_loras(
    system_integration = Depends(get_system_integration)
):
    """
    List all available LoRA files
    """
    try:
        lora_manager = system_integration.get_lora_manager()
        available_loras = lora_manager.list_available_loras()
        
        lora_list = []
        for name, info in available_loras.items():
            lora_list.append(LoRAInfo(
                name=name,
                filename=info["filename"],
                path=info["path"],
                size_mb=round(info["size_mb"], 2),
                modified_time=info["modified_time"],
                is_loaded=info["is_loaded"],
                is_applied=info["is_applied"],
                current_strength=info["current_strength"]
            ))
        
        return LoRAListResponse(
            loras=lora_list,
            total_count=len(lora_list),
            total_size_mb=round(sum(lora.size_mb for lora in lora_list), 2)
        )
        
    except Exception as e:
        logger.error(f"Error listing LoRAs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "lora_list_failed",
                "message": "Failed to list LoRA files",
                "details": str(e),
                "suggestions": [
                    "Check if the LoRA directory is accessible",
                    "Verify file permissions",
                    "Try refreshing the page"
                ]
            }
        )

@router.post("/upload", response_model=LoRAUploadResponse)
async def upload_lora(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    system_integration = Depends(get_system_integration)
):
    """
    Upload a new LoRA file
    """
    try:
        # Validate file extension
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Filename is required"
            )
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in SUPPORTED_LORA_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_file_format",
                    "message": f"Unsupported file format: {file_extension}",
                    "details": f"Supported formats: {', '.join(SUPPORTED_LORA_EXTENSIONS)}",
                    "suggestions": [
                        "Convert your LoRA to a supported format",
                        "Use .safetensors format for best compatibility"
                    ]
                }
            )
        
        # Read file content to validate size
        file_content = await file.read()
        if len(file_content) > MAX_LORA_SIZE:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "file_too_large",
                    "message": f"File too large: {len(file_content) / (1024*1024):.1f}MB",
                    "details": f"Maximum allowed size: {MAX_LORA_SIZE / (1024*1024):.0f}MB",
                    "suggestions": [
                        "Use a smaller LoRA file",
                        "Compress the LoRA if possible"
                    ]
                }
            )
        
        # Reset file pointer
        await file.seek(0)
        
        # Use provided name or derive from filename
        lora_name = name or Path(file.filename).stem
        
        # Get LoRA manager
        lora_manager = system_integration.get_lora_manager()
        
        # Create temporary file for upload
        temp_id = str(uuid.uuid4())
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / f"{temp_id}_{file.filename}"
        
        try:
            # Save uploaded file temporarily
            with open(temp_file_path, "wb") as temp_file:
                shutil.copyfileobj(file.file, temp_file)
            
            # Use LoRA manager to handle the upload
            result = lora_manager.upload_lora_file(str(temp_file_path), f"{lora_name}{file_extension}")
            
            if "error" in result:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "lora_upload_failed",
                        "message": result["error"],
                        "suggestions": [
                            "Check if the LoRA file is valid",
                            "Try a different LoRA file",
                            "Ensure the file is not corrupted"
                        ]
                    }
                )
            
            return LoRAUploadResponse(
                success=True,
                message=result["message"],
                lora_name=lora_name,
                file_path=result.get("file_path", ""),
                size_mb=len(file_content) / (1024 * 1024)
            )
            
        finally:
            # Clean up temporary file
            if temp_file_path.exists():
                temp_file_path.unlink()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading LoRA: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "lora_upload_error",
                "message": "Failed to upload LoRA file",
                "details": str(e),
                "suggestions": [
                    "Check if there's enough disk space",
                    "Verify file permissions",
                    "Try uploading again"
                ]
            }
        )

@router.delete("/{lora_name}")
async def delete_lora(
    lora_name: str,
    system_integration = Depends(get_system_integration)
):
    """
    Delete a LoRA file
    """
    try:
        lora_manager = system_integration.get_lora_manager()
        
        # Check if LoRA exists
        lora_status = lora_manager.get_lora_status(lora_name)
        if not lora_status["exists"]:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "lora_not_found",
                    "message": f"LoRA '{lora_name}' not found",
                    "suggestions": [
                        "Check the LoRA name spelling",
                        "Refresh the LoRA list"
                    ]
                }
            )
        
        # Use LoRA manager to delete
        result = lora_manager.delete_lora_file(lora_name)
        
        if "error" in result:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "lora_delete_failed",
                    "message": result["error"],
                    "suggestions": [
                        "Check if the file is currently in use",
                        "Verify file permissions"
                    ]
                }
            )
        
        return {
            "success": True,
            "message": result["message"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting LoRA {lora_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "lora_delete_error",
                "message": "Failed to delete LoRA file",
                "details": str(e)
            }
        )

@router.get("/{lora_name}/status", response_model=LoRAStatusResponse)
async def get_lora_status(
    lora_name: str,
    system_integration = Depends(get_system_integration)
):
    """
    Get detailed status information for a specific LoRA
    """
    try:
        lora_manager = system_integration.get_lora_manager()
        status = lora_manager.get_lora_status(lora_name)
        
        return LoRAStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Error getting LoRA status for {lora_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "lora_status_error",
                "message": "Failed to get LoRA status",
                "details": str(e)
            }
        )

@router.post("/{lora_name}/preview")
async def generate_lora_preview(
    lora_name: str,
    base_prompt: str = Form(...),
    system_integration = Depends(get_system_integration)
):
    """
    Generate a preview of what a LoRA might produce with a given prompt
    """
    try:
        lora_manager = system_integration.get_lora_manager()
        
        # Check if LoRA exists
        lora_status = lora_manager.get_lora_status(lora_name)
        if not lora_status["exists"]:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "lora_not_found",
                    "message": f"LoRA '{lora_name}' not found"
                }
            )
        
        # Generate enhanced prompt as preview
        enhanced_prompt = lora_manager.get_fallback_prompt_enhancement(base_prompt, lora_name)
        
        # Generate style indicators based on LoRA name
        style_indicators = _generate_style_indicators(lora_name)
        
        return {
            "lora_name": lora_name,
            "base_prompt": base_prompt,
            "enhanced_prompt": enhanced_prompt,
            "style_indicators": style_indicators,
            "preview_note": "This is a preview based on LoRA name analysis. Actual results may vary."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating LoRA preview for {lora_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "lora_preview_error",
                "message": "Failed to generate LoRA preview",
                "details": str(e)
            }
        )

@router.get("/{lora_name}/memory-impact")
async def estimate_lora_memory_impact(
    lora_name: str,
    system_integration = Depends(get_system_integration)
):
    """
    Estimate the memory impact of loading a specific LoRA
    """
    try:
        lora_manager = system_integration.get_lora_manager()
        
        # Check if LoRA exists
        lora_status = lora_manager.get_lora_status(lora_name)
        if not lora_status["exists"]:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "lora_not_found",
                    "message": f"LoRA '{lora_name}' not found"
                }
            )
        
        # Estimate memory impact based on file size
        file_size_mb = lora_status.get("size_mb", 0)
        
        # Rough estimation: LoRA memory usage is typically 1.5-2x file size
        estimated_memory_mb = file_size_mb * 1.8
        
        # Get current system stats for context
        system_stats = system_integration.get_system_stats()
        vram_available_mb = system_stats["vram_total_mb"] - system_stats["vram_used_mb"]
        
        can_load = estimated_memory_mb < vram_available_mb * 0.8  # Leave 20% buffer
        
        return {
            "lora_name": lora_name,
            "file_size_mb": file_size_mb,
            "estimated_memory_mb": round(estimated_memory_mb, 1),
            "vram_available_mb": round(vram_available_mb, 1),
            "can_load": can_load,
            "memory_impact": "low" if estimated_memory_mb < 100 else "medium" if estimated_memory_mb < 300 else "high",
            "recommendation": _get_memory_recommendation(estimated_memory_mb, vram_available_mb)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error estimating memory impact for {lora_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "memory_estimation_error",
                "message": "Failed to estimate memory impact",
                "details": str(e)
            }
        )

def _generate_style_indicators(lora_name: str) -> List[str]:
    """Generate style indicators based on LoRA name analysis"""
    name_lower = lora_name.lower()
    indicators = []
    
    # Style categories
    if any(keyword in name_lower for keyword in ['anime', 'manga', 'cartoon']):
        indicators.extend(['Anime Style', 'Stylized', 'Vibrant Colors'])
    
    if any(keyword in name_lower for keyword in ['realistic', 'photo', 'real']):
        indicators.extend(['Photorealistic', 'Detailed', 'Natural Lighting'])
    
    if any(keyword in name_lower for keyword in ['art', 'painting', 'artistic']):
        indicators.extend(['Artistic', 'Painterly', 'Creative'])
    
    if any(keyword in name_lower for keyword in ['portrait', 'face', 'character']):
        indicators.extend(['Character Focus', 'Facial Details'])
    
    if any(keyword in name_lower for keyword in ['landscape', 'scenery', 'environment']):
        indicators.extend(['Environmental', 'Scenic'])
    
    # Quality indicators
    if any(keyword in name_lower for keyword in ['detail', 'hd', 'quality']):
        indicators.extend(['High Detail', 'Enhanced Quality'])
    
    # Default if no specific indicators found
    if not indicators:
        indicators = ['Style Enhancement', 'Custom Look']
    
    return list(set(indicators))  # Remove duplicates

def _get_memory_recommendation(estimated_mb: float, available_mb: float) -> str:
    """Get memory usage recommendation"""
    if estimated_mb > available_mb:
        return "Not enough VRAM available. Consider freeing memory or using a smaller LoRA."
    elif estimated_mb > available_mb * 0.8:
        return "High VRAM usage expected. Monitor system performance."
    elif estimated_mb > available_mb * 0.5:
        return "Moderate VRAM usage. Should work well with current system."
    else:
        return "Low VRAM usage. Safe to load with current system."
