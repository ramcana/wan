"""
Advanced optimization presets and recommendations API
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional

from backend.schemas.schemas import OptimizationSettings, SystemStats, QuantizationLevel
from backend.repositories.database import get_db
from backend.core.system_integration import get_system_integration, SystemIntegration

logger = logging.getLogger(__name__)

router = APIRouter()

# Advanced optimization presets
OPTIMIZATION_PRESETS = {
    "balanced": {
        "name": "Balanced",
        "description": "Good balance of quality and performance for most users",
        "settings": {
            "quantization": QuantizationLevel.FP16,
            "enable_offload": False,
            "vae_tile_size": 512,
            "max_vram_usage_gb": 12.0
        },
        "vram_savings_gb": 2.0,
        "performance_impact": "low",
        "quality_impact": "minimal",
        "recommended_for": ["RTX 3080", "RTX 4070", "RTX 4080"],
        "min_vram_gb": 10.0
    },
    "memory_efficient": {
        "name": "Memory Efficient",
        "description": "Optimized for lower VRAM usage with good quality",
        "settings": {
            "quantization": QuantizationLevel.BF16,
            "enable_offload": True,
            "vae_tile_size": 256,
            "max_vram_usage_gb": 8.0
        },
        "vram_savings_gb": 4.5,
        "performance_impact": "medium",
        "quality_impact": "low",
        "recommended_for": ["RTX 3060", "RTX 3070", "RTX 4060"],
        "min_vram_gb": 6.0
    },
    "ultra_efficient": {
        "name": "Ultra Efficient",
        "description": "Maximum memory savings for low-end GPUs",
        "settings": {
            "quantization": QuantizationLevel.INT8,
            "enable_offload": True,
            "vae_tile_size": 128,
            "max_vram_usage_gb": 6.0
        },
        "vram_savings_gb": 6.0,
        "performance_impact": "high",
        "quality_impact": "medium",
        "recommended_for": ["GTX 1660", "RTX 3050", "RTX 4050"],
        "min_vram_gb": 4.0
    },
    "high_performance": {
        "name": "High Performance",
        "description": "Best quality for high-end GPUs with ample VRAM",
        "settings": {
            "quantization": QuantizationLevel.FP16,
            "enable_offload": False,
            "vae_tile_size": 512,
            "max_vram_usage_gb": 20.0
        },
        "vram_savings_gb": 1.0,
        "performance_impact": "low",
        "quality_impact": "none",
        "recommended_for": ["RTX 4090", "RTX 4080 Super", "RTX 5090"],
        "min_vram_gb": 16.0
    },
    "quality_focused": {
        "name": "Quality Focused",
        "description": "Prioritizes output quality over speed",
        "settings": {
            "quantization": QuantizationLevel.FP16,
            "enable_offload": False,
            "vae_tile_size": 512,
            "max_vram_usage_gb": 16.0
        },
        "vram_savings_gb": 1.5,
        "performance_impact": "low",
        "quality_impact": "none",
        "recommended_for": ["RTX 4070 Ti", "RTX 4080", "RTX 4090"],
        "min_vram_gb": 12.0
    },
    "speed_focused": {
        "name": "Speed Focused",
        "description": "Optimized for fastest generation times",
        "settings": {
            "quantization": QuantizationLevel.BF16,
            "enable_offload": False,
            "vae_tile_size": 256,
            "max_vram_usage_gb": 14.0
        },
        "vram_savings_gb": 3.0,
        "performance_impact": "low",
        "quality_impact": "minimal",
        "recommended_for": ["RTX 4070", "RTX 4080", "RTX 4090"],
        "min_vram_gb": 10.0
    }
}

@router.get("/optimization/presets")
async def get_optimization_presets(
    integration: SystemIntegration = Depends(get_system_integration)
):
    """
    Get all available optimization presets with system compatibility
    Requirement 4.2: Create advanced optimization presets and recommendations
    """
    try:
        # Get current system stats to determine compatibility
        stats = await integration.get_enhanced_system_stats()
        current_vram_gb = 0.0
        
        if stats:
            current_vram_gb = stats.get("vram_total_mb", 0) / 1024
        
        # Add compatibility information to presets
        enhanced_presets = {}
        for preset_id, preset in OPTIMIZATION_PRESETS.items():
            enhanced_preset = preset.copy()
            
            # Check compatibility
            is_compatible = current_vram_gb >= preset["min_vram_gb"]
            enhanced_preset["compatible"] = is_compatible
            enhanced_preset["compatibility_reason"] = (
                "Compatible with your system" if is_compatible 
                else f"Requires at least {preset['min_vram_gb']}GB VRAM (you have {current_vram_gb:.1f}GB)"
            )
            
            # Calculate estimated generation time impact
            performance_multipliers = {
                "low": 1.0,
                "medium": 1.3,
                "high": 1.8
            }
            enhanced_preset["estimated_time_multiplier"] = performance_multipliers.get(
                preset["performance_impact"], 1.0
            )
            
            enhanced_presets[preset_id] = enhanced_preset
        
        return {
            "presets": enhanced_presets,
            "system_info": {
                "total_vram_gb": current_vram_gb,
                "recommended_presets": _get_recommended_presets(current_vram_gb)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting optimization presets: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not retrieve optimization presets")

@router.get("/optimization/recommendations")
async def get_optimization_recommendations(
    integration: SystemIntegration = Depends(get_system_integration)
):
    """
    Get personalized optimization recommendations based on current system state
    Requirement 4.3: Provide real-time VRAM usage estimates and recommendations
    """
    try:
        # Get current system stats
        stats = await integration.get_enhanced_system_stats()
        
        if not stats:
            raise HTTPException(status_code=503, detail="Could not retrieve system statistics")
        
        current_vram_gb = stats.get("vram_total_mb", 0) / 1024
        used_vram_gb = stats.get("vram_used_mb", 0) / 1024
        vram_percent = stats.get("vram_percent", 0)
        
        recommendations = []
        priority_recommendations = []
        
        # Critical VRAM usage
        if vram_percent > 95:
            priority_recommendations.append({
                "type": "critical",
                "title": "Critical VRAM Usage",
                "message": "VRAM usage is critically high. Generation may fail.",
                "actions": [
                    "Apply 'Ultra Efficient' preset immediately",
                    "Enable model offloading",
                    "Use int8 quantization",
                    "Reduce VAE tile size to 128"
                ],
                "estimated_savings_gb": 6.0
            })
        
        # High VRAM usage
        elif vram_percent > 85:
            recommendations.append({
                "type": "warning",
                "title": "High VRAM Usage",
                "message": "Consider optimizing to prevent generation failures.",
                "actions": [
                    "Apply 'Memory Efficient' preset",
                    "Enable model offloading",
                    "Use bf16 quantization"
                ],
                "estimated_savings_gb": 4.5
            })
        
        # Performance optimization opportunities
        if current_vram_gb >= 16.0 and vram_percent < 60:
            recommendations.append({
                "type": "info",
                "title": "Performance Opportunity",
                "message": "Your system can handle higher quality settings.",
                "actions": [
                    "Try 'High Performance' preset",
                    "Increase VAE tile size to 512",
                    "Disable model offloading for faster generation"
                ],
                "estimated_impact": "20-30% faster generation"
            })
        
        # GPU-specific recommendations
        gpu_recommendations = _get_gpu_specific_recommendations(current_vram_gb, vram_percent)
        recommendations.extend(gpu_recommendations)
        
        # Model-specific recommendations
        model_recommendations = _get_model_specific_recommendations(stats)
        recommendations.extend(model_recommendations)
        
        # Determine best preset for current system
        best_preset_id = _determine_best_preset(current_vram_gb, vram_percent)
        best_preset = OPTIMIZATION_PRESETS.get(best_preset_id)
        
        return {
            "priority_recommendations": priority_recommendations,
            "recommendations": recommendations,
            "best_preset": {
                "id": best_preset_id,
                "preset": best_preset
            },
            "system_analysis": {
                "total_vram_gb": current_vram_gb,
                "used_vram_gb": used_vram_gb,
                "vram_percent": vram_percent,
                "vram_category": _categorize_vram(current_vram_gb),
                "optimization_potential": _calculate_optimization_potential(stats)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting optimization recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not generate optimization recommendations")

@router.post("/optimization/apply-preset/{preset_id}")
async def apply_optimization_preset(
    preset_id: str,
    integration: SystemIntegration = Depends(get_system_integration)
):
    """
    Apply a specific optimization preset
    """
    try:
        if preset_id not in OPTIMIZATION_PRESETS:
            raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found")
        
        preset = OPTIMIZATION_PRESETS[preset_id]
        
        # Check system compatibility
        stats = await integration.get_enhanced_system_stats()
        if stats:
            current_vram_gb = stats.get("vram_total_mb", 0) / 1024
            if current_vram_gb < preset["min_vram_gb"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Preset requires at least {preset['min_vram_gb']}GB VRAM, but system has {current_vram_gb:.1f}GB"
                )
        
        # Apply the preset settings
        settings = OptimizationSettings(**preset["settings"])
        
        # Here you would typically save to database or apply to system
        # For now, we'll return the applied settings
        
        return {
            "message": f"Applied '{preset['name']}' optimization preset",
            "preset": preset,
            "applied_settings": settings,
            "expected_benefits": {
                "vram_savings_gb": preset["vram_savings_gb"],
                "performance_impact": preset["performance_impact"],
                "quality_impact": preset["quality_impact"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying optimization preset: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not apply optimization preset")

@router.get("/optimization/analysis")
async def get_optimization_analysis(
    integration: SystemIntegration = Depends(get_system_integration)
):
    """
    Get detailed optimization analysis for the current system
    """
    try:
        stats = await integration.get_enhanced_system_stats()
        
        if not stats:
            raise HTTPException(status_code=503, detail="Could not retrieve system statistics")
        
        current_vram_gb = stats.get("vram_total_mb", 0) / 1024
        used_vram_gb = stats.get("vram_used_mb", 0) / 1024
        vram_percent = stats.get("vram_percent", 0)
        
        # Analyze each preset's impact
        preset_analysis = {}
        for preset_id, preset in OPTIMIZATION_PRESETS.items():
            estimated_vram_after = max(0, used_vram_gb - preset["vram_savings_gb"])
            estimated_percent_after = (estimated_vram_after / current_vram_gb) * 100 if current_vram_gb > 0 else 0
            
            preset_analysis[preset_id] = {
                "preset": preset,
                "current_compatibility": current_vram_gb >= preset["min_vram_gb"],
                "estimated_vram_usage_after": estimated_vram_after,
                "estimated_vram_percent_after": estimated_percent_after,
                "vram_reduction": preset["vram_savings_gb"],
                "safety_margin": max(0, current_vram_gb - preset["min_vram_gb"]),
                "recommended_score": _calculate_preset_score(preset, current_vram_gb, vram_percent)
            }
        
        # Overall system optimization potential
        max_savings = max(preset["vram_savings_gb"] for preset in OPTIMIZATION_PRESETS.values())
        optimization_potential = min(100, (max_savings / current_vram_gb) * 100) if current_vram_gb > 0 else 0
        
        return {
            "system_overview": {
                "total_vram_gb": current_vram_gb,
                "used_vram_gb": used_vram_gb,
                "free_vram_gb": current_vram_gb - used_vram_gb,
                "vram_percent": vram_percent,
                "vram_category": _categorize_vram(current_vram_gb),
                "optimization_potential_percent": optimization_potential
            },
            "preset_analysis": preset_analysis,
            "recommendations": {
                "top_preset": max(preset_analysis.keys(), key=lambda k: preset_analysis[k]["recommended_score"]),
                "most_compatible": [k for k, v in preset_analysis.items() if v["current_compatibility"]],
                "highest_savings": max(OPTIMIZATION_PRESETS.keys(), key=lambda k: OPTIMIZATION_PRESETS[k]["vram_savings_gb"])
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting optimization analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not perform optimization analysis")

def _get_recommended_presets(vram_gb: float) -> List[str]:
    """Get recommended presets based on VRAM amount"""
    if vram_gb >= 16.0:
        return ["high_performance", "quality_focused", "balanced"]
    elif vram_gb >= 12.0:
        return ["balanced", "quality_focused", "speed_focused"]
    elif vram_gb >= 8.0:
        return ["memory_efficient", "balanced", "speed_focused"]
    else:
        return ["ultra_efficient", "memory_efficient"]

def _get_gpu_specific_recommendations(vram_gb: float, vram_percent: float) -> List[Dict[str, Any]]:
    """Get GPU-specific optimization recommendations"""
    recommendations = []
    
    # RTX 4090 specific
    if vram_gb >= 24.0:
        recommendations.append({
            "type": "info",
            "title": "RTX 4090 Optimization",
            "message": "Your high-end GPU can handle maximum quality settings.",
            "actions": [
                "Use 'High Performance' preset",
                "Consider larger batch sizes",
                "Enable advanced features"
            ]
        })
    
    # RTX 4080 specific
    elif vram_gb >= 16.0:
        recommendations.append({
            "type": "info",
            "title": "RTX 4080 Optimization",
            "message": "Excellent balance of performance and efficiency available.",
            "actions": [
                "Use 'Balanced' or 'Quality Focused' preset",
                "fp16 quantization recommended",
                "Model offloading not necessary"
            ]
        })
    
    # Mid-range GPU
    elif vram_gb >= 8.0:
        recommendations.append({
            "type": "info",
            "title": "Mid-Range GPU Optimization",
            "message": "Consider memory-efficient settings for best experience.",
            "actions": [
                "Use 'Memory Efficient' preset",
                "Enable model offloading",
                "Use bf16 quantization"
            ]
        })
    
    # Low-end GPU
    else:
        recommendations.append({
            "type": "warning",
            "title": "Low VRAM GPU Detected",
            "message": "Aggressive optimization required for stable generation.",
            "actions": [
                "Use 'Ultra Efficient' preset",
                "Enable all memory optimizations",
                "Consider lower resolution outputs"
            ]
        })
    
    return recommendations

def _get_model_specific_recommendations(stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get model-specific optimization recommendations"""
    recommendations = []
    
    # These would be based on actual model usage patterns
    # For now, providing general recommendations
    
    recommendations.append({
        "type": "info",
        "title": "Model Optimization",
        "message": "Consider model-specific optimizations.",
        "actions": [
            "Use appropriate quantization for your models",
            "Enable model caching for frequently used models",
            "Consider model pruning for production use"
        ]
    })
    
    return recommendations

def _determine_best_preset(vram_gb: float, vram_percent: float) -> str:
    """Determine the best preset for current system state"""
    if vram_percent > 90:
        return "ultra_efficient"
    elif vram_percent > 80:
        return "memory_efficient"
    elif vram_gb >= 16.0 and vram_percent < 60:
        return "high_performance"
    elif vram_gb >= 12.0:
        return "balanced"
    elif vram_gb >= 8.0:
        return "memory_efficient"
    else:
        return "ultra_efficient"

def _categorize_vram(vram_gb: float) -> str:
    """Categorize VRAM amount"""
    if vram_gb >= 20.0:
        return "high_end"
    elif vram_gb >= 12.0:
        return "mid_high"
    elif vram_gb >= 8.0:
        return "mid_range"
    elif vram_gb >= 6.0:
        return "low_mid"
    else:
        return "low_end"

def _calculate_optimization_potential(stats: Dict[str, Any]) -> float:
    """Calculate optimization potential as a percentage"""
    vram_percent = stats.get("vram_percent", 0)
    
    if vram_percent > 90:
        return 90.0  # High potential for optimization
    elif vram_percent > 80:
        return 70.0
    elif vram_percent > 60:
        return 40.0
    else:
        return 20.0  # Low potential, system is already efficient

def _calculate_preset_score(preset: Dict[str, Any], vram_gb: float, vram_percent: float) -> float:
    """Calculate a score for how well a preset fits the current system"""
    score = 0.0
    
    # Compatibility score
    if vram_gb >= preset["min_vram_gb"]:
        score += 30.0
    else:
        score -= 20.0
    
    # VRAM usage appropriateness
    if vram_percent > 90:
        # High usage - prefer efficient presets
        if preset["vram_savings_gb"] >= 4.0:
            score += 25.0
        elif preset["vram_savings_gb"] >= 2.0:
            score += 15.0
    elif vram_percent < 60:
        # Low usage - can afford performance presets
        if preset["performance_impact"] == "low":
            score += 20.0
    
    # VRAM category appropriateness
    vram_category = _categorize_vram(vram_gb)
    if vram_category == "high_end" and preset["name"] in ["High Performance", "Quality Focused"]:
        score += 15.0
    elif vram_category == "mid_range" and preset["name"] in ["Balanced", "Memory Efficient"]:
        score += 15.0
    elif vram_category == "low_end" and preset["name"] == "Ultra Efficient":
        score += 15.0
    
    return max(0.0, score)