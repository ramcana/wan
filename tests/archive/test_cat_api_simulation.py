#!/usr/bin/env python3
"""
Simulate API call for "A cat walking in the park" generation
Tests the complete generation pipeline without requiring a running server
"""

import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

async def simulate_generation_request():
    """Simulate a complete generation request"""
    print("🎬 Simulating Generation Request")
    print("=" * 50)
    
    try:
        # Import the enhanced generation components
        from api.enhanced_generation import (
            ModelDetectionService, 
            PromptEnhancementService,
            GenerationRequest,
            GenerationResponse
        )
        
        # Create request data
        prompt = "A cat walking in the park"
        print(f"📝 Prompt: '{prompt}'")
        
        # Step 1: Auto-detect model type
        detected_model_type = ModelDetectionService.detect_model_type(
            prompt, 
            has_image=False,
            has_end_image=False
        )
        print(f"🤖 Auto-detected model: {detected_model_type}")
        
        # Step 2: Get model requirements
        model_requirements = ModelDetectionService.get_model_requirements(detected_model_type)
        print(f"📊 Model requirements: {json.dumps(model_requirements, indent=2)}")
        
        # Step 3: Enhance prompt
        enhanced_prompt = PromptEnhancementService.enhance_prompt(
            prompt, detected_model_type, {
                "enhance_quality": True,
                "enhance_technical": True
            }
        )
        print(f"✨ Enhanced prompt: '{enhanced_prompt}'")
        
        # Step 4: Create generation parameters (mock)
        class MockGenerationParams:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        generation_params = MockGenerationParams(
            prompt=enhanced_prompt,
            model_type=detected_model_type,
            image_path=None,
            end_image_path=None,
            resolution="1280x720",
            num_inference_steps=50,
            lora_path=None,
            lora_strength=1.0
        )
        
        print(f"🔧 Generation parameters created:")
        print(f"   Model: {generation_params.model_type}")
        print(f"   Resolution: {generation_params.resolution}")
        print(f"   Steps: {generation_params.num_inference_steps}")
        
        # Step 5: Estimate generation time
        frames_count = 16  # Default frame count
        time_per_frame = model_requirements.get("estimated_time_per_frame", 1.0)
        estimated_time_minutes = (frames_count * time_per_frame) / 60.0
        print(f"⏱️ Estimated time: {estimated_time_minutes:.1f} minutes")
        
        # Step 6: Create mock response
        task_id = f"gen_cat_park_{datetime.now().strftime('%H%M%S')}"
        
        applied_optimizations = [
            "Hardware-specific quantization",
            "Memory optimization", 
            "Pipeline caching"
        ]
        
        if enhanced_prompt != prompt:
            applied_optimizations.append("Prompt enhancement")
        
        response = GenerationResponse(
            success=True,
            task_id=task_id,
            message=f"Generation task submitted successfully with {detected_model_type}",
            detected_model_type=detected_model_type,
            estimated_time_minutes=estimated_time_minutes,
            queue_position=0,
            enhanced_prompt=enhanced_prompt if enhanced_prompt != prompt else None,
            applied_optimizations=applied_optimizations
        )
        
        print(f"\n📋 Generation Response:")
        print(f"   Success: {response.success}")
        print(f"   Task ID: {response.task_id}")
        print(f"   Message: {response.message}")
        print(f"   Queue Position: {response.queue_position}")
        print(f"   Applied Optimizations: {', '.join(response.applied_optimizations)}")
        
        print("\n✅ Generation request simulation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Generation request simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_model_detection_endpoint():
    """Test the model detection endpoint"""
    print("\n🔍 Testing Model Detection Endpoint")
    print("=" * 50)
    
    try:
        from api.enhanced_generation import ModelDetectionService
        
        prompt = "A cat walking in the park"
        has_image = False
        has_end_image = False
        
        # Simulate the /models/detect endpoint
        detected_model = ModelDetectionService.detect_model_type(prompt, has_image, has_end_image)
        requirements = ModelDetectionService.get_model_requirements(detected_model)
        
        # Generate explanation
        explanation = []
        if has_image and has_end_image:
            explanation.append("Both start and end images provided - TI2V recommended for interpolation")
        elif has_image:
            text_indicators = any(keyword in prompt.lower() for keyword in [
                "transform", "change", "evolve", "morph", "animate"
            ])
            if text_indicators:
                explanation.append("Image + text transformation keywords detected - TI2V recommended")
            else:
                explanation.append("Single image provided - I2V recommended for pure image animation")
        else:
            explanation.append("Text-only input - T2V recommended for pure text-to-video generation")
        
        response = {
            "detected_model_type": detected_model,
            "confidence": 0.9,
            "explanation": explanation,
            "requirements": requirements,
            "alternatives": [
                model for model in ["T2V-A14B", "I2V-A14B", "TI2V-5B"] 
                if model != detected_model
            ]
        }
        
        print(f"📊 Model Detection Response:")
        print(json.dumps(response, indent=2))
        
        print("\n✅ Model detection endpoint test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model detection endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_prompt_enhancement_endpoint():
    """Test the prompt enhancement endpoint"""
    print("\n✨ Testing Prompt Enhancement Endpoint")
    print("=" * 50)
    
    try:
        from api.enhanced_generation import PromptEnhancementService
        
        prompt = "A cat walking in the park"
        model_type = "T2V-A14B"
        enhance_quality = True
        enhance_technical = True
        
        # Simulate the /prompt/enhance endpoint
        enhanced_prompt = PromptEnhancementService.enhance_prompt(
            prompt, model_type, {
                "enhance_quality": enhance_quality,
                "enhance_technical": enhance_technical
            }
        )
        
        enhancements_applied = []
        if enhanced_prompt != prompt:
            # Detect what enhancements were applied
            added_parts = enhanced_prompt.replace(prompt, "").strip(", ")
            if added_parts:
                enhancements_applied = [part.strip() for part in added_parts.split(",")]
        
        response = {
            "original_prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "enhancements_applied": enhancements_applied,
            "model_type": model_type,
            "character_count": {
                "original": len(prompt),
                "enhanced": len(enhanced_prompt),
                "difference": len(enhanced_prompt) - len(prompt)
            }
        }
        
        print(f"📝 Prompt Enhancement Response:")
        print(json.dumps(response, indent=2))
        
        print("\n✅ Prompt enhancement endpoint test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Prompt enhancement endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_capabilities_endpoint():
    """Test the capabilities endpoint"""
    print("\n🔧 Testing Capabilities Endpoint")
    print("=" * 50)
    
    try:
        from api.enhanced_generation import ModelDetectionService
        
        # Simulate the /capabilities endpoint
        models_info = {}
        for model_type in ["T2V-A14B", "I2V-A14B", "TI2V-5B"]:
            models_info[model_type] = ModelDetectionService.get_model_requirements(model_type)
        
        response = {
            "supported_models": list(models_info.keys()),
            "models_info": models_info,
            "supported_resolutions": ["854x480", "1024x576", "1280x720", "1920x1080"],
            "supported_formats": ["mp4"],
            "max_steps": 100,
            "min_steps": 1,
            "default_steps": 50,
            "max_prompt_length": 500,
            "max_image_size_mb": 10,
            "supported_image_formats": ["JPEG", "PNG", "WebP"],
            "features": {
                "auto_model_detection": True,
                "prompt_enhancement": True,
                "lora_support": True,
                "hardware_optimization": True,
                "real_time_progress": True,
                "queue_management": True
            }
        }
        
        print(f"🔧 Capabilities Response:")
        print(json.dumps(response, indent=2))
        
        print("\n✅ Capabilities endpoint test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Capabilities endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all API simulation tests"""
    print("🐱 API Simulation: 'A cat walking in the park'")
    print("=" * 60)
    
    results = []
    
    # Test generation request simulation
    gen_result = await simulate_generation_request()
    results.append(("Generation Request Simulation", gen_result))
    
    # Test model detection endpoint
    detection_result = await test_model_detection_endpoint()
    results.append(("Model Detection Endpoint", detection_result))
    
    # Test prompt enhancement endpoint
    enhancement_result = await test_prompt_enhancement_endpoint()
    results.append(("Prompt Enhancement Endpoint", enhancement_result))
    
    # Test capabilities endpoint
    capabilities_result = await test_capabilities_endpoint()
    results.append(("Capabilities Endpoint", capabilities_result))
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 API SIMULATION RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}   {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All API simulations passed! Ready to generate 'A cat walking in the park'")
        print("\n📋 Next Steps:")
        print("1. Start the backend server: python backend/app.py")
        print("2. Test with curl: curl -X POST http://localhost:9001/api/v1/generation/submit \\")
        print("   -F 'prompt=A cat walking in the park' \\")
        print("   -F 'model_type=T2V-A14B' \\")
        print("   -F 'resolution=1280x720'")
        print("3. Or use the CLI: python cli/main.py wan generate 'A cat walking in the park'")
    else:
        print(f"\n⚠️ {total-passed} simulation(s) failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
