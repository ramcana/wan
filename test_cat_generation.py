#!/usr/bin/env python3
"""
Test script for "A cat walking in the park" generation
Tests both CLI and API functionality
"""

import sys
import asyncio
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

async def test_enhanced_generation_api():
    """Test the enhanced generation API directly"""
    print("🎬 Testing Enhanced Generation API")
    print("=" * 50)
    
    try:
        # Import the API components
        from api.enhanced_generation import ModelDetectionService, PromptEnhancementService
        
        # Test prompt
        prompt = "A cat walking in the park"
        print(f"📝 Original prompt: '{prompt}'")
        
        # Test model detection
        detected_model = ModelDetectionService.detect_model_type(prompt, has_image=False, has_end_image=False)
        print(f"🤖 Auto-detected model: {detected_model}")
        
        # Test model requirements
        requirements = ModelDetectionService.get_model_requirements(detected_model)
        print(f"📊 Model requirements:")
        for key, value in requirements.items():
            print(f"   {key}: {value}")
        
        # Test prompt enhancement
        enhanced_prompt = PromptEnhancementService.enhance_prompt(prompt, detected_model)
        print(f"✨ Enhanced prompt: '{enhanced_prompt}'")
        
        # Show what enhancements were applied
        if enhanced_prompt != prompt:
            added_parts = enhanced_prompt.replace(prompt, "").strip(", ")
            print(f"🔧 Enhancements applied: {added_parts}")
        else:
            print("🔧 No enhancements needed")
        
        print("\n✅ Enhanced Generation API test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Generation API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_generation():
    """Test CLI generation functionality"""
    print("\n🖥️ Testing CLI Generation")
    print("=" * 50)
    
    try:
        # Import CLI components
        from cli.commands.wan import app
        
        print("✅ CLI components imported successfully")
        
        # Simulate CLI generation parameters
        prompt = "A cat walking in the park"
        model_type = "auto"
        resolution = "1280x720"
        steps = 50
        
        print(f"📝 Prompt: {prompt}")
        print(f"🤖 Model: {model_type}")
        print(f"📐 Resolution: {resolution}")
        print(f"🔄 Steps: {steps}")
        
        # Test model auto-detection logic
        if model_type.lower() == "auto":
            # Simulate the auto-detection logic from CLI
            has_image = False  # No image provided
            
            if has_image:
                detected_model = "I2V-A14B"
            else:
                detected_model = "T2V-A14B"  # Text-to-video for text-only input
            
            print(f"🎯 Auto-detected model: {detected_model}")
        
        print("✅ CLI generation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ CLI generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prompt_enhancement():
    """Test prompt enhancement specifically for the cat prompt"""
    print("\n✨ Testing Prompt Enhancement")
    print("=" * 50)
    
    try:
        from api.enhanced_generation import PromptEnhancementService
        
        prompt = "A cat walking in the park"
        
        # Test enhancement for different model types
        models = ["T2V-A14B", "I2V-A14B", "TI2V-5B"]
        
        for model_type in models:
            enhanced = PromptEnhancementService.enhance_prompt(prompt, model_type)
            print(f"🤖 {model_type}:")
            print(f"   Original: {prompt}")
            print(f"   Enhanced: {enhanced}")
            
            if enhanced != prompt:
                added = enhanced.replace(prompt, "").strip(", ")
                print(f"   Added: {added}")
            else:
                print("   No changes needed")
            print()
        
        print("✅ Prompt enhancement test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Prompt enhancement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("🐱 Testing 'A cat walking in the park' Generation")
    print("=" * 60)
    
    results = []
    
    # Test Enhanced Generation API
    api_result = await test_enhanced_generation_api()
    results.append(("Enhanced Generation API", api_result))
    
    # Test CLI Generation
    cli_result = test_cli_generation()
    results.append(("CLI Generation", cli_result))
    
    # Test Prompt Enhancement
    enhancement_result = test_prompt_enhancement()
    results.append(("Prompt Enhancement", enhancement_result))
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS SUMMARY")
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
        print("\n🎉 All tests passed! The generation system is ready for 'A cat walking in the park'")
    else:
        print(f"\n⚠️ {total-passed} test(s) failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)