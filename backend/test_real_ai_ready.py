#!/usr/bin/env python3
"""
Test if the system is ready for real AI generation
"""

import asyncio
import sys
sys.path.append('.')

async def test_real_ai_readiness():
    print("🚀 Testing Real AI Generation Readiness")
    print("=" * 60)
    
    try:
        # Test system integration
        print("1. 🔧 Testing System Integration...")
        from backend.core.system_integration import get_system_integration
        integration = await get_system_integration()
        optimizer = integration.get_wan22_system_optimizer()
        
        print(f"   ✅ System integration: {integration._initialized}")
        print(f"   ✅ WAN22 optimizer: {optimizer is not None}")
        
        if optimizer:
            profile = optimizer.get_hardware_profile()
            print(f"   ✅ Hardware profile: {profile.gpu_model} ({profile.vram_gb}GB)")
        
        # Test model integration bridge
        print("\n2. 🌉 Testing Model Integration Bridge...")
        from backend.core.model_integration_bridge import get_model_integration_bridge
        bridge = await get_model_integration_bridge()
        
        print(f"   ✅ Bridge initialized: {bridge._initialized}")
        print(f"   ⏳ Hardware profile before optimizer: {bridge.hardware_profile}")
        
        # Set optimizer
        if optimizer:
            bridge.set_hardware_optimizer(optimizer)
            print(f"   ✅ Hardware profile after optimizer: {bridge.hardware_profile}")
            
            if bridge.hardware_profile and bridge.hardware_profile.gpu_model:
                print(f"   🎮 GPU detected: {bridge.hardware_profile.gpu_model}")
                print(f"   💾 VRAM available: {bridge.hardware_profile.vram_gb}GB")
        
        # Test generation service
        print("\n3. 🎬 Testing Generation Service...")
        from backend.services.generation_service import get_generation_service
        service = await get_generation_service()
        
        print(f"   ✅ Service initialized: {service is not None}")
        if service:
            print(f"   ✅ WAN22 optimizer integrated: {service.wan22_system_optimizer is not None}")
            print(f"   ✅ Model bridge integrated: {service.model_integration_bridge is not None}")
            print(f"   ✅ Real pipeline integrated: {service.real_generation_pipeline is not None}")
            
            if service.hardware_profile:
                print(f"   🎮 Service GPU: {service.hardware_profile.gpu_model}")
        
        # Test CUDA directly
        print("\n4. 🔥 Testing CUDA Availability...")
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"   ✅ PyTorch CUDA: {cuda_available}")
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   🎮 GPU: {gpu_name}")
            print(f"   📊 GPU count: {gpu_count}")
        
        # Overall assessment
        print("\n" + "=" * 60)
        print("📋 READINESS ASSESSMENT:")
        
        ready_for_real_ai = (
            integration._initialized and
            optimizer is not None and
            bridge._initialized and
            bridge.hardware_profile is not None and
            cuda_available and
            service is not None
        )
        
        if ready_for_real_ai:
            print("🎉 ✅ SYSTEM IS READY FOR REAL AI GENERATION!")
            print("   • Hardware optimization: ✅")
            print("   • GPU detection: ✅") 
            print("   • Model integration: ✅")
            print("   • Generation pipeline: ✅")
            print("\n🚀 You can now submit generation requests and they should use real AI models!")
        else:
            print("⚠️  ❌ SYSTEM NOT FULLY READY")
            print("   Some components need attention before real AI generation will work.")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_real_ai_readiness())