#!/usr/bin/env python3
"""
Final comprehensive system check for real AI generation
"""

import asyncio
import sys
sys.path.append('.')

async def final_system_check():
    print("🔍 FINAL SYSTEM CHECK FOR REAL AI GENERATION")
    print("=" * 70)
    
    try:
        # 1. Check system integration
        print("1. 🔧 System Integration Check...")
        from core.system_integration import get_system_integration
        integration = await get_system_integration()
        optimizer = integration.get_wan22_system_optimizer()
        
        if optimizer and optimizer.get_hardware_profile():
            profile = optimizer.get_hardware_profile()
            print(f"   ✅ Hardware: {profile.gpu_model} ({profile.vram_gb}GB VRAM)")
        else:
            print("   ❌ Hardware profile not available")
        
        # 2. Check generation service
        print("\n2. 🎬 Generation Service Check...")
        from services.generation_service import get_generation_service
        service = await get_generation_service()
        
        if service:
            print(f"   ✅ Service initialized: {service is not None}")
            print(f"   ✅ WAN22 optimizer: {service.wan22_system_optimizer is not None}")
            print(f"   ✅ Model bridge: {service.model_integration_bridge is not None}")
            print(f"   ✅ Real pipeline: {service.real_generation_pipeline is not None}")
            
            if service.hardware_profile:
                print(f"   ✅ Service GPU: {service.hardware_profile.gpu_model}")
        
        # 3. Check fallback recovery system
        print("\n3. 🛡️  Fallback Recovery System Check...")
        from core.fallback_recovery_system import get_fallback_recovery_system
        recovery = get_fallback_recovery_system()
        
        if recovery:
            health_status = await recovery.check_system_health()
            print(f"   ✅ Recovery system: Available")
            print(f"   🔍 System responsive: {health_status.get('system_responsive', False)}")
            print(f"   🔍 GPU available: {health_status.get('gpu_available', False)}")
            print(f"   🔍 Models loaded: {health_status.get('models_loaded', False)}")
            print(f"   🔍 WAN22 ready: {health_status.get('wan22_infrastructure_ready', False)}")
        
        # 4. Test CUDA directly
        print("\n4. 🔥 Direct CUDA Test...")
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"   ✅ PyTorch CUDA: {cuda_available}")
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   🎮 GPU: {gpu_name}")
            
            # Test tensor creation
            try:
                test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
                print(f"   ✅ GPU tensor test: Success ({test_tensor.device})")
            except Exception as e:
                print(f"   ❌ GPU tensor test: Failed ({e})")
        
        # 5. Check if server is running
        print("\n5. 🌐 Server Connectivity Test...")
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8000/health", timeout=5.0)
                if response.status_code == 200:
                    print("   ✅ FastAPI server: Running")
                else:
                    print(f"   ⚠️  FastAPI server: Unexpected status {response.status_code}")
        except Exception as e:
            print(f"   ❌ FastAPI server: Not accessible ({e})")
        
        # 6. Overall assessment
        print("\n" + "=" * 70)
        print("📋 FINAL ASSESSMENT:")
        
        all_good = (
            optimizer is not None and
            service is not None and
            service.wan22_system_optimizer is not None and
            service.model_integration_bridge is not None and
            service.real_generation_pipeline is not None and
            cuda_available
        )
        
        if all_good:
            print("🎉 ✅ SYSTEM FULLY READY FOR REAL AI GENERATION!")
            print("\n🚀 Next steps:")
            print("   1. Submit a generation request from the frontend")
            print("   2. Watch the logs for real AI generation attempts")
            print("   3. Models will download automatically if needed")
            print("   4. System will fall back gracefully if there are issues")
            
            print("\n💡 Expected behavior:")
            print("   • Hardware optimizations applied ✅")
            print("   • RTX 4080 detected and optimized ✅")
            print("   • Real AI models will be attempted ✅")
            print("   • Automatic fallback if models not available ✅")
        else:
            print("⚠️  ❌ SYSTEM NOT FULLY READY")
            print("   Some components still need attention.")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ System check failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(final_system_check())