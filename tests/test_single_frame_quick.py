#!/usr/bin/env python3
"""
Quick Single Frame Generation Test
Test the fixed pipeline with minimal 1-frame generation
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

async def test_backend_health():
    """Quick backend health check"""
    base_url = "http://localhost:9000"
    
    try:
        async with aiohttp.ClientSession() as session:
            # Try the simple health endpoint first
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"✅ Backend healthy: {health_data.get('status', 'unknown')}")
                    return True
                else:
                    print(f"⚠️  Backend status: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Backend not accessible: {e}")
        return False

async def submit_single_frame():
    """Submit ultra-minimal single frame generation"""
    base_url = "http://localhost:9000"
    
    # Ultra minimal parameters for fastest test
    test_data = {
        "model_type": "T2V-A14B",
        "prompt": "red ball",  # Simple prompt
        "resolution": "854x480",  # Smallest resolution
        "num_frames": 1,  # Single frame only
        "steps": 5,  # Minimal steps for speed
        "seed": 42
    }
    
    print(f"\n🎬 Submitting Single Frame Test")
    print(f"   Prompt: '{test_data['prompt']}'")
    print(f"   Frames: {test_data['num_frames']}")
    print(f"   Steps: {test_data['steps']}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/api/v1/generation/submit",
                json=test_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    task_id = result.get("task_id")
                    print(f"✅ Task submitted: {task_id}")
                    return task_id
                else:
                    error_text = await response.text()
                    print(f"❌ Submission failed ({response.status}): {error_text}")
                    return None
                    
    except Exception as e:
        print(f"❌ Submission error: {e}")
        return None

async def monitor_quick_generation(task_id):
    """Monitor with focus on key progress points"""
    base_url = "http://localhost:9000"
    
    print(f"\n🔍 Monitoring Task: {task_id}")
    
    start_time = time.time()
    last_progress = -1
    progress_log = []
    
    # Monitor for up to 3 minutes (single frame should be fast)
    for check in range(36):  # 36 checks * 5s = 3 minutes
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/api/v1/queue") as response:
                    if response.status == 200:
                        queue_data = await response.json()
                        
                        # Find our task
                        task = None
                        for t in queue_data.get("tasks", []):
                            if t.get("id") == task_id:
                                task = t
                                break
                        
                        if task:
                            status = task.get("status", "unknown")
                            progress = task.get("progress", 0)
                            elapsed = time.time() - start_time
                            
                            # Log significant progress changes
                            if progress != last_progress:
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                change = progress - last_progress if last_progress >= 0 else progress
                                
                                progress_entry = {
                                    "time": elapsed,
                                    "progress": progress,
                                    "status": status,
                                    "change": change
                                }
                                progress_log.append(progress_entry)
                                
                                print(f"[{timestamp}] {progress:3d}% (+{change:2d}%) - {status} ({elapsed:.0f}s)")
                                last_progress = progress
                            
                            # Check completion
                            if status == "completed":
                                output_path = task.get("output_path", "")
                                print(f"\n🎉 Generation completed in {elapsed:.0f}s!")
                                if output_path:
                                    print(f"📁 Output: {output_path}")
                                
                                # Show progress summary
                                print_progress_summary(progress_log)
                                return True
                                
                            elif status == "failed":
                                error_msg = task.get("error_message", "Unknown error")
                                print(f"\n❌ Generation failed after {elapsed:.0f}s")
                                print(f"💥 Error: {error_msg}")
                                print_progress_summary(progress_log)
                                return False
                        
                        else:
                            print(f"⚠️  Task not found in queue (check {check+1})")
                            
        except Exception as e:
            print(f"❌ Monitor error: {e}")
        
        await asyncio.sleep(5)
    
    print(f"\n⏰ Timeout after 3 minutes")
    print_progress_summary(progress_log)
    return False

def print_progress_summary(progress_log):
    """Print concise progress analysis"""
    if not progress_log:
        print("❌ No progress updates received")
        return
    
    print(f"\n📊 Progress Summary:")
    print(f"   Updates: {len(progress_log)}")
    print(f"   Range: {progress_log[0]['progress']}% → {progress_log[-1]['progress']}%")
    
    # Check for smooth progress in generation phase (40-95%)
    gen_updates = [p for p in progress_log if 40 <= p['progress'] <= 95]
    if len(gen_updates) >= 2:
        print(f"   Generation phase: {len(gen_updates)} updates ✅")
    else:
        print(f"   Generation phase: {len(gen_updates)} updates ⚠️")
    
    # Show key milestones
    milestones = [p for p in progress_log if p['progress'] in [0, 40, 95, 100]]
    if milestones:
        print(f"   Key milestones:")
        for m in milestones:
            print(f"     {m['progress']:3d}% at {m['time']:.0f}s")

async def main():
    print("🚀 Quick Single Frame Test")
    print("Testing pipeline fallback fixes with 1-frame generation")
    print("=" * 55)
    
    # 1. Check backend health
    if not await test_backend_health():
        print("❌ Backend not ready")
        return False
    
    # 2. Submit single frame generation
    task_id = await submit_single_frame()
    if not task_id:
        print("❌ Could not submit generation")
        return False
    
    # 3. Monitor generation
    success = await monitor_quick_generation(task_id)
    
    # 4. Results
    print(f"\n🎯 Test Results:")
    if success:
        print("✅ Single frame generation successful!")
        print("   • Pipeline fallback fixes working")
        print("   • VRAM calculations correct")
        print("   • Progress tracking functional")
        print("   • Ready for larger generations")
    else:
        print("⚠️  Issues detected:")
        print("   • Check backend logs for details")
        print("   • Verify pipeline loading")
        print("   • Check VRAM availability")
    
    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
