"""
Demo script for Progress Tracker Integration
Tests the progress tracker with simulated video generation
"""

import time
import logging
from progress_tracker import (
    get_progress_tracker, 
    create_progress_callback,
    GenerationPhase
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simulate_video_generation():
    """Simulate a video generation process with progress tracking"""
    print("🎬 Starting video generation simulation...")
    
    # Get progress tracker
    config = {
        "progress_update_interval": 1.0,  # Update every second
        "enable_system_monitoring": True
    }
    tracker = get_progress_tracker(config)
    
    # Create callback to display progress
    def display_progress(progress_data):
        print(f"\n📊 Progress Update:")
        print(f"   Step: {progress_data.current_step}/{progress_data.total_steps}")
        print(f"   Progress: {progress_data.progress_percentage:.1f}%")
        print(f"   Phase: {progress_data.current_phase}")
        print(f"   Elapsed: {progress_data.elapsed_time:.1f}s")
        print(f"   ETA: {progress_data.estimated_remaining:.1f}s")
        print(f"   Frames: {progress_data.frames_processed}")
        print(f"   Speed: {progress_data.processing_speed:.2f} fps")
    
    tracker.add_update_callback(display_progress)
    
    # Start tracking
    total_steps = 50
    tracker.start_progress_tracking("demo_generation", total_steps)
    
    # Simulate generation phases
    phases = [
        (GenerationPhase.INITIALIZATION, 2),
        (GenerationPhase.MODEL_LOADING, 5),
        (GenerationPhase.PREPROCESSING, 8),
        (GenerationPhase.GENERATION, 30),
        (GenerationPhase.POSTPROCESSING, 3),
        (GenerationPhase.ENCODING, 2)
    ]
    
    current_step = 0
    frames_processed = 0
    
    for phase, steps_in_phase in phases:
        print(f"\n🔄 Entering phase: {phase.value}")
        tracker.update_phase(phase.value)
        
        for i in range(steps_in_phase):
            current_step += 1
            frames_processed += 2  # Simulate 2 frames per step
            
            # Simulate work
            time.sleep(0.2)
            
            # Update progress
            tracker.update_progress(
                step=current_step,
                frames_processed=frames_processed,
                additional_data={
                    "phase_step": i + 1,
                    "phase_total": steps_in_phase,
                    "memory_usage": 1024 + (current_step * 10)  # Simulate increasing memory
                }
            )
    
    # Complete tracking
    final_stats = tracker.complete_progress_tracking({
        "total_frames": frames_processed,
        "output_file": "demo_video.mp4",
        "file_size_mb": 125.5
    })
    
    print(f"\n✅ Generation completed!")
    print(f"   Total time: {final_stats.elapsed_seconds:.1f}s")
    print(f"   Total frames: {final_stats.frames_processed}")
    print(f"   Average FPS: {final_stats.frames_per_second:.2f}")
    print(f"   Phase durations: {final_stats.phase_durations}")
    
    # Display final HTML
    print(f"\n📊 Final Progress HTML:")
    print("=" * 50)
    html = tracker.get_progress_html()
    # Strip HTML tags for console display
    import re
    text_content = re.sub('<[^<]+?>', '', html)
    print(text_content)
    print("=" * 50)

def test_progress_callback_integration():
    """Test progress callback integration with generation functions"""
    print("\n🧪 Testing progress callback integration...")
    
    tracker = get_progress_tracker()
    
    # Create progress callback
    progress_callback = create_progress_callback(tracker)
    
    # Start tracking
    tracker.start_progress_tracking("callback_test", 20)
    
    # Simulate generation with callback
    for step in range(1, 21):
        progress_callback(
            step, 20,
            phase=GenerationPhase.GENERATION.value,
            frames_processed=step * 3,
            memory_usage=500 + (step * 25)
        )
        time.sleep(0.1)
    
    # Complete
    final_stats = tracker.complete_progress_tracking()
    print(f"✅ Callback test completed in {final_stats.elapsed_seconds:.1f}s")

def test_error_handling():
    """Test error handling during progress tracking"""
    print("\n🚨 Testing error handling...")
    
    tracker = get_progress_tracker()
    
    # Add a callback that will fail
    def failing_callback(data):
        if data.current_step == 5:
            raise Exception("Simulated callback error")
    
    tracker.add_update_callback(failing_callback)
    
    # Start tracking
    tracker.start_progress_tracking("error_test", 10)
    
    # Update progress (should handle callback error gracefully)
    for step in range(1, 11):
        tracker.update_progress(step)
        time.sleep(0.1)
    
    # Complete
    final_stats = tracker.complete_progress_tracking()
    print(f"✅ Error handling test completed - no crashes!")

def test_html_generation():
    """Test HTML generation for different progress states"""
    print("\n🎨 Testing HTML generation...")
    
    # Create a fresh tracker for this test
    tracker = ProgressTracker()
    
    # Test with no data
    html = tracker.get_progress_html()
    assert html == "", "Empty HTML should be returned when no data"
    print("✅ Empty state HTML test passed")
    
    # Test with progress data
    tracker.start_progress_tracking("html_test", 100)
    tracker.update_progress(25, GenerationPhase.GENERATION.value, 50)
    
    html = tracker.get_progress_html()
    assert "25.0%" in html, "Progress percentage should be in HTML"
    assert "Generation" in html, "Phase should be in HTML"
    assert "Step 25 / 100" in html, "Step info should be in HTML"
    print("✅ Progress HTML generation test passed")
    
    tracker.complete_progress_tracking()

if __name__ == "__main__":
    print("🚀 Progress Tracker Integration Demo")
    print("=" * 50)
    
    try:
        # Run tests
        simulate_video_generation()
        test_progress_callback_integration()
        test_error_handling()
        test_html_generation()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()