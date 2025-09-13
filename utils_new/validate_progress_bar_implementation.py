"""
Validation script for Progress Bar with Generation Statistics Implementation
Demonstrates all required functionality from the task requirements
"""

import time
import logging
from progress_tracker import (
    get_progress_tracker, 
    create_progress_callback,
    GenerationPhase,
    ProgressTracker
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def validate_requirement_11_1():
    """
    Requirement 11.1: WHEN video generation starts THEN the system SHALL display a progress bar showing completion percentage
    """
    print("\n🧪 Validating Requirement 11.1: Progress bar with completion percentage")
    
    tracker = ProgressTracker()
    tracker.start_progress_tracking("req_11_1", 100)
    
    # Update progress to 25%
    tracker.update_progress(25)
    html = tracker.get_progress_html()
    
    # Validate progress bar is displayed with percentage
    assert "25.0%" in html, "Progress percentage should be displayed"
    assert "progress" in html.lower(), "Progress bar should be present"
    assert "Step 25 / 100" in html, "Step information should be displayed"
    
    tracker.complete_progress_tracking()
    print("✅ Requirement 11.1 validated: Progress bar displays completion percentage")

def validate_requirement_11_2():
    """
    Requirement 11.2: WHEN generation is in progress THEN the system SHALL show current step number and total steps
    """
    print("\n🧪 Validating Requirement 11.2: Current step and total steps display")
    
    tracker = ProgressTracker()
    tracker.start_progress_tracking("req_11_2", 50)
    
    # Update to step 30
    tracker.update_progress(30)
    html = tracker.get_progress_html()
    
    # Validate step information is displayed
    assert "Step 30 / 50" in html, "Current step and total steps should be displayed"
    assert "60.0%" in html, "Progress percentage should be calculated correctly"
    
    tracker.complete_progress_tracking()
    print("✅ Requirement 11.2 validated: Current step and total steps are displayed")

def validate_requirement_11_3():
    """
    Requirement 11.3: WHEN generation is running THEN the system SHALL display estimated time remaining
    """
    print("\n🧪 Validating Requirement 11.3: Estimated time remaining display")
    
    tracker = ProgressTracker()
    tracker.start_progress_tracking("req_11_3", 20)
    
    # Add some elapsed time and progress
    time.sleep(0.2)
    tracker.update_progress(5)
    
    html = tracker.get_progress_html()
    
    # Validate ETA is displayed
    assert "Estimated Remaining" in html, "ETA section should be present"
    assert tracker.progress_data.estimated_remaining >= 0, "ETA should be calculated"
    
    tracker.complete_progress_tracking()
    print("✅ Requirement 11.3 validated: Estimated time remaining is displayed")

def validate_requirement_11_4():
    """
    Requirement 11.4: WHEN generation progresses THEN the system SHALL update statistics in real-time 
    (frames processed, current phase)
    """
    print("\n🧪 Validating Requirement 11.4: Real-time statistics updates")
    
    tracker = ProgressTracker()
    updates_received = []
    
    def capture_updates(progress_data):
        updates_received.append({
            'step': progress_data.current_step,
            'frames': progress_data.frames_processed,
            'phase': progress_data.current_phase
        })
    
    tracker.add_update_callback(capture_updates)
    tracker.start_progress_tracking("req_11_4", 10)
    
    # Simulate real-time updates with different phases
    phases = [GenerationPhase.MODEL_LOADING, GenerationPhase.GENERATION, GenerationPhase.ENCODING]
    
    for i, phase in enumerate(phases):
        step = (i + 1) * 3
        frames = step * 4
        tracker.update_progress(step, phase.value, frames)
        time.sleep(0.1)  # Simulate processing time
    
    # Validate real-time updates
    assert len(updates_received) >= 3, "Should receive real-time updates"
    assert updates_received[-1]['frames'] == 36, "Frames processed should be updated"
    assert updates_received[-1]['phase'] == GenerationPhase.ENCODING.value, "Phase should be updated"
    
    # Validate HTML shows current statistics
    html = tracker.get_progress_html()
    assert "36" in html, "Frames processed should be displayed"
    assert "Encoding" in html, "Current phase should be displayed"
    
    tracker.complete_progress_tracking()
    print("✅ Requirement 11.4 validated: Real-time statistics updates working")

def validate_requirement_11_5():
    """
    Requirement 11.5: WHEN generation completes THEN the system SHALL show final statistics 
    (total time, frames per second processed)
    """
    print("\n🧪 Validating Requirement 11.5: Final completion statistics")
    
    tracker = ProgressTracker()
    tracker.start_progress_tracking("req_11_5", 25)
    
    # Simulate full generation process
    for step in range(1, 26):
        frames = step * 2
        tracker.update_progress(step, GenerationPhase.GENERATION.value, frames)
        time.sleep(0.02)  # Small delay to accumulate time
    
    # Complete with final statistics
    final_stats = tracker.complete_progress_tracking({
        "output_file": "test_video.mp4",
        "total_frames": 50,
        "file_size_mb": 125.5
    })
    
    # Validate final statistics
    assert final_stats.elapsed_seconds > 0, "Total time should be recorded"
    assert final_stats.frames_processed == 50, "Total frames should be recorded"
    assert final_stats.frames_per_second > 0, "FPS should be calculated"
    assert "output_file" in final_stats.performance_metrics, "Final metrics should be stored"
    
    # Validate final HTML shows completion
    html = tracker.get_progress_html()
    assert "100.0%" in html, "Should show 100% completion"
    assert "Completion" in html, "Should show completion phase"
    
    print(f"   📊 Final Statistics:")
    print(f"      Total time: {final_stats.elapsed_seconds:.2f}s")
    print(f"      Frames processed: {final_stats.frames_processed}")
    print(f"      Average FPS: {final_stats.frames_per_second:.2f}")
    
    print("✅ Requirement 11.5 validated: Final completion statistics displayed")

def validate_generation_phase_tracking():
    """
    Validate generation phase tracking (initialization, processing, encoding)
    """
    print("\n🧪 Validating Generation Phase Tracking")
    
    tracker = ProgressTracker()
    tracker.start_progress_tracking("phase_tracking", 60)
    
    # Test all required phases
    phases = [
        (GenerationPhase.INITIALIZATION, 5),
        (GenerationPhase.MODEL_LOADING, 10),
        (GenerationPhase.PREPROCESSING, 15),
        (GenerationPhase.GENERATION, 40),
        (GenerationPhase.POSTPROCESSING, 55),
        (GenerationPhase.ENCODING, 60)
    ]
    
    for phase, step in phases:
        tracker.update_progress(step, phase.value)
        html = tracker.get_progress_html()
        
        # Validate phase is displayed correctly
        phase_display = phase.value.replace('_', ' ').title()
        assert phase_display in html, f"Phase {phase_display} should be displayed"
        
        time.sleep(0.05)  # Small delay between phases
    
    # Complete and check phase durations
    final_stats = tracker.complete_progress_tracking()
    assert len(final_stats.phase_durations) > 0, "Phase durations should be recorded"
    
    print("   📊 Phase durations recorded:")
    for phase, duration in final_stats.phase_durations.items():
        print(f"      {phase}: {duration:.3f}s")
    
    print("✅ Generation phase tracking validated")

def validate_performance_metrics():
    """
    Validate performance metrics display (frames processed, processing speed)
    """
    print("\n🧪 Validating Performance Metrics Display")
    
    tracker = ProgressTracker()
    tracker.start_progress_tracking("performance", 30)
    
    # Simulate processing with performance metrics
    time.sleep(0.1)  # Ensure some elapsed time
    
    for step in range(1, 16):
        frames = step * 3
        tracker.update_progress(
            step, 
            GenerationPhase.GENERATION.value,
            frames,
            additional_data={
                "memory_usage": 512 + (step * 20),
                "gpu_utilization": min(95, 60 + step * 2)
            }
        )
        time.sleep(0.05)
    
    html = tracker.get_progress_html()
    
    # Validate performance metrics are displayed
    assert "Processing Speed" in html, "Processing speed should be displayed"
    assert "fps" in html, "FPS unit should be displayed"
    assert "Memory Usage" in html, "Memory usage should be displayed"
    assert "MB" in html, "Memory unit should be displayed"
    
    # Validate metrics are calculated
    assert tracker.progress_data.processing_speed > 0, "Processing speed should be calculated"
    assert tracker.progress_data.frames_processed == 45, "Frames should be tracked"
    
    tracker.complete_progress_tracking()
    print("✅ Performance metrics display validated")

def validate_progress_callback_integration():
    """
    Validate integration with generation function progress callbacks
    """
    print("\n🧪 Validating Progress Callback Integration")
    
    tracker = ProgressTracker()
    callback = create_progress_callback(tracker)
    
    # Start tracking
    tracker.start_progress_tracking("callback_test", 20)
    
    # Simulate generation function using the callback
    for step in range(1, 21):
        callback(
            step, 20,
            phase=GenerationPhase.GENERATION.value,
            frames_processed=step * 2,
            memory_usage=400 + step * 15
        )
        time.sleep(0.02)
    
    # Validate callback integration worked
    assert tracker.progress_data.current_step == 20, "Callback should update progress"
    assert tracker.progress_data.frames_processed == 40, "Callback should update frames"
    assert tracker.progress_data.progress_percentage == 100.0, "Should reach 100%"
    
    tracker.complete_progress_tracking()
    print("✅ Progress callback integration validated")

def demonstrate_complete_functionality():
    """
    Demonstrate complete progress bar functionality with all features
    """
    print("\n🎬 Demonstrating Complete Progress Bar Functionality")
    print("=" * 60)
    
    tracker = ProgressTracker({
        "progress_update_interval": 0.5,
        "enable_system_monitoring": True
    })
    
    # Add callback to show real-time updates
    def show_progress(progress_data):
        print(f"📊 {progress_data.current_phase.replace('_', ' ').title()}: "
              f"{progress_data.progress_percentage:.1f}% "
              f"({progress_data.current_step}/{progress_data.total_steps}) "
              f"- {progress_data.frames_processed} frames @ {progress_data.processing_speed:.1f} fps")
    
    tracker.add_update_callback(show_progress)
    
    # Start comprehensive generation simulation
    tracker.start_progress_tracking("complete_demo", 100)
    
    # Simulate all phases of video generation
    phases = [
        (GenerationPhase.INITIALIZATION, 5, "Initializing system..."),
        (GenerationPhase.MODEL_LOADING, 15, "Loading AI model..."),
        (GenerationPhase.PREPROCESSING, 25, "Processing input data..."),
        (GenerationPhase.GENERATION, 80, "Generating video frames..."),
        (GenerationPhase.POSTPROCESSING, 95, "Post-processing video..."),
        (GenerationPhase.ENCODING, 100, "Encoding final video...")
    ]
    
    current_step = 0
    for phase, end_step, description in phases:
        print(f"\n🔄 {description}")
        tracker.update_phase(phase.value)
        
        while current_step < end_step:
            current_step += 1
            frames = current_step * 2
            
            tracker.update_progress(
                current_step,
                frames_processed=frames,
                additional_data={
                    "phase_description": description,
                    "memory_usage": 512 + (current_step * 8),
                    "gpu_utilization": min(95, 40 + current_step)
                }
            )
            
            time.sleep(0.1)  # Simulate processing time
    
    # Complete with final statistics
    final_stats = tracker.complete_progress_tracking({
        "output_file": "demo_video.mp4",
        "resolution": "1920x1080",
        "duration": "4.0s",
        "file_size_mb": 156.7,
        "codec": "H.264"
    })
    
    print(f"\n✅ Generation Complete!")
    print(f"   📊 Final Statistics:")
    print(f"      Total time: {final_stats.elapsed_seconds:.1f}s")
    print(f"      Frames processed: {final_stats.frames_processed}")
    print(f"      Average FPS: {final_stats.frames_per_second:.1f}")
    print(f"      Output: {final_stats.performance_metrics.get('output_file', 'N/A')}")
    print(f"      File size: {final_stats.performance_metrics.get('file_size_mb', 'N/A')} MB")
    
    # Show final HTML
    print(f"\n📊 Final Progress Display HTML:")
    print("-" * 40)
    html = tracker.get_progress_html()
    # Extract key information from HTML for display
    if "100.0%" in html:
        print("✅ Progress: 100% Complete")
    if "Completion" in html:
        print("✅ Phase: Completion")
    if str(final_stats.frames_processed) in html:
        print(f"✅ Frames: {final_stats.frames_processed} processed")
    print("-" * 40)

def main():
    """Run all validation tests"""
    print("🚀 Progress Bar with Generation Statistics - Validation Suite")
    print("=" * 70)
    
    try:
        # Validate all requirements
        validate_requirement_11_1()
        validate_requirement_11_2()
        validate_requirement_11_3()
        validate_requirement_11_4()
        validate_requirement_11_5()
        
        # Validate additional functionality
        validate_generation_phase_tracking()
        validate_performance_metrics()
        validate_progress_callback_integration()
        
        # Demonstrate complete functionality
        demonstrate_complete_functionality()
        
        print(f"\n🎉 All Validations Passed!")
        print("=" * 70)
        print("✅ Progress bar component creation - IMPLEMENTED")
        print("✅ Real-time statistics display - IMPLEMENTED")
        print("✅ Generation phase tracking - IMPLEMENTED")
        print("✅ Performance metrics display - IMPLEMENTED")
        print("✅ All requirements 11.1-11.5 - VALIDATED")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
