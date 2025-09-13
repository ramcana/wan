"""
Test suite for Progress Tracker
Tests progress tracking functionality, statistics collection, and HTML generation
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from progress_tracker import (
    ProgressTracker, ProgressData, GenerationStats, GenerationPhase,
    get_progress_tracker, create_progress_callback
)

class TestProgressData:
    """Test ProgressData dataclass"""
    
    def test_progress_data_initialization(self):
        """Test ProgressData initialization with default values"""
        data = ProgressData()
        
        assert data.current_step == 0
        assert data.total_steps == 0
        assert data.progress_percentage == 0.0
        assert data.elapsed_time == 0.0
        assert data.estimated_remaining == 0.0
        assert data.current_phase == GenerationPhase.INITIALIZATION.value
        assert data.frames_processed == 0
        assert data.processing_speed == 0.0
    
    def test_progress_data_with_values(self):
        """Test ProgressData initialization with custom values"""
        data = ProgressData(
            current_step=25,
            total_steps=100,
            progress_percentage=25.0,
            current_phase=GenerationPhase.GENERATION.value,
            frames_processed=50
        )
        
        assert data.current_step == 25
        assert data.total_steps == 100
        assert data.progress_percentage == 25.0
        assert data.current_phase == GenerationPhase.GENERATION.value
        assert data.frames_processed == 50

class TestGenerationStats:
    """Test GenerationStats dataclass"""
    
    def test_generation_stats_initialization(self):
        """Test GenerationStats initialization"""
        stats = GenerationStats()
        
        assert isinstance(stats.start_time, datetime)
        assert isinstance(stats.current_time, datetime)
        assert stats.elapsed_seconds == 0.0
        assert stats.current_step == 0
        assert stats.total_steps == 0
        assert stats.frames_processed == 0
        assert isinstance(stats.phase_durations, dict)
        assert isinstance(stats.performance_metrics, dict)

class TestProgressTracker:
    """Test ProgressTracker class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = {
            "progress_update_interval": 0.1,  # Fast updates for testing
            "enable_system_monitoring": False  # Disable for testing
        }
        self.tracker = ProgressTracker(self.config)
    
    def teardown_method(self):
        """Clean up after tests"""
        if self.tracker.is_tracking:
            self.tracker.complete_progress_tracking()
    
    def test_tracker_initialization(self):
        """Test ProgressTracker initialization"""
        assert self.tracker.config == self.config
        assert self.tracker.current_task is None
        assert self.tracker.progress_data is None
        assert self.tracker.generation_stats is None
        assert not self.tracker.is_tracking
        assert self.tracker.update_interval == 0.1
        assert not self.tracker.enable_system_monitoring
    
    def test_add_update_callback(self):
        """Test adding update callbacks"""
        callback = Mock()
        self.tracker.add_update_callback(callback)
        
        assert callback in self.tracker.update_callbacks
    
    def test_start_progress_tracking(self):
        """Test starting progress tracking"""
        task_id = "test_task_123"
        total_steps = 50
        
        self.tracker.start_progress_tracking(task_id, total_steps)
        
        assert self.tracker.current_task == task_id
        assert self.tracker.is_tracking
        assert self.tracker.progress_data is not None
        assert self.tracker.progress_data.total_steps == total_steps
        assert self.tracker.generation_stats is not None
        assert self.tracker.generation_stats.total_steps == total_steps
        assert self.tracker.update_thread is not None
        assert self.tracker.update_thread.is_alive()
    
    def test_update_progress_basic(self):
        """Test basic progress updates"""
        self.tracker.start_progress_tracking("test_task", 100)
        
        # Update progress
        self.tracker.update_progress(25)
        
        assert self.tracker.progress_data.current_step == 25
        assert self.tracker.progress_data.progress_percentage == 25.0
        assert self.tracker.progress_data.elapsed_time > 0
    
    def test_update_progress_with_phase(self):
        """Test progress updates with phase changes"""
        self.tracker.start_progress_tracking("test_task", 100)
        
        # Update with phase
        self.tracker.update_progress(30, phase=GenerationPhase.GENERATION.value)
        
        assert self.tracker.progress_data.current_step == 30
        assert self.tracker.progress_data.current_phase == GenerationPhase.GENERATION.value
    
    def test_update_progress_with_frames(self):
        """Test progress updates with frame processing"""
        self.tracker.start_progress_tracking("test_task", 100)
        
        # Simulate some elapsed time
        time.sleep(0.1)
        
        # Update with frames
        self.tracker.update_progress(40, frames_processed=80)
        
        assert self.tracker.progress_data.current_step == 40
        assert self.tracker.progress_data.frames_processed == 80
        assert self.tracker.progress_data.processing_speed > 0
    
    def test_update_progress_eta_calculation(self):
        """Test ETA calculation in progress updates"""
        self.tracker.start_progress_tracking("test_task", 100)
        
        # Simulate some elapsed time
        time.sleep(0.1)
        
        # Update progress
        self.tracker.update_progress(20)
        
        assert self.tracker.progress_data.estimated_remaining > 0
    
    def test_update_phase(self):
        """Test phase updates"""
        self.tracker.start_progress_tracking("test_task", 100)
        
        # Update phase
        self.tracker.update_phase(GenerationPhase.PREPROCESSING.value, 0.5)
        
        assert self.tracker.progress_data.current_phase == GenerationPhase.PREPROCESSING.value
        assert self.tracker.progress_data.phase_progress == 0.5
    
    def test_complete_progress_tracking(self):
        """Test completing progress tracking"""
        self.tracker.start_progress_tracking("test_task", 100)
        
        # Complete tracking
        final_stats = self.tracker.complete_progress_tracking({"final_metric": 123})
        
        assert not self.tracker.is_tracking
        assert isinstance(final_stats, GenerationStats)
        assert final_stats.performance_metrics.get("final_metric") == 123
        assert self.tracker.progress_data.progress_percentage == 100.0
        assert self.tracker.progress_data.current_phase == GenerationPhase.COMPLETION.value
    
    def test_progress_callbacks(self):
        """Test progress update callbacks"""
        callback = Mock()
        self.tracker.add_update_callback(callback)
        
        self.tracker.start_progress_tracking("test_task", 100)
        
        # Callback should be called on start
        callback.assert_called()
        callback.reset_mock()
        
        # Update progress
        self.tracker.update_progress(50)
        
        # Callback should be called on update
        callback.assert_called()
        assert isinstance(callback.call_args[0][0], ProgressData)
    
    def test_callback_error_handling(self):
        """Test error handling in callbacks"""
        # Create a callback that raises an exception
        def failing_callback(data):
            raise Exception("Callback error")
        
        self.tracker.add_update_callback(failing_callback)
        
        # Should not raise exception
        self.tracker.start_progress_tracking("test_task", 100)
        self.tracker.update_progress(25)

        assert True  # TODO: Add proper assertion
    
    def test_get_progress_html(self):
        """Test HTML generation for progress display"""
        self.tracker.start_progress_tracking("test_task", 100)
        self.tracker.update_progress(30, frames_processed=60)
        
        html = self.tracker.get_progress_html()
        
        assert "Generation Progress" in html
        assert "30.0%" in html
        assert "Step 30 / 100" in html
        assert "60" in html  # frames processed
        assert "progress-bar" in html.lower() or "background:" in html
    
    def test_get_progress_html_no_data(self):
        """Test HTML generation when no progress data exists"""
        html = self.tracker.get_progress_html()
        assert html == ""
    
    def test_format_duration(self):
        """Test duration formatting"""
        # Test seconds
        assert "30s" in self.tracker._format_duration(30)
        
        # Test minutes
        assert "2m" in self.tracker._format_duration(125)
        
        # Test hours
        assert "1h" in self.tracker._format_duration(3665)
        
        # Test zero
        assert self.tracker._format_duration(0) == "0s"
        
        # Test negative
        assert self.tracker._format_duration(-10) == "0s"
    
    @patch('psutil.Process')
    def test_system_monitoring(self, mock_process_class):
        """Test system metrics monitoring"""
        # Enable system monitoring
        self.tracker.enable_system_monitoring = True
        
        # Mock system metrics
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 500  # 500 MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process
        
        self.tracker.start_progress_tracking("test_task", 100)
        self.tracker.update_progress(25)
        
        assert self.tracker.progress_data.memory_usage_mb == 500.0
    
    def test_phase_duration_tracking(self):
        """Test tracking of phase durations"""
        self.tracker.start_progress_tracking("test_task", 100)
        
        # Change phases with small delays
        self.tracker.update_phase(GenerationPhase.MODEL_LOADING.value)
        time.sleep(0.05)
        
        self.tracker.update_phase(GenerationPhase.GENERATION.value)
        time.sleep(0.05)
        
        self.tracker.update_phase(GenerationPhase.ENCODING.value)
        
        # Check that phase durations were recorded
        assert GenerationPhase.MODEL_LOADING.value in self.tracker.generation_stats.phase_durations
        assert GenerationPhase.GENERATION.value in self.tracker.generation_stats.phase_durations
        
        # Durations should be positive
        assert self.tracker.generation_stats.phase_durations[GenerationPhase.MODEL_LOADING.value] > 0
        assert self.tracker.generation_stats.phase_durations[GenerationPhase.GENERATION.value] > 0

class TestGlobalFunctions:
    """Test global utility functions"""
    
    def test_get_progress_tracker_singleton(self):
        """Test global progress tracker singleton"""
        tracker1 = get_progress_tracker()
        tracker2 = get_progress_tracker()
        
        assert tracker1 is tracker2
        assert isinstance(tracker1, ProgressTracker)
    
    def test_get_progress_tracker_with_config(self):
        """Test global progress tracker with config"""
        config = {"test_setting": True}
        tracker = get_progress_tracker(config)
        
        assert tracker.config == config
    
    def test_create_progress_callback(self):
        """Test creating progress callback function"""
        tracker = ProgressTracker()
        callback = create_progress_callback(tracker)
        
        assert callable(callback)
        
        # Mock the tracker's update_progress method
        tracker.update_progress = Mock()
        
        # Call the callback
        callback(25, 100, phase="test_phase", frames_processed=50)
        
        # Verify the tracker was called correctly
        tracker.update_progress.assert_called_once_with(
            step=25,
            phase="test_phase",
            frames_processed=50,
            additional_data={
                'phase': 'test_phase',
                'frames_processed': 50
            }
        )

class TestIntegration:
    """Integration tests for progress tracking"""
    
    def test_full_generation_simulation(self):
        """Test simulating a full generation process"""
        tracker = ProgressTracker({"progress_update_interval": 0.05})
        callback_data = []
        
        def capture_callback(data):
            callback_data.append(data.progress_percentage)
        
        tracker.add_update_callback(capture_callback)
        
        # Start tracking
        tracker.start_progress_tracking("integration_test", 50)
        
        # Simulate generation phases
        phases = [
            GenerationPhase.MODEL_LOADING,
            GenerationPhase.PREPROCESSING,
            GenerationPhase.GENERATION,
            GenerationPhase.POSTPROCESSING,
            GenerationPhase.ENCODING
        ]
        
        steps_per_phase = 10
        for i, phase in enumerate(phases):
            tracker.update_phase(phase.value)
            
            for step in range(steps_per_phase):
                current_step = i * steps_per_phase + step + 1
                frames = current_step * 2
                
                tracker.update_progress(
                    current_step,
                    frames_processed=frames
                )
                time.sleep(0.01)  # Small delay to simulate work
        
        # Complete tracking
        final_stats = tracker.complete_progress_tracking()
        
        # Verify results
        assert len(callback_data) > 0
        assert callback_data[-1] == 100.0  # Final progress should be 100%
        assert final_stats.elapsed_seconds > 0
        assert len(final_stats.phase_durations) > 0
        assert final_stats.frames_processed == 100  # 50 steps * 2 frames per step
    
    def test_error_recovery_during_tracking(self):
        """Test error handling during progress tracking"""
        tracker = ProgressTracker()
        
        # Start tracking
        tracker.start_progress_tracking("error_test", 100)
        
        # Simulate normal progress
        tracker.update_progress(25)
        
        # Simulate error condition (e.g., callback failure)
        def failing_callback(data):
            raise RuntimeError("Simulated error")
        
        tracker.add_update_callback(failing_callback)
        
        # Should not raise exception
        tracker.update_progress(50)
        
        # Should still be able to complete
        final_stats = tracker.complete_progress_tracking()
        assert isinstance(final_stats, GenerationStats)

if __name__ == "__main__":
    pytest.main([__file__])
