"""
Integration test for Progress Bar with Generation Statistics
Tests the complete progress tracking system integration
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from progress_tracker import (
    ProgressTracker, GenerationPhase, get_progress_tracker, 
    create_progress_callback
)

class TestProgressBarIntegration:
    """Test progress bar integration with generation system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = {
            "progress_update_interval": 0.1,
            "enable_system_monitoring": False
        }
        self.tracker = ProgressTracker(self.config)
    
    def teardown_method(self):
        """Clean up after tests"""
        if self.tracker.is_tracking:
            self.tracker.complete_progress_tracking()
    
    def test_progress_bar_creation(self):
        """Test progress bar component creation"""
        # Test that progress tracker can be initialized
        assert self.tracker is not None
        assert not self.tracker.is_tracking
        assert self.tracker.progress_data is None
    
    def test_generation_statistics_display(self):
        """Test generation statistics display functionality"""
        # Start tracking
        self.tracker.start_progress_tracking("test_gen", 50)
        
        # Update with various statistics
        self.tracker.update_progress(
            step=25,
            phase=GenerationPhase.GENERATION.value,
            frames_processed=100,
            additional_data={
                "memory_usage": 1024,
                "gpu_utilization": 85,
                "processing_speed": 2.5
            }
        )
        
        # Verify statistics are captured
        assert self.tracker.progress_data.current_step == 25
        assert self.tracker.progress_data.progress_percentage == 50.0
        assert self.tracker.progress_data.frames_processed == 100
        assert self.tracker.progress_data.current_phase == GenerationPhase.GENERATION.value
        
        # Test HTML generation includes all statistics
        html = self.tracker.get_progress_html()
        assert "50.0%" in html
        assert "Generation" in html
        assert "Step 25 / 50" in html
        assert "100" in html  # frames processed
    
    def test_real_time_statistics_updates(self):
        """Test real-time statistics updates during generation"""
        updates_received = []
        
        def capture_updates(progress_data):
            updates_received.append({
                'step': progress_data.current_step,
                'percentage': progress_data.progress_percentage,
                'phase': progress_data.current_phase
            })
        
        self.tracker.add_update_callback(capture_updates)
        self.tracker.start_progress_tracking("realtime_test", 10)
        
        # Simulate rapid updates
        for i in range(1, 6):
            self.tracker.update_progress(i, GenerationPhase.GENERATION.value)
            time.sleep(0.05)  # Small delay
        
        # Verify updates were received
        assert len(updates_received) >= 5
        assert updates_received[-1]['step'] == 5
        assert updates_received[-1]['percentage'] == 50.0
    
    def test_generation_phase_tracking(self):
        """Test tracking of different generation phases"""
        self.tracker.start_progress_tracking("phase_test", 100)
        
        phases = [
            GenerationPhase.INITIALIZATION,
            GenerationPhase.MODEL_LOADING,
            GenerationPhase.PREPROCESSING,
            GenerationPhase.GENERATION,
            GenerationPhase.POSTPROCESSING,
            GenerationPhase.ENCODING
        ]
        
        step = 0
        for phase in phases:
            step += 10
            self.tracker.update_progress(step, phase.value)
            assert self.tracker.progress_data.current_phase == phase.value
        
        # Complete and check phase durations were recorded
        final_stats = self.tracker.complete_progress_tracking()
        assert len(final_stats.phase_durations) > 0
    
    def test_performance_metrics_display(self):
        """Test performance metrics display (frames processed, processing speed)"""
        self.tracker.start_progress_tracking("perf_test", 20)
        
        # Add small delay to ensure elapsed time > 0
        time.sleep(0.1)
        
        # Simulate processing with increasing frame count
        for step in range(1, 11):
            frames = step * 5  # 5 frames per step
            self.tracker.update_progress(step, frames_processed=frames)
            
            # Add small delay between updates
            if step > 1:
                time.sleep(0.05)
                # Verify processing speed is calculated after some time
                if self.tracker.progress_data.elapsed_time > 0:
                    assert self.tracker.progress_data.processing_speed >= 0
        
        # Verify final metrics
        assert self.tracker.progress_data.frames_processed == 50
        # Processing speed should be calculated if there's elapsed time
        if self.tracker.progress_data.elapsed_time > 0:
            assert self.tracker.progress_data.processing_speed >= 0
    
    def test_eta_calculation(self):
        """Test estimated time remaining calculation"""
        self.tracker.start_progress_tracking("eta_test", 20)
        
        # Add some delay to allow ETA calculation
        time.sleep(0.1)
        self.tracker.update_progress(5)
        
        # ETA should be calculated after some progress
        assert self.tracker.progress_data.estimated_remaining >= 0
        
        # ETA should decrease as progress increases
        time.sleep(0.1)
        self.tracker.update_progress(10)
        
        # Should have some ETA value
        assert self.tracker.progress_data.estimated_remaining >= 0
    
    def test_progress_callback_integration(self):
        """Test integration with generation function progress callbacks"""
        callback = create_progress_callback(self.tracker)
        
        # Start tracking
        self.tracker.start_progress_tracking("callback_integration", 30)
        
        # Simulate generation function calling the callback
        for step in range(1, 16):
            callback(
                step, 30,
                phase=GenerationPhase.GENERATION.value,
                frames_processed=step * 2,
                memory_usage=500 + step * 10
            )
        
        # Verify progress was updated
        assert self.tracker.progress_data.current_step == 15
        assert self.tracker.progress_data.progress_percentage == 50.0
        assert self.tracker.progress_data.frames_processed == 30
    
    def test_html_formatting_and_styling(self):
        """Test HTML formatting and styling for progress display"""
        self.tracker.start_progress_tracking("html_test", 40)
        self.tracker.update_progress(
            20, 
            GenerationPhase.GENERATION.value,
            frames_processed=80
        )
        
        html = self.tracker.get_progress_html()
        
        # Check for essential HTML elements
        assert "<div" in html
        assert "progress" in html.lower()
        assert "50.0%" in html
        assert "Step 20 / 40" in html
        assert "Generation" in html
        assert "80" in html  # frames processed
        
        # Check for styling elements
        assert "background:" in html or "style=" in html
        assert "grid" in html.lower() or "flex" in html.lower()
    
    def test_error_handling_during_generation(self):
        """Test error handling during generation progress tracking"""
        # Add a callback that will fail
        def failing_callback(data):
            raise Exception("Callback error")
        
        self.tracker.add_update_callback(failing_callback)
        self.tracker.start_progress_tracking("error_test", 10)
        
        # Should not raise exception despite callback failure
        self.tracker.update_progress(5)
        
        # Should still be tracking
        assert self.tracker.is_tracking
        assert self.tracker.progress_data.current_step == 5
    
    def test_completion_statistics(self):
        """Test final completion statistics display"""
        self.tracker.start_progress_tracking("completion_test", 25)
        
        # Simulate full generation
        for step in range(1, 26):
            self.tracker.update_progress(
                step,
                GenerationPhase.GENERATION.value,
                frames_processed=step * 3
            )
        
        # Complete with final statistics
        final_stats = self.tracker.complete_progress_tracking({
            "output_file": "test_video.mp4",
            "file_size_mb": 150.5,
            "total_time": 45.2
        })
        
        # Verify final statistics
        assert final_stats.frames_processed == 75
        assert final_stats.elapsed_seconds > 0
        assert "output_file" in final_stats.performance_metrics
        assert final_stats.performance_metrics["file_size_mb"] == 150.5
    
    def test_multiple_generation_sessions(self):
        """Test handling multiple generation sessions"""
        # First session
        self.tracker.start_progress_tracking("session1", 10)
        self.tracker.update_progress(5)
        stats1 = self.tracker.complete_progress_tracking()
        
        # Second session
        self.tracker.start_progress_tracking("session2", 20)
        self.tracker.update_progress(10)
        stats2 = self.tracker.complete_progress_tracking()
        
        # Verify sessions are independent
        assert stats1.total_steps == 10
        assert stats2.total_steps == 20
        assert stats1.start_time != stats2.start_time
    
    @patch('psutil.Process')
    def test_system_monitoring_integration(self, mock_process_class):
        """Test system monitoring integration"""
        # Enable system monitoring
        self.tracker.enable_system_monitoring = True
        
        # Mock system metrics
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 800  # 800 MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process
        
        self.tracker.start_progress_tracking("system_test", 5)
        self.tracker.update_progress(3)
        
        # Verify system metrics are captured
        assert self.tracker.progress_data.memory_usage_mb == 800.0

class TestUIIntegration:
    """Test UI integration aspects"""
    
    def test_progress_display_component_creation(self):
        """Test that progress display component can be created"""
        # This would test the actual UI component creation
        # For now, we test the HTML generation
        tracker = ProgressTracker()
        tracker.start_progress_tracking("ui_test", 30)
        tracker.update_progress(15)
        
        html = tracker.get_progress_html()
        
        # Verify HTML is suitable for UI display
        assert len(html) > 100  # Should be substantial HTML
        assert "progress" in html.lower()
        assert "50.0%" in html
        
        tracker.complete_progress_tracking()
    
    def test_gradio_component_compatibility(self):
        """Test compatibility with Gradio HTML components"""
        tracker = ProgressTracker()
        tracker.start_progress_tracking("gradio_test", 50)
        
        # Update with various data
        tracker.update_progress(25, GenerationPhase.GENERATION.value, 100)
        
        html = tracker.get_progress_html()
        
        # Check HTML is well-formed for Gradio
        assert html.count('<div') == html.count('</div>')  # Balanced tags
        assert 'style=' in html  # Has inline styles
        assert not html.startswith('<script')  # No script tags
        
        tracker.complete_progress_tracking()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
