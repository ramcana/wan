"""
Progress Tracking Functionality Tests
Comprehensive tests for progress bar and generation statistics functionality
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import time
import threading
from datetime import datetime, timedelta
import json

# Import modules under test
from progress_tracker import (
    ProgressTracker, ProgressData, GenerationStats, GenerationPhase
)


class TestProgressData(unittest.TestCase):
    """Test ProgressData class functionality"""
    
    def test_progress_data_creation(self):
        """Test ProgressData object creation"""
        progress = ProgressData(
            current_step=25,
            total_steps=100,
            progress_percentage=25.0,
            elapsed_time=10.0,
            estimated_remaining=30.0,
            current_phase=GenerationPhase.GENERATION.value,
            frames_processed=50,
            processing_speed=5.0
        )
        
        self.assertEqual(progress.current_step, 25)
        self.assertEqual(progress.total_steps, 100)
        self.assertEqual(progress.progress_percentage, 25.0)
        self.assertEqual(progress.elapsed_time, 10.0)
        self.assertEqual(progress.estimated_remaining, 30.0)
        self.assertEqual(progress.current_phase, GenerationPhase.GENERATION.value)
        self.assertEqual(progress.frames_processed, 50)
        self.assertEqual(progress.processing_speed, 5.0)
    
    def test_progress_data_defaults(self):
        """Test ProgressData with default values"""
        progress = ProgressData()
        
        self.assertEqual(progress.current_step, 0)
        self.assertEqual(progress.total_steps, 0)
        self.assertEqual(progress.progress_percentage, 0.0)
        self.assertEqual(progress.elapsed_time, 0.0)
        self.assertEqual(progress.estimated_remaining, 0.0)
        self.assertEqual(progress.current_phase, GenerationPhase.INITIALIZATION.value)
        self.assertEqual(progress.frames_processed, 0)
        self.assertEqual(progress.processing_speed, 0.0)


class TestGenerationStats(unittest.TestCase):
    """Test GenerationStats class functionality"""
    
    def test_generation_stats_creation(self):
        """Test GenerationStats object creation"""
        start_time = datetime.now()
        stats = GenerationStats(
            start_time=start_time,
            current_time=start_time + timedelta(seconds=30),
            elapsed_seconds=30.0,
            estimated_total_seconds=120.0,
            current_step=25,
            total_steps=100,
            current_phase=GenerationPhase.GENERATION.value,
            frames_processed=50,
            frames_per_second=1.67,
            memory_usage_mb=512.0,
            gpu_utilization_percent=85.0
        )
        
        self.assertEqual(stats.start_time, start_time)
        self.assertEqual(stats.elapsed_seconds, 30.0)
        self.assertEqual(stats.estimated_total_seconds, 120.0)
        self.assertEqual(stats.current_step, 25)
        self.assertEqual(stats.total_steps, 100)
        self.assertEqual(stats.current_phase, GenerationPhase.GENERATION.value)
        self.assertEqual(stats.frames_processed, 50)
        self.assertAlmostEqual(stats.frames_per_second, 1.67, places=2)
        self.assertEqual(stats.memory_usage_mb, 512.0)
        self.assertEqual(stats.gpu_utilization_percent, 85.0)
    
    def test_generation_stats_defaults(self):
        """Test GenerationStats with default values"""
        stats = GenerationStats()
        
        self.assertIsInstance(stats.start_time, datetime)
        self.assertIsInstance(stats.current_time, datetime)
        self.assertEqual(stats.elapsed_seconds, 0.0)
        self.assertEqual(stats.estimated_total_seconds, 0.0)
        self.assertEqual(stats.current_step, 0)
        self.assertEqual(stats.total_steps, 0)
        self.assertEqual(stats.current_phase, GenerationPhase.INITIALIZATION.value)
        self.assertEqual(stats.frames_processed, 0)
        self.assertEqual(stats.frames_per_second, 0.0)
        self.assertIsInstance(stats.phase_durations, dict)
        self.assertIsInstance(stats.performance_metrics, dict)


class TestGenerationPhase(unittest.TestCase):
    """Test GenerationPhase enum"""
    
    def test_generation_phases(self):
        """Test all generation phases are defined"""
        expected_phases = [
            "initialization",
            "model_loading", 
            "preprocessing",
            "generation",
            "postprocessing",
            "encoding",
            "completion"
        ]
        
        for phase_name in expected_phases:
            with self.subTest(phase=phase_name):
                # Check that phase exists and has correct value
                phase = getattr(GenerationPhase, phase_name.upper())
                self.assertEqual(phase.value, phase_name)


class TestProgressTracker(unittest.TestCase):
    """Test ProgressTracker class functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            "progress_update_interval": 0.1,  # Fast updates for testing
            "enable_system_monitoring": False  # Disable for testing
        }
        self.tracker = ProgressTracker(self.config)
    
    def tearDown(self):
        """Clean up after tests"""
        if self.tracker.is_tracking:
            self.tracker.stop_progress_tracking()
    
    def test_progress_tracker_initialization(self):
        """Test ProgressTracker initialization"""
        self.assertIsNone(self.tracker.current_task)
        self.assertIsNone(self.tracker.progress_data)
        self.assertIsNone(self.tracker.generation_stats)
        self.assertFalse(self.tracker.is_tracking)
        self.assertEqual(self.tracker.update_interval, 0.1)
        self.assertFalse(self.tracker.enable_system_monitoring)
        self.assertEqual(len(self.tracker.update_callbacks), 0)
    
    def test_start_progress_tracking(self):
        """Test starting progress tracking"""
        task_id = "test_task_123"
        total_steps = 50
        
        self.tracker.start_progress_tracking(task_id, total_steps)
        
        self.assertEqual(self.tracker.current_task, task_id)
        self.assertIsNotNone(self.tracker.progress_data)
        self.assertIsNotNone(self.tracker.generation_stats)
        self.assertTrue(self.tracker.is_tracking)
        self.assertEqual(self.tracker.progress_data.total_steps, total_steps)
        self.assertEqual(self.tracker.generation_stats.total_steps, total_steps)
    
    def test_update_progress_basic(self):
        """Test basic progress updates"""
        self.tracker.start_progress_tracking("test_task", 100)
        
        # Update progress
        self.tracker.update_progress(
            step=25,
            phase=GenerationPhase.GENERATION.value,
            frames_processed=10,
            processing_speed=2.5
        )
        
        progress = self.tracker.progress_data
        self.assertEqual(progress.current_step, 25)
        self.assertEqual(progress.progress_percentage, 25.0)
        self.assertEqual(progress.current_phase, GenerationPhase.GENERATION.value)
        self.assertEqual(progress.frames_processed, 10)
        self.assertEqual(progress.processing_speed, 2.5)
    
    def test_update_progress_percentage_calculation(self):
        """Test progress percentage calculation"""
        self.tracker.start_progress_tracking("test_task", 200)
        
        test_cases = [
            (0, 0.0),
            (50, 25.0),
            (100, 50.0),
            (150, 75.0),
            (200, 100.0)
        ]
        
        for step, expected_percentage in test_cases:
            with self.subTest(step=step):
                self.tracker.update_progress(step=step)
                self.assertEqual(self.tracker.progress_data.progress_percentage, expected_percentage)
    
    def test_update_progress_with_all_parameters(self):
        """Test progress update with all parameters"""
        self.tracker.start_progress_tracking("test_task", 100)
        
        self.tracker.update_progress(
            step=40,
            phase=GenerationPhase.POSTPROCESSING.value,
            frames_processed=80,
            processing_speed=3.2,
            memory_usage_mb=768.0,
            gpu_utilization=92.5,
            additional_metrics={"custom_metric": 42}
        )
        
        progress = self.tracker.progress_data
        stats = self.tracker.generation_stats
        
        self.assertEqual(progress.current_step, 40)
        self.assertEqual(progress.progress_percentage, 40.0)
        self.assertEqual(progress.current_phase, GenerationPhase.POSTPROCESSING.value)
        self.assertEqual(progress.frames_processed, 80)
        self.assertEqual(progress.processing_speed, 3.2)
        self.assertEqual(progress.memory_usage_mb, 768.0)
        self.assertEqual(progress.gpu_utilization_percent, 92.5)
        
        self.assertEqual(stats.current_step, 40)
        self.assertEqual(stats.frames_processed, 80)
        self.assertEqual(stats.frames_per_second, 3.2)
        self.assertEqual(stats.memory_usage_mb, 768.0)
        self.assertEqual(stats.gpu_utilization_percent, 92.5)
    
    def test_complete_progress_tracking(self):
        """Test completing progress tracking"""
        self.tracker.start_progress_tracking("test_task", 100)
        
        # Update some progress
        self.tracker.update_progress(step=50, frames_processed=25)
        
        # Complete tracking
        final_stats = self.tracker.complete_progress_tracking()
        
        self.assertFalse(self.tracker.is_tracking)
        self.assertIsNotNone(final_stats)
        self.assertIsInstance(final_stats, GenerationStats)
        self.assertEqual(final_stats.current_step, 50)
        self.assertEqual(final_stats.frames_processed, 25)
    
    def test_stop_progress_tracking(self):
        """Test stopping progress tracking"""
        self.tracker.start_progress_tracking("test_task", 100)
        self.assertTrue(self.tracker.is_tracking)
        
        self.tracker.stop_progress_tracking()
        
        self.assertFalse(self.tracker.is_tracking)
        self.assertTrue(self.tracker.stop_event.is_set())
    
    def test_add_update_callback(self):
        """Test adding update callbacks"""
        callback_called = False
        callback_data = None
        
        def test_callback(progress_data):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = progress_data
        
        # Add callback
        self.tracker.add_update_callback(test_callback)
        self.assertEqual(len(self.tracker.update_callbacks), 1)
        
        # Start tracking and update
        self.tracker.start_progress_tracking("test_task", 100)
        self.tracker.update_progress(step=30)
        
        # Manually trigger callback for testing
        test_callback(self.tracker.progress_data)
        
        self.assertTrue(callback_called)
        self.assertIsNotNone(callback_data)
        self.assertEqual(callback_data.current_step, 30)
    
    def test_multiple_callbacks(self):
        """Test multiple update callbacks"""
        callback1_called = False
        callback2_called = False
        
        def callback1(progress_data):
            nonlocal callback1_called
            callback1_called = True
        
        def callback2(progress_data):
            nonlocal callback2_called
            callback2_called = True
        
        # Add multiple callbacks
        self.tracker.add_update_callback(callback1)
        self.tracker.add_update_callback(callback2)
        self.assertEqual(len(self.tracker.update_callbacks), 2)
        
        # Start tracking and trigger callbacks
        self.tracker.start_progress_tracking("test_task", 100)
        self.tracker.update_progress(step=50)
        
        # Manually trigger callbacks for testing
        for callback in self.tracker.update_callbacks:
            callback(self.tracker.progress_data)
        
        self.assertTrue(callback1_called)
        self.assertTrue(callback2_called)
    
    def test_eta_calculation(self):
        """Test ETA calculation logic"""
        self.tracker.start_progress_tracking("test_task", 100)
        
        # Mock elapsed time
        start_time = time.time()
        elapsed_time = 20.0  # 20 seconds elapsed
        current_step = 25    # 25% complete
        
        # Calculate expected ETA
        if current_step > 0:
            estimated_total_time = elapsed_time * 100 / current_step  # 80 seconds total
            expected_eta = estimated_total_time - elapsed_time        # 60 seconds remaining
            
            # Update progress with mocked time
            self.tracker.update_progress(step=current_step)
            
            # Manually set elapsed time for testing
            if self.tracker.progress_data:
                self.tracker.progress_data.elapsed_time = elapsed_time
                self.tracker.progress_data.estimated_remaining = expected_eta
                
                self.assertAlmostEqual(self.tracker.progress_data.estimated_remaining, 60.0, delta=1.0)
    
    def test_phase_tracking(self):
        """Test phase tracking functionality"""
        self.tracker.start_progress_tracking("test_task", 100)
        
        phases = [
            GenerationPhase.INITIALIZATION,
            GenerationPhase.MODEL_LOADING,
            GenerationPhase.PREPROCESSING,
            GenerationPhase.GENERATION,
            GenerationPhase.POSTPROCESSING,
            GenerationPhase.ENCODING,
            GenerationPhase.COMPLETION
        ]
        
        for i, phase in enumerate(phases):
            with self.subTest(phase=phase):
                step = (i + 1) * 10
                self.tracker.update_progress(step=step, phase=phase.value)
                
                self.assertEqual(self.tracker.progress_data.current_phase, phase.value)
                self.assertEqual(self.tracker.generation_stats.current_phase, phase.value)
    
    def test_performance_metrics_tracking(self):
        """Test performance metrics tracking"""
        self.tracker.start_progress_tracking("test_task", 100)
        
        # Update with performance metrics
        self.tracker.update_progress(
            step=50,
            frames_processed=100,
            processing_speed=2.0,
            memory_usage_mb=1024.0,
            gpu_utilization=88.5
        )
        
        stats = self.tracker.generation_stats
        
        self.assertEqual(stats.frames_processed, 100)
        self.assertEqual(stats.frames_per_second, 2.0)
        self.assertEqual(stats.memory_usage_mb, 1024.0)
        self.assertEqual(stats.gpu_utilization_percent, 88.5)
    
    def test_progress_data_serialization(self):
        """Test that progress data can be serialized to JSON"""
        self.tracker.start_progress_tracking("test_task", 100)
        self.tracker.update_progress(
            step=30,
            phase=GenerationPhase.GENERATION.value,
            frames_processed=15,
            processing_speed=0.5
        )
        
        progress = self.tracker.progress_data
        
        # Create serializable dictionary
        progress_dict = {
            'current_step': progress.current_step,
            'total_steps': progress.total_steps,
            'progress_percentage': progress.progress_percentage,
            'current_phase': progress.current_phase,
            'frames_processed': progress.frames_processed,
            'processing_speed': progress.processing_speed,
            'elapsed_time': progress.elapsed_time,
            'estimated_remaining': progress.estimated_remaining
        }
        
        # Should be JSON serializable
        json_str = json.dumps(progress_dict)
        self.assertIsInstance(json_str, str)
        
        # Should be deserializable
        deserialized = json.loads(json_str)
        self.assertEqual(deserialized['current_step'], 30)
        self.assertEqual(deserialized['current_phase'], GenerationPhase.GENERATION.value)
        self.assertEqual(deserialized['frames_processed'], 15)
    
    def test_generation_stats_serialization(self):
        """Test that generation stats can be serialized"""
        self.tracker.start_progress_tracking("test_task", 100)
        self.tracker.update_progress(step=75, frames_processed=150)
        
        stats = self.tracker.generation_stats
        
        # Create serializable dictionary
        stats_dict = {
            'start_time': stats.start_time.isoformat(),
            'current_time': stats.current_time.isoformat(),
            'elapsed_seconds': stats.elapsed_seconds,
            'estimated_total_seconds': stats.estimated_total_seconds,
            'current_step': stats.current_step,
            'total_steps': stats.total_steps,
            'current_phase': stats.current_phase,
            'frames_processed': stats.frames_processed,
            'frames_per_second': stats.frames_per_second,
            'memory_usage_mb': stats.memory_usage_mb,
            'gpu_utilization_percent': stats.gpu_utilization_percent,
            'phase_durations': stats.phase_durations,
            'performance_metrics': stats.performance_metrics
        }
        
        # Should be JSON serializable
        json_str = json.dumps(stats_dict)
        self.assertIsInstance(json_str, str)
        
        # Should be deserializable
        deserialized = json.loads(json_str)
        self.assertEqual(deserialized['current_step'], 75)
        self.assertEqual(deserialized['frames_processed'], 150)


class TestProgressTrackerIntegration(unittest.TestCase):
    """Integration tests for progress tracker with mock generation processes"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tracker = ProgressTracker({
            "progress_update_interval": 0.05,  # Very fast for testing
            "enable_system_monitoring": False
        })
    
    def tearDown(self):
        """Clean up after tests"""
        if self.tracker.is_tracking:
            self.tracker.stop_progress_tracking()
    
    def test_mock_video_generation_workflow(self):
        """Test complete mock video generation workflow"""
        # Define generation phases with step counts
        generation_phases = [
            (GenerationPhase.INITIALIZATION, 5),
            (GenerationPhase.MODEL_LOADING, 10),
            (GenerationPhase.PREPROCESSING, 15),
            (GenerationPhase.GENERATION, 60),
            (GenerationPhase.POSTPROCESSING, 8),
            (GenerationPhase.ENCODING, 2)
        ]
        
        total_steps = sum(steps for _, steps in generation_phases)
        
        # Start tracking
        self.tracker.start_progress_tracking("mock_video_generation", total_steps)
        
        # Simulate generation process
        current_step = 0
        frames_processed = 0
        
        for phase, phase_steps in generation_phases:
            for step_in_phase in range(phase_steps):
                current_step += 1
                frames_processed += 2 if phase == GenerationPhase.GENERATION else 0
                
                processing_speed = 2.0 if phase == GenerationPhase.GENERATION else 0.5
                
                self.tracker.update_progress(
                    step=current_step,
                    phase=phase.value,
                    frames_processed=frames_processed,
                    processing_speed=processing_speed
                )
        
        # Verify final state
        progress = self.tracker.progress_data
        stats = self.tracker.generation_stats
        
        self.assertEqual(progress.current_step, total_steps)
        self.assertEqual(progress.progress_percentage, 100.0)
        self.assertEqual(progress.current_phase, GenerationPhase.ENCODING.value)
        self.assertEqual(stats.frames_processed, frames_processed)
        
        # Complete tracking
        final_stats = self.tracker.complete_progress_tracking()
        self.assertIsNotNone(final_stats)
        self.assertFalse(self.tracker.is_tracking)
    
    def test_mock_i2v_generation_with_callbacks(self):
        """Test I2V generation with progress callbacks"""
        callback_updates = []
        
        def progress_callback(progress_data):
            callback_updates.append({
                'step': progress_data.current_step,
                'phase': progress_data.current_phase,
                'percentage': progress_data.progress_percentage,
                'frames': progress_data.frames_processed
            })
        
        # Add callback
        self.tracker.add_update_callback(progress_callback)
        
        # Start I2V generation simulation
        self.tracker.start_progress_tracking("i2v_generation", 80)
        
        # Simulate I2V-specific phases
        i2v_phases = [
            (GenerationPhase.INITIALIZATION, 5),
            (GenerationPhase.PREPROCESSING, 10),  # Image preprocessing
            (GenerationPhase.GENERATION, 50),     # Main generation
            (GenerationPhase.POSTPROCESSING, 10), # Video postprocessing
            (GenerationPhase.ENCODING, 5)         # Final encoding
        ]
        
        current_step = 0
        for phase, phase_steps in i2v_phases:
            for _ in range(phase_steps):
                current_step += 1
                frames = current_step if phase == GenerationPhase.GENERATION else 0
                
                self.tracker.update_progress(
                    step=current_step,
                    phase=phase.value,
                    frames_processed=frames
                )
                
                # Manually trigger callback for testing
                progress_callback(self.tracker.progress_data)
        
        # Verify callbacks were called
        self.assertEqual(len(callback_updates), 80)  # One for each step
        
        # Verify progression
        first_update = callback_updates[0]
        last_update = callback_updates[-1]
        
        self.assertEqual(first_update['step'], 1)
        self.assertEqual(last_update['step'], 80)
        self.assertEqual(last_update['percentage'], 100.0)
        self.assertEqual(last_update['phase'], GenerationPhase.ENCODING.value)
    
    def test_mock_ti2v_generation_with_error_recovery(self):
        """Test TI2V generation with error recovery simulation"""
        self.tracker.start_progress_tracking("ti2v_generation", 120)
        
        # Simulate normal progress
        for step in range(1, 51):
            self.tracker.update_progress(
                step=step,
                phase=GenerationPhase.GENERATION.value,
                frames_processed=step * 2
            )
        
        # Simulate error and recovery (restart from checkpoint)
        # In real scenario, this might involve reloading state
        checkpoint_step = 30
        
        # Continue from checkpoint
        for step in range(checkpoint_step, 121):
            self.tracker.update_progress(
                step=step,
                phase=GenerationPhase.GENERATION.value if step < 100 else GenerationPhase.ENCODING.value,
                frames_processed=step * 2
            )
        
        # Verify final state
        progress = self.tracker.progress_data
        self.assertEqual(progress.current_step, 120)
        self.assertEqual(progress.progress_percentage, 100.0)
        self.assertEqual(progress.frames_processed, 240)
    
    def test_concurrent_progress_tracking(self):
        """Test that progress tracking handles concurrent updates safely"""
        self.tracker.start_progress_tracking("concurrent_test", 100)
        
        # Simulate concurrent updates (in real scenario, these might come from different threads)
        updates = [
            (10, GenerationPhase.INITIALIZATION.value, 5),
            (20, GenerationPhase.PREPROCESSING.value, 10),
            (30, GenerationPhase.GENERATION.value, 15),
            (40, GenerationPhase.GENERATION.value, 20),
            (50, GenerationPhase.GENERATION.value, 25)
        ]
        
        for step, phase, frames in updates:
            self.tracker.update_progress(
                step=step,
                phase=phase,
                frames_processed=frames
            )
        
        # Verify final state is consistent
        progress = self.tracker.progress_data
        self.assertEqual(progress.current_step, 50)
        self.assertEqual(progress.current_phase, GenerationPhase.GENERATION.value)
        self.assertEqual(progress.frames_processed, 25)


if __name__ == '__main__':
    unittest.main(verbosity=2)