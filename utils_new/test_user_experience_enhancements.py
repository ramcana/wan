"""
User Acceptance Tests for User Experience Enhancements

This module contains comprehensive tests for the user experience enhancements
including generation history, error resolution, parameter recommendations,
and queue management.
"""

import pytest
import tempfile
import json
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from generation_history_manager import (
    GenerationHistoryManager, GenerationHistoryEntry, GenerationStatus
)
from interactive_error_resolver import (
    InteractiveErrorResolver, ResolutionStatus, ResolutionStep
)
from hardware_parameter_recommender import (
    HardwareParameterRecommender, HardwareClass, RecommendationConfidence
)
from generation_queue_manager import (
    GenerationQueueManager, QueuePriority, RequestStatus
)
from error_handler import UserFriendlyError, ErrorCategory, ErrorSeverity

class TestGenerationHistoryManager:
    """Test generation history tracking and retry capabilities"""
    
    @pytest.fixture
    def temp_history_file(self):
        """Create temporary history file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def history_manager(self, temp_history_file):
        """Create history manager with temporary file"""
        config = {"history_file": temp_history_file}
        return GenerationHistoryManager(config)
    
    def test_add_generation_entry(self, history_manager):
        """Test adding new generation entries"""
        entry_id = history_manager.add_entry(
            model_type="t2v",
            prompt="A beautiful sunset over mountains",
            resolution="1280x720",
            steps=50
        )
        
        assert entry_id != ""
        assert len(history_manager.history) == 1
        
        entry = history_manager.get_entry(entry_id)
        assert entry is not None
        assert entry.model_type == "t2v"
        assert entry.prompt == "A beautiful sunset over mountains"
        assert entry.status == GenerationStatus.PENDING
    
    def test_update_entry_status(self, history_manager):
        """Test updating entry status"""
        entry_id = history_manager.add_entry(
            model_type="t2v",
            prompt="Test prompt"
        )
        
        # Update to completed
        history_manager.update_entry_status(
            entry_id,
            GenerationStatus.COMPLETED,
            output_path="output.mp4",
            generation_time=120.5
        )
        
        entry = history_manager.get_entry(entry_id)
        assert entry.status == GenerationStatus.COMPLETED
        assert entry.output_path == "output.mp4"
        assert entry.generation_time == 120.5
    
    def test_retry_failed_entry(self, history_manager):
        """Test retrying failed generations"""
        # Add failed entry
        entry_id = history_manager.add_entry(
            model_type="t2v",
            prompt="Test prompt"
        )
        
        history_manager.update_entry_status(
            entry_id,
            GenerationStatus.FAILED,
            error_message="Test error"
        )
        
        # Retry the entry
        retry_id = history_manager.retry_entry(entry_id)
        assert retry_id is not None
        assert retry_id != entry_id
        
        # Check original entry
        original = history_manager.get_entry(entry_id)
        assert original.retry_count == 1
        assert original.status == GenerationStatus.RETRYING
        
        # Check retry entry
        retry_entry = history_manager.get_entry(retry_id)
        assert retry_entry.prompt == original.prompt
        assert retry_entry.status == GenerationStatus.PENDING
    
    def test_get_statistics(self, history_manager):
        """Test generation statistics"""
        # Add various entries
        entries = []
        for i in range(5):
            entry_id = history_manager.add_entry(
                model_type="t2v",
                prompt=f"Test prompt {i}"
            )
            entries.append(entry_id)
        
        # Mark some as completed
        for i in range(3):
            history_manager.update_entry_status(
                entries[i],
                GenerationStatus.COMPLETED,
                generation_time=60.0 + i * 10
            )
        
        # Mark one as failed
        history_manager.update_entry_status(
            entries[3],
            GenerationStatus.FAILED,
            error_message="Test error"
        )
        
        stats = history_manager.get_statistics()
        assert stats["total"] == 5
        assert stats["by_status"]["completed"] == 3
        assert stats["by_status"]["failed"] == 1
        assert stats["by_status"]["pending"] == 1
        assert stats["success_rate"] == 60.0  # 3/5 * 100
    
    def test_search_entries(self, history_manager):
        """Test searching history entries"""
        # Add entries with different prompts
        history_manager.add_entry(model_type="t2v", prompt="Beautiful sunset")
        history_manager.add_entry(model_type="i2v", prompt="Dancing cat")
        history_manager.add_entry(model_type="t2v", prompt="Mountain landscape")
        
        # Search for "sunset"
        results = history_manager.search_entries("sunset")
        assert len(results) == 1
        assert "sunset" in results[0].prompt.lower()
        
        # Search for "t2v"
        results = history_manager.search_entries("t2v")
        assert len(results) == 2
    
    def test_user_rating(self, history_manager):
        """Test adding user ratings"""
        entry_id = history_manager.add_entry(
            model_type="t2v",
            prompt="Test prompt"
        )
        
        # Add rating
        history_manager.add_user_rating(entry_id, 4, "Good quality")
        
        entry = history_manager.get_entry(entry_id)
        assert entry.user_rating == 4
        assert entry.user_notes == "Good quality"
    
    def test_export_history(self, history_manager, temp_history_file):
        """Test exporting history"""
        # Add some entries
        for i in range(3):
            history_manager.add_entry(
                model_type="t2v",
                prompt=f"Test prompt {i}"
            )
        
        # Export to file
        export_path = temp_history_file + ".export"
        success = history_manager.export_history(export_path)
        assert success
        
        # Verify export file
        with open(export_path, 'r') as f:
            export_data = json.load(f)
        
        assert export_data["total_entries"] == 3
        assert len(export_data["entries"]) == 3
        
        Path(export_path).unlink()

class TestInteractiveErrorResolver:
    """Test interactive error resolution system"""
    
    @pytest.fixture
    def error_resolver(self):
        """Create error resolver"""
        return InteractiveErrorResolver()
    
    @pytest.fixture
    def sample_error(self):
        """Create sample error"""
        from error_handler import RecoveryAction
        return UserFriendlyError(
            category=ErrorCategory.VRAM_MEMORY,
            severity=ErrorSeverity.HIGH,
            title="Insufficient VRAM",
            message="Not enough GPU memory for generation",
            recovery_suggestions=["Reduce resolution", "Enable CPU offloading"],
            recovery_actions=[
                RecoveryAction(
                    action_type="optimize_vram",
                    description="Optimize VRAM settings",
                    parameters={},
                    automatic=True
                )
            ]
        )
    
    def test_start_resolution_session(self, error_resolver, sample_error):
        """Test starting error resolution session"""
        session_id = error_resolver.start_resolution_session(sample_error)
        assert session_id != ""
        
        status = error_resolver.get_session_status(session_id)
        assert status is not None
        assert status["status"] == "pending"
        assert status["error_category"] == "vram_memory"
    
    def test_execute_automatic_steps(self, error_resolver, sample_error):
        """Test executing automatic resolution steps"""
        session_id = error_resolver.start_resolution_session(sample_error)
        
        # Mock the VRAM check to return success
        with patch.object(error_resolver, '_check_vram_usage', return_value=(True, "VRAM OK")):
            results = error_resolver.execute_automatic_steps(session_id)
        
        assert len(results) > 0
        assert any(result["success"] for result in results)
    
    def test_get_resolution_suggestions(self, error_resolver, sample_error):
        """Test getting resolution suggestions"""
        session_id = error_resolver.start_resolution_session(sample_error)
        
        suggestions = error_resolver.get_resolution_suggestions(session_id)
        assert len(suggestions) > 0
        
        # Check suggestion structure
        suggestion = suggestions[0]
        assert "id" in suggestion
        assert "title" in suggestion
        assert "description" in suggestion
        assert "success_probability" in suggestion
    
    def test_apply_user_selected_fix(self, error_resolver, sample_error):
        """Test applying user-selected fixes"""
        session_id = error_resolver.start_resolution_session(sample_error)
        
        # Mock a fix execution
        with patch.object(error_resolver, '_optimize_vram_settings', return_value=(True, "VRAM optimized")):
            success, message = error_resolver.apply_user_selected_fix(
                session_id, "optimize_vram_settings"
            )
        
        assert success
        assert "optimized" in message.lower()
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated', return_value=4 * 1024**3)  # 4GB
    def test_vram_check_resolution_step(self, mock_allocated, mock_props, mock_available, error_resolver):
        """Test VRAM checking resolution step"""
        # Mock GPU properties
        mock_device = Mock()
        mock_device.total_memory = 8 * 1024**3  # 8GB
        mock_props.return_value = mock_device
        
        success, message = error_resolver._check_vram_usage()
        assert success
        assert "4.0GB / 8.0GB" in message

class TestHardwareParameterRecommender:
    """Test hardware-based parameter recommendations"""
    
    @pytest.fixture
    def recommender(self):
        """Create parameter recommender"""
        return HardwareParameterRecommender()
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_name', return_value="RTX 4080")
    @patch('torch.cuda.get_device_properties')
    def test_hardware_detection(self, mock_props, mock_name, mock_available):
        """Test hardware detection"""
        # Mock GPU properties
        mock_device = Mock()
        mock_device.total_memory = 16 * 1024**3  # 16GB
        mock_props.return_value = mock_device
        
        recommender = HardwareParameterRecommender()
        profile = recommender.get_hardware_profile()
        
        assert profile is not None
        assert profile.gpu_name == "RTX 4080"
        assert profile.gpu_memory_gb == 16.0
        assert profile.hardware_class in [HardwareClass.HIGH_END, HardwareClass.ENTHUSIAST]
    
    def test_parameter_recommendations(self, recommender):
        """Test getting parameter recommendations"""
        # Mock hardware profile
        recommender.hardware_profile.gpu_memory_gb = 8.0
        recommender.hardware_profile.hardware_class = HardwareClass.MID_RANGE
        
        current_params = {
            "resolution": "1920x1080",
            "steps": 100,
            "enable_cpu_offload": False
        }
        
        recommendations = recommender.get_parameter_recommendations("t2v", current_params)
        
        assert len(recommendations) > 0
        
        # Should recommend CPU offloading for 8GB VRAM
        offload_rec = next(
            (r for r in recommendations if r.parameter == "enable_cpu_offload"),
            None
        )
        assert offload_rec is not None
        assert offload_rec.recommended_value is True
    
    def test_generation_profiles(self, recommender):
        """Test getting generation profiles"""
        # Mock hardware profile
        recommender.hardware_profile.gpu_memory_gb = 12.0
        recommender.hardware_profile.hardware_class = HardwareClass.HIGH_END
        
        profiles = recommender.get_generation_profiles("t2v")
        
        assert len(profiles) >= 2  # At least speed and balanced
        
        # Check profile structure
        profile = profiles[0]
        assert hasattr(profile, 'name')
        assert hasattr(profile, 'parameters')
        assert hasattr(profile, 'estimated_time_minutes')
        assert hasattr(profile, 'quality_score')
        assert hasattr(profile, 'success_probability')
    
    def test_optimization_suggestions(self, recommender):
        """Test getting optimization suggestions"""
        # Mock hardware profile
        recommender.hardware_profile.gpu_memory_gb = 6.0
        recommender.hardware_profile.hardware_class = HardwareClass.LOW_END
        
        current_params = {"steps": 80, "resolution": "1920x1080"}
        
        # Test speed optimization
        suggestions = recommender.get_optimization_suggestions(current_params, "speed")
        assert len(suggestions) > 0
        assert any("steps" in s.lower() for s in suggestions)
        
        # Test memory optimization
        suggestions = recommender.get_optimization_suggestions(current_params, "memory")
        assert len(suggestions) > 0
        assert any("offload" in s.lower() for s in suggestions)

class TestGenerationQueueManager:
    """Test generation queue management"""
    
    @pytest.fixture
    def queue_manager(self):
        """Create queue manager"""
        config = {
            "max_concurrent_requests": 2,
            "max_queue_size": 10,
            "save_queue_state": False
        }
        manager = GenerationQueueManager(config)
        yield manager
        manager.shutdown()
    
    def test_submit_request(self, queue_manager):
        """Test submitting generation requests"""
        request_id = queue_manager.submit_request(
            model_type="t2v",
            prompt="Test prompt",
            priority=QueuePriority.NORMAL
        )
        
        assert request_id != ""
        
        status = queue_manager.get_request_status(request_id)
        assert status is not None
        assert status["status"] in ["queued", "processing"]
    
    def test_queue_priority(self, queue_manager):
        """Test queue priority handling"""
        # Submit low priority request
        low_id = queue_manager.submit_request(
            model_type="t2v",
            prompt="Low priority",
            priority=QueuePriority.LOW
        )
        
        # Submit high priority request
        high_id = queue_manager.submit_request(
            model_type="t2v",
            prompt="High priority",
            priority=QueuePriority.HIGH
        )
        
        # High priority should be processed first
        time.sleep(0.5)  # Allow processing to start
        
        low_status = queue_manager.get_request_status(low_id)
        high_status = queue_manager.get_request_status(high_id)
        
        # High priority should be processing or completed first
        assert high_status["status"] in ["processing", "completed"]
    
    def test_cancel_request(self, queue_manager):
        """Test cancelling requests"""
        request_id = queue_manager.submit_request(
            model_type="t2v",
            prompt="Test prompt"
        )
        
        # Cancel the request
        success = queue_manager.cancel_request(request_id)
        assert success
        
        status = queue_manager.get_request_status(request_id)
        assert status["status"] == "cancelled"
    
    def test_queue_status(self, queue_manager):
        """Test getting queue status"""
        # Submit some requests
        for i in range(3):
            queue_manager.submit_request(
                model_type="t2v",
                prompt=f"Test prompt {i}"
            )
        
        status = queue_manager.get_queue_status()
        
        assert "statistics" in status
        assert "active_requests" in status
        assert "worker_status" in status
        
        stats = status["statistics"]
        assert stats["total_requests"] >= 3
    
    def test_user_requests(self, queue_manager):
        """Test getting user-specific requests"""
        user_id = "test_user"
        
        # Submit requests for specific user
        request_ids = []
        for i in range(2):
            request_id = queue_manager.submit_request(
                model_type="t2v",
                prompt=f"User prompt {i}",
                user_id=user_id
            )
            request_ids.append(request_id)
        
        # Get user requests
        user_requests = queue_manager.get_user_requests(user_id)
        
        assert len(user_requests) >= 2
        assert all(req["user_id"] == user_id for req in user_requests)
    
    def test_retry_failed_request(self, queue_manager):
        """Test retrying failed requests"""
        # Mock a request that will fail
        with patch.object(queue_manager, '_execute_generation', return_value=(False, None, "Test error")):
            request_id = queue_manager.submit_request(
                model_type="t2v",
                prompt="Test prompt"
            )
            
            # Wait for processing
            time.sleep(1)
            
            # Check if it failed
            status = queue_manager.get_request_status(request_id)
            if status and status["status"] == "failed":
                # Retry the request
                retry_id = queue_manager.retry_failed_request(request_id)
                assert retry_id is not None
                assert retry_id != request_id
    
    def test_pause_resume_queue(self, queue_manager):
        """Test pausing and resuming queue"""
        # Pause the queue
        queue_manager.pause_queue()
        
        # Submit a request
        request_id = queue_manager.submit_request(
            model_type="t2v",
            prompt="Test prompt"
        )
        
        # Should remain queued
        time.sleep(0.5)
        status = queue_manager.get_request_status(request_id)
        assert status["status"] == "queued"
        
        # Resume the queue
        queue_manager.resume_queue()
        
        # Should start processing
        time.sleep(0.5)
        status = queue_manager.get_request_status(request_id)
        assert status["status"] in ["processing", "completed"]
    
    def test_clear_queue(self, queue_manager):
        """Test clearing the queue"""
        # Submit multiple requests
        for i in range(3):
            queue_manager.submit_request(
                model_type="t2v",
                prompt=f"Test prompt {i}"
            )
        
        # Clear the queue
        queue_manager.clear_queue()
        
        # Check queue is empty
        status = queue_manager.get_queue_status()
        assert status["statistics"]["queued_requests"] == 0

class TestIntegratedUserExperience:
    """Test integrated user experience scenarios"""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated system with all components"""
        history_manager = GenerationHistoryManager()
        error_resolver = InteractiveErrorResolver()
        recommender = HardwareParameterRecommender()
        queue_manager = GenerationQueueManager({
            "max_concurrent_requests": 1,
            "save_queue_state": False
        })
        
        yield {
            "history": history_manager,
            "resolver": error_resolver,
            "recommender": recommender,
            "queue": queue_manager
        }
        
        queue_manager.shutdown()
    
    def test_end_to_end_generation_flow(self, integrated_system):
        """Test complete generation flow with all enhancements"""
        history = integrated_system["history"]
        recommender = integrated_system["recommender"]
        queue = integrated_system["queue"]
        
        # Get parameter recommendations
        current_params = {"steps": 100, "resolution": "1920x1080"}
        recommendations = recommender.get_parameter_recommendations("t2v", current_params)
        
        # Apply recommendations
        optimized_params = current_params.copy()
        for rec in recommendations:
            optimized_params[rec.parameter] = rec.recommended_value
        
        # Add to history (filter parameters to only include supported ones)
        history_params = {
            k: v for k, v in optimized_params.items() 
            if k in ["resolution", "steps"]
        }
        history_id = history.add_entry(
            model_type="t2v",
            prompt="Test generation with optimizations",
            **history_params
        )
        
        # Submit to queue
        queue_id = queue.submit_request(
            model_type="t2v",
            prompt="Test generation with optimizations",
            parameters=optimized_params
        )
        
        # Verify integration
        assert history_id != ""
        assert queue_id != ""
        
        # Check history entry
        history_entry = history.get_entry(history_id)
        assert history_entry is not None
        
        # Check queue status
        queue_status = queue.get_request_status(queue_id)
        assert queue_status is not None
    
    def test_error_recovery_with_retry(self, integrated_system):
        """Test error recovery with automatic retry"""
        history = integrated_system["history"]
        resolver = integrated_system["resolver"]
        queue = integrated_system["queue"]
        
        # Simulate a failed generation
        history_id = history.add_entry(
            model_type="t2v",
            prompt="Test failed generation"
        )
        
        # Update as failed
        history.update_entry_status(
            history_id,
            GenerationStatus.FAILED,
            error_message="VRAM insufficient"
        )
        
        # Create error for resolution
        from error_handler import RecoveryAction
        error = UserFriendlyError(
            category=ErrorCategory.VRAM_MEMORY,
            severity=ErrorSeverity.HIGH,
            title="VRAM Error",
            message="Insufficient VRAM",
            recovery_suggestions=["Optimize VRAM"],
            recovery_actions=[
                RecoveryAction(
                    action_type="optimize_vram",
                    description="Optimize VRAM settings",
                    parameters={},
                    automatic=True
                )
            ]
        )
        
        # Start resolution session
        session_id = resolver.start_resolution_session(error)
        assert session_id != ""
        
        # Get suggestions
        suggestions = resolver.get_resolution_suggestions(session_id)
        assert len(suggestions) > 0
        
        # Retry the failed generation
        retry_id = history.retry_entry(history_id)
        assert retry_id is not None
    
    def test_hardware_based_queue_optimization(self, integrated_system):
        """Test hardware-based queue optimization"""
        recommender = integrated_system["recommender"]
        queue = integrated_system["queue"]
        
        # Mock low-end hardware
        if recommender.hardware_profile:
            recommender.hardware_profile.hardware_class = HardwareClass.LOW_END
            recommender.hardware_profile.gpu_memory_gb = 6.0
        
        # Get optimized profiles
        profiles = recommender.get_generation_profiles("t2v")
        speed_profile = next(p for p in profiles if "speed" in p.name.lower())
        
        # Submit request with optimized parameters
        request_id = queue.submit_request(
            model_type="t2v",
            prompt="Hardware optimized generation",
            parameters=speed_profile.parameters,
            priority=QueuePriority.HIGH
        )
        
        assert request_id != ""
        
        # Verify parameters are optimized for low-end hardware
        assert speed_profile.parameters.get("enable_cpu_offload", False)
        assert speed_profile.parameters.get("steps", 50) <= 30

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
