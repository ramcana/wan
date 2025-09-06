"""
Test suite for GPU Load Balancer

Tests load balancing strategies, task assignment, and multi-GPU management.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from gpu_load_balancer import (
    GPULoadBalancer, GPUTask, GPUAssignment, LoadBalancingStrategy
)
from vram_manager import GPUInfo, VRAMUsage, VRAMManager


class TestGPULoadBalancer:
    """Test cases for GPULoadBalancer class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.mock_vram_manager = Mock(spec=VRAMManager)
        self.balancer = GPULoadBalancer(vram_manager=self.mock_vram_manager)
        
        # Create test GPUs
        self.test_gpus = [
            GPUInfo(0, "RTX 4080", 16384, "535.98", is_available=True),
            GPUInfo(1, "RTX 3080", 10240, "535.98", is_available=True),
            GPUInfo(2, "RTX 2080", 8192, "535.98", is_available=True)
        ]
    
    def teardown_method(self):
        """Cleanup after each test method"""
        if hasattr(self, 'balancer'):
            self.balancer.cleanup()
    
    def test_initialization_success(self):
        """Test successful load balancer initialization"""
        self.mock_vram_manager.detect_vram_capacity.return_value = self.test_gpus
        
        result = self.balancer.initialize()
        
        assert result is True
        assert len(self.balancer.available_gpus) == 3
        assert len(self.balancer.gpu_assignments) == 3
        assert len(self.balancer.gpu_weights) == 3
        
        # Check that weights are calculated based on VRAM
        assert self.balancer.gpu_weights[0] == 1.0  # RTX 4080 has most VRAM
        assert self.balancer.gpu_weights[1] < 1.0   # RTX 3080 has less VRAM
        assert self.balancer.gpu_weights[2] < 1.0   # RTX 2080 has least VRAM
    
    def test_initialization_no_gpus(self):
        """Test initialization with no GPUs detected"""
        self.mock_vram_manager.detect_vram_capacity.return_value = []
        
        result = self.balancer.initialize()
        
        assert result is False
        assert len(self.balancer.available_gpus) == 0
    
    def test_initialization_failure(self):
        """Test initialization failure"""
        self.mock_vram_manager.detect_vram_capacity.side_effect = Exception("Detection failed")
        
        result = self.balancer.initialize()
        
        assert result is False
    
    def test_submit_task(self):
        """Test task submission"""
        task = GPUTask(
            task_id="test_task_1",
            task_type="image_generation",
            estimated_vram_mb=2048
        )
        
        task_id = self.balancer.submit_task(task)
        
        assert task_id == "test_task_1"
        assert self.balancer.task_queue.qsize() == 1
    
    def test_round_robin_strategy(self):
        """Test round-robin load balancing strategy"""
        self.balancer.available_gpus = self.test_gpus
        self.balancer.strategy = LoadBalancingStrategy.ROUND_ROBIN
        
        # Initialize assignments
        for gpu in self.test_gpus:
            self.balancer.gpu_assignments[gpu.index] = []
        
        task = GPUTask("test", "generation", 1024)
        
        # First task should go to GPU 0
        gpu_index = self.balancer._select_round_robin(self.test_gpus)
        assert gpu_index == 0
        
        # Simulate assignment
        self.balancer.gpu_assignments[0].append(Mock())
        
        # Second task should go to GPU 1
        gpu_index = self.balancer._select_round_robin(self.test_gpus)
        assert gpu_index == 1
    
    def test_least_loaded_strategy(self):
        """Test least loaded load balancing strategy"""
        self.balancer.available_gpus = self.test_gpus
        
        # Mock VRAM usage - GPU 1 has lowest usage
        usage_data = [
            VRAMUsage(0, 8192, 8192, 16384, 50.0, datetime.now()),  # 50% usage
            VRAMUsage(1, 2048, 8192, 10240, 20.0, datetime.now()),  # 20% usage (lowest)
            VRAMUsage(2, 4096, 4096, 8192, 50.0, datetime.now())    # 50% usage
        ]
        
        def mock_get_usage(gpu_index):
            for usage in usage_data:
                if usage.gpu_index == gpu_index:
                    return [usage]
            return []
        
        self.mock_vram_manager.get_current_vram_usage.side_effect = mock_get_usage
        
        gpu_index = self.balancer._select_least_loaded(self.test_gpus)
        assert gpu_index == 1  # GPU 1 has lowest usage
    
    def test_most_vram_strategy(self):
        """Test most VRAM load balancing strategy"""
        self.balancer.available_gpus = self.test_gpus
        
        # Mock VRAM usage - GPU 0 has most free VRAM
        usage_data = [
            VRAMUsage(0, 4096, 12288, 16384, 25.0, datetime.now()),  # 12GB free (most)
            VRAMUsage(1, 6144, 4096, 10240, 60.0, datetime.now()),   # 4GB free
            VRAMUsage(2, 4096, 4096, 8192, 50.0, datetime.now())     # 4GB free
        ]
        
        def mock_get_usage(gpu_index):
            for usage in usage_data:
                if usage.gpu_index == gpu_index:
                    return [usage]
            return []
        
        self.mock_vram_manager.get_current_vram_usage.side_effect = mock_get_usage
        
        gpu_index = self.balancer._select_most_vram(self.test_gpus)
        assert gpu_index == 0  # GPU 0 has most free VRAM
    
    def test_weighted_strategy(self):
        """Test weighted load balancing strategy"""
        self.balancer.available_gpus = self.test_gpus
        
        # Set up weights and performance history
        self.balancer.gpu_weights = {0: 1.0, 1: 0.8, 2: 0.6}
        self.balancer.gpu_performance_history = {
            0: [100.0, 95.0, 105.0],  # Good performance
            1: [80.0, 85.0, 90.0],    # Moderate performance
            2: [60.0, 65.0, 70.0]     # Lower performance
        }
        
        # Mock low usage for all GPUs
        usage_data = [
            VRAMUsage(0, 2048, 14336, 16384, 12.5, datetime.now()),
            VRAMUsage(1, 2048, 8192, 10240, 20.0, datetime.now()),
            VRAMUsage(2, 2048, 6144, 8192, 25.0, datetime.now())
        ]
        
        def mock_get_usage(gpu_index):
            for usage in usage_data:
                if usage.gpu_index == gpu_index:
                    return [usage]
            return []
        
        self.mock_vram_manager.get_current_vram_usage.side_effect = mock_get_usage
        
        gpu_index = self.balancer._select_weighted(self.test_gpus)
        assert gpu_index == 0  # GPU 0 should have highest weighted score
    
    def test_filter_suitable_gpus(self):
        """Test filtering of suitable GPUs for a task"""
        self.balancer.available_gpus = self.test_gpus
        self.balancer.max_vram_usage_percent = 80.0
        
        # Task requiring 4GB VRAM
        task = GPUTask("test", "generation", 4096)
        
        # Mock VRAM usage - GPU 1 is too loaded
        usage_data = [
            VRAMUsage(0, 4096, 12288, 16384, 25.0, datetime.now()),  # Suitable
            VRAMUsage(1, 8192, 2048, 10240, 80.0, datetime.now()),   # Too loaded
            VRAMUsage(2, 2048, 6144, 8192, 25.0, datetime.now())     # Suitable
        ]
        
        def mock_get_usage(gpu_index):
            for usage in usage_data:
                if usage.gpu_index == gpu_index:
                    return [usage]
            return []
        
        self.mock_vram_manager.get_current_vram_usage.side_effect = mock_get_usage
        
        suitable_gpus = self.balancer._filter_suitable_gpus(task)
        
        # Should return GPUs 0 and 2 (GPU 1 is too loaded)
        suitable_indices = [gpu.index for gpu in suitable_gpus]
        assert 0 in suitable_indices
        assert 1 not in suitable_indices  # Too loaded
        assert 2 in suitable_indices
    
    def test_filter_suitable_gpus_insufficient_vram(self):
        """Test filtering when task requires more VRAM than GPU has"""
        self.balancer.available_gpus = self.test_gpus
        
        # Task requiring 20GB VRAM (more than any GPU has)
        task = GPUTask("test", "generation", 20480)
        
        suitable_gpus = self.balancer._filter_suitable_gpus(task)
        
        # No GPUs should be suitable
        assert len(suitable_gpus) == 0
    
    def test_get_optimal_gpu(self):
        """Test getting optimal GPU for a task"""
        self.balancer.available_gpus = self.test_gpus
        self.balancer.strategy = LoadBalancingStrategy.LEAST_LOADED
        
        # Initialize assignments
        for gpu in self.test_gpus:
            self.balancer.gpu_assignments[gpu.index] = []
        
        task = GPUTask("test", "generation", 2048)
        
        # Mock VRAM usage
        usage_data = [
            VRAMUsage(0, 4096, 12288, 16384, 25.0, datetime.now()),
            VRAMUsage(1, 2048, 8192, 10240, 20.0, datetime.now()),  # Least loaded
            VRAMUsage(2, 4096, 4096, 8192, 50.0, datetime.now())
        ]
        
        def mock_get_usage(gpu_index):
            for usage in usage_data:
                if usage.gpu_index == gpu_index:
                    return [usage]
            return []
        
        self.mock_vram_manager.get_current_vram_usage.side_effect = mock_get_usage
        
        optimal_gpu = self.balancer.get_optimal_gpu(task)
        assert optimal_gpu == 1  # GPU 1 is least loaded
    
    def test_get_optimal_gpu_no_suitable(self):
        """Test getting optimal GPU when no suitable GPU available"""
        self.balancer.available_gpus = self.test_gpus
        
        # Task requiring more VRAM than available
        task = GPUTask("test", "generation", 50000)
        
        optimal_gpu = self.balancer.get_optimal_gpu(task)
        assert optimal_gpu is None
    
    def test_start_stop_balancing(self):
        """Test starting and stopping load balancing"""
        self.mock_vram_manager.detect_vram_capacity.return_value = self.test_gpus
        
        # Initialize first
        self.balancer.initialize()
        
        # Start balancing
        self.balancer.start_balancing()
        assert self.balancer.balancer_active is True
        assert self.balancer.balancer_thread is not None
        assert self.balancer.health_monitor_thread is not None
        
        # Stop balancing
        self.balancer.stop_balancing()
        assert self.balancer.balancer_active is False
    
    def test_process_task_success(self):
        """Test successful task processing"""
        self.balancer.available_gpus = self.test_gpus
        
        # Initialize assignments
        for gpu in self.test_gpus:
            self.balancer.gpu_assignments[gpu.index] = []
        
        task = GPUTask("test_task", "generation", 2048)
        
        # Mock get_optimal_gpu to return GPU 0
        with patch.object(self.balancer, 'get_optimal_gpu', return_value=0):
            self.balancer._process_task(task)
        
        # Check that task was assigned to GPU 0
        assert len(self.balancer.gpu_assignments[0]) == 1
        assignment = self.balancer.gpu_assignments[0][0]
        assert assignment.task.task_id == "test_task"
        assert assignment.gpu_index == 0
    
    def test_process_task_no_suitable_gpu(self):
        """Test task processing when no suitable GPU available"""
        self.balancer.available_gpus = self.test_gpus
        
        task = GPUTask("test_task", "generation", 2048)
        
        # Mock get_optimal_gpu to return None
        with patch.object(self.balancer, 'get_optimal_gpu', return_value=None):
            self.balancer._process_task(task)
        
        # Check that task was marked as failed
        assert "test_task" in self.balancer.failed_tasks
        assert "No suitable GPU available" in self.balancer.failed_tasks["test_task"]
    
    def test_cleanup_assignments_timeout(self):
        """Test cleanup of timed-out assignments"""
        self.balancer.task_timeout_seconds = 1.0  # Short timeout for testing
        
        # Create a timed-out assignment
        old_task = GPUTask("old_task", "generation", 2048)
        old_assignment = GPUAssignment(
            gpu_index=0,
            task=old_task,
            assigned_at=time.time() - 2.0  # 2 seconds ago
        )
        
        self.balancer.gpu_assignments[0] = [old_assignment]
        
        # Run cleanup
        self.balancer._cleanup_assignments()
        
        # Check that assignment was removed and task marked as failed
        assert len(self.balancer.gpu_assignments[0]) == 0
        assert "old_task" in self.balancer.failed_tasks
        assert "timeout" in self.balancer.failed_tasks["old_task"].lower()
    
    def test_update_gpu_health(self):
        """Test GPU health monitoring"""
        self.balancer.available_gpus = self.test_gpus
        
        # Mock critical VRAM usage for GPU 1
        usage_data = [
            VRAMUsage(0, 4096, 12288, 16384, 25.0, datetime.now()),  # Healthy
            VRAMUsage(1, 9830, 410, 10240, 96.0, datetime.now()),    # Critical usage
            VRAMUsage(2, 4096, 4096, 8192, 50.0, datetime.now())     # Healthy
        ]
        
        self.mock_vram_manager.get_current_vram_usage.return_value = usage_data
        
        self.balancer._update_gpu_health()
        
        # Check that GPU 1 is marked as unavailable
        assert self.balancer.available_gpus[0].is_available is True   # GPU 0
        assert self.balancer.available_gpus[1].is_available is False  # GPU 1 (critical)
        assert self.balancer.available_gpus[2].is_available is True   # GPU 2
    
    def test_get_load_balancing_stats(self):
        """Test getting load balancing statistics"""
        self.balancer.available_gpus = self.test_gpus
        
        # Add some test data
        self.balancer.gpu_assignments[0] = [Mock(), Mock()]  # 2 assignments
        self.balancer.gpu_assignments[1] = [Mock()]          # 1 assignment
        self.balancer.gpu_assignments[2] = []                # 0 assignments
        
        self.balancer.completed_tasks["task1"] = Mock()
        self.balancer.failed_tasks["task2"] = "Error"
        
        self.balancer.gpu_performance_history[0] = [100.0, 95.0, 105.0]
        
        stats = self.balancer.get_load_balancing_stats()
        
        assert stats["total_gpus"] == 3
        assert stats["completed_tasks"] == 1
        assert stats["failed_tasks"] == 1
        assert stats["gpu_assignments"][0] == 2
        assert stats["gpu_assignments"][1] == 1
        assert stats["gpu_assignments"][2] == 0
        assert 0 in stats["performance_history"]
        assert stats["performance_history"][0]["avg_performance"] == 100.0
    
    def test_set_strategy(self):
        """Test setting load balancing strategy"""
        self.balancer.set_strategy(LoadBalancingStrategy.WEIGHTED)
        assert self.balancer.strategy == LoadBalancingStrategy.WEIGHTED
    
    def test_set_gpu_weight(self):
        """Test setting GPU weight"""
        self.balancer.gpu_weights[0] = 1.0
        self.balancer.set_gpu_weight(0, 1.5)
        assert self.balancer.gpu_weights[0] == 1.5


class TestGPUTask:
    """Test cases for GPUTask class"""
    
    def test_gpu_task_creation(self):
        """Test GPUTask creation"""
        task = GPUTask(
            task_id="test_task",
            task_type="image_generation",
            estimated_vram_mb=4096,
            priority=2,
            metadata={"model": "stable_diffusion"}
        )
        
        assert task.task_id == "test_task"
        assert task.task_type == "image_generation"
        assert task.estimated_vram_mb == 4096
        assert task.priority == 2
        assert task.metadata["model"] == "stable_diffusion"
    
    def test_gpu_task_defaults(self):
        """Test GPUTask with default values"""
        task = GPUTask(
            task_id="test_task",
            task_type="image_generation",
            estimated_vram_mb=2048
        )
        
        assert task.priority == 1
        assert task.metadata is None


class TestGPUAssignment:
    """Test cases for GPUAssignment class"""
    
    def test_gpu_assignment_creation(self):
        """Test GPUAssignment creation"""
        task = GPUTask("test", "generation", 2048)
        assignment = GPUAssignment(
            gpu_index=0,
            task=task,
            assigned_at=time.time()
        )
        
        assert assignment.gpu_index == 0
        assert assignment.task.task_id == "test"
        assert assignment.assigned_at > 0
        assert assignment.estimated_completion is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])