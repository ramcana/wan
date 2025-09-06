"""
GPU Load Balancer for WAN22 System

Provides intelligent load balancing across multiple GPUs for optimal performance.
"""

import logging
import threading
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from queue import Queue, Empty
import json

from vram_manager import VRAMManager, GPUInfo, VRAMUsage


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    MOST_VRAM = "most_vram"
    WEIGHTED = "weighted"


@dataclass
class GPUTask:
    """Represents a task to be executed on a GPU"""
    task_id: str
    task_type: str
    estimated_vram_mb: int
    priority: int = 1
    metadata: Dict[str, Any] = None


@dataclass
class GPUAssignment:
    """Represents a GPU assignment for a task"""
    gpu_index: int
    task: GPUTask
    assigned_at: float
    estimated_completion: Optional[float] = None


class GPULoadBalancer:
    """
    Intelligent GPU load balancer for multi-GPU setups
    
    Features:
    - Multiple load balancing strategies
    - Real-time VRAM usage monitoring
    - Task queue management
    - GPU health monitoring
    - Automatic failover
    """
    
    def __init__(self, vram_manager: Optional[VRAMManager] = None):
        self.logger = logging.getLogger(__name__)
        self.vram_manager = vram_manager or VRAMManager()
        
        # Load balancing configuration
        self.strategy = LoadBalancingStrategy.LEAST_LOADED
        self.max_vram_usage_percent = 85.0
        self.task_timeout_seconds = 300.0
        self.health_check_interval = 5.0
        
        # State tracking
        self.available_gpus: List[GPUInfo] = []
        self.gpu_assignments: Dict[int, List[GPUAssignment]] = {}
        self.task_queue: Queue[GPUTask] = Queue()
        self.completed_tasks: Dict[str, GPUAssignment] = {}
        self.failed_tasks: Dict[str, str] = {}
        
        # Threading
        self.balancer_active = False
        self.balancer_thread: Optional[threading.Thread] = None
        self.health_monitor_thread: Optional[threading.Thread] = None
        
        # GPU weights for weighted strategy
        self.gpu_weights: Dict[int, float] = {}
        
        # Performance tracking
        self.gpu_performance_history: Dict[int, List[float]] = {}
        
    def initialize(self) -> bool:
        """
        Initialize the load balancer
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Detect available GPUs
            self.available_gpus = self.vram_manager.detect_vram_capacity()
            
            if not self.available_gpus:
                self.logger.error("No GPUs detected for load balancing")
                return False
            
            # Initialize GPU assignments
            for gpu in self.available_gpus:
                self.gpu_assignments[gpu.index] = []
                self.gpu_weights[gpu.index] = 1.0  # Default weight
                self.gpu_performance_history[gpu.index] = []
            
            # Calculate initial weights based on VRAM capacity
            self._calculate_initial_weights()
            
            self.logger.info(f"Load balancer initialized with {len(self.available_gpus)} GPUs")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize load balancer: {e}")
            return False
    
    def start_balancing(self) -> None:
        """Start the load balancing service"""
        if self.balancer_active:
            self.logger.warning("Load balancer already active")
            return
        
        if not self.available_gpus:
            if not self.initialize():
                raise RuntimeError("Failed to initialize load balancer")
        
        self.balancer_active = True
        
        # Start balancer thread
        self.balancer_thread = threading.Thread(
            target=self._balancing_loop,
            daemon=True
        )
        self.balancer_thread.start()
        
        # Start health monitor thread
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True
        )
        self.health_monitor_thread.start()
        
        self.logger.info("GPU load balancer started")
    
    def stop_balancing(self) -> None:
        """Stop the load balancing service"""
        if not self.balancer_active:
            return
        
        self.balancer_active = False
        
        # Wait for threads to finish
        if self.balancer_thread:
            self.balancer_thread.join(timeout=5.0)
        
        if self.health_monitor_thread:
            self.health_monitor_thread.join(timeout=5.0)
        
        self.logger.info("GPU load balancer stopped")
    
    def submit_task(self, task: GPUTask) -> str:
        """
        Submit a task for GPU processing
        
        Args:
            task: GPUTask to be processed
            
        Returns:
            Task ID for tracking
        """
        self.task_queue.put(task)
        self.logger.info(f"Task {task.task_id} submitted to queue")
        return task.task_id
    
    def get_optimal_gpu(self, task: GPUTask) -> Optional[int]:
        """
        Get the optimal GPU for a given task
        
        Args:
            task: Task to be assigned
            
        Returns:
            GPU index or None if no suitable GPU available
        """
        if not self.available_gpus:
            return None
        
        # Filter GPUs that can handle the task
        suitable_gpus = self._filter_suitable_gpus(task)
        
        if not suitable_gpus:
            return None
        
        # Apply load balancing strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._select_round_robin(suitable_gpus)
        elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
            return self._select_least_loaded(suitable_gpus)
        elif self.strategy == LoadBalancingStrategy.MOST_VRAM:
            return self._select_most_vram(suitable_gpus)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED:
            return self._select_weighted(suitable_gpus)
        else:
            return suitable_gpus[0].index
    
    def _filter_suitable_gpus(self, task: GPUTask) -> List[GPUInfo]:
        """Filter GPUs that can handle the given task"""
        suitable_gpus = []
        
        for gpu in self.available_gpus:
            if not gpu.is_available:
                continue
            
            # Check VRAM capacity
            if task.estimated_vram_mb > gpu.total_memory_mb:
                continue
            
            # Check current VRAM usage
            try:
                usage = self.vram_manager.get_current_vram_usage(gpu.index)
                if usage:
                    current_usage = usage[0]
                    available_vram = current_usage.free_mb
                    
                    if task.estimated_vram_mb > available_vram:
                        continue
                    
                    # Check if adding this task would exceed usage threshold
                    projected_usage = ((current_usage.used_mb + task.estimated_vram_mb) / current_usage.total_mb) * 100
                    if projected_usage > self.max_vram_usage_percent:
                        continue
            except:
                # If we can't get usage info, assume GPU is available
                pass
            
            suitable_gpus.append(gpu)
        
        return suitable_gpus
    
    def _select_round_robin(self, gpus: List[GPUInfo]) -> int:
        """Select GPU using round-robin strategy"""
        # Simple round-robin based on total assignments
        min_assignments = float('inf')
        selected_gpu = gpus[0].index
        
        for gpu in gpus:
            assignment_count = len(self.gpu_assignments.get(gpu.index, []))
            if assignment_count < min_assignments:
                min_assignments = assignment_count
                selected_gpu = gpu.index
        
        return selected_gpu
    
    def _select_least_loaded(self, gpus: List[GPUInfo]) -> int:
        """Select GPU with least current load"""
        min_load = float('inf')
        selected_gpu = gpus[0].index
        
        for gpu in gpus:
            try:
                usage = self.vram_manager.get_current_vram_usage(gpu.index)
                if usage:
                    load = usage[0].usage_percent
                else:
                    load = 0.0
                
                if load < min_load:
                    min_load = load
                    selected_gpu = gpu.index
            except:
                # If we can't get usage, assume low load
                if 0.0 < min_load:
                    min_load = 0.0
                    selected_gpu = gpu.index
        
        return selected_gpu
    
    def _select_most_vram(self, gpus: List[GPUInfo]) -> int:
        """Select GPU with most available VRAM"""
        max_vram = 0
        selected_gpu = gpus[0].index
        
        for gpu in gpus:
            try:
                usage = self.vram_manager.get_current_vram_usage(gpu.index)
                if usage:
                    available_vram = usage[0].free_mb
                else:
                    available_vram = gpu.total_memory_mb
                
                if available_vram > max_vram:
                    max_vram = available_vram
                    selected_gpu = gpu.index
            except:
                # If we can't get usage, use total VRAM
                if gpu.total_memory_mb > max_vram:
                    max_vram = gpu.total_memory_mb
                    selected_gpu = gpu.index
        
        return selected_gpu
    
    def _select_weighted(self, gpus: List[GPUInfo]) -> int:
        """Select GPU using weighted strategy"""
        best_score = -1.0
        selected_gpu = gpus[0].index
        
        for gpu in gpus:
            try:
                # Calculate score based on weight and current load
                weight = self.gpu_weights.get(gpu.index, 1.0)
                
                usage = self.vram_manager.get_current_vram_usage(gpu.index)
                if usage:
                    load_factor = 1.0 - (usage[0].usage_percent / 100.0)
                else:
                    load_factor = 1.0
                
                # Include performance history
                perf_history = self.gpu_performance_history.get(gpu.index, [])
                if perf_history:
                    avg_performance = sum(perf_history) / len(perf_history)
                    perf_factor = min(avg_performance / 100.0, 2.0)  # Cap at 2x
                else:
                    perf_factor = 1.0
                
                score = weight * load_factor * perf_factor
                
                if score > best_score:
                    best_score = score
                    selected_gpu = gpu.index
                    
            except Exception as e:
                self.logger.debug(f"Error calculating score for GPU {gpu.index}: {e}")
                continue
        
        return selected_gpu
    
    def _calculate_initial_weights(self) -> None:
        """Calculate initial GPU weights based on capabilities"""
        if not self.available_gpus:
            return
        
        # Base weights on VRAM capacity
        max_vram = max(gpu.total_memory_mb for gpu in self.available_gpus)
        
        for gpu in self.available_gpus:
            vram_ratio = gpu.total_memory_mb / max_vram
            self.gpu_weights[gpu.index] = vram_ratio
        
        self.logger.info(f"Initial GPU weights: {self.gpu_weights}")
    
    def _balancing_loop(self) -> None:
        """Main load balancing loop"""
        while self.balancer_active:
            try:
                # Process pending tasks
                try:
                    task = self.task_queue.get(timeout=1.0)
                    self._process_task(task)
                except Empty:
                    continue
                
                # Clean up completed assignments
                self._cleanup_assignments()
                
            except Exception as e:
                self.logger.error(f"Error in balancing loop: {e}")
                time.sleep(1.0)
    
    def _process_task(self, task: GPUTask) -> None:
        """Process a single task"""
        try:
            optimal_gpu = self.get_optimal_gpu(task)
            
            if optimal_gpu is None:
                self.logger.warning(f"No suitable GPU found for task {task.task_id}")
                self.failed_tasks[task.task_id] = "No suitable GPU available"
                return
            
            # Create assignment
            assignment = GPUAssignment(
                gpu_index=optimal_gpu,
                task=task,
                assigned_at=time.time()
            )
            
            # Add to assignments
            if optimal_gpu not in self.gpu_assignments:
                self.gpu_assignments[optimal_gpu] = []
            
            self.gpu_assignments[optimal_gpu].append(assignment)
            
            self.logger.info(f"Task {task.task_id} assigned to GPU {optimal_gpu}")
            
        except Exception as e:
            self.logger.error(f"Failed to process task {task.task_id}: {e}")
            self.failed_tasks[task.task_id] = str(e)
    
    def _cleanup_assignments(self) -> None:
        """Clean up completed and timed-out assignments"""
        current_time = time.time()
        
        for gpu_index, assignments in self.gpu_assignments.items():
            completed_assignments = []
            
            for assignment in assignments[:]:  # Copy list to avoid modification during iteration
                # Check for timeout
                if current_time - assignment.assigned_at > self.task_timeout_seconds:
                    self.logger.warning(f"Task {assignment.task.task_id} timed out on GPU {gpu_index}")
                    self.failed_tasks[assignment.task.task_id] = "Task timeout"
                    assignments.remove(assignment)
                    completed_assignments.append(assignment)
            
            # Update performance history for completed tasks
            for assignment in completed_assignments:
                execution_time = current_time - assignment.assigned_at
                if gpu_index not in self.gpu_performance_history:
                    self.gpu_performance_history[gpu_index] = []
                
                # Store performance metric (tasks per second * 100 for easier handling)
                performance_metric = (1.0 / execution_time) * 100
                self.gpu_performance_history[gpu_index].append(performance_metric)
                
                # Keep only last 100 entries
                if len(self.gpu_performance_history[gpu_index]) > 100:
                    self.gpu_performance_history[gpu_index] = self.gpu_performance_history[gpu_index][-100:]
    
    def _health_monitor_loop(self) -> None:
        """Monitor GPU health and update availability"""
        while self.balancer_active:
            try:
                self._update_gpu_health()
                time.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Error in health monitor loop: {e}")
                time.sleep(self.health_check_interval)
    
    def _update_gpu_health(self) -> None:
        """Update GPU health status"""
        try:
            current_usage = self.vram_manager.get_current_vram_usage()
            
            for usage in current_usage:
                gpu_index = usage.gpu_index
                
                # Find corresponding GPU info
                gpu_info = None
                for gpu in self.available_gpus:
                    if gpu.index == gpu_index:
                        gpu_info = gpu
                        break
                
                if not gpu_info:
                    continue
                
                # Check health conditions
                is_healthy = True
                
                # Check VRAM usage
                if usage.usage_percent > 95.0:
                    is_healthy = False
                    self.logger.warning(f"GPU {gpu_index} VRAM usage critical: {usage.usage_percent:.1f}%")
                
                # Check temperature if available
                if hasattr(gpu_info, 'temperature') and gpu_info.temperature:
                    if gpu_info.temperature > 85.0:
                        is_healthy = False
                        self.logger.warning(f"GPU {gpu_index} temperature critical: {gpu_info.temperature}°C")
                
                # Update availability
                gpu_info.is_available = is_healthy
                
        except Exception as e:
            self.logger.debug(f"Error updating GPU health: {e}")
    
    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics"""
        stats = {
            "strategy": self.strategy.value,
            "available_gpus": len([gpu for gpu in self.available_gpus if gpu.is_available]),
            "total_gpus": len(self.available_gpus),
            "pending_tasks": self.task_queue.qsize(),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "gpu_assignments": {},
            "gpu_weights": self.gpu_weights.copy(),
            "performance_history": {}
        }
        
        # Add per-GPU stats
        for gpu_index, assignments in self.gpu_assignments.items():
            stats["gpu_assignments"][gpu_index] = len(assignments)
        
        # Add performance history summary
        for gpu_index, history in self.gpu_performance_history.items():
            if history:
                stats["performance_history"][gpu_index] = {
                    "avg_performance": sum(history) / len(history),
                    "samples": len(history)
                }
        
        return stats
    
    def set_strategy(self, strategy: LoadBalancingStrategy) -> None:
        """Set load balancing strategy"""
        self.strategy = strategy
        self.logger.info(f"Load balancing strategy set to: {strategy.value}")
    
    def set_gpu_weight(self, gpu_index: int, weight: float) -> None:
        """Set weight for a specific GPU"""
        if gpu_index in self.gpu_weights:
            self.gpu_weights[gpu_index] = weight
            self.logger.info(f"GPU {gpu_index} weight set to: {weight}")
    
    def cleanup(self) -> None:
        """Cleanup load balancer resources"""
        self.stop_balancing()


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    balancer = GPULoadBalancer()
    
    try:
        if balancer.initialize():
            balancer.start_balancing()
            
            # Submit some test tasks
            for i in range(5):
                task = GPUTask(
                    task_id=f"test_task_{i}",
                    task_type="image_generation",
                    estimated_vram_mb=2048,
                    priority=1
                )
                balancer.submit_task(task)
            
            # Let it run for a bit
            time.sleep(10)
            
            # Print stats
            stats = balancer.get_load_balancing_stats()
            print("Load Balancing Stats:")
            print(json.dumps(stats, indent=2))
            
        else:
            print("Failed to initialize load balancer")
    
    finally:
        balancer.cleanup()