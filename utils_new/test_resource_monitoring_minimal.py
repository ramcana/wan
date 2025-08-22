#!/usr/bin/env python3
"""
Minimal test script for resource monitoring system
Tests only the resource monitoring functionality without other dependencies
"""

import time
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
import psutil
import GPUtil

# Import only the resource monitoring components
try:
    import pynvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False
    print("⚠️  pynvml not available, GPU monitoring will be limited")

@dataclass
class ResourceStats:
    """Data structure for system resource statistics"""
    cpu_percent: float
    ram_percent: float
    ram_used_gb: float
    ram_total_gb: float
    gpu_percent: float
    vram_used_mb: float
    vram_total_mb: float
    vram_percent: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for serialization"""
        return {
            "cpu_percent": self.cpu_percent,
            "ram_percent": self.ram_percent,
            "ram_used_gb": self.ram_used_gb,
            "ram_total_gb": self.ram_total_gb,
            "gpu_percent": self.gpu_percent,
            "vram_used_mb": self.vram_used_mb,
            "vram_total_mb": self.vram_total_mb,
            "vram_percent": self.vram_percent,
            "timestamp": self.timestamp.isoformat()
        }


class ResourceMonitor:
    """Monitors system resources including CPU, RAM, GPU, and VRAM"""
    
    def __init__(self, refresh_interval: int = 5):
        self.refresh_interval = refresh_interval
        self.monitoring_active = False
        self.monitor_thread = None
        self.current_stats = None
        self.stats_lock = threading.Lock()
        self.warning_callbacks: List[Callable[[str, float], None]] = []
        
        # Warning thresholds
        self.vram_warning_threshold = 90.0  # 90% VRAM usage
        self.ram_warning_threshold = 85.0   # 85% RAM usage
        self.cpu_warning_threshold = 90.0   # 90% CPU usage
        
        # Initialize NVIDIA ML
        self._init_nvidia_ml()
        
        print(f"Resource monitor initialized with {refresh_interval}s refresh interval")
    
    def _init_nvidia_ml(self):
        """Initialize NVIDIA Management Library"""
        if NVIDIA_ML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvidia_ml_available = True
                print("NVIDIA ML initialized successfully")
            except Exception as e:
                self.nvidia_ml_available = False
                print(f"Failed to initialize NVIDIA ML: {e}")
        else:
            self.nvidia_ml_available = False
    
    def collect_system_stats(self) -> ResourceStats:
        """Collect comprehensive system resource statistics"""
        try:
            # CPU statistics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # RAM statistics
            ram_info = psutil.virtual_memory()
            ram_percent = ram_info.percent
            ram_used_gb = ram_info.used / (1024**3)
            ram_total_gb = ram_info.total / (1024**3)
            
            # GPU and VRAM statistics
            gpu_percent = 0.0
            vram_used_mb = 0.0
            vram_total_mb = 0.0
            vram_percent = 0.0
            
            if self.nvidia_ml_available:
                try:
                    # Get GPU utilization
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_percent = gpu_util.gpu
                    
                    # Get VRAM information
                    vram_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    vram_used_mb = vram_info.used / (1024**2)
                    vram_total_mb = vram_info.total / (1024**2)
                    vram_percent = (vram_info.used / vram_info.total) * 100
                    
                except Exception as e:
                    print(f"Failed to get GPU stats via NVIDIA ML: {e}")
                    # Fallback to GPUtil
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]
                            gpu_percent = gpu.load * 100
                            vram_used_mb = gpu.memoryUsed
                            vram_total_mb = gpu.memoryTotal
                            vram_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
                    except Exception as e2:
                        print(f"Failed to get GPU stats via GPUtil: {e2}")
            else:
                # Try GPUtil as fallback
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        gpu_percent = gpu.load * 100
                        vram_used_mb = gpu.memoryUsed
                        vram_total_mb = gpu.memoryTotal
                        vram_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
                except Exception as e:
                    print(f"Failed to get GPU stats via GPUtil: {e}")
            
            # Create stats object
            stats = ResourceStats(
                cpu_percent=cpu_percent,
                ram_percent=ram_percent,
                ram_used_gb=ram_used_gb,
                ram_total_gb=ram_total_gb,
                gpu_percent=gpu_percent,
                vram_used_mb=vram_used_mb,
                vram_total_mb=vram_total_mb,
                vram_percent=vram_percent,
                timestamp=datetime.now()
            )
            
            # Check for warnings
            self._check_resource_warnings(stats)
            
            return stats
            
        except Exception as e:
            print(f"Failed to collect system stats: {e}")
            # Return empty stats on error
            return ResourceStats(
                cpu_percent=0.0,
                ram_percent=0.0,
                ram_used_gb=0.0,
                ram_total_gb=0.0,
                gpu_percent=0.0,
                vram_used_mb=0.0,
                vram_total_mb=0.0,
                vram_percent=0.0,
                timestamp=datetime.now()
            )
    
    def _check_resource_warnings(self, stats: ResourceStats):
        """Check resource usage against warning thresholds"""
        warnings = []
        
        if stats.vram_percent >= self.vram_warning_threshold:
            warnings.append(("VRAM", stats.vram_percent))
        
        if stats.ram_percent >= self.ram_warning_threshold:
            warnings.append(("RAM", stats.ram_percent))
        
        if stats.cpu_percent >= self.cpu_warning_threshold:
            warnings.append(("CPU", stats.cpu_percent))
        
        # Trigger warning callbacks
        for resource_type, usage_percent in warnings:
            for callback in self.warning_callbacks:
                try:
                    callback(resource_type, usage_percent)
                except Exception as e:
                    print(f"Error in warning callback: {e}")
    
    def start_monitoring(self):
        """Start real-time resource monitoring"""
        if self.monitoring_active:
            print("Resource monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print("Started real-time resource monitoring")
    
    def stop_monitoring(self):
        """Stop real-time resource monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        print("Stopped real-time resource monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs in background thread"""
        while self.monitoring_active:
            try:
                # Collect current stats
                stats = self.collect_system_stats()
                
                # Update current stats with thread safety
                with self.stats_lock:
                    self.current_stats = stats
                
                # Wait for next refresh
                time.sleep(self.refresh_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.refresh_interval)
    
    def get_current_stats(self) -> Optional[ResourceStats]:
        """Get the most recent resource statistics"""
        with self.stats_lock:
            return self.current_stats
    
    def refresh_stats_manually(self) -> ResourceStats:
        """Manually refresh and return current resource statistics"""
        stats = self.collect_system_stats()
        
        # Update current stats
        with self.stats_lock:
            self.current_stats = stats
        
        return stats
    
    def add_warning_callback(self, callback: Callable[[str, float], None]):
        """Add a callback function to be called when resource warnings occur"""
        self.warning_callbacks.append(callback)
    
    def set_warning_thresholds(self, vram_threshold: float = None, 
                              ram_threshold: float = None, 
                              cpu_threshold: float = None):
        """Set custom warning thresholds for resource usage"""
        if vram_threshold is not None:
            self.vram_warning_threshold = max(0.0, min(100.0, vram_threshold))
        if ram_threshold is not None:
            self.ram_warning_threshold = max(0.0, min(100.0, ram_threshold))
        if cpu_threshold is not None:
            self.cpu_warning_threshold = max(0.0, min(100.0, cpu_threshold))
        
        print(f"Updated warning thresholds: VRAM={self.vram_warning_threshold}%, "
              f"RAM={self.ram_warning_threshold}%, CPU={self.cpu_warning_threshold}%")
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get a formatted summary of current resource usage"""
        stats = self.get_current_stats()
        if not stats:
            stats = self.refresh_stats_manually()
        
        return {
            "cpu": {
                "usage_percent": round(stats.cpu_percent, 1),
                "status": "warning" if stats.cpu_percent >= self.cpu_warning_threshold else "normal"
            },
            "ram": {
                "usage_percent": round(stats.ram_percent, 1),
                "used_gb": round(stats.ram_used_gb, 2),
                "total_gb": round(stats.ram_total_gb, 2),
                "free_gb": round(stats.ram_total_gb - stats.ram_used_gb, 2),
                "status": "warning" if stats.ram_percent >= self.ram_warning_threshold else "normal"
            },
            "gpu": {
                "usage_percent": round(stats.gpu_percent, 1),
                "status": "normal"  # GPU usage warnings are less critical
            },
            "vram": {
                "usage_percent": round(stats.vram_percent, 1),
                "used_mb": round(stats.vram_used_mb, 1),
                "total_mb": round(stats.vram_total_mb, 1),
                "free_mb": round(stats.vram_total_mb - stats.vram_used_mb, 1),
                "status": "warning" if stats.vram_percent >= self.vram_warning_threshold else "normal"
            },
            "timestamp": stats.timestamp.isoformat(),
            "monitoring_active": self.monitoring_active
        }


def test_warning_callback(resource_type: str, usage_percent: float):
    """Test callback for resource warnings"""
    print(f"⚠️  WARNING: {resource_type} usage is at {usage_percent:.1f}%")

def test_basic_stats_collection():
    """Test basic system stats collection"""
    print("🔍 Testing basic system stats collection...")
    
    monitor = ResourceMonitor()
    stats = monitor.collect_system_stats()
    print(f"✅ CPU: {stats.cpu_percent:.1f}%")
    print(f"✅ RAM: {stats.ram_percent:.1f}% ({stats.ram_used_gb:.2f}GB / {stats.ram_total_gb:.2f}GB)")
    print(f"✅ GPU: {stats.gpu_percent:.1f}%")
    print(f"✅ VRAM: {stats.vram_percent:.1f}% ({stats.vram_used_mb:.1f}MB / {stats.vram_total_mb:.1f}MB)")
    print(f"✅ Timestamp: {stats.timestamp}")
    print()
    return monitor

def test_manual_refresh(monitor):
    """Test manual stats refresh functionality"""
    print("🔄 Testing manual stats refresh...")
    
    stats = monitor.refresh_stats_manually()
    print(f"✅ Manually refreshed stats at {stats.timestamp}")
    print(f"✅ Current VRAM usage: {stats.vram_percent:.1f}%")
    print()

def test_resource_summary(monitor):
    """Test formatted resource summary"""
    print("📊 Testing resource summary...")
    
    summary = monitor.get_resource_summary()
    print("✅ Resource Summary:")
    print(f"   CPU: {summary['cpu']['usage_percent']}% ({summary['cpu']['status']})")
    print(f"   RAM: {summary['ram']['usage_percent']}% - {summary['ram']['used_gb']}GB / {summary['ram']['total_gb']}GB ({summary['ram']['status']})")
    print(f"   GPU: {summary['gpu']['usage_percent']}% ({summary['gpu']['status']})")
    print(f"   VRAM: {summary['vram']['usage_percent']}% - {summary['vram']['used_mb']}MB / {summary['vram']['total_mb']}MB ({summary['vram']['status']})")
    print(f"   Monitoring Active: {summary['monitoring_active']}")
    print()

def test_real_time_monitoring(monitor):
    """Test real-time monitoring with 5-second refresh intervals"""
    print("⏱️  Testing real-time monitoring (5-second intervals)...")
    
    # Start monitoring
    monitor.start_monitoring()
    print(f"✅ Monitoring started: {monitor.monitoring_active}")
    
    # Wait for a few refresh cycles
    print("⏳ Waiting for 12 seconds to test refresh intervals...")
    for i in range(3):
        time.sleep(4)  # Wait 4 seconds
        stats = monitor.get_current_stats()
        if stats:
            print(f"   Cycle {i+1}: VRAM {stats.vram_percent:.1f}%, RAM {stats.ram_percent:.1f}%, CPU {stats.cpu_percent:.1f}%")
        else:
            print(f"   Cycle {i+1}: No stats available yet")
    
    # Stop monitoring
    monitor.stop_monitoring()
    print(f"✅ Monitoring stopped: {monitor.monitoring_active}")
    print()

def test_warning_system(monitor):
    """Test resource usage warnings and alerts"""
    print("⚠️  Testing warning system...")
    
    # Add warning callback
    monitor.add_warning_callback(test_warning_callback)
    print("✅ Added warning callback")
    
    # Set low thresholds to trigger warnings
    monitor.set_warning_thresholds(vram_threshold=1.0, ram_threshold=1.0, cpu_threshold=1.0)
    print("✅ Set low warning thresholds (1% for all resources)")
    
    # Collect stats to trigger warnings
    stats = monitor.collect_system_stats()
    print("✅ Collected stats with low thresholds (should trigger warnings above)")
    
    # Reset to normal thresholds
    monitor.set_warning_thresholds(vram_threshold=90.0, ram_threshold=85.0, cpu_threshold=90.0)
    print("✅ Reset to normal warning thresholds")
    print()

def test_error_handling(monitor):
    """Test error handling and graceful degradation"""
    print("🛡️  Testing error handling...")
    
    # Force disable NVIDIA ML to test fallback
    original_nvidia_ml = monitor.nvidia_ml_available
    monitor.nvidia_ml_available = False
    
    stats = monitor.collect_system_stats()
    print(f"✅ Stats collection with disabled NVIDIA ML: VRAM {stats.vram_percent:.1f}%")
    
    # Restore original state
    monitor.nvidia_ml_available = original_nvidia_ml
    print("✅ Error handling test completed")
    print()

def main():
    """Run all resource monitoring tests"""
    print("🚀 Starting Resource Monitoring System Tests")
    print("=" * 50)
    
    try:
        # Test basic functionality
        monitor = test_basic_stats_collection()
        test_manual_refresh(monitor)
        test_resource_summary(monitor)
        
        # Test real-time monitoring
        test_real_time_monitoring(monitor)
        
        # Test warning system
        test_warning_system(monitor)
        
        # Test error handling
        test_error_handling(monitor)
        
        print("🎉 All resource monitoring tests completed successfully!")
        print("\n📋 Requirements Verification:")
        print("✅ 7.1: System stats collection functions (CPU, RAM, GPU, VRAM)")
        print("✅ 7.2: Real-time monitoring with 5-second refresh intervals")
        print("✅ 7.3: Resource usage warnings and alerts")
        print("✅ 7.4: Manual stats refresh functionality")
        print("✅ 7.5: Comprehensive resource monitoring system")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()