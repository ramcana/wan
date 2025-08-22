#!/usr/bin/env python3
"""
Centralized CPU monitoring to avoid conflicts between multiple monitoring systems
"""

import psutil
import threading
import time
from typing import Optional
from datetime import datetime, timedelta

class CPUMonitor:
    """Centralized CPU monitoring to avoid conflicts between multiple psutil calls"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._last_reading = None
        self._last_reading_time = None
        self._reading_interval = 2.0  # Minimum interval between readings
        self._reading_count = 0
        self._warmup_readings = 3  # Ignore first few readings
        
        # Initialize psutil CPU monitoring
        self._initialize_cpu_monitoring()
        
    def get_cpu_percent(self) -> float:
        """Get CPU percentage with proper synchronization"""
        with self._lock:
            current_time = datetime.now()
            
            # If we have a recent reading, return it
            if (self._last_reading is not None and 
                self._last_reading_time is not None and
                (current_time - self._last_reading_time).total_seconds() < self._reading_interval):
                return self._last_reading
            
            # Get a new reading
            try:
                # Use non-blocking call first, then blocking call for accuracy
                psutil.cpu_percent()  # Initialize
                time.sleep(0.1)  # Short delay
                cpu_percent = psutil.cpu_percent()
                
                # Increment reading count
                self._reading_count += 1
                
                # If this is one of the first few readings and it's suspiciously high, 
                # return a reasonable default
                if self._reading_count <= self._warmup_readings and cpu_percent > 50:
                    cpu_percent = 2.0  # Return reasonable default during warmup
                
                # Store the reading
                self._last_reading = cpu_percent
                self._last_reading_time = current_time
                
                return cpu_percent
                
            except Exception as e:
                # Fallback
                return 2.0
    
    def _initialize_cpu_monitoring(self):
        """Initialize CPU monitoring to avoid the first 100% reading"""
        try:
            # Make the first call to initialize psutil's internal state
            psutil.cpu_percent()
        except Exception:
            pass

# Global instance
_cpu_monitor = None
_cpu_monitor_lock = threading.Lock()

def get_cpu_monitor() -> CPUMonitor:
    """Get the global CPU monitor instance"""
    global _cpu_monitor
    
    if _cpu_monitor is None:
        with _cpu_monitor_lock:
            if _cpu_monitor is None:
                _cpu_monitor = CPUMonitor()
    
    return _cpu_monitor

def get_cpu_percent() -> float:
    """Get current CPU percentage using the centralized monitor"""
    monitor = get_cpu_monitor()
    return monitor.get_cpu_percent()