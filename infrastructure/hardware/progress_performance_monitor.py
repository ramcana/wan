#!/usr/bin/env python3
"""
Progress Performance Monitor for WAN22
Monitors and optimizes progress tracking system performance
"""

import time
import threading
import psutil
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
import statistics

@dataclass
class ProgressMetrics:
    """Metrics for progress tracking performance"""
    timestamp: datetime
    update_latency_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    active_trackers: int
    updates_per_second: float
    queue_size: int

@dataclass
class PerformanceAlert:
    """Performance alert for progress tracking issues"""
    timestamp: datetime
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    metrics: ProgressMetrics
    suggested_action: str

class ProgressPerformanceMonitor:
    """Monitors progress tracking system performance"""
    
    def __init__(self, alert_thresholds: Dict[str, float] = None):
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'update_latency_ms': 100.0,  # Alert if updates take > 100ms
            'memory_usage_mb': 500.0,    # Alert if memory > 500MB
            'cpu_usage_percent': 80.0,   # Alert if CPU > 80%
            'updates_per_second': 0.5,   # Alert if < 0.5 updates/sec
            'queue_size': 100            # Alert if queue > 100 items
        }
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread = None
        self._metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 metrics
        self._alerts: List[PerformanceAlert] = []
        self._active_trackers = 0
        self._update_times: deque = deque(maxlen=100)  # Track last 100 update times
        self._last_update_count = 0
        self._update_count = 0
        
        # Performance optimization settings
        self._optimization_enabled = True
        self._adaptive_update_interval = 1.0  # Start with 1 second
        self._min_update_interval = 0.1
        self._max_update_interval = 5.0
        
        # Callbacks for performance events
        self._alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
    def start_monitoring(self, interval: float = 1.0):
        """Start performance monitoring"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,), 
            daemon=True
        )
        self._monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            
    def register_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Register callback for performance alerts"""
        self._alert_callbacks.append(callback)
        
    def track_update_start(self) -> str:
        """Track the start of a progress update operation"""
        update_id = f"update_{time.time()}_{threading.current_thread().ident}"
        self._update_times.append((update_id, time.time()))
        return update_id
        
    def track_update_end(self, update_id: str):
        """Track the end of a progress update operation"""
        end_time = time.time()
        
        # Find matching start time
        for i, entry in enumerate(self._update_times):
            if len(entry) >= 2 and entry[0] == update_id:
                stored_id, start_time = entry[0], entry[1]
                latency_ms = (end_time - start_time) * 1000
                self._update_times[i] = (stored_id, start_time, latency_ms)
                self._update_count += 1
                break
                
    def register_tracker(self):
        """Register a new active progress tracker"""
        self._active_trackers += 1
        
    def unregister_tracker(self):
        """Unregister an active progress tracker"""
        self._active_trackers = max(0, self._active_trackers - 1)
        
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                self._metrics_history.append(metrics)
                
                # Check for performance issues
                alerts = self._check_performance_alerts(metrics)
                for alert in alerts:
                    self._handle_alert(alert)
                    
                # Adaptive interval adjustment
                if self._optimization_enabled:
                    interval = self._adjust_monitoring_interval(metrics)
                    
            except Exception as e:
                print(f"Error in progress performance monitoring: {e}")
                
            time.sleep(interval)
            
    def _collect_metrics(self) -> ProgressMetrics:
        """Collect current performance metrics"""
        current_time = datetime.now()
        
        # Calculate average update latency
        recent_updates = [
            entry[2] for entry in self._update_times 
            if len(entry) == 3  # Only completed updates
        ]
        avg_latency = statistics.mean(recent_updates) if recent_updates else 0.0
        
        # Calculate updates per second
        time_window = 10.0  # 10 second window
        recent_time = current_time - timedelta(seconds=time_window)
        recent_update_count = 0
        
        for entry in self._update_times:
            if len(entry) >= 2:
                try:
                    # Extract timestamp from update_id format: "update_{timestamp}_{thread_id}"
                    timestamp_str = entry[0].split('_')[1]
                    timestamp = float(timestamp_str)
                    if datetime.fromtimestamp(timestamp) > recent_time:
                        recent_update_count += 1
                except (IndexError, ValueError):
                    # Skip entries with invalid format
                    continue
                    
        updates_per_second = recent_update_count / time_window
        
        # Get system metrics
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        cpu_percent = process.cpu_percent()
        
        # Estimate queue size (simplified)
        queue_size = max(0, self._active_trackers - 1)
        
        return ProgressMetrics(
            timestamp=current_time,
            update_latency_ms=avg_latency,
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            active_trackers=self._active_trackers,
            updates_per_second=updates_per_second,
            queue_size=queue_size
        )
        
    def _check_performance_alerts(self, metrics: ProgressMetrics) -> List[PerformanceAlert]:
        """Check for performance issues and generate alerts"""
        alerts = []
        
        # Check update latency
        if metrics.update_latency_ms > self.alert_thresholds['update_latency_ms']:
            severity = 'high' if metrics.update_latency_ms > 200 else 'medium'
            alerts.append(PerformanceAlert(
                timestamp=metrics.timestamp,
                alert_type='high_latency',
                severity=severity,
                message=f"Progress update latency is high: {metrics.update_latency_ms:.1f}ms",
                metrics=metrics,
                suggested_action="Consider reducing update frequency or optimizing update operations"
            ))
            
        # Check memory usage
        if metrics.memory_usage_mb > self.alert_thresholds['memory_usage_mb']:
            severity = 'critical' if metrics.memory_usage_mb > 1000 else 'high'
            alerts.append(PerformanceAlert(
                timestamp=metrics.timestamp,
                alert_type='high_memory',
                severity=severity,
                message=f"High memory usage in progress tracking: {metrics.memory_usage_mb:.1f}MB",
                metrics=metrics,
                suggested_action="Check for memory leaks in progress tracking components"
            ))
            
        # Check CPU usage
        if metrics.cpu_usage_percent > self.alert_thresholds['cpu_usage_percent']:
            alerts.append(PerformanceAlert(
                timestamp=metrics.timestamp,
                alert_type='high_cpu',
                severity='medium',
                message=f"High CPU usage in progress tracking: {metrics.cpu_usage_percent:.1f}%",
                metrics=metrics,
                suggested_action="Consider optimizing progress calculation algorithms"
            ))
            
        # Check update frequency
        if metrics.updates_per_second < self.alert_thresholds['updates_per_second']:
            alerts.append(PerformanceAlert(
                timestamp=metrics.timestamp,
                alert_type='low_update_rate',
                severity='medium',
                message=f"Low progress update rate: {metrics.updates_per_second:.2f} updates/sec",
                metrics=metrics,
                suggested_action="Check for blocked or slow progress update operations"
            ))
            
        # Check queue size
        if metrics.queue_size > self.alert_thresholds['queue_size']:
            severity = 'critical' if metrics.queue_size > 200 else 'high'
            alerts.append(PerformanceAlert(
                timestamp=metrics.timestamp,
                alert_type='large_queue',
                severity=severity,
                message=f"Large progress tracking queue: {metrics.queue_size} items",
                metrics=metrics,
                suggested_action="Increase processing capacity or reduce tracking granularity"
            ))
            
        return alerts
        
    def _handle_alert(self, alert: PerformanceAlert):
        """Handle a performance alert"""
        self._alerts.append(alert)
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self._alerts = [a for a in self._alerts if a.timestamp > cutoff_time]
        
        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Error in alert callback: {e}")
                
        # Auto-optimization for critical alerts
        if alert.severity == 'critical' and self._optimization_enabled:
            self._apply_emergency_optimization(alert)
            
    def _apply_emergency_optimization(self, alert: PerformanceAlert):
        """Apply emergency optimizations for critical alerts"""
        if alert.alert_type == 'high_memory':
            # Reduce tracking granularity
            print("EMERGENCY: Reducing progress tracking granularity due to high memory usage")
            
        elif alert.alert_type == 'large_queue':
            # Increase update interval to reduce queue pressure
            self._adaptive_update_interval = min(self._max_update_interval, 
                                               self._adaptive_update_interval * 2)
            print(f"EMERGENCY: Increased update interval to {self._adaptive_update_interval}s")
            
    def _adjust_monitoring_interval(self, metrics: ProgressMetrics) -> float:
        """Dynamically adjust monitoring interval based on performance"""
        # Increase interval if system is under stress
        if (metrics.cpu_usage_percent > 70 or 
            metrics.memory_usage_mb > 400 or 
            metrics.update_latency_ms > 50):
            self._adaptive_update_interval = min(
                self._max_update_interval,
                self._adaptive_update_interval * 1.1
            )
        else:
            # Decrease interval if system is performing well
            self._adaptive_update_interval = max(
                self._min_update_interval,
                self._adaptive_update_interval * 0.95
            )
            
        return self._adaptive_update_interval
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self._metrics_history:
            return {}
            
        recent_metrics = list(self._metrics_history)[-100:]  # Last 100 measurements
        
        return {
            'monitoring_duration_minutes': len(self._metrics_history) * self._adaptive_update_interval / 60,
            'average_latency_ms': statistics.mean(m.update_latency_ms for m in recent_metrics),
            'max_latency_ms': max(m.update_latency_ms for m in recent_metrics),
            'average_memory_mb': statistics.mean(m.memory_usage_mb for m in recent_metrics),
            'peak_memory_mb': max(m.memory_usage_mb for m in recent_metrics),
            'average_cpu_percent': statistics.mean(m.cpu_usage_percent for m in recent_metrics),
            'peak_cpu_percent': max(m.cpu_usage_percent for m in recent_metrics),
            'average_updates_per_second': statistics.mean(m.updates_per_second for m in recent_metrics),
            'total_alerts': len(self._alerts),
            'critical_alerts': len([a for a in self._alerts if a.severity == 'critical']),
            'active_trackers': self._active_trackers,
            'current_update_interval': self._adaptive_update_interval
        }
        
    def export_metrics(self, filename: str = None) -> str:
        """Export performance metrics to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"progress_performance_metrics_{timestamp}.json"
            
        data = {
            'summary': self.get_performance_summary(),
            'metrics': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'update_latency_ms': m.update_latency_ms,
                    'memory_usage_mb': m.memory_usage_mb,
                    'cpu_usage_percent': m.cpu_usage_percent,
                    'active_trackers': m.active_trackers,
                    'updates_per_second': m.updates_per_second,
                    'queue_size': m.queue_size
                }
                for m in self._metrics_history
            ],
            'alerts': [
                {
                    'timestamp': a.timestamp.isoformat(),
                    'alert_type': a.alert_type,
                    'severity': a.severity,
                    'message': a.message,
                    'suggested_action': a.suggested_action
                }
                for a in self._alerts
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        return filename
        
    def print_performance_report(self):
        """Print comprehensive performance report"""
        summary = self.get_performance_summary()
        
        print("\n" + "="*60)
        print("PROGRESS TRACKING PERFORMANCE REPORT")
        print("="*60)
        
        if not summary:
            print("No performance data available")
            return
            
        print(f"Monitoring Duration: {summary.get('monitoring_duration_minutes', 0):.1f} minutes")
        print(f"Active Trackers: {summary.get('active_trackers', 0)}")
        print(f"Current Update Interval: {summary.get('current_update_interval', 0):.2f}s")
        
        print(f"\nLATENCY METRICS:")
        print(f"  Average: {summary.get('average_latency_ms', 0):.1f}ms")
        print(f"  Peak: {summary.get('max_latency_ms', 0):.1f}ms")
        
        print(f"\nMEMORY METRICS:")
        print(f"  Average: {summary.get('average_memory_mb', 0):.1f}MB")
        print(f"  Peak: {summary.get('peak_memory_mb', 0):.1f}MB")
        
        print(f"\nCPU METRICS:")
        print(f"  Average: {summary.get('average_cpu_percent', 0):.1f}%")
        print(f"  Peak: {summary.get('peak_cpu_percent', 0):.1f}%")
        
        print(f"\nUPDATE RATE:")
        print(f"  Average: {summary.get('average_updates_per_second', 0):.2f} updates/sec")
        
        print(f"\nALERTS:")
        print(f"  Total: {summary.get('total_alerts', 0)}")
        print(f"  Critical: {summary.get('critical_alerts', 0)}")
        
        # Show recent alerts
        recent_alerts = [a for a in self._alerts if 
                        a.timestamp > datetime.now() - timedelta(hours=1)]
        if recent_alerts:
            print(f"\nRECENT ALERTS (Last Hour):")
            for alert in recent_alerts[-5:]:  # Show last 5
                print(f"  [{alert.severity.upper()}] {alert.message}")
                
        print("="*60)

# Global monitor instance
_global_monitor: Optional[ProgressPerformanceMonitor] = None
_monitor_lock = threading.Lock()

def get_global_monitor() -> ProgressPerformanceMonitor:
    """Get global progress performance monitor"""
    global _global_monitor
    
    with _monitor_lock:
        if _global_monitor is None:
            _global_monitor = ProgressPerformanceMonitor()
            # Don't auto-start monitoring to prevent initialization loops
        return _global_monitor

def track_progress_update(func):
    """Decorator to track progress update performance"""
    def wrapper(*args, **kwargs):
        monitor = get_global_monitor()
        update_id = monitor.track_update_start()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            monitor.track_update_end(update_id)
    return wrapper

if __name__ == "__main__":
    # Test the performance monitor
    monitor = ProgressPerformanceMonitor()
    
    def test_alert_callback(alert: PerformanceAlert):
        print(f"ALERT: {alert.message}")
        
    monitor.register_alert_callback(test_alert_callback)
    monitor.start_monitoring(0.5)  # Monitor every 500ms
    
    # Simulate some progress tracking activity
    for i in range(10):
        monitor.register_tracker()
        update_id = monitor.track_update_start()
        time.sleep(0.1)  # Simulate update work
        monitor.track_update_end(update_id)
        
    time.sleep(5)  # Let monitor collect data
    
    monitor.print_performance_report()
    monitor.stop_monitoring()
