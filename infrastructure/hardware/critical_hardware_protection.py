"""
WAN22 Critical Hardware Protection System

Provides critical hardware protection with safe shutdown capabilities,
user-configurable alert thresholds, and automatic recovery mechanisms
to prevent hardware damage during intensive AI workloads.
"""

import time
import threading
import logging
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
import signal
import sys

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from health_monitor import HealthMonitor, SystemMetrics, HealthAlert, SafetyThresholds


class ProtectionLevel(Enum):
    """Hardware protection levels"""
    NORMAL = "normal"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    EMERGENCY = "emergency"


class ShutdownReason(Enum):
    """Reasons for system shutdown"""
    GPU_OVERHEAT = "gpu_overheat"
    GPU_MEMORY_CRITICAL = "gpu_memory_critical"
    CPU_OVERHEAT = "cpu_overheat"
    SYSTEM_MEMORY_CRITICAL = "system_memory_critical"
    POWER_LIMIT = "power_limit"
    MANUAL_TRIGGER = "manual_trigger"
    MULTIPLE_FAILURES = "multiple_failures"


@dataclass
class CriticalThresholds:
    """Critical hardware protection thresholds"""
    # GPU thresholds
    gpu_temperature_emergency: float = 90.0  # Celsius - immediate shutdown
    gpu_temperature_critical: float = 85.0   # Celsius - workload reduction
    vram_usage_emergency: float = 98.0       # Percent - immediate shutdown
    vram_usage_critical: float = 95.0        # Percent - workload reduction
    
    # CPU thresholds
    cpu_temperature_emergency: float = 85.0  # Celsius - immediate shutdown
    cpu_temperature_critical: float = 80.0   # Celsius - workload reduction
    cpu_usage_emergency: float = 98.0        # Percent - sustained high usage
    cpu_usage_critical: float = 95.0         # Percent - workload reduction
    
    # Memory thresholds
    memory_usage_emergency: float = 98.0     # Percent - immediate shutdown
    memory_usage_critical: float = 95.0      # Percent - workload reduction
    
    # Disk thresholds
    disk_usage_emergency: float = 98.0       # Percent - prevent new operations
    disk_usage_critical: float = 95.0        # Percent - cleanup warning
    
    # Time-based thresholds
    critical_duration_seconds: float = 30.0  # How long critical conditions can persist
    emergency_response_seconds: float = 5.0  # Response time for emergency conditions
    
    # Recovery thresholds (when to consider system recovered)
    recovery_margin_percent: float = 10.0    # How much below critical to consider recovered


@dataclass
class ProtectionAction:
    """Hardware protection action"""
    timestamp: datetime
    reason: ShutdownReason
    severity: str  # 'critical', 'emergency'
    action_taken: str
    metrics_snapshot: Dict[str, Any]
    recovery_time: Optional[datetime] = None
    success: bool = True


class CriticalHardwareProtection:
    """
    Critical hardware protection system with safe shutdown capabilities,
    configurable thresholds, and automatic recovery mechanisms.
    """
    
    def __init__(self,
                 health_monitor: HealthMonitor,
                 thresholds: Optional[CriticalThresholds] = None,
                 protection_level: ProtectionLevel = ProtectionLevel.NORMAL,
                 config_file: Optional[str] = None):
        """
        Initialize critical hardware protection
        
        Args:
            health_monitor: HealthMonitor instance for metrics
            thresholds: Critical protection thresholds
            protection_level: Protection aggressiveness level
            config_file: Path to configuration file
        """
        self.health_monitor = health_monitor
        self.thresholds = thresholds or CriticalThresholds()
        self.protection_level = protection_level
        self.config_file = config_file or "critical_protection_config.json"
        
        # Protection state
        self.is_active = False
        self.protection_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Critical state tracking
        self.critical_conditions: Dict[str, datetime] = {}
        self.emergency_shutdown_triggered = False
        self.workload_reduced = False
        self.system_paused = False
        
        # Action history
        self.protection_actions: List[ProtectionAction] = []
        
        # Callbacks for different protection actions
        self.workload_reduction_callbacks: List[Callable[[str, float], None]] = []
        self.system_pause_callbacks: List[Callable[[str], None]] = []
        self.emergency_shutdown_callbacks: List[Callable[[ShutdownReason, Dict], None]] = []
        self.recovery_callbacks: List[Callable[[str], None]] = []
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Load configuration if exists
        self._load_configuration()
        
        # Adjust thresholds based on protection level
        self._adjust_thresholds_for_protection_level()
        
        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()
        
    def _load_configuration(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    
                # Update thresholds from config
                if 'thresholds' in config:
                    threshold_data = config['thresholds']
                    for key, value in threshold_data.items():
                        if hasattr(self.thresholds, key):
                            setattr(self.thresholds, key, value)
                            
                # Update protection level
                if 'protection_level' in config:
                    try:
                        self.protection_level = ProtectionLevel(config['protection_level'])
                    except ValueError:
                        self.logger.warning(f"Invalid protection level in config: {config['protection_level']}")
                        
                self.logger.info(f"Configuration loaded from {self.config_file}")
                
            except Exception as e:
                self.logger.error(f"Failed to load configuration: {e}")
                
    def save_configuration(self):
        """Save current configuration to file"""
        try:
            config = {
                'thresholds': asdict(self.thresholds),
                'protection_level': self.protection_level.value,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
            self.logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            
    def _adjust_thresholds_for_protection_level(self):
        """Adjust thresholds based on protection level"""
        if self.protection_level == ProtectionLevel.CONSERVATIVE:
            # Lower thresholds for more conservative protection
            self.thresholds.gpu_temperature_emergency -= 5.0
            self.thresholds.gpu_temperature_critical -= 5.0
            self.thresholds.vram_usage_emergency -= 3.0
            self.thresholds.vram_usage_critical -= 5.0
            self.thresholds.cpu_temperature_emergency -= 5.0
            self.thresholds.cpu_temperature_critical -= 5.0
            self.thresholds.critical_duration_seconds = 15.0
            
        elif self.protection_level == ProtectionLevel.AGGRESSIVE:
            # Higher thresholds for less aggressive protection
            self.thresholds.gpu_temperature_emergency += 3.0
            self.thresholds.gpu_temperature_critical += 3.0
            self.thresholds.vram_usage_emergency += 1.0
            self.thresholds.vram_usage_critical += 2.0
            self.thresholds.cpu_temperature_emergency += 3.0
            self.thresholds.cpu_temperature_critical += 3.0
            self.thresholds.critical_duration_seconds = 60.0
            
        elif self.protection_level == ProtectionLevel.EMERGENCY:
            # Very conservative thresholds for emergency mode
            self.thresholds.gpu_temperature_emergency -= 10.0
            self.thresholds.gpu_temperature_critical -= 10.0
            self.thresholds.vram_usage_emergency -= 5.0
            self.thresholds.vram_usage_critical -= 10.0
            self.thresholds.cpu_temperature_emergency -= 10.0
            self.thresholds.cpu_temperature_critical -= 10.0
            self.thresholds.critical_duration_seconds = 5.0
            
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating safe shutdown")
            self.trigger_emergency_shutdown(ShutdownReason.MANUAL_TRIGGER)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def start_protection(self):
        """Start critical hardware protection monitoring"""
        if self.is_active:
            self.logger.warning("Critical protection already active")
            return
            
        self.is_active = True
        self.stop_event.clear()
        self.emergency_shutdown_triggered = False
        
        self.protection_thread = threading.Thread(target=self._protection_loop, daemon=True)
        self.protection_thread.start()
        
        self.logger.info(f"Critical hardware protection started (level: {self.protection_level.value})")
        
    def stop_protection(self):
        """Stop critical hardware protection monitoring"""
        if not self.is_active:
            return
            
        self.is_active = False
        self.stop_event.set()
        
        if self.protection_thread and self.protection_thread.is_alive():
            self.protection_thread.join(timeout=10.0)
            
        self.logger.info("Critical hardware protection stopped")
        
    def _protection_loop(self):
        """Main protection monitoring loop"""
        while not self.stop_event.wait(1.0):  # Check every second for critical conditions
            try:
                current_metrics = self.health_monitor.get_current_metrics()
                if current_metrics:
                    self._check_critical_conditions(current_metrics)
                    self._check_recovery_conditions(current_metrics)
                    
            except Exception as e:
                self.logger.error(f"Error in protection loop: {e}")
                
    def _check_critical_conditions(self, metrics: SystemMetrics):
        """Check for critical hardware conditions"""
        current_time = datetime.now()
        emergency_actions = []
        critical_actions = []
        
        # Check GPU temperature
        if metrics.gpu_temperature > 0:  # Only check if we have valid GPU data
            if metrics.gpu_temperature >= self.thresholds.gpu_temperature_emergency:
                emergency_actions.append(('gpu_temperature', metrics.gpu_temperature, 
                                        self.thresholds.gpu_temperature_emergency))
            elif metrics.gpu_temperature >= self.thresholds.gpu_temperature_critical:
                critical_actions.append(('gpu_temperature', metrics.gpu_temperature,
                                       self.thresholds.gpu_temperature_critical))
                
        # Check VRAM usage
        if metrics.vram_usage_percent >= self.thresholds.vram_usage_emergency:
            emergency_actions.append(('vram_usage', metrics.vram_usage_percent,
                                    self.thresholds.vram_usage_emergency))
        elif metrics.vram_usage_percent >= self.thresholds.vram_usage_critical:
            critical_actions.append(('vram_usage', metrics.vram_usage_percent,
                                   self.thresholds.vram_usage_critical))
            
        # Check CPU usage (sustained high usage)
        if metrics.cpu_usage_percent >= self.thresholds.cpu_usage_emergency:
            emergency_actions.append(('cpu_usage', metrics.cpu_usage_percent,
                                    self.thresholds.cpu_usage_emergency))
        elif metrics.cpu_usage_percent >= self.thresholds.cpu_usage_critical:
            critical_actions.append(('cpu_usage', metrics.cpu_usage_percent,
                                   self.thresholds.cpu_usage_critical))
            
        # Check memory usage
        if metrics.memory_usage_percent >= self.thresholds.memory_usage_emergency:
            emergency_actions.append(('memory_usage', metrics.memory_usage_percent,
                                    self.thresholds.memory_usage_emergency))
        elif metrics.memory_usage_percent >= self.thresholds.memory_usage_critical:
            critical_actions.append(('memory_usage', metrics.memory_usage_percent,
                                   self.thresholds.memory_usage_critical))
            
        # Check disk usage
        if metrics.disk_usage_percent >= self.thresholds.disk_usage_emergency:
            emergency_actions.append(('disk_usage', metrics.disk_usage_percent,
                                    self.thresholds.disk_usage_emergency))
        elif metrics.disk_usage_percent >= self.thresholds.disk_usage_critical:
            critical_actions.append(('disk_usage', metrics.disk_usage_percent,
                                   self.thresholds.disk_usage_critical))
            
        # Handle emergency conditions (immediate action)
        if emergency_actions:
            self._handle_emergency_conditions(emergency_actions, metrics)
            
        # Handle critical conditions (time-based action)
        if critical_actions:
            self._handle_critical_conditions(critical_actions, metrics, current_time)
            
    def _handle_emergency_conditions(self, emergency_actions: List[Tuple], metrics: SystemMetrics):
        """Handle emergency conditions requiring immediate action"""
        if self.emergency_shutdown_triggered:
            return  # Already handling emergency
            
        self.logger.critical(f"EMERGENCY CONDITIONS DETECTED: {emergency_actions}")
        
        # Determine primary reason for shutdown
        reason_map = {
            'gpu_temperature': ShutdownReason.GPU_OVERHEAT,
            'vram_usage': ShutdownReason.GPU_MEMORY_CRITICAL,
            'cpu_usage': ShutdownReason.CPU_OVERHEAT,
            'memory_usage': ShutdownReason.SYSTEM_MEMORY_CRITICAL,
            'disk_usage': ShutdownReason.POWER_LIMIT  # Treat as resource limit
        }
        
        primary_reason = ShutdownReason.MULTIPLE_FAILURES
        if len(emergency_actions) == 1:
            condition_type = emergency_actions[0][0]
            primary_reason = reason_map.get(condition_type, ShutdownReason.MULTIPLE_FAILURES)
            
        # Trigger emergency shutdown
        self.trigger_emergency_shutdown(primary_reason, metrics.to_dict())
        
    def _handle_critical_conditions(self, critical_actions: List[Tuple], 
                                  metrics: SystemMetrics, current_time: datetime):
        """Handle critical conditions with time-based escalation"""
        for condition_type, value, threshold in critical_actions:
            if condition_type not in self.critical_conditions:
                # First time seeing this critical condition
                self.critical_conditions[condition_type] = current_time
                self.logger.warning(f"Critical condition started: {condition_type} = {value} (threshold: {threshold})")
                
                # Immediate workload reduction
                if not self.workload_reduced:
                    self._trigger_workload_reduction(condition_type, value)
                    
            else:
                # Check if critical condition has persisted too long
                duration = (current_time - self.critical_conditions[condition_type]).total_seconds()
                if duration >= self.thresholds.critical_duration_seconds:
                    self.logger.critical(f"Critical condition persisted for {duration:.1f}s: {condition_type}")
                    
                    # Escalate to emergency
                    self._handle_emergency_conditions([(condition_type, value, threshold)], metrics)
                    
    def _check_recovery_conditions(self, metrics: SystemMetrics):
        """Check if system has recovered from critical conditions"""
        current_time = datetime.now()
        recovered_conditions = []
        
        for condition_type, start_time in list(self.critical_conditions.items()):
            current_value = self._get_metric_value(metrics, condition_type)
            threshold = self._get_critical_threshold(condition_type)
            recovery_threshold = threshold - self.thresholds.recovery_margin_percent
            
            if current_value < recovery_threshold:
                recovered_conditions.append(condition_type)
                self.logger.info(f"Recovered from critical condition: {condition_type} = {current_value}")
                
        # Remove recovered conditions
        for condition_type in recovered_conditions:
            del self.critical_conditions[condition_type]
            
        # Check for full system recovery
        if not self.critical_conditions and (self.workload_reduced or self.system_paused):
            self._trigger_system_recovery()
            
    def _get_metric_value(self, metrics: SystemMetrics, condition_type: str) -> float:
        """Get metric value by condition type"""
        value_map = {
            'gpu_temperature': metrics.gpu_temperature,
            'vram_usage': metrics.vram_usage_percent,
            'cpu_usage': metrics.cpu_usage_percent,
            'memory_usage': metrics.memory_usage_percent,
            'disk_usage': metrics.disk_usage_percent
        }
        return value_map.get(condition_type, 0.0)
        
    def _get_critical_threshold(self, condition_type: str) -> float:
        """Get critical threshold by condition type"""
        threshold_map = {
            'gpu_temperature': self.thresholds.gpu_temperature_critical,
            'vram_usage': self.thresholds.vram_usage_critical,
            'cpu_usage': self.thresholds.cpu_usage_critical,
            'memory_usage': self.thresholds.memory_usage_critical,
            'disk_usage': self.thresholds.disk_usage_critical
        }
        return threshold_map.get(condition_type, 100.0)
        
    def _trigger_workload_reduction(self, reason: str, value: float):
        """Trigger workload reduction to prevent hardware damage"""
        if self.workload_reduced:
            return  # Already reduced
            
        self.workload_reduced = True
        action = ProtectionAction(
            timestamp=datetime.now(),
            reason=ShutdownReason.GPU_OVERHEAT if 'gpu' in reason else ShutdownReason.CPU_OVERHEAT,
            severity='critical',
            action_taken=f'workload_reduction_{reason}',
            metrics_snapshot={'condition': reason, 'value': value}
        )
        self.protection_actions.append(action)
        
        self.logger.warning(f"WORKLOAD REDUCTION TRIGGERED: {reason} = {value}")
        
        # Notify callbacks
        for callback in self.workload_reduction_callbacks:
            try:
                callback(reason, value)
            except Exception as e:
                self.logger.error(f"Error in workload reduction callback: {e}")
                
    def _trigger_system_recovery(self):
        """Trigger system recovery after critical conditions resolve"""
        self.logger.info("System recovery initiated - critical conditions resolved")
        
        # Reset protection state
        self.workload_reduced = False
        self.system_paused = False
        
        # Record recovery action
        action = ProtectionAction(
            timestamp=datetime.now(),
            reason=ShutdownReason.MANUAL_TRIGGER,  # Recovery action
            severity='info',
            action_taken='system_recovery',
            metrics_snapshot={'recovered_conditions': list(self.critical_conditions.keys())},
            recovery_time=datetime.now()
        )
        self.protection_actions.append(action)
        
        # Notify callbacks
        for callback in self.recovery_callbacks:
            try:
                callback("system_recovered")
            except Exception as e:
                self.logger.error(f"Error in recovery callback: {e}")
                
    def trigger_emergency_shutdown(self, reason: ShutdownReason, 
                                 metrics_snapshot: Optional[Dict] = None):
        """Trigger emergency shutdown to protect hardware"""
        if self.emergency_shutdown_triggered:
            return  # Already triggered
            
        self.emergency_shutdown_triggered = True
        
        # Record emergency action
        action = ProtectionAction(
            timestamp=datetime.now(),
            reason=reason,
            severity='emergency',
            action_taken='emergency_shutdown',
            metrics_snapshot=metrics_snapshot or {}
        )
        self.protection_actions.append(action)
        
        self.logger.critical(f"EMERGENCY SHUTDOWN TRIGGERED: {reason.value}")
        
        # Notify emergency callbacks
        for callback in self.emergency_shutdown_callbacks:
            try:
                callback(reason, metrics_snapshot or {})
            except Exception as e:
                self.logger.error(f"Error in emergency shutdown callback: {e}")
                
        # Perform emergency actions
        self._perform_emergency_shutdown(reason)
        
    def _perform_emergency_shutdown(self, reason: ShutdownReason):
        """Perform actual emergency shutdown procedures"""
        try:
            # 1. Stop health monitoring to prevent further alerts
            if self.health_monitor.is_monitoring:
                self.health_monitor.stop_monitoring()
                
            # 2. Clear GPU memory if possible
            self._emergency_gpu_cleanup()
            
            # 3. Save protection state
            self._save_emergency_state(reason)
            
            # 4. Pause system operations
            self.system_paused = True
            
            # 5. Log final state
            self.logger.critical(f"Emergency shutdown completed for reason: {reason.value}")
            
            # Note: We don't actually shut down the system here as that would be too aggressive
            # Instead, we pause operations and wait for manual intervention or recovery
            
        except Exception as e:
            self.logger.critical(f"Error during emergency shutdown: {e}")
            
    def _emergency_gpu_cleanup(self):
        """Emergency GPU memory cleanup"""
        try:
            # Try to clear GPU cache if torch is available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    self.logger.info("GPU cache cleared during emergency shutdown")
            except ImportError:
                pass
                
        except Exception as e:
            self.logger.error(f"Failed to clear GPU cache: {e}")
            
    def _save_emergency_state(self, reason: ShutdownReason):
        """Save emergency state for recovery"""
        try:
            emergency_state = {
                'timestamp': datetime.now().isoformat(),
                'reason': reason.value,
                'protection_level': self.protection_level.value,
                'thresholds': asdict(self.thresholds),
                'critical_conditions': {
                    k: v.isoformat() for k, v in self.critical_conditions.items()
                },
                'actions_taken': [
                    {
                        'timestamp': action.timestamp.isoformat(),
                        'reason': action.reason.value,
                        'severity': action.severity,
                        'action_taken': action.action_taken,
                        'metrics_snapshot': action.metrics_snapshot
                    }
                    for action in self.protection_actions[-10:]  # Last 10 actions
                ]
            }
            
            emergency_file = f"emergency_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(emergency_file, 'w') as f:
                json.dump(emergency_state, f, indent=2)
                
            self.logger.info(f"Emergency state saved to {emergency_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save emergency state: {e}")
            
    def update_thresholds(self, new_thresholds: CriticalThresholds):
        """Update critical protection thresholds"""
        self.thresholds = new_thresholds
        self._adjust_thresholds_for_protection_level()
        self.save_configuration()
        self.logger.info("Critical protection thresholds updated")
        
    def set_protection_level(self, level: ProtectionLevel):
        """Set protection level and adjust thresholds accordingly"""
        self.protection_level = level
        self._adjust_thresholds_for_protection_level()
        self.save_configuration()
        self.logger.info(f"Protection level set to {level.value}")
        
    def add_workload_reduction_callback(self, callback: Callable[[str, float], None]):
        """Add callback for workload reduction events"""
        self.workload_reduction_callbacks.append(callback)
        
    def add_system_pause_callback(self, callback: Callable[[str], None]):
        """Add callback for system pause events"""
        self.system_pause_callbacks.append(callback)
        
    def add_emergency_shutdown_callback(self, callback: Callable[[ShutdownReason, Dict], None]):
        """Add callback for emergency shutdown events"""
        self.emergency_shutdown_callbacks.append(callback)
        
    def add_recovery_callback(self, callback: Callable[[str], None]):
        """Add callback for system recovery events"""
        self.recovery_callbacks.append(callback)
        
    def get_protection_status(self) -> Dict[str, Any]:
        """Get current protection system status"""
        return {
            'is_active': self.is_active,
            'protection_level': self.protection_level.value,
            'emergency_shutdown_triggered': self.emergency_shutdown_triggered,
            'workload_reduced': self.workload_reduced,
            'system_paused': self.system_paused,
            'critical_conditions': {
                k: v.isoformat() for k, v in self.critical_conditions.items()
            },
            'thresholds': asdict(self.thresholds),
            'recent_actions': [
                {
                    'timestamp': action.timestamp.isoformat(),
                    'reason': action.reason.value,
                    'severity': action.severity,
                    'action_taken': action.action_taken,
                    'success': action.success
                }
                for action in self.protection_actions[-5:]  # Last 5 actions
            ]
        }
        
    def reset_protection_state(self):
        """Reset protection state (use with caution)"""
        if self.is_active:
            self.logger.warning("Cannot reset protection state while active")
            return False
            
        self.emergency_shutdown_triggered = False
        self.workload_reduced = False
        self.system_paused = False
        self.critical_conditions.clear()
        
        self.logger.info("Protection state reset")
        return True
        
    def get_action_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get protection action history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            {
                'timestamp': action.timestamp.isoformat(),
                'reason': action.reason.value,
                'severity': action.severity,
                'action_taken': action.action_taken,
                'metrics_snapshot': action.metrics_snapshot,
                'recovery_time': action.recovery_time.isoformat() if action.recovery_time else None,
                'success': action.success
            }
            for action in self.protection_actions
            if action.timestamp >= cutoff_time
        ]
        
    def __enter__(self):
        """Context manager entry"""
        self.start_protection()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_protection()


# Demo and utility functions
def create_demo_protection_system(health_monitor: Optional[HealthMonitor] = None) -> CriticalHardwareProtection:
    """Create a demo protection system for testing"""
    if health_monitor is None:
        from health_monitor import create_demo_health_monitor
        health_monitor = create_demo_health_monitor()
        
    # Create conservative thresholds for demo
    thresholds = CriticalThresholds(
        gpu_temperature_emergency=80.0,
        gpu_temperature_critical=75.0,
        vram_usage_emergency=95.0,
        vram_usage_critical=90.0,
        cpu_usage_emergency=95.0,
        cpu_usage_critical=90.0,
        critical_duration_seconds=10.0  # Shorter for demo
    )
    
    protection = CriticalHardwareProtection(
        health_monitor=health_monitor,
        thresholds=thresholds,
        protection_level=ProtectionLevel.CONSERVATIVE
    )
    
    # Add demo callbacks
    def workload_reduction_handler(reason: str, value: float):
        print(f"🔥 WORKLOAD REDUCTION: {reason} = {value}")
        
    def emergency_shutdown_handler(reason: ShutdownReason, metrics: Dict):
        print(f"🚨 EMERGENCY SHUTDOWN: {reason.value}")
        print(f"   Metrics: {metrics}")
        
    def recovery_handler(message: str):
        print(f"✅ RECOVERY: {message}")
        
    protection.add_workload_reduction_callback(workload_reduction_handler)
    protection.add_emergency_shutdown_callback(emergency_shutdown_handler)
    protection.add_recovery_callback(recovery_handler)
    
    return protection


def run_protection_demo():
    """Run a demo of the critical hardware protection system"""
    print("WAN22 Critical Hardware Protection Demo")
    print("=" * 50)
    
    # Create demo components
    from health_monitor import create_demo_health_monitor
    
    monitor = create_demo_health_monitor()
    protection = create_demo_protection_system(monitor)
    
    try:
        # Start systems
        monitor.start_monitoring()
        protection.start_protection()
        
        print("Protection system started...")
        print("Monitoring for critical conditions...")
        
        # Monitor for a short time
        for i in range(10):
            time.sleep(2)
            
            status = protection.get_protection_status()
            print(f"\nStatus check {i+1}:")
            print(f"  Active: {status['is_active']}")
            print(f"  Level: {status['protection_level']}")
            print(f"  Critical conditions: {len(status['critical_conditions'])}")
            print(f"  Workload reduced: {status['workload_reduced']}")
            
            if status['recent_actions']:
                print(f"  Recent actions: {len(status['recent_actions'])}")
                
        # Test manual emergency trigger
        print("\nTesting manual emergency trigger...")
        protection.trigger_emergency_shutdown(ShutdownReason.MANUAL_TRIGGER)
        
        time.sleep(2)
        
        final_status = protection.get_protection_status()
        print(f"\nFinal status:")
        print(f"  Emergency triggered: {final_status['emergency_shutdown_triggered']}")
        print(f"  System paused: {final_status['system_paused']}")
        
    except KeyboardInterrupt:
        print("\nStopping protection demo...")
    finally:
        protection.stop_protection()
        monitor.stop_monitoring()
        
    print("Demo completed!")


if __name__ == "__main__":
    run_protection_demo()