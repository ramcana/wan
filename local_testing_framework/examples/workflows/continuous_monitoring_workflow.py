#!/usr/bin/env python3
"""
Continuous Monitoring Workflow

This script sets up continuous monitoring for the Wan2.2 system during
development or production. It monitors performance, resources, and system
health with configurable alerts and reporting.

Usage:
    python continuous_monitoring_workflow.py [--duration 3600] [--alerts] [--profile dev|prod]
"""

import sys
import subprocess
import argparse
import json
import time
import signal
from pathlib import Path
from datetime import datetime, timedelta
from threading import Thread, Event

class ContinuousMonitor:
    def __init__(self, duration=3600, enable_alerts=False, profile="dev"):
        self.duration = duration
        self.enable_alerts = enable_alerts
        self.profile = profile
        self.stop_event = Event()
        self.monitoring_data = []
        self.alert_count = 0
        
        # Profile-specific settings
        self.profiles = {
            "dev": {
                "interval": 10,
                "alert_thresholds": {
                    "cpu_percent": 80,
                    "memory_percent": 85,
                    "vram_percent": 90,
                    "gpu_temp": 80
                },
                "report_interval": 300,  # 5 minutes
                "cleanup_interval": 600   # 10 minutes
            },
            "prod": {
                "interval": 5,
                "alert_thresholds": {
                    "cpu_percent": 70,
                    "memory_percent": 80,
                    "vram_percent": 85,
                    "gpu_temp": 75
                },
                "report_interval": 60,   # 1 minute
                "cleanup_interval": 300  # 5 minutes
            }
        }
        
        self.config = self.profiles.get(profile, self.profiles["dev"])
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}. Shutting down gracefully...")
            self.stop_event.set()
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def run_command(self, command, capture_output=True):
        """Run a command and return result"""
        try:
            result = subprocess.run(command, shell=True, capture_output=capture_output, text=True)
            if result.returncode == 0:
                return True, result.stdout if capture_output else None
            else:
                return False, result.stderr if capture_output else None
        except Exception as e:
            return False, str(e)

    def collect_metrics(self):
        """Collect system metrics"""
        success, output = self.run_command(
            "python -m local_testing_framework diagnose --system --json"
        )
        
        if success:
            try:
                metrics = json.loads(output)
                metrics["timestamp"] = datetime.now().isoformat()
                return metrics
            except json.JSONDecodeError:
                return None
        return None

    def check_alerts(self, metrics):
        """Check metrics against alert thresholds"""
        if not self.enable_alerts or not metrics:
            return []
            
        alerts = []
        thresholds = self.config["alert_thresholds"]
        
        # Check CPU usage
        if metrics.get("cpu_percent", 0) > thresholds["cpu_percent"]:
            alerts.append({
                "type": "CPU_HIGH",
                "value": metrics["cpu_percent"],
                "threshold": thresholds["cpu_percent"],
                "message": f"CPU usage {metrics['cpu_percent']}% exceeds threshold {thresholds['cpu_percent']}%"
            })
            
        # Check memory usage
        if metrics.get("memory_percent", 0) > thresholds["memory_percent"]:
            alerts.append({
                "type": "MEMORY_HIGH",
                "value": metrics["memory_percent"],
                "threshold": thresholds["memory_percent"],
                "message": f"Memory usage {metrics['memory_percent']}% exceeds threshold {thresholds['memory_percent']}%"
            })
            
        # Check VRAM usage
        if metrics.get("vram_percent", 0) > thresholds["vram_percent"]:
            alerts.append({
                "type": "VRAM_HIGH",
                "value": metrics["vram_percent"],
                "threshold": thresholds["vram_percent"],
                "message": f"VRAM usage {metrics['vram_percent']}% exceeds threshold {thresholds['vram_percent']}%"
            })
            
        return alerts

    def handle_alerts(self, alerts):
        """Handle triggered alerts"""
        for alert in alerts:
            self.alert_count += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            print(f"\n🚨 ALERT [{timestamp}] {alert['type']}: {alert['message']}")
            
            # Log alert to file
            alert_log = {
                "timestamp": datetime.now().isoformat(),
                "alert": alert
            }
            
            with open("monitoring_alerts.log", "a") as f:
                f.write(json.dumps(alert_log) + "\n")
                
            # Suggest remediation
            self.suggest_remediation(alert)

    def suggest_remediation(self, alert):
        """Suggest remediation actions for alerts"""
        remediation = {
            "CPU_HIGH": [
                "Check for runaway processes",
                "Consider reducing concurrent operations",
                "Enable CPU offload in configuration"
            ],
            "MEMORY_HIGH": [
                "Clear system cache",
                "Enable memory optimizations",
                "Restart application if necessary"
            ],
            "VRAM_HIGH": [
                "Enable attention slicing",
                "Reduce VAE tile size",
                "Clear GPU cache: torch.cuda.empty_cache()"
            ]
        }
        
        suggestions = remediation.get(alert["type"], ["Check system resources"])
        print("💡 Suggested actions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")

    def cleanup_resources(self):
        """Perform periodic resource cleanup"""
        print("🧹 Performing resource cleanup...")
        
        # Clear GPU cache
        cleanup_success, _ = self.run_command(
            "python -c 'import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None'"
        )
        
        if cleanup_success:
            print("✅ GPU cache cleared")
        else:
            print("⚠️  Failed to clear GPU cache")
            
        # Clean temporary files
        temp_cleanup_success, _ = self.run_command(
            "python -m local_testing_framework cleanup --temp-files"
        )
        
        if temp_cleanup_success:
            print("✅ Temporary files cleaned")

    def generate_periodic_report(self):
        """Generate periodic monitoring report"""
        if not self.monitoring_data:
            return
            
        report_data = {
            "monitoring_session": {
                "start_time": self.monitoring_data[0]["timestamp"],
                "current_time": datetime.now().isoformat(),
                "profile": self.profile,
                "alert_count": self.alert_count,
                "data_points": len(self.monitoring_data)
            },
            "recent_metrics": self.monitoring_data[-10:],  # Last 10 data points
            "summary": self.calculate_summary()
        }
        
        # Save report
        report_filename = f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, "w") as f:
            json.dump(report_data, f, indent=2)
            
        print(f"📊 Periodic report saved: {report_filename}")

    def calculate_summary(self):
        """Calculate summary statistics from monitoring data"""
        if not self.monitoring_data:
            return {}
            
        cpu_values = [d.get("cpu_percent", 0) for d in self.monitoring_data]
        memory_values = [d.get("memory_percent", 0) for d in self.monitoring_data]
        vram_values = [d.get("vram_percent", 0) for d in self.monitoring_data if d.get("vram_percent")]
        
        summary = {
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "max": max(cpu_values) if cpu_values else 0,
                "min": min(cpu_values) if cpu_values else 0
            },
            "memory": {
                "avg": sum(memory_values) / len(memory_values) if memory_values else 0,
                "max": max(memory_values) if memory_values else 0,
                "min": min(memory_values) if memory_values else 0
            }
        }
        
        if vram_values:
            summary["vram"] = {
                "avg": sum(vram_values) / len(vram_values),
                "max": max(vram_values),
                "min": min(vram_values)
            }
            
        return summary

    def print_status(self, metrics):
        """Print current system status"""
        if not metrics:
            print("❌ Failed to collect metrics")
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        cpu = metrics.get("cpu_percent", 0)
        memory = metrics.get("memory_percent", 0)
        vram = metrics.get("vram_percent", 0)
        
        # Color coding based on thresholds
        def get_color(value, threshold):
            if value > threshold:
                return "🔴"
            elif value > threshold * 0.8:
                return "🟡"
            else:
                return "🟢"
                
        thresholds = self.config["alert_thresholds"]
        cpu_color = get_color(cpu, thresholds["cpu_percent"])
        memory_color = get_color(memory, thresholds["memory_percent"])
        vram_color = get_color(vram, thresholds["vram_percent"])
        
        print(f"[{timestamp}] {cpu_color} CPU: {cpu:5.1f}% | {memory_color} Memory: {memory:5.1f}% | {vram_color} VRAM: {vram:5.1f}%")

    def run_monitoring(self):
        """Main monitoring loop"""
        print("🚀 STARTING CONTINUOUS MONITORING")
        print(f"Profile: {self.profile}")
        print(f"Duration: {self.duration} seconds")
        print(f"Interval: {self.config['interval']} seconds")
        print(f"Alerts: {'Enabled' if self.enable_alerts else 'Disabled'}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        start_time = time.time()
        last_report_time = start_time
        last_cleanup_time = start_time
        
        try:
            while not self.stop_event.is_set():
                current_time = time.time()
                
                # Check if duration exceeded
                if current_time - start_time >= self.duration:
                    print(f"\n⏰ Monitoring duration ({self.duration}s) completed")
                    break
                
                # Collect metrics
                metrics = self.collect_metrics()
                if metrics:
                    self.monitoring_data.append(metrics)
                    self.print_status(metrics)
                    
                    # Check for alerts
                    if self.enable_alerts:
                        alerts = self.check_alerts(metrics)
                        if alerts:
                            self.handle_alerts(alerts)
                
                # Periodic report generation
                if current_time - last_report_time >= self.config["report_interval"]:
                    self.generate_periodic_report()
                    last_report_time = current_time
                
                # Periodic cleanup
                if current_time - last_cleanup_time >= self.config["cleanup_interval"]:
                    self.cleanup_resources()
                    last_cleanup_time = current_time
                
                # Wait for next interval
                self.stop_event.wait(self.config["interval"])
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring interrupted by user")
        except Exception as e:
            print(f"\n💥 Monitoring error: {e}")
        
        # Generate final report
        self.generate_final_report()

    def generate_final_report(self):
        """Generate final monitoring report"""
        print("\n📊 Generating final monitoring report...")
        
        end_time = datetime.now()
        duration = len(self.monitoring_data) * self.config["interval"]
        
        final_report = {
            "monitoring_session": {
                "profile": self.profile,
                "start_time": self.monitoring_data[0]["timestamp"] if self.monitoring_data else None,
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "data_points": len(self.monitoring_data),
                "alert_count": self.alert_count,
                "interval_seconds": self.config["interval"]
            },
            "summary_statistics": self.calculate_summary(),
            "configuration": self.config,
            "all_data": self.monitoring_data
        }
        
        # Save final report
        final_report_filename = f"final_monitoring_report_{end_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(final_report_filename, "w") as f:
            json.dump(final_report, f, indent=2)
        
        print(f"📄 Final report saved: {final_report_filename}")
        
        # Print summary
        print("\n" + "="*60)
        print("📋 MONITORING SESSION SUMMARY")
        print("="*60)
        print(f"Duration: {duration} seconds ({duration//60} minutes)")
        print(f"Data Points: {len(self.monitoring_data)}")
        print(f"Alerts Triggered: {self.alert_count}")
        
        if self.monitoring_data:
            summary = self.calculate_summary()
            print(f"Average CPU: {summary['cpu']['avg']:.1f}%")
            print(f"Average Memory: {summary['memory']['avg']:.1f}%")
            if 'vram' in summary:
                print(f"Average VRAM: {summary['vram']['avg']:.1f}%")
        
        print(f"Final Report: {final_report_filename}")
        print("🎉 Monitoring session completed!")

def main():
    parser = argparse.ArgumentParser(description="Continuous Monitoring Workflow")
    parser.add_argument(
        "--duration", 
        type=int, 
        default=3600, 
        help="Monitoring duration in seconds (default: 3600)"
    )
    parser.add_argument(
        "--alerts", 
        action="store_true", 
        help="Enable alert notifications"
    )
    parser.add_argument(
        "--profile", 
        choices=["dev", "prod"], 
        default="dev",
        help="Monitoring profile (dev or prod)"
    )
    args = parser.parse_args()

    monitor = ContinuousMonitor(args.duration, args.alerts, args.profile)
    monitor.setup_signal_handlers()
    monitor.run_monitoring()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())