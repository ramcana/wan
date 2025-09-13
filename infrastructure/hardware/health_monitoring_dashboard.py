"""
WAN22 Health Monitoring Dashboard

Provides a comprehensive dashboard for system health monitoring with
real-time metrics display, historical trend tracking, and integration
with external monitoring tools.
"""

import json
import subprocess
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    mdates = None
    FuncAnimation = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from health_monitor import HealthMonitor, SystemMetrics, HealthAlert, SafetyThresholds


@dataclass
class DashboardConfig:
    """Configuration for health monitoring dashboard"""
    update_interval: float = 5.0  # seconds
    history_hours: int = 24
    chart_width: int = 12
    chart_height: int = 8
    enable_nvidia_smi: bool = True
    enable_real_time_charts: bool = True
    export_format: str = 'json'  # 'json', 'csv', 'html'
    alert_sound: bool = False


class HealthDashboard:
    """
    Comprehensive health monitoring dashboard with real-time metrics,
    historical trends, and external tool integration.
    """
    
    def __init__(self, 
                 health_monitor: HealthMonitor,
                 config: Optional[DashboardConfig] = None):
        """
        Initialize health dashboard
        
        Args:
            health_monitor: HealthMonitor instance to display data from
            config: Dashboard configuration
        """
        self.health_monitor = health_monitor
        self.config = config or DashboardConfig()
        self.logger = logging.getLogger(__name__)
        
        # Dashboard state
        self.is_running = False
        self.last_update = None
        
        # External tool integration
        self.nvidia_smi_available = self._check_nvidia_smi()
        
        # Chart management
        self.charts = {}
        self.animation = None
        
    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available"""
        try:
            result = subprocess.run(['nvidia-smi', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False
            
    def get_nvidia_smi_data(self) -> Optional[Dict[str, Any]]:
        """Get detailed GPU information from nvidia-smi"""
        if not self.nvidia_smi_available or not self.config.enable_nvidia_smi:
            return None
            
        try:
            # Get GPU information in XML format for easier parsing
            result = subprocess.run([
                'nvidia-smi', '-q', '-x'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return None
                
            # For simplicity, get basic info with CSV format
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return None
                
            lines = result.stdout.strip().split('\n')
            gpus = []
            
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    gpus.append({
                        'gpu_id': i,
                        'name': parts[0],
                        'temperature': float(parts[1]) if parts[1] != '[Not Supported]' else 0,
                        'utilization': float(parts[2]) if parts[2] != '[Not Supported]' else 0,
                        'memory_used': float(parts[3]) if parts[3] != '[Not Supported]' else 0,
                        'memory_total': float(parts[4]) if parts[4] != '[Not Supported]' else 0,
                        'power_draw': float(parts[5]) if parts[5] != '[Not Supported]' else 0
                    })
                    
            return {
                'timestamp': datetime.now().isoformat(),
                'gpu_count': len(gpus),
                'gpus': gpus
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to get nvidia-smi data: {e}")
            return None
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        current_metrics = self.health_monitor.get_current_metrics()
        active_alerts = self.health_monitor.get_active_alerts()
        health_summary = self.health_monitor.get_health_summary()
        
        # Get nvidia-smi data if available
        nvidia_data = self.get_nvidia_smi_data()
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'health_summary': health_summary,
            'current_metrics': current_metrics.to_dict() if current_metrics else None,
            'active_alerts': [
                {
                    'timestamp': alert.timestamp.isoformat(),
                    'severity': alert.severity,
                    'component': alert.component,
                    'metric': alert.metric,
                    'current_value': alert.current_value,
                    'threshold_value': alert.threshold_value,
                    'message': alert.message
                }
                for alert in active_alerts
            ],
            'nvidia_smi_data': nvidia_data,
            'dashboard_config': {
                'update_interval': self.config.update_interval,
                'history_hours': self.config.history_hours,
                'nvidia_smi_available': self.nvidia_smi_available
            }
        }
        
        return status
        
    def print_dashboard(self):
        """Print text-based dashboard to console"""
        status = self.get_system_status()
        current_metrics = status['current_metrics']
        active_alerts = status['active_alerts']
        health_summary = status['health_summary']
        
        print("\n" + "=" * 80)
        print("WAN22 SYSTEM HEALTH DASHBOARD")
        print("=" * 80)
        print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Overall Status: {health_summary.get('status', 'unknown').upper()}")
        
        if current_metrics:
            print("\nCURRENT METRICS:")
            print("-" * 40)
            print(f"GPU Temperature:  {current_metrics['gpu_temperature']:.1f}°C")
            print(f"GPU Utilization:  {current_metrics['gpu_utilization']:.1f}%")
            print(f"VRAM Usage:       {current_metrics['vram_usage_mb']:,} MB ({current_metrics['vram_usage_percent']:.1f}%)")
            print(f"CPU Usage:        {current_metrics['cpu_usage_percent']:.1f}%")
            print(f"Memory Usage:     {current_metrics['memory_usage_gb']:.1f} GB ({current_metrics['memory_usage_percent']:.1f}%)")
            print(f"Disk Usage:       {current_metrics['disk_usage_percent']:.1f}%")
            
        if active_alerts:
            print(f"\nACTIVE ALERTS ({len(active_alerts)}):")
            print("-" * 40)
            for alert in active_alerts:
                severity_symbol = "🔴" if alert['severity'] == 'critical' else "🟡"
                print(f"{severity_symbol} [{alert['severity'].upper()}] {alert['message']}")
        else:
            print("\nACTIVE ALERTS: None ✅")
            
        # Show nvidia-smi data if available
        nvidia_data = status.get('nvidia_smi_data')
        if nvidia_data and nvidia_data['gpus']:
            print(f"\nNVIDIA-SMI DATA ({nvidia_data['gpu_count']} GPU(s)):")
            print("-" * 40)
            for gpu in nvidia_data['gpus']:
                print(f"GPU {gpu['gpu_id']}: {gpu['name']}")
                print(f"  Temperature: {gpu['temperature']:.1f}°C")
                print(f"  Utilization: {gpu['utilization']:.1f}%")
                print(f"  Memory: {gpu['memory_used']:.0f}/{gpu['memory_total']:.0f} MB")
                if gpu['power_draw'] > 0:
                    print(f"  Power Draw: {gpu['power_draw']:.1f}W")
                    
        print("=" * 80)
        
    def create_historical_charts(self, save_path: Optional[str] = None) -> bool:
        """Create historical trend charts"""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available - charts disabled")
            return False
            
        try:
            # Get historical data
            history = self.health_monitor.get_metrics_history(
                duration_minutes=self.config.history_hours * 60
            )
            
            if not history:
                self.logger.warning("No historical data available for charts")
                return False
                
            # Convert to lists for plotting
            timestamps = [m.timestamp for m in history]
            gpu_temps = [m.gpu_temperature for m in history]
            vram_usage = [m.vram_usage_percent for m in history]
            cpu_usage = [m.cpu_usage_percent for m in history]
            memory_usage = [m.memory_usage_percent for m in history]
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, 
                                                        figsize=(self.config.chart_width, self.config.chart_height))
            fig.suptitle('WAN22 System Health Trends', fontsize=16)
            
            # GPU Temperature
            ax1.plot(timestamps, gpu_temps, 'r-', linewidth=2, label='GPU Temperature')
            ax1.axhline(y=self.health_monitor.thresholds.gpu_temperature_warning, 
                       color='orange', linestyle='--', alpha=0.7, label='Warning')
            ax1.axhline(y=self.health_monitor.thresholds.gpu_temperature_critical, 
                       color='red', linestyle='--', alpha=0.7, label='Critical')
            ax1.set_title('GPU Temperature')
            ax1.set_ylabel('Temperature (°C)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # VRAM Usage
            ax2.plot(timestamps, vram_usage, 'b-', linewidth=2, label='VRAM Usage')
            ax2.axhline(y=self.health_monitor.thresholds.vram_usage_warning, 
                       color='orange', linestyle='--', alpha=0.7, label='Warning')
            ax2.axhline(y=self.health_monitor.thresholds.vram_usage_critical, 
                       color='red', linestyle='--', alpha=0.7, label='Critical')
            ax2.set_title('VRAM Usage')
            ax2.set_ylabel('Usage (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # CPU Usage
            ax3.plot(timestamps, cpu_usage, 'g-', linewidth=2, label='CPU Usage')
            ax3.axhline(y=self.health_monitor.thresholds.cpu_usage_warning, 
                       color='orange', linestyle='--', alpha=0.7, label='Warning')
            ax3.axhline(y=self.health_monitor.thresholds.cpu_usage_critical, 
                       color='red', linestyle='--', alpha=0.7, label='Critical')
            ax3.set_title('CPU Usage')
            ax3.set_ylabel('Usage (%)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Memory Usage
            ax4.plot(timestamps, memory_usage, 'm-', linewidth=2, label='Memory Usage')
            ax4.axhline(y=self.health_monitor.thresholds.memory_usage_warning, 
                       color='orange', linestyle='--', alpha=0.7, label='Warning')
            ax4.axhline(y=self.health_monitor.thresholds.memory_usage_critical, 
                       color='red', linestyle='--', alpha=0.7, label='Critical')
            ax4.set_title('Memory Usage')
            ax4.set_ylabel('Usage (%)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Format x-axis for all subplots
            for ax in [ax1, ax2, ax3, ax4]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Charts saved to {save_path}")
            else:
                plt.show()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create historical charts: {e}")
            return False
            
    def export_data(self, filepath: str, format_type: Optional[str] = None) -> bool:
        """Export dashboard data to file"""
        format_type = format_type or self.config.export_format
        
        try:
            if format_type.lower() == 'json':
                return self._export_json(filepath)
            elif format_type.lower() == 'csv' and PANDAS_AVAILABLE:
                return self._export_csv(filepath)
            elif format_type.lower() == 'html':
                return self._export_html(filepath)
            else:
                self.logger.error(f"Unsupported export format: {format_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to export data: {e}")
            return False
            
    def _export_json(self, filepath: str) -> bool:
        """Export data as JSON"""
        try:
            status = self.get_system_status()
            
            # Add historical data
            history = self.health_monitor.get_metrics_history(
                duration_minutes=self.config.history_hours * 60
            )
            status['historical_metrics'] = [m.to_dict() for m in history]
            
            # Add alert history
            alert_history = self.health_monitor.get_alert_history(
                duration_hours=self.config.history_hours
            )
            status['alert_history'] = [
                {
                    'timestamp': alert.timestamp.isoformat(),
                    'severity': alert.severity,
                    'component': alert.component,
                    'metric': alert.metric,
                    'current_value': alert.current_value,
                    'threshold_value': alert.threshold_value,
                    'message': alert.message,
                    'resolved': alert.resolved,
                    'resolved_timestamp': alert.resolved_timestamp.isoformat() if alert.resolved_timestamp else None
                }
                for alert in alert_history
            ]
            
            with open(filepath, 'w') as f:
                json.dump(status, f, indent=2, default=str)
                
            self.logger.info(f"Data exported to JSON: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export JSON: {e}")
            return False
            
    def _export_csv(self, filepath: str) -> bool:
        """Export metrics data as CSV"""
        try:
            history = self.health_monitor.get_metrics_history(
                duration_minutes=self.config.history_hours * 60
            )
            
            if not history:
                self.logger.warning("No historical data to export")
                return False
                
            # Convert to DataFrame
            data = []
            for m in history:
                data.append({
                    'timestamp': m.timestamp,
                    'gpu_temperature': m.gpu_temperature,
                    'gpu_utilization': m.gpu_utilization,
                    'vram_usage_mb': m.vram_usage_mb,
                    'vram_usage_percent': m.vram_usage_percent,
                    'cpu_usage_percent': m.cpu_usage_percent,
                    'memory_usage_gb': m.memory_usage_gb,
                    'memory_usage_percent': m.memory_usage_percent,
                    'disk_usage_percent': m.disk_usage_percent
                })
                
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            
            self.logger.info(f"Data exported to CSV: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export CSV: {e}")
            return False
            
    def _export_html(self, filepath: str) -> bool:
        """Export dashboard as HTML report"""
        try:
            status = self.get_system_status()
            current_metrics = status['current_metrics']
            active_alerts = status['active_alerts']
            health_summary = status['health_summary']
            
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>WAN22 Health Dashboard Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .status-healthy {{ color: green; font-weight: bold; }}
        .status-warning {{ color: orange; font-weight: bold; }}
        .status-critical {{ color: red; font-weight: bold; }}
        .metrics-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .metrics-table th {{ background-color: #f2f2f2; }}
        .alert {{ padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .alert-warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; }}
        .alert-critical {{ background-color: #f8d7da; border-left: 4px solid #dc3545; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>WAN22 System Health Dashboard</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Status: <span class="status-{health_summary.get('status', 'unknown')}">{health_summary.get('status', 'unknown').upper()}</span></p>
    </div>
    
    <h2>Current Metrics</h2>
"""
            
            if current_metrics:
                html_content += """
    <table class="metrics-table">
        <tr><th>Metric</th><th>Value</th></tr>
"""
                html_content += f"""
        <tr><td>GPU Temperature</td><td>{current_metrics['gpu_temperature']:.1f}°C</td></tr>
        <tr><td>GPU Utilization</td><td>{current_metrics['gpu_utilization']:.1f}%</td></tr>
        <tr><td>VRAM Usage</td><td>{current_metrics['vram_usage_mb']:,} MB ({current_metrics['vram_usage_percent']:.1f}%)</td></tr>
        <tr><td>CPU Usage</td><td>{current_metrics['cpu_usage_percent']:.1f}%</td></tr>
        <tr><td>Memory Usage</td><td>{current_metrics['memory_usage_gb']:.1f} GB ({current_metrics['memory_usage_percent']:.1f}%)</td></tr>
        <tr><td>Disk Usage</td><td>{current_metrics['disk_usage_percent']:.1f}%</td></tr>
    </table>
"""
            else:
                html_content += "<p>No current metrics available.</p>"
                
            html_content += f"""
    <h2>Active Alerts ({len(active_alerts)})</h2>
"""
            
            if active_alerts:
                for alert in active_alerts:
                    alert_class = f"alert-{alert['severity']}"
                    html_content += f"""
    <div class="alert {alert_class}">
        <strong>[{alert['severity'].upper()}]</strong> {alert['message']}<br>
        <small>Component: {alert['component']} | Metric: {alert['metric']} | Value: {alert['current_value']} | Threshold: {alert['threshold_value']}</small>
    </div>
"""
            else:
                html_content += "<p>No active alerts.</p>"
                
            # Add nvidia-smi data if available
            nvidia_data = status.get('nvidia_smi_data')
            if nvidia_data and nvidia_data['gpus']:
                html_content += f"""
    <h2>GPU Information ({nvidia_data['gpu_count']} GPU(s))</h2>
    <table class="metrics-table">
        <tr><th>GPU</th><th>Name</th><th>Temperature</th><th>Utilization</th><th>Memory</th><th>Power</th></tr>
"""
                for gpu in nvidia_data['gpus']:
                    power_info = f"{gpu['power_draw']:.1f}W" if gpu['power_draw'] > 0 else "N/A"
                    html_content += f"""
        <tr>
            <td>GPU {gpu['gpu_id']}</td>
            <td>{gpu['name']}</td>
            <td>{gpu['temperature']:.1f}°C</td>
            <td>{gpu['utilization']:.1f}%</td>
            <td>{gpu['memory_used']:.0f}/{gpu['memory_total']:.0f} MB</td>
            <td>{power_info}</td>
        </tr>
"""
                html_content += "</table>"
                
            html_content += """
</body>
</html>
"""
            
            with open(filepath, 'w') as f:
                f.write(html_content)
                
            self.logger.info(f"Dashboard exported to HTML: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export HTML: {e}")
            return False
            
    def start_real_time_monitoring(self, duration_seconds: Optional[int] = None):
        """Start real-time monitoring with periodic dashboard updates"""
        if self.is_running:
            self.logger.warning("Dashboard already running")
            return
            
        self.is_running = True
        start_time = time.time()
        
        try:
            print("Starting real-time health monitoring dashboard...")
            print("Press Ctrl+C to stop")
            
            while self.is_running:
                # Clear screen (works on most terminals)
                print("\033[2J\033[H", end="")
                
                # Print dashboard
                self.print_dashboard()
                
                # Check if duration limit reached
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    break
                    
                # Wait for next update
                time.sleep(self.config.update_interval)
                
        except KeyboardInterrupt:
            print("\nStopping dashboard...")
        finally:
            self.is_running = False
            
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_running = False
        
    def generate_health_report(self, output_dir: str = "health_reports") -> Dict[str, str]:
        """Generate comprehensive health report with multiple formats"""
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"health_report_{timestamp}"
        
        generated_files = {}
        
        # Generate JSON report
        json_path = os.path.join(output_dir, f"{base_filename}.json")
        if self.export_data(json_path, 'json'):
            generated_files['json'] = json_path
            
        # Generate HTML report
        html_path = os.path.join(output_dir, f"{base_filename}.html")
        if self.export_data(html_path, 'html'):
            generated_files['html'] = html_path
            
        # Generate CSV report if pandas available
        if PANDAS_AVAILABLE:
            csv_path = os.path.join(output_dir, f"{base_filename}.csv")
            if self.export_data(csv_path, 'csv'):
                generated_files['csv'] = csv_path
                
        # Generate charts if matplotlib available
        if MATPLOTLIB_AVAILABLE:
            chart_path = os.path.join(output_dir, f"{base_filename}_charts.png")
            if self.create_historical_charts(chart_path):
                generated_files['charts'] = chart_path
                
        self.logger.info(f"Health report generated: {len(generated_files)} files in {output_dir}")
        return generated_files


# Demo and utility functions
def create_demo_dashboard(health_monitor: Optional[HealthMonitor] = None) -> HealthDashboard:
    """Create a demo dashboard for testing"""
    if health_monitor is None:
        from health_monitor import create_demo_health_monitor
        health_monitor = create_demo_health_monitor()
        
    config = DashboardConfig(
        update_interval=2.0,
        history_hours=1,  # Shorter for demo
        enable_nvidia_smi=True,
        enable_real_time_charts=True
    )
    
    return HealthDashboard(health_monitor, config)


def run_dashboard_demo():
    """Run a demo of the health dashboard"""
    print("WAN22 Health Dashboard Demo")
    print("=" * 40)
    
    # Create demo components
    from health_monitor import create_demo_health_monitor
    
    monitor = create_demo_health_monitor()
    dashboard = create_demo_dashboard(monitor)
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Wait for some data to be collected
        print("Collecting initial data...")
        time.sleep(5)
        
        # Show single dashboard update
        print("\nDashboard snapshot:")
        dashboard.print_dashboard()
        
        # Test export functionality
        print("\nTesting export functionality...")
        reports = dashboard.generate_health_report("demo_reports")
        print(f"Generated reports: {list(reports.keys())}")
        
        # Test nvidia-smi integration
        nvidia_data = dashboard.get_nvidia_smi_data()
        if nvidia_data:
            print(f"\nNVIDIA-SMI integration: {nvidia_data['gpu_count']} GPU(s) detected")
        else:
            print("\nNVIDIA-SMI integration: Not available")
            
    finally:
        monitor.stop_monitoring()
        
    print("\nDemo completed!")


if __name__ == "__main__":
    run_dashboard_demo()
