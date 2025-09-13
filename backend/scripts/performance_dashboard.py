#!/usr/bin/env python3
"""
Performance dashboard for monitoring real AI model integration performance.
Provides real-time monitoring and analysis of generation performance.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from backend.core.performance_monitor import get_performance_monitor
from backend.core.system_integration import SystemIntegration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceDashboard:
    """Real-time performance dashboard for AI model integration."""
    
    def __init__(self):
        self.performance_monitor = None
        self.system_integration = None
        self.running = False
        
    async def initialize(self) -> bool:
        """Initialize dashboard components."""
        try:
            self.performance_monitor = get_performance_monitor()
            self.system_integration = SystemIntegration()
            await self.system_integration.initialize()
            
            logger.info("Performance dashboard initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize dashboard: {e}")
            return False
    
    def display_system_status(self):
        """Display current system status."""
        try:
            status = self.performance_monitor.get_current_system_status()
            
            print("\n" + "="*60)
            print("SYSTEM PERFORMANCE STATUS")
            print("="*60)
            
            if "error" in status:
                print(f"❌ Error: {status['error']}")
                return
            
            # System resources
            print(f"🖥️  CPU Usage: {status.get('cpu_usage_percent', 0):.1f}%")
            print(f"💾 RAM Usage: {status.get('ram_usage_mb', 0):.0f}MB "
                  f"(Available: {status.get('ram_available_mb', 0):.0f}MB)")
            print(f"💿 Disk Usage: {status.get('disk_usage_percent', 0):.1f}%")
            
            # GPU resources
            if status.get('gpu_available', False):
                print(f"🎮 GPU Usage: {status.get('gpu_usage_percent', 0):.1f}%")
                print(f"🎯 VRAM Usage: {status.get('vram_usage_mb', 0):.0f}MB "
                      f"(Available: {status.get('vram_available_mb', 0):.0f}MB)")
                
                if status.get('temperature_celsius', 0) > 0:
                    temp = status.get('temperature_celsius', 0)
                    temp_icon = "🔥" if temp > 80 else "🌡️"
                    print(f"{temp_icon} GPU Temperature: {temp}°C")
            else:
                print("🚫 GPU not available")
            
            # Active tasks
            active_tasks = status.get('active_tasks', 0)
            print(f"⚡ Active Generation Tasks: {active_tasks}")
            
            # Timestamp
            timestamp = datetime.fromtimestamp(status.get('timestamp', time.time()))
            print(f"🕐 Last Updated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"❌ Error displaying system status: {e}")
    
    def display_performance_analysis(self, hours: int = 24):
        """Display performance analysis."""
        try:
            analysis = self.performance_monitor.get_performance_analysis(hours)
            
            print("\n" + "="*60)
            print(f"PERFORMANCE ANALYSIS (Last {hours} hours)")
            print("="*60)
            
            # Basic metrics
            print(f"📊 Average Generation Time: {analysis.average_generation_time:.1f}s")
            print(f"✅ Success Rate: {analysis.success_rate:.1%}")
            print(f"⚡ Resource Efficiency: {analysis.resource_efficiency:.1%}")
            
            # Bottleneck analysis
            bottlenecks = analysis.bottleneck_analysis
            print("\n🔍 BOTTLENECK ANALYSIS:")
            
            if bottlenecks.get("vram_bottleneck"):
                print("  ⚠️  VRAM bottleneck detected")
            if bottlenecks.get("ram_bottleneck"):
                print("  ⚠️  RAM bottleneck detected")
            if bottlenecks.get("cpu_bottleneck"):
                print("  ⚠️  CPU bottleneck detected")
            if bottlenecks.get("model_load_bottleneck"):
                print("  ⚠️  Model loading bottleneck detected")
            
            if not any(bottlenecks.get(k, False) for k in ["vram_bottleneck", "ram_bottleneck", "cpu_bottleneck", "model_load_bottleneck"]):
                print("  ✅ No significant bottlenecks detected")
            
            # Optimization recommendations
            recommendations = analysis.optimization_recommendations
            if recommendations:
                print("\n💡 OPTIMIZATION RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
                    print(f"  {i}. {rec}")
            else:
                print("\n✅ No optimization recommendations at this time")
            
            # Performance trends
            trends = analysis.performance_trends
            if trends.get("generation_times"):
                recent_times = trends["generation_times"][-3:]  # Last 3 data points
                if len(recent_times) >= 2:
                    trend = "📈" if recent_times[-1] > recent_times[0] else "📉"
                    print(f"\n{trend} Generation Time Trend: {recent_times}")
            
        except Exception as e:
            print(f"❌ Error displaying performance analysis: {e}")
    
    def display_recent_tasks(self, count: int = 10):
        """Display recent generation tasks."""
        try:
            # Get recent metrics
            cutoff_time = time.time() - (24 * 3600)  # Last 24 hours
            recent_metrics = [
                m for m in self.performance_monitor.metrics_history
                if m.start_time >= cutoff_time
            ]
            
            # Sort by start time (most recent first)
            recent_metrics.sort(key=lambda m: m.start_time, reverse=True)
            recent_metrics = recent_metrics[:count]
            
            print("\n" + "="*60)
            print(f"RECENT GENERATION TASKS (Last {count})")
            print("="*60)
            
            if not recent_metrics:
                print("No recent generation tasks found")
                return
            
            # Table header
            print(f"{'Task ID':<12} {'Model':<12} {'Resolution':<10} {'Time':<8} {'Status':<8} {'VRAM':<8}")
            print("-" * 60)
            
            for m in recent_metrics:
                task_id = m.task_id[:8] + "..." if len(m.task_id) > 8 else m.task_id
                model = m.model_type[:10] if m.model_type else "Unknown"
                resolution = m.resolution[:8] if m.resolution else "Unknown"
                gen_time = f"{m.generation_time_seconds:.1f}s" if m.generation_time_seconds > 0 else "N/A"
                status = "✅" if m.success else "❌"
                vram = f"{m.peak_vram_usage_mb:.0f}MB" if m.peak_vram_usage_mb > 0 else "N/A"
                
                print(f"{task_id:<12} {model:<12} {resolution:<10} {gen_time:<8} {status:<8} {vram:<8}")
            
        except Exception as e:
            print(f"❌ Error displaying recent tasks: {e}")
    
    def display_benchmarks(self):
        """Display performance benchmarks and targets."""
        try:
            analysis = self.performance_monitor.get_performance_analysis(24)
            
            print("\n" + "="*60)
            print("PERFORMANCE BENCHMARKS")
            print("="*60)
            
            # Generation time benchmarks
            print("🎯 GENERATION TIME TARGETS:")
            print(f"  720p Target: 5 minutes (300s)")
            print(f"  1080p Target: 15 minutes (900s)")
            
            current_time = analysis.average_generation_time
            if current_time > 0:
                if current_time <= 300:
                    status = "✅ Excellent"
                elif current_time <= 450:
                    status = "⚠️ Acceptable"
                else:
                    status = "❌ Needs Improvement"
                print(f"  Current Average: {current_time:.1f}s ({status})")
            
            # Success rate benchmarks
            print("\n📊 SUCCESS RATE TARGETS:")
            print(f"  Target: 98%")
            print(f"  Minimum: 95%")
            
            current_success = analysis.success_rate
            if current_success >= 0.98:
                status = "✅ Excellent"
            elif current_success >= 0.95:
                status = "⚠️ Acceptable"
            else:
                status = "❌ Needs Improvement"
            print(f"  Current: {current_success:.1%} ({status})")
            
            # Resource efficiency benchmarks
            print("\n⚡ RESOURCE EFFICIENCY TARGETS:")
            print(f"  Target: 85%")
            print(f"  Minimum: 70%")
            
            current_efficiency = analysis.resource_efficiency
            if current_efficiency >= 0.85:
                status = "✅ Excellent"
            elif current_efficiency >= 0.70:
                status = "⚠️ Acceptable"
            else:
                status = "❌ Needs Improvement"
            print(f"  Current: {current_efficiency:.1%} ({status})")
            
        except Exception as e:
            print(f"❌ Error displaying benchmarks: {e}")
    
    def display_menu(self):
        """Display interactive menu."""
        print("\n" + "="*60)
        print("PERFORMANCE DASHBOARD MENU")
        print("="*60)
        print("1. System Status")
        print("2. Performance Analysis (24h)")
        print("3. Performance Analysis (1h)")
        print("4. Recent Tasks")
        print("5. Benchmarks")
        print("6. Export Performance Data")
        print("7. Continuous Monitoring")
        print("0. Exit")
        print("="*60)
    
    async def export_performance_data(self):
        """Export performance data to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_export_{timestamp}.json"
            
            self.performance_monitor.export_metrics(filename, 24)
            print(f"✅ Performance data exported to: {filename}")
            
        except Exception as e:
            print(f"❌ Error exporting performance data: {e}")
    
    async def continuous_monitoring(self, interval: int = 10):
        """Run continuous monitoring display."""
        print(f"\n🔄 Starting continuous monitoring (updates every {interval}s)")
        print("Press Ctrl+C to stop...")
        
        try:
            while True:
                # Clear screen (works on most terminals)
                print("\033[2J\033[H")
                
                print(f"🔄 CONTINUOUS MONITORING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                self.display_system_status()
                self.display_performance_analysis(1)  # Last hour
                
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Continuous monitoring stopped")
    
    async def run_interactive(self):
        """Run interactive dashboard."""
        if not await self.initialize():
            print("❌ Failed to initialize dashboard")
            return
        
        print("🚀 Performance Dashboard Started")
        
        while True:
            try:
                self.display_menu()
                choice = input("\nEnter your choice (0-7): ").strip()
                
                if choice == "0":
                    print("👋 Goodbye!")
                    break
                elif choice == "1":
                    self.display_system_status()
                elif choice == "2":
                    self.display_performance_analysis(24)
                elif choice == "3":
                    self.display_performance_analysis(1)
                elif choice == "4":
                    self.display_recent_tasks()
                elif choice == "5":
                    self.display_benchmarks()
                elif choice == "6":
                    await self.export_performance_data()
                elif choice == "7":
                    await self.continuous_monitoring()
                else:
                    print("❌ Invalid choice. Please try again.")
                
                if choice != "7":  # Don't pause after continuous monitoring
                    input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                input("\nPress Enter to continue...")

async def main():
    """Main dashboard function."""
    dashboard = PerformanceDashboard()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "status":
            if await dashboard.initialize():
                dashboard.display_system_status()
        elif command == "analysis":
            if await dashboard.initialize():
                hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
                dashboard.display_performance_analysis(hours)
        elif command == "tasks":
            if await dashboard.initialize():
                count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
                dashboard.display_recent_tasks(count)
        elif command == "benchmarks":
            if await dashboard.initialize():
                dashboard.display_benchmarks()
        elif command == "export":
            if await dashboard.initialize():
                await dashboard.export_performance_data()
        elif command == "monitor":
            if await dashboard.initialize():
                interval = int(sys.argv[2]) if len(sys.argv) > 2 else 10
                await dashboard.continuous_monitoring(interval)
        else:
            print(f"Unknown command: {command}")
            print("Available commands: status, analysis, tasks, benchmarks, export, monitor")
    else:
        # Run interactive mode
        await dashboard.run_interactive()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        sys.exit(1)
