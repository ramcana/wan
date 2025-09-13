#!/usr/bin/env python3
"""
WAN Model Monitoring Script

Command-line interface for monitoring deployed WAN models with real-time
health checking, alerting, and performance metrics.
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.deployment import MonitoringService, DeploymentConfig


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/monitoring.log')
        ]
    )


class MonitoringCLI:
    """CLI wrapper for monitoring service"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.monitoring_service = MonitoringService(config)
        self.running = False
    
    async def start_monitoring(self, deployment_id: str, models: List[str]):
        """Start monitoring for a deployment"""
        print(f"📈 Starting monitoring for deployment: {deployment_id}")
        print(f"Models: {', '.join(models)}")
        
        # Add custom alert handler for CLI output
        self.monitoring_service.add_alert_handler(self._handle_alert)
        
        # Start monitoring
        await self.monitoring_service.start_monitoring(deployment_id, models)
        
        self.running = True
        print("✅ Monitoring started. Press Ctrl+C to stop.")
        
        # Keep running until interrupted
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping monitoring...")
            await self.monitoring_service.stop_monitoring(deployment_id)
            await self.monitoring_service.shutdown()
            print("✅ Monitoring stopped.")
    
    async def _handle_alert(self, alert):
        """Handle alerts by printing to console"""
        level_emoji = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨"
        }
        
        emoji = level_emoji.get(alert.level.value, "❓")
        print(f"\n{emoji} ALERT [{alert.level.value.upper()}]: {alert.message}")
        print(f"   Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        if alert.model_name:
            print(f"   Model: {alert.model_name}")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False


async def start_monitoring(args):
    """Start monitoring command"""
    config = DeploymentConfig(
        source_models_path=args.source_path,
        target_models_path=args.target_path,
        backup_path=args.backup_path,
        monitoring_enabled=True,
        health_check_interval=args.interval
    )
    
    cli = MonitoringCLI(config)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, stopping monitoring...")
        cli.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await cli.start_monitoring(args.deployment_id, args.models)
        return 0
    except Exception as e:
        print(f"❌ Monitoring failed: {str(e)}")
        logging.error(f"Monitoring exception: {str(e)}", exc_info=True)
        return 1


async def get_health_status(args):
    """Get current health status"""
    config = DeploymentConfig(
        source_models_path=args.source_path,
        target_models_path=args.target_path,
        backup_path=args.backup_path
    )
    
    monitoring_service = MonitoringService(config)
    
    try:
        health_status = await monitoring_service.get_health_status()
        
        print(f"\n🏥 Health Status Report")
        print(f"Generated: {health_status['timestamp']}")
        print(f"Overall Status: {health_status['overall_status'].upper()}")
        print(f"Monitored Deployments: {health_status['monitored_deployments']}")
        print(f"Active Alerts: {health_status['active_alerts']}")
        
        if health_status['deployments']:
            print("\n📊 Deployment Details:")
            for deployment_id, deployment_info in health_status['deployments'].items():
                status_emoji = {
                    "healthy": "✅",
                    "warning": "⚠️",
                    "critical": "❌",
                    "unknown": "❓"
                }.get(deployment_info['status'], "❓")
                
                print(f"  {status_emoji} {deployment_id}")
                print(f"     Models: {', '.join(deployment_info['models'])}")
                print(f"     Status: {deployment_info['status']}")
                print(f"     Uptime: {deployment_info['uptime_seconds']:.0f}s")
                if deployment_info['last_check']:
                    print(f"     Last Check: {deployment_info['last_check']}")
        
        # Save health report if requested
        if args.output_report:
            with open(args.output_report, 'w') as f:
                json.dump(health_status, f, indent=2, default=str)
            print(f"\n📄 Health report saved to: {args.output_report}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to get health status: {str(e)}")
        logging.error(f"Health status exception: {str(e)}", exc_info=True)
        return 1


async def get_model_history(args):
    """Get health history for a specific model"""
    config = DeploymentConfig(
        source_models_path=args.source_path,
        target_models_path=args.target_path,
        backup_path=args.backup_path
    )
    
    monitoring_service = MonitoringService(config)
    
    try:
        history = await monitoring_service.get_model_health_history(
            args.model, args.hours
        )
        
        if not history:
            print(f"No health history found for model {args.model} in the last {args.hours} hours.")
            return 0
        
        print(f"\n📈 Health History for {args.model}")
        print(f"Time Range: Last {args.hours} hours")
        print(f"Total Reports: {len(history)}")
        print("-" * 80)
        
        for report in history[-10:]:  # Show last 10 reports
            status_emoji = {
                "healthy": "✅",
                "warning": "⚠️", 
                "critical": "❌",
                "unknown": "❓"
            }.get(report.overall_status.value, "❓")
            
            print(f"{status_emoji} {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {report.overall_status.value.upper()}")
            
            if report.alerts:
                for alert in report.alerts:
                    print(f"   ⚠️  {alert}")
            
            # Show key metrics
            key_metrics = [m for m in report.metrics if m.name in ['cpu_usage', 'memory_usage', 'inference_time']]
            if key_metrics:
                metrics_str = ", ".join([f"{m.name}: {m.value:.1f}{m.unit}" for m in key_metrics])
                print(f"   📊 {metrics_str}")
            
            print()
        
        # Save history report if requested
        if args.output_report:
            history_data = {
                "model_name": args.model,
                "time_range_hours": args.hours,
                "total_reports": len(history),
                "reports": [
                    {
                        "timestamp": report.timestamp.isoformat(),
                        "status": report.overall_status.value,
                        "uptime_seconds": report.uptime_seconds,
                        "alerts": report.alerts,
                        "metrics": [
                            {
                                "name": m.name,
                                "value": m.value,
                                "unit": m.unit,
                                "status": m.status.value
                            } for m in report.metrics
                        ]
                    } for report in history
                ]
            }
            
            with open(args.output_report, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            print(f"📄 History report saved to: {args.output_report}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to get model history: {str(e)}")
        logging.error(f"Model history exception: {str(e)}", exc_info=True)
        return 1


async def list_alerts(args):
    """List active alerts"""
    config = DeploymentConfig(
        source_models_path=args.source_path,
        target_models_path=args.target_path,
        backup_path=args.backup_path
    )
    
    monitoring_service = MonitoringService(config)
    
    try:
        alerts = await monitoring_service.get_active_alerts()
        
        if not alerts:
            print("✅ No active alerts.")
            return 0
        
        print(f"\n🚨 Active Alerts ({len(alerts)}):")
        print("-" * 80)
        
        for alert in alerts:
            level_emoji = {
                "info": "ℹ️",
                "warning": "⚠️",
                "critical": "🚨"
            }.get(alert.level.value, "❓")
            
            print(f"{level_emoji} [{alert.level.value.upper()}] {alert.alert_id}")
            print(f"   Message: {alert.message}")
            print(f"   Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            if alert.model_name:
                print(f"   Model: {alert.model_name}")
            print()
        
        # Save alerts report if requested
        if args.output_report:
            alerts_data = {
                "generated_at": asyncio.get_event_loop().time(),
                "total_alerts": len(alerts),
                "alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "level": alert.level.value,
                        "message": alert.message,
                        "model_name": alert.model_name,
                        "timestamp": alert.timestamp.isoformat(),
                        "resolved": alert.resolved
                    } for alert in alerts
                ]
            }
            
            with open(args.output_report, 'w') as f:
                json.dump(alerts_data, f, indent=2)
            
            print(f"📄 Alerts report saved to: {args.output_report}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to list alerts: {str(e)}")
        logging.error(f"List alerts exception: {str(e)}", exc_info=True)
        return 1


async def export_health_report(args):
    """Export comprehensive health report"""
    config = DeploymentConfig(
        source_models_path=args.source_path,
        target_models_path=args.target_path,
        backup_path=args.backup_path
    )
    
    monitoring_service = MonitoringService(config)
    
    try:
        await monitoring_service.export_health_report(args.output_file, args.hours)
        print(f"✅ Health report exported to: {args.output_file}")
        return 0
        
    except Exception as e:
        print(f"❌ Export failed: {str(e)}")
        logging.error(f"Export exception: {str(e)}", exc_info=True)
        return 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="WAN Model Monitoring CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start monitoring a deployment
  python monitor_wan_models.py start --deployment-id deployment_123 --models t2v-A14B i2v-A14B
  
  # Get current health status
  python monitor_wan_models.py status --output-report health_status.json
  
  # Get model health history
  python monitor_wan_models.py history --model t2v-A14B --hours 24
  
  # List active alerts
  python monitor_wan_models.py alerts
  
  # Export comprehensive health report
  python monitor_wan_models.py export --output-file health_report.json --hours 48
        """
    )
    
    # Global arguments
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--source-path', default='models', help='Source models directory')
    parser.add_argument('--target-path', default='models', help='Target models directory')
    parser.add_argument('--backup-path', default='backups/models', help='Backup directory')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start monitoring command
    start_parser = subparsers.add_parser('start', help='Start monitoring a deployment')
    start_parser.add_argument('--deployment-id', required=True, help='Deployment ID to monitor')
    start_parser.add_argument('--models', nargs='+', required=True, help='Model names to monitor')
    start_parser.add_argument('--interval', type=int, default=300, help='Health check interval in seconds')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Get current health status')
    status_parser.add_argument('--output-report', help='Output health status report file')
    
    # History command
    history_parser = subparsers.add_parser('history', help='Get model health history')
    history_parser.add_argument('--model', required=True, help='Model name')
    history_parser.add_argument('--hours', type=int, default=24, help='Hours of history to retrieve')
    history_parser.add_argument('--output-report', help='Output history report file')
    
    # Alerts command
    alerts_parser = subparsers.add_parser('alerts', help='List active alerts')
    alerts_parser.add_argument('--output-report', help='Output alerts report file')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export comprehensive health report')
    export_parser.add_argument('--output-file', required=True, help='Output file for health report')
    export_parser.add_argument('--hours', type=int, default=24, help='Hours of data to include')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)
    
    # Run the appropriate command
    try:
        if args.command == 'start':
            return asyncio.run(start_monitoring(args))
        elif args.command == 'status':
            return asyncio.run(get_health_status(args))
        elif args.command == 'history':
            return asyncio.run(get_model_history(args))
        elif args.command == 'alerts':
            return asyncio.run(list_alerts(args))
        elif args.command == 'export':
            return asyncio.run(export_health_report(args))
        else:
            print(f"Unknown command: {args.command}")
            return 1
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
