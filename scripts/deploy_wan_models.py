#!/usr/bin/env python3
"""
WAN Model Deployment Script

Command-line interface for deploying WAN models from placeholder to production
with comprehensive validation, rollback, and monitoring capabilities.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.deployment import (
    DeploymentManager, DeploymentConfig, DeploymentStatus
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/deployment.log')
        ]
    )


async def deploy_models(args):
    """Deploy WAN models"""
    print(f"🚀 Starting WAN model deployment...")
    print(f"Models to deploy: {', '.join(args.models)}")
    
    # Create deployment configuration
    config = DeploymentConfig(
        source_models_path=args.source_path,
        target_models_path=args.target_path,
        backup_path=args.backup_path,
        validation_enabled=not args.skip_validation,
        rollback_enabled=not args.skip_rollback,
        monitoring_enabled=not args.skip_monitoring,
        health_check_interval=args.health_check_interval,
        max_deployment_time=args.max_deployment_time
    )
    
    # Initialize deployment manager
    deployment_manager = DeploymentManager(config)
    
    try:
        # Start deployment
        result = await deployment_manager.deploy_models(
            models=args.models,
            deployment_id=args.deployment_id
        )
        
        # Print results
        print(f"\n📊 Deployment Results:")
        print(f"Deployment ID: {result.deployment_id}")
        print(f"Status: {result.status.value}")
        print(f"Models deployed: {len(result.models_deployed)}")
        print(f"Duration: {(result.end_time - result.start_time).total_seconds():.2f}s")
        
        if result.status == DeploymentStatus.COMPLETED:
            print("✅ Deployment completed successfully!")
            
            if result.validation_results:
                print("\n🔍 Validation Results:")
                for validation_type, results in result.validation_results.items():
                    print(f"  {validation_type}: {'✅ PASSED' if results.get('is_valid') else '❌ FAILED'}")
            
            if config.monitoring_enabled:
                print("\n📈 Monitoring started for deployed models")
                
        elif result.status == DeploymentStatus.FAILED:
            print("❌ Deployment failed!")
            if result.error_message:
                print(f"Error: {result.error_message}")
                
        elif result.status == DeploymentStatus.ROLLED_BACK:
            print("🔄 Deployment was rolled back due to validation failures")
            if result.error_message:
                print(f"Reason: {result.error_message}")
        
        # Save deployment report
        if args.output_report:
            report_data = {
                "deployment_id": result.deployment_id,
                "status": result.status.value,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat() if result.end_time else None,
                "models_deployed": result.models_deployed,
                "validation_results": result.validation_results,
                "rollback_available": result.rollback_available,
                "error_message": result.error_message
            }
            
            with open(args.output_report, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            print(f"\n📄 Deployment report saved to: {args.output_report}")
        
        return 0 if result.status == DeploymentStatus.COMPLETED else 1
        
    except Exception as e:
        print(f"❌ Deployment failed with exception: {str(e)}")
        logging.error(f"Deployment exception: {str(e)}", exc_info=True)
        return 1


async def list_deployments(args):
    """List deployment history"""
    config = DeploymentConfig(
        source_models_path=args.source_path,
        target_models_path=args.target_path,
        backup_path=args.backup_path
    )
    
    deployment_manager = DeploymentManager(config)
    
    status_filter = None
    if args.status:
        status_filter = DeploymentStatus(args.status)
    
    deployments = await deployment_manager.list_deployments(
        status_filter=status_filter,
        limit=args.limit
    )
    
    if not deployments:
        print("No deployments found.")
        return 0
    
    print(f"\n📋 Deployment History ({len(deployments)} deployments):")
    print("-" * 80)
    
    for deployment in deployments:
        status_emoji = {
            DeploymentStatus.COMPLETED: "✅",
            DeploymentStatus.FAILED: "❌",
            DeploymentStatus.ROLLED_BACK: "🔄",
            DeploymentStatus.IN_PROGRESS: "⏳",
            DeploymentStatus.PENDING: "⏸️"
        }.get(deployment.status, "❓")
        
        print(f"{status_emoji} {deployment.deployment_id}")
        print(f"   Status: {deployment.status.value}")
        print(f"   Models: {', '.join(deployment.models_deployed)}")
        print(f"   Started: {deployment.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if deployment.end_time:
            duration = (deployment.end_time - deployment.start_time).total_seconds()
            print(f"   Duration: {duration:.2f}s")
        if deployment.error_message:
            print(f"   Error: {deployment.error_message}")
        print()
    
    return 0


async def rollback_deployment(args):
    """Rollback a deployment"""
    print(f"🔄 Rolling back deployment: {args.deployment_id}")
    
    config = DeploymentConfig(
        source_models_path=args.source_path,
        target_models_path=args.target_path,
        backup_path=args.backup_path
    )
    
    deployment_manager = DeploymentManager(config)
    
    try:
        result = await deployment_manager.rollback_deployment(
            deployment_id=args.deployment_id,
            reason=args.reason or "Manual rollback via CLI"
        )
        
        if result.success:
            print("✅ Rollback completed successfully!")
            print(f"Models restored: {', '.join(result.models_restored)}")
            print(f"Rollback time: {result.rollback_time:.2f}s")
        else:
            print("❌ Rollback failed!")
            if result.error:
                print(f"Error: {result.error}")
            return 1
        
    except Exception as e:
        print(f"❌ Rollback failed with exception: {str(e)}")
        logging.error(f"Rollback exception: {str(e)}", exc_info=True)
        return 1
    
    return 0


async def check_health(args):
    """Check health status of deployed models"""
    config = DeploymentConfig(
        source_models_path=args.source_path,
        target_models_path=args.target_path,
        backup_path=args.backup_path
    )
    
    deployment_manager = DeploymentManager(config)
    
    try:
        health_status = await deployment_manager.get_health_status()
        
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
        
        # Export health report if requested
        if args.output_report:
            with open(args.output_report, 'w') as f:
                json.dump(health_status, f, indent=2, default=str)
            print(f"\n📄 Health report saved to: {args.output_report}")
        
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        logging.error(f"Health check exception: {str(e)}", exc_info=True)
        return 1
    
    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="WAN Model Deployment and Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy specific models
  python deploy_wan_models.py deploy --models t2v-A14B i2v-A14B
  
  # Deploy with custom paths
  python deploy_wan_models.py deploy --models ti2v-5B --source-path ./models --target-path ./production
  
  # List recent deployments
  python deploy_wan_models.py list --limit 10
  
  # Rollback a deployment
  python deploy_wan_models.py rollback --deployment-id deployment_20241201_143022
  
  # Check health status
  python deploy_wan_models.py health --output-report health_report.json
        """
    )
    
    # Global arguments
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--source-path', default='models', help='Source models directory')
    parser.add_argument('--target-path', default='models', help='Target models directory')
    parser.add_argument('--backup-path', default='backups/models', help='Backup directory')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy WAN models')
    deploy_parser.add_argument('--models', nargs='+', required=True, 
                              help='Model names to deploy')
    deploy_parser.add_argument('--deployment-id', help='Custom deployment ID')
    deploy_parser.add_argument('--skip-validation', action='store_true',
                              help='Skip validation steps')
    deploy_parser.add_argument('--skip-rollback', action='store_true',
                              help='Skip rollback capability')
    deploy_parser.add_argument('--skip-monitoring', action='store_true',
                              help='Skip monitoring setup')
    deploy_parser.add_argument('--health-check-interval', type=int, default=300,
                              help='Health check interval in seconds')
    deploy_parser.add_argument('--max-deployment-time', type=int, default=3600,
                              help='Maximum deployment time in seconds')
    deploy_parser.add_argument('--output-report', help='Output deployment report file')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List deployment history')
    list_parser.add_argument('--status', choices=['pending', 'in_progress', 'completed', 'failed', 'rolled_back'],
                            help='Filter by deployment status')
    list_parser.add_argument('--limit', type=int, default=20, help='Maximum number of deployments to show')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback a deployment')
    rollback_parser.add_argument('--deployment-id', required=True, help='Deployment ID to rollback')
    rollback_parser.add_argument('--reason', help='Reason for rollback')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Check health status')
    health_parser.add_argument('--output-report', help='Output health report file')
    
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
        if args.command == 'deploy':
            return asyncio.run(deploy_models(args))
        elif args.command == 'list':
            return asyncio.run(list_deployments(args))
        elif args.command == 'rollback':
            return asyncio.run(rollback_deployment(args))
        elif args.command == 'health':
            return asyncio.run(check_health(args))
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