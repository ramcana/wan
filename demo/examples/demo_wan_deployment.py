#!/usr/bin/env python3
"""
WAN Model Deployment System Demo

Demonstrates the key features of the WAN Model Deployment and Migration system.
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path

from infrastructure.deployment import (
    DeploymentManager, DeploymentConfig, DeploymentStatus
)


async def demo_deployment_system():
    """Demonstrate the WAN Model Deployment System"""
    print("🚀 WAN Model Deployment System Demo")
    print("=" * 50)
    
    # Setup demo environment
    temp_dir = Path(tempfile.mkdtemp())
    source_path = temp_dir / "source"
    target_path = temp_dir / "target"
    backup_path = temp_dir / "backup"
    
    # Create directories
    source_path.mkdir(parents=True)
    target_path.mkdir(parents=True)
    backup_path.mkdir(parents=True)
    
    # Create demo model
    demo_model_dir = source_path / "demo-model"
    demo_model_dir.mkdir()
    
    (demo_model_dir / "config.json").write_text(json.dumps({
        "model_type": "text-to-video",
        "architecture": "transformer",
        "version": "1.0.0"
    }))
    
    (demo_model_dir / "model.bin").write_text("Demo model data")
    
    print(f"📁 Demo environment created at: {temp_dir}")
    
    # Create deployment configuration
    config = DeploymentConfig(
        source_models_path=str(source_path),
        target_models_path=str(target_path),
        backup_path=str(backup_path),
        validation_enabled=True,
        rollback_enabled=True,
        monitoring_enabled=False,  # Disable for demo
        health_check_interval=60
    )
    
    print(f"⚙️  Configuration created")
    
    # Initialize deployment manager
    deployment_manager = DeploymentManager(config)
    print(f"🎛️  Deployment manager initialized")
    
    try:
        # Demo 1: Deploy a model
        print(f"\n📦 Demo 1: Deploying model 'demo-model'")
        result = await deployment_manager.deploy_models(["demo-model"])
        
        print(f"   Deployment ID: {result.deployment_id}")
        print(f"   Status: {result.status.value}")
        print(f"   Models deployed: {result.models_deployed}")
        print(f"   Duration: {(result.end_time - result.start_time).total_seconds():.2f}s")
        
        if result.validation_results:
            print(f"   Validation results: {len(result.validation_results)} checks")
        
        # Demo 2: List deployments
        print(f"\n📋 Demo 2: Listing deployment history")
        deployments = await deployment_manager.list_deployments(limit=5)
        
        for deployment in deployments:
            print(f"   • {deployment.deployment_id} - {deployment.status.value}")
        
        # Demo 3: Get deployment status
        print(f"\n📊 Demo 3: Getting deployment status")
        status = await deployment_manager.get_deployment_status(result.deployment_id)
        
        if status:
            print(f"   Deployment: {status.deployment_id}")
            print(f"   Status: {status.status.value}")
            print(f"   Models: {status.models_deployed}")
        
        # Demo 4: Health check
        print(f"\n🏥 Demo 4: Health status check")
        health = await deployment_manager.get_health_status()
        
        print(f"   Overall status: {health.get('overall_status', 'unknown')}")
        print(f"   Monitored deployments: {health.get('monitored_deployments', 0)}")
        print(f"   Active alerts: {health.get('active_alerts', 0)}")
        
        # Demo 5: Rollback (if deployment was successful)
        if result.status == DeploymentStatus.COMPLETED and result.rollback_available:
            print(f"\n🔄 Demo 5: Rolling back deployment")
            rollback_result = await deployment_manager.rollback_deployment(
                result.deployment_id, 
                "Demo rollback"
            )
            
            print(f"   Rollback success: {rollback_result.success}")
            if rollback_result.success:
                print(f"   Models restored: {rollback_result.models_restored}")
                print(f"   Rollback time: {rollback_result.rollback_time:.2f}s")
        
        print(f"\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        logging.error(f"Demo error: {str(e)}", exc_info=True)
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"🧹 Demo environment cleaned up")


async def demo_validation_system():
    """Demonstrate the validation system"""
    print(f"\n🔍 Validation System Demo")
    print("=" * 30)
    
    from infrastructure.deployment import ValidationService
    
    # Create temporary config for demo
    temp_dir = Path(tempfile.mkdtemp())
    config = DeploymentConfig(
        source_models_path=str(temp_dir / "source"),
        target_models_path=str(temp_dir / "target"),
        backup_path=str(temp_dir / "backup")
    )
    
    validation_service = ValidationService(config)
    
    try:
        # Demo pre-deployment validation
        print(f"📋 Running pre-deployment validation...")
        result = await validation_service.validate_pre_deployment(["demo-model"])
        
        print(f"   Overall status: {'✅ PASSED' if result.is_valid else '❌ FAILED'}")
        print(f"   Checks performed: {len(result.checks_performed)}")
        print(f"   Passed checks: {len(result.passed_checks)}")
        print(f"   Failed checks: {len(result.failed_checks)}")
        print(f"   Warnings: {len(result.warnings)}")
        
        if result.performance_metrics:
            print(f"   Performance metrics: {len(result.performance_metrics)} collected")
        
    except Exception as e:
        print(f"❌ Validation demo failed: {str(e)}")
    
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


async def demo_monitoring_system():
    """Demonstrate the monitoring system"""
    print(f"\n📈 Monitoring System Demo")
    print("=" * 30)
    
    from infrastructure.deployment import MonitoringService
    
    # Create temporary config for demo
    temp_dir = Path(tempfile.mkdtemp())
    config = DeploymentConfig(
        source_models_path=str(temp_dir / "source"),
        target_models_path=str(temp_dir / "target"),
        backup_path=str(temp_dir / "backup"),
        monitoring_enabled=True,
        health_check_interval=1  # Fast for demo
    )
    
    monitoring_service = MonitoringService(config)
    
    try:
        # Demo monitoring start
        print(f"🎯 Starting monitoring for demo deployment...")
        await monitoring_service.start_monitoring("demo-deployment", ["demo-model"])
        
        # Wait a bit for monitoring to collect data
        print(f"⏳ Collecting monitoring data...")
        await asyncio.sleep(3)
        
        # Get health status
        health_status = await monitoring_service.get_health_status()
        print(f"   Overall status: {health_status.get('overall_status', 'unknown')}")
        print(f"   Monitored deployments: {health_status.get('monitored_deployments', 0)}")
        
        # Stop monitoring
        await monitoring_service.stop_monitoring("demo-deployment")
        print(f"⏹️  Monitoring stopped")
        
    except Exception as e:
        print(f"❌ Monitoring demo failed: {str(e)}")
    
    finally:
        await monitoring_service.shutdown()
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


async def main():
    """Run all demos"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("🎭 WAN Model Deployment System - Complete Demo")
    print("=" * 60)
    
    try:
        # Run deployment demo
        await demo_deployment_system()
        
        # Run validation demo
        await demo_validation_system()
        
        # Run monitoring demo
        await demo_monitoring_system()
        
        print(f"\n🎉 All demos completed successfully!")
        print(f"\n📚 For more information, see:")
        print(f"   • docs/WAN_MODEL_DEPLOYMENT_GUIDE.md")
        print(f"   • WAN_MODEL_DEPLOYMENT_IMPLEMENTATION_SUMMARY.md")
        print(f"   • config_templates/deployment_config.json")
        
    except Exception as e:
        print(f"\n💥 Demo failed: {str(e)}")
        logging.error(f"Demo error: {str(e)}", exc_info=True)


if __name__ == '__main__':
    asyncio.run(main())