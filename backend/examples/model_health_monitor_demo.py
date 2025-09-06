"""
Model Health Monitor Demo
Demonstrates the functionality of the Model Health Monitor system.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Import the health monitor
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.model_health_monitor import (
    ModelHealthMonitor,
    HealthCheckConfig,
    HealthStatus,
    CorruptionType
)


async def create_sample_model(models_dir: Path, model_id: str, healthy: bool = True):
    """Create a sample model for testing"""
    model_path = models_dir / model_id
    model_path.mkdir(parents=True, exist_ok=True)
    
    if healthy:
        # Create a healthy model structure
        config_data = {
            "model_type": "text-to-video",
            "version": "1.0",
            "_class_name": "WAN22Pipeline"
        }
        (model_path / "config.json").write_text(json.dumps(config_data, indent=2))
        
        model_index_data = {
            "_class_name": "WAN22Pipeline",
            "text_encoder": ["transformers", "CLIPTextModel"],
            "unet": ["diffusers", "UNet3DConditionModel"],
            "vae": ["diffusers", "AutoencoderKL"]
        }
        (model_path / "model_index.json").write_text(json.dumps(model_index_data, indent=2))
        
        # Create component directories
        for component in ["text_encoder", "unet", "vae"]:
            comp_dir = model_path / component
            comp_dir.mkdir(exist_ok=True)
            
            comp_config = {"_class_name": f"Test{component.title()}"}
            (comp_dir / "config.json").write_text(json.dumps(comp_config, indent=2))
            
            # Create a model file
            (comp_dir / "pytorch_model.bin").write_bytes(b"fake model data" * 1000)
        
        print(f"✅ Created healthy model: {model_id}")
    else:
        # Create a corrupted model
        (model_path / "config.json").write_text("invalid json {")
        # Missing model_index.json
        (model_path / "empty_file.bin").write_bytes(b"")  # Empty file
        (model_path / "temp.tmp").write_bytes(b"temporary data")  # Temp file
        
        print(f"❌ Created corrupted model: {model_id}")


async def demonstrate_integrity_checking(monitor: ModelHealthMonitor):
    """Demonstrate model integrity checking"""
    print("\n" + "="*60)
    print("INTEGRITY CHECKING DEMONSTRATION")
    print("="*60)
    
    # Check healthy model
    print("\n1. Checking healthy model...")
    result = await monitor.check_model_integrity("healthy-t2v-model")
    
    print(f"   Model ID: {result.model_id}")
    print(f"   Health Status: {result.health_status.value}")
    print(f"   Is Healthy: {result.is_healthy}")
    print(f"   File Count: {result.file_count}")
    print(f"   Total Size: {result.total_size_mb:.2f} MB")
    print(f"   Issues: {len(result.issues)}")
    if result.issues:
        for issue in result.issues:
            print(f"     - {issue}")
    
    # Check corrupted model
    print("\n2. Checking corrupted model...")
    result = await monitor.check_model_integrity("corrupted-model")
    
    print(f"   Model ID: {result.model_id}")
    print(f"   Health Status: {result.health_status.value}")
    print(f"   Is Healthy: {result.is_healthy}")
    print(f"   Issues: {len(result.issues)}")
    for issue in result.issues:
        print(f"     - {issue}")
    
    print(f"   Corruption Types: {[ct.value for ct in result.corruption_types]}")
    print(f"   Repair Suggestions:")
    for suggestion in result.repair_suggestions:
        print(f"     - {suggestion}")
    
    # Check missing model
    print("\n3. Checking missing model...")
    result = await monitor.check_model_integrity("nonexistent-model")
    
    print(f"   Model ID: {result.model_id}")
    print(f"   Health Status: {result.health_status.value}")
    print(f"   Issues: {result.issues}")


async def demonstrate_performance_monitoring(monitor: ModelHealthMonitor):
    """Demonstrate performance monitoring"""
    print("\n" + "="*60)
    print("PERFORMANCE MONITORING DEMONSTRATION")
    print("="*60)
    
    model_id = "performance-test-model"
    
    print(f"\n1. Monitoring performance for {model_id}...")
    
    # Simulate multiple generation runs with varying performance
    performance_scenarios = [
        {
            "name": "Good Performance",
            "metrics": {
                "load_time_seconds": 5.0,
                "generation_time_seconds": 30.0,
                "memory_usage_mb": 2048.0,
                "vram_usage_mb": 8192.0,
                "cpu_usage_percent": 45.0,
                "throughput_fps": 2.5,
                "quality_score": 0.85,
                "error_rate": 0.0
            }
        },
        {
            "name": "Degraded Performance",
            "metrics": {
                "load_time_seconds": 8.0,
                "generation_time_seconds": 45.0,
                "memory_usage_mb": 3072.0,
                "vram_usage_mb": 12288.0,
                "cpu_usage_percent": 75.0,
                "throughput_fps": 1.8,
                "quality_score": 0.75,
                "error_rate": 0.05
            }
        },
        {
            "name": "Poor Performance",
            "metrics": {
                "load_time_seconds": 15.0,
                "generation_time_seconds": 120.0,
                "memory_usage_mb": 4096.0,
                "vram_usage_mb": 15000.0,
                "cpu_usage_percent": 95.0,
                "throughput_fps": 0.8,
                "quality_score": 0.60,
                "error_rate": 0.15
            }
        }
    ]
    
    for i, scenario in enumerate(performance_scenarios, 1):
        print(f"\n   Scenario {i}: {scenario['name']}")
        
        health = await monitor.monitor_model_performance(model_id, scenario['metrics'])
        
        print(f"     Overall Score: {health.overall_score:.2f}")
        print(f"     Performance Trend: {health.performance_trend}")
        
        if health.bottlenecks:
            print(f"     Bottlenecks:")
            for bottleneck in health.bottlenecks:
                print(f"       - {bottleneck}")
        
        if health.recommendations:
            print(f"     Recommendations:")
            for rec in health.recommendations:
                print(f"       - {rec}")


async def demonstrate_corruption_detection(monitor: ModelHealthMonitor):
    """Demonstrate corruption detection"""
    print("\n" + "="*60)
    print("CORRUPTION DETECTION DEMONSTRATION")
    print("="*60)
    
    models_to_check = ["healthy-t2v-model", "corrupted-model"]
    
    for model_id in models_to_check:
        print(f"\n1. Analyzing {model_id} for corruption...")
        
        report = await monitor.detect_corruption(model_id)
        
        print(f"   Corruption Detected: {report.corruption_detected}")
        
        if report.corruption_detected:
            print(f"   Severity: {report.severity}")
            print(f"   Corruption Types: {[ct.value for ct in report.corruption_types]}")
            
            if report.affected_files:
                print(f"   Affected Files:")
                for file in report.affected_files[:5]:  # Show first 5
                    print(f"     - {file}")
                if len(report.affected_files) > 5:
                    print(f"     ... and {len(report.affected_files) - 5} more")
            
            print(f"   Repair Possible: {report.repair_possible}")
            
            if report.repair_actions:
                print(f"   Repair Actions:")
                for action in report.repair_actions:
                    print(f"     - {action}")


async def demonstrate_system_health_report(monitor: ModelHealthMonitor):
    """Demonstrate system health reporting"""
    print("\n" + "="*60)
    print("SYSTEM HEALTH REPORT DEMONSTRATION")
    print("="*60)
    
    print("\nGenerating comprehensive system health report...")
    
    report = await monitor.get_health_report()
    
    print(f"\n📊 SYSTEM HEALTH SUMMARY")
    print(f"   Overall Health Score: {report.overall_health_score:.2f}/1.0")
    print(f"   Last Updated: {report.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n📈 MODEL STATUS BREAKDOWN")
    print(f"   ✅ Healthy Models: {report.models_healthy}")
    print(f"   ⚠️  Degraded Models: {report.models_degraded}")
    print(f"   ❌ Corrupted Models: {report.models_corrupted}")
    print(f"   🔍 Missing Models: {report.models_missing}")
    
    total_models = (report.models_healthy + report.models_degraded + 
                   report.models_corrupted + report.models_missing)
    print(f"   📦 Total Models: {total_models}")
    
    if report.storage_usage_percent > 0:
        print(f"\n💾 STORAGE USAGE")
        print(f"   Storage Usage: {report.storage_usage_percent:.1f}%")
    
    if report.recommendations:
        print(f"\n💡 RECOMMENDATIONS")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"   {i}. {rec}")
    
    if report.detailed_reports:
        print(f"\n📋 DETAILED MODEL REPORTS")
        for model_id, detail in report.detailed_reports.items():
            status_emoji = {
                HealthStatus.HEALTHY: "✅",
                HealthStatus.DEGRADED: "⚠️",
                HealthStatus.CORRUPTED: "❌",
                HealthStatus.MISSING: "🔍",
                HealthStatus.UNKNOWN: "❓"
            }.get(detail.health_status, "❓")
            
            print(f"   {status_emoji} {model_id}: {detail.health_status.value}")
            if detail.issues:
                print(f"      Issues: {len(detail.issues)}")


async def demonstrate_health_callbacks(monitor: ModelHealthMonitor):
    """Demonstrate health monitoring callbacks"""
    print("\n" + "="*60)
    print("HEALTH CALLBACKS DEMONSTRATION")
    print("="*60)
    
    # Set up callbacks
    health_events = []
    corruption_events = []
    
    def health_callback(result):
        health_events.append({
            'model_id': result.model_id,
            'status': result.health_status.value,
            'timestamp': datetime.now()
        })
        print(f"   🔔 Health Alert: {result.model_id} is {result.health_status.value}")
    
    def corruption_callback(report):
        corruption_events.append({
            'model_id': report.model_id,
            'detected': report.corruption_detected,
            'severity': report.severity,
            'timestamp': datetime.now()
        })
        if report.corruption_detected:
            print(f"   🚨 Corruption Alert: {report.model_id} - {report.severity} severity")
    
    # Register callbacks
    monitor.add_health_callback(health_callback)
    monitor.add_corruption_callback(corruption_callback)
    
    print("\n1. Registered health and corruption callbacks")
    print("2. Running health checks to trigger callbacks...")
    
    # Trigger callbacks by running checks
    await monitor.check_model_integrity("healthy-t2v-model")
    await monitor.check_model_integrity("corrupted-model")
    await monitor.detect_corruption("corrupted-model")
    
    print(f"\n📊 CALLBACK SUMMARY")
    print(f"   Health Events: {len(health_events)}")
    print(f"   Corruption Events: {len(corruption_events)}")


async def main():
    """Main demonstration function"""
    print("🏥 MODEL HEALTH MONITOR DEMONSTRATION")
    print("=" * 80)
    
    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        models_dir = Path(temp_dir)
        
        # Configure health monitor
        config = HealthCheckConfig(
            check_interval_hours=24,
            performance_monitoring_enabled=True,
            automatic_repair_enabled=True,
            corruption_detection_enabled=True
        )
        
        # Initialize health monitor
        monitor = ModelHealthMonitor(str(models_dir), config)
        
        print(f"📁 Using temporary models directory: {models_dir}")
        
        # Create sample models
        print("\n🔧 Setting up demo models...")
        await create_sample_model(models_dir, "healthy-t2v-model", healthy=True)
        await create_sample_model(models_dir, "healthy-i2v-model", healthy=True)
        await create_sample_model(models_dir, "corrupted-model", healthy=False)
        
        # Run demonstrations
        await demonstrate_integrity_checking(monitor)
        await demonstrate_performance_monitoring(monitor)
        await demonstrate_corruption_detection(monitor)
        await demonstrate_system_health_report(monitor)
        await demonstrate_health_callbacks(monitor)
        
        # Cleanup
        await monitor.cleanup()
        
        print("\n" + "="*80)
        print("✅ MODEL HEALTH MONITOR DEMONSTRATION COMPLETED")
        print("="*80)
        
        print("\n📝 SUMMARY:")
        print("   • Integrity checking detects missing files and corruption")
        print("   • Performance monitoring tracks generation metrics and trends")
        print("   • Corruption detection identifies various types of model issues")
        print("   • System health reports provide comprehensive status overview")
        print("   • Callbacks enable real-time monitoring and alerting")
        print("   • Automatic repair can fix common issues")


if __name__ == "__main__":
    asyncio.run(main())