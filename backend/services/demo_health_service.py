"""
Demo script showing how to use the Model Health Service.

This demonstrates the basic usage of the health monitoring endpoint
for the Model Orchestrator.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path

from backend.services.model_health_service import (
    ModelHealthService, initialize_model_health_service
)
from backend.core.model_orchestrator import (
    ModelRegistry, ModelResolver, ModelEnsurer, ModelStatus, ModelStatusInfo
)


class MockModelRegistry:
    """Mock model registry for demo purposes."""
    
    def list_models(self):
        return ["t2v-A14B@2.2.0", "i2v-A14B@2.2.0", "ti2v-5b@2.2.0"]


class MockModelResolver:
    """Mock model resolver for demo purposes."""
    
    def __init__(self, models_root):
        self.models_root = models_root
    
    def local_dir(self, model_id, variant=None):
        return str(Path(self.models_root) / model_id)


class MockModelEnsurer:
    """Mock model ensurer for demo purposes."""
    
    def __init__(self, models_root):
        self.models_root = Path(models_root)
    
    def status(self, model_id, variant=None):
        """Mock status method that simulates different model states."""
        model_path = self.models_root / model_id
        
        if model_id == "t2v-A14B@2.2.0":
            # Simulate complete model
            return ModelStatusInfo(
                status=ModelStatus.COMPLETE,
                local_path=str(model_path),
                missing_files=[],
                bytes_needed=0
            )
        elif model_id == "i2v-A14B@2.2.0":
            # Simulate missing model
            return ModelStatusInfo(
                status=ModelStatus.NOT_PRESENT,
                local_path=str(model_path),
                missing_files=["config.json", "unet/model.safetensors", "vae/model.safetensors"],
                bytes_needed=8500000000  # 8.5GB
            )
        else:  # ti2v-5b@2.2.0
            # Simulate partial model
            return ModelStatusInfo(
                status=ModelStatus.PARTIAL,
                local_path=str(model_path),
                missing_files=["text_encoder/model.safetensors"],
                bytes_needed=2100000000  # 2.1GB
            )


async def demo_health_service():
    """Demonstrate the health service functionality."""
    print("🏥 Model Health Service Demo")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        models_root = Path(temp_dir)
        
        # Create some mock model directories and files
        print("📁 Setting up mock model environment...")
        
        # Create t2v model (complete)
        t2v_dir = models_root / "t2v-A14B@2.2.0"
        t2v_dir.mkdir(parents=True)
        (t2v_dir / "config.json").write_text('{"model_type": "t2v"}')
        (t2v_dir / "model_index.json").write_text('{"_class_name": "WanT2VPipeline"}')
        
        # Create verification file for t2v
        verification_data = {
            "model_id": "t2v-A14B@2.2.0",
            "verified_at": time.time() - 3600,  # 1 hour ago
            "files": [
                {"path": "config.json", "size": 25},
                {"path": "model_index.json", "size": 35}
            ]
        }
        
        verification_file = t2v_dir / ".verified.json"
        with open(verification_file, 'w') as f:
            json.dump(verification_data, f)
        
        # Create mock components
        registry = MockModelRegistry()
        resolver = MockModelResolver(models_root)
        ensurer = MockModelEnsurer(models_root)
        
        # Initialize health service
        print("🔧 Initializing health service...")
        health_service = initialize_model_health_service(
            registry=registry,
            resolver=resolver,
            ensurer=ensurer,
            timeout_ms=100.0
        )
        
        print(f"✅ Health service initialized with {health_service.timeout_ms}ms timeout")
        print()
        
        # Test overall health status
        print("🔍 Checking overall health status...")
        start_time = time.time()
        
        health_response = await health_service.get_health_status(dry_run=True)
        
        end_time = time.time()
        actual_time_ms = (end_time - start_time) * 1000
        
        print(f"⏱️  Response time: {actual_time_ms:.1f}ms (service reported: {health_response.response_time_ms:.1f}ms)")
        print(f"📊 Overall status: {health_response.status.upper()}")
        print(f"📈 Models summary:")
        print(f"   • Total: {health_response.total_models}")
        print(f"   • Healthy: {health_response.healthy_models}")
        print(f"   • Missing: {health_response.missing_models}")
        print(f"   • Partial: {health_response.partial_models}")
        print(f"   • Corrupt: {health_response.corrupt_models}")
        print(f"💾 Total bytes needed: {health_response.total_bytes_needed / (1024**3):.1f} GB")
        print()
        
        # Show individual model details
        print("📋 Individual model status:")
        for model_id, model_health in health_response.models.items():
            status_emoji = {
                "COMPLETE": "✅",
                "NOT_PRESENT": "❌",
                "PARTIAL": "⚠️",
                "CORRUPT": "💥",
                "ERROR": "🚨"
            }.get(model_health.status, "❓")
            
            print(f"   {status_emoji} {model_id}: {model_health.status}")
            
            if model_health.missing_files:
                print(f"      Missing files: {len(model_health.missing_files)}")
                for file in model_health.missing_files[:3]:  # Show first 3
                    print(f"        - {file}")
                if len(model_health.missing_files) > 3:
                    print(f"        ... and {len(model_health.missing_files) - 3} more")
            
            if model_health.bytes_needed > 0:
                print(f"      Bytes needed: {model_health.bytes_needed / (1024**3):.1f} GB")
            
            if model_health.last_verified:
                hours_ago = (time.time() - model_health.last_verified) / 3600
                print(f"      Last verified: {hours_ago:.1f} hours ago")
            
            print()
        
        # Test individual model health
        print("🔍 Testing individual model health check...")
        individual_health = await health_service.get_model_health("t2v-A14B@2.2.0")
        
        print(f"   Model: {individual_health.model_id}")
        print(f"   Status: {individual_health.status}")
        print(f"   Path: {individual_health.local_path}")
        print(f"   Last verified: {'Yes' if individual_health.last_verified else 'No'}")
        print()
        
        # Test JSON serialization
        print("📄 Testing JSON serialization...")
        json_data = health_service.to_dict(health_response)
        json_str = json.dumps(json_data, indent=2)
        print(f"   JSON size: {len(json_str)} characters")
        print(f"   Contains {len(json_data['models'])} model entries")
        print()
        
        # Performance test
        print("⚡ Performance test (multiple calls)...")
        times = []
        for i in range(5):
            start = time.time()
            await health_service.get_health_status(dry_run=True)
            end = time.time()
            times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        print(f"   Average: {avg_time:.1f}ms")
        print(f"   Min: {min_time:.1f}ms")
        print(f"   Max: {max_time:.1f}ms")
        print(f"   All under 100ms: {'✅' if max_time < 100 else '❌'}")
        print()
        
        print("🎉 Demo completed successfully!")
        print()
        print("💡 Key features demonstrated:")
        print("   • Fast health checks (<100ms for 3 models)")
        print("   • Dry-run mode (no side effects)")
        print("   • Structured JSON output")
        print("   • Individual model status")
        print("   • Verification cache reading")
        print("   • Timeout protection")
        print("   • Error handling")


if __name__ == "__main__":
    asyncio.run(demo_health_service())