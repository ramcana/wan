"""
Intelligent Fallback Manager Demo

This script demonstrates the key features of the Intelligent Fallback Manager:
- Model compatibility scoring algorithms
- Alternative model suggestions based on requirements
- Fallback strategy decision engine with multiple options
- Request queuing for downloading models
- Estimated wait time calculations for model downloads
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.intelligent_fallback_manager import (
    IntelligentFallbackManager,
    GenerationRequirements,
    FallbackType,
    initialize_intelligent_fallback_manager
)


class DemoAvailabilityManager:
    """Demo availability manager that simulates different model states"""
    
    def __init__(self):
        self.model_states = {
            "t2v-A14B": {"status": "available", "size_mb": 12000},
            "i2v-A14B": {"status": "missing", "size_mb": 14000},
            "ti2v-5B": {"status": "available", "size_mb": 8000}
        }
    
    async def _check_single_model_availability(self, model_id: str):
        """Mock model availability check"""
        from dataclasses import dataclass
        from enum import Enum
        
        class MockStatus(Enum):
            AVAILABLE = "available"
            MISSING = "missing"
            DOWNLOADING = "downloading"
        
        @dataclass
        class MockModelStatus:
            model_id: str
            availability_status: MockStatus
            size_mb: float = 0.0
        
        state = self.model_states.get(model_id, {"status": "missing", "size_mb": 10000})
        status_enum = MockStatus.AVAILABLE if state["status"] == "available" else MockStatus.MISSING
        
        return MockModelStatus(
            model_id=model_id,
            availability_status=status_enum,
            size_mb=state["size_mb"]
        )
    
    async def get_comprehensive_model_status(self):
        """Mock comprehensive status"""
        result = {}
        for model_id in self.model_states.keys():
            result[model_id] = await self._check_single_model_availability(model_id)
        return result


async def demo_model_compatibility_scoring():
    """Demonstrate model compatibility scoring algorithms"""
    print("🔍 Model Compatibility Scoring Demo")
    print("=" * 50)
    
    # Create fallback manager with demo availability manager
    availability_manager = DemoAvailabilityManager()
    fallback_manager = IntelligentFallbackManager(availability_manager)
    
    # Test different requirement scenarios
    scenarios = [
        {
            "name": "High Quality Text-to-Video",
            "requested": "i2v-A14B",  # Missing model
            "requirements": GenerationRequirements(
                model_type="i2v-A14B",
                quality="high",
                speed="medium",
                resolution="1920x1080"
            )
        },
        {
            "name": "Fast Generation with Lower Quality",
            "requested": "t2v-A14B",  # Available model, but let's see alternatives
            "requirements": GenerationRequirements(
                model_type="t2v-A14B",
                quality="medium",
                speed="fast",
                resolution="1280x720"
            )
        },
        {
            "name": "Image-to-Video with Quality Constraints",
            "requested": "i2v-A14B",  # Missing model
            "requirements": GenerationRequirements(
                model_type="i2v-A14B",
                quality="high",
                speed="medium",
                resolution="1280x720",
                allow_alternatives=True
            )
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"   Requested Model: {scenario['requested']}")
        print(f"   Requirements: {scenario['requirements'].quality}/{scenario['requirements'].speed} @ {scenario['requirements'].resolution}")
        
        try:
            suggestion = await asyncio.wait_for(
                fallback_manager.suggest_alternative_model(
                    scenario['requested'], 
                    scenario['requirements']
                ),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            print(f"   ⏰ Timeout getting suggestion")
            continue
        except Exception as e:
            print(f"   ❌ Error getting suggestion: {e}")
            continue
        
        print(f"   ✅ Suggested Alternative: {suggestion.suggested_model}")
        print(f"   📊 Compatibility Score: {suggestion.compatibility_score:.2f}")
        print(f"   🎯 Quality Difference: {suggestion.estimated_quality_difference}")
        print(f"   💾 VRAM Requirement: {suggestion.vram_requirement_gb:.1f}GB")
        print(f"   ⏱️  Estimated Time: {suggestion.estimated_generation_time}")
        print(f"   📝 Reason: {suggestion.reason}")
        
        if suggestion.capabilities_match:
            print(f"   ✅ Matching Capabilities: {', '.join(suggestion.capabilities_match)}")
        if suggestion.capabilities_missing:
            print(f"   ❌ Missing Capabilities: {', '.join(suggestion.capabilities_missing)}")


async def demo_fallback_strategies():
    """Demonstrate fallback strategy decision engine"""
    print("\n\n🛡️  Fallback Strategy Decision Engine Demo")
    print("=" * 50)
    
    availability_manager = DemoAvailabilityManager()
    fallback_manager = IntelligentFallbackManager(availability_manager)
    
    # Test different failure scenarios
    failure_scenarios = [
        {
            "name": "Model Loading Failure",
            "model": "i2v-A14B",
            "context": {
                "failure_type": "model_loading_failure",
                "error_message": "Model not found",
                "requirements": GenerationRequirements(
                    model_type="i2v-A14B",
                    quality="high",
                    allow_alternatives=True
                )
            }
        },
        {
            "name": "VRAM Exhaustion",
            "model": "t2v-A14B",
            "context": {
                "failure_type": "vram_exhaustion",
                "error_message": "CUDA out of memory",
                "requirements": GenerationRequirements(
                    model_type="t2v-A14B",
                    quality="high",
                    resolution="1920x1080",
                    allow_quality_reduction=True
                )
            }
        },
        {
            "name": "Network Error",
            "model": "ti2v-5B",
            "context": {
                "failure_type": "network_error",
                "error_message": "Download failed: Connection timeout",
                "requirements": GenerationRequirements(
                    model_type="ti2v-5B",
                    quality="medium",
                    allow_alternatives=True
                )
            }
        }
    ]
    
    for scenario in failure_scenarios:
        print(f"\n🚨 Failure Scenario: {scenario['name']}")
        print(f"   Failed Model: {scenario['model']}")
        print(f"   Error: {scenario['context']['error_message']}")
        
        try:
            strategy = await asyncio.wait_for(
                fallback_manager.get_fallback_strategy(
                    scenario['model'], 
                    scenario['context']
                ),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            print(f"   ⏰ Timeout getting strategy")
            continue
        except Exception as e:
            print(f"   ❌ Error getting strategy: {e}")
            continue
        
        print(f"   🎯 Strategy Type: {strategy.strategy_type.value}")
        print(f"   📋 Recommended Action: {strategy.recommended_action}")
        print(f"   💬 User Message: {strategy.user_message}")
        print(f"   🎲 Confidence Score: {strategy.confidence_score:.2f}")
        
        if strategy.alternative_model:
            print(f"   🔄 Alternative Model: {strategy.alternative_model}")
        
        if strategy.estimated_wait_time:
            print(f"   ⏰ Estimated Wait Time: {strategy.estimated_wait_time}")
        
        if strategy.can_queue_request:
            print(f"   📥 Can Queue Request: Yes")
        
        if strategy.requirements_adjustments:
            print(f"   ⚙️  Requirement Adjustments: {strategy.requirements_adjustments}")


async def demo_request_queuing():
    """Demonstrate request queuing for downloading models"""
    print("\n\n📥 Request Queuing Demo")
    print("=" * 50)
    
    availability_manager = DemoAvailabilityManager()
    fallback_manager = IntelligentFallbackManager(availability_manager)
    
    # Queue several requests with different priorities
    requests = [
        {
            "model": "i2v-A14B",
            "requirements": GenerationRequirements(
                model_type="i2v-A14B",
                priority="high",
                quality="high"
            ),
            "description": "High priority image-to-video request"
        },
        {
            "model": "i2v-A14B",
            "requirements": GenerationRequirements(
                model_type="i2v-A14B",
                priority="normal",
                quality="medium"
            ),
            "description": "Normal priority request"
        }
    ]
    
    print("📝 Queuing requests...")
    queued_requests = []
    
    for i, request in enumerate(requests):
        print(f"\n   Request {i+1}: {request['description']}")
        print(f"   Model: {request['model']}, Priority: {request['requirements'].priority}")
        
        try:
            result = await asyncio.wait_for(
                fallback_manager.queue_request_for_downloading_model(
                    request['model'],
                    request['requirements']
                ),
                timeout=5.0  # 5 second timeout
            )
            
            if result.success:
                print(f"   ✅ Queued successfully - Position: {result.queue_position}")
                if result.estimated_wait_time:
                    print(f"   ⏰ Estimated Wait: {result.estimated_wait_time}")
                queued_requests.append(result.request_id)
            else:
                print(f"   ❌ Failed to queue: {result.error}")
        except asyncio.TimeoutError:
            print(f"   ⏰ Timeout queuing request")
        except Exception as e:
            print(f"   ❌ Error queuing request: {e}")
    
    # Show queue status
    print(f"\n📊 Queue Status:")
    try:
        status = await asyncio.wait_for(fallback_manager.get_queue_status(), timeout=2.0)
        print(f"   Total Requests: {status['total_queued_requests']}")
        print(f"   Queue Utilization: {status['queue_utilization']:.1%}")
        
        for model_id, model_requests in status['queue_by_model'].items():
            print(f"\n   Model: {model_id}")
            for req in model_requests:
                wait_minutes = req['wait_time_minutes']
                print(f"     - Request {req['request_id']}: {req['priority']} priority, waiting {wait_minutes:.1f}min")
    except Exception as e:
        print(f"   ❌ Error getting queue status: {e}")


async def demo_wait_time_estimation():
    """Demonstrate wait time estimation algorithms"""
    print("\n\n⏰ Wait Time Estimation Demo")
    print("=" * 50)
    
    availability_manager = DemoAvailabilityManager()
    fallback_manager = IntelligentFallbackManager(availability_manager)
    
    models_to_test = ["t2v-A14B", "i2v-A14B", "ti2v-5B", "unknown-model"]
    
    for model_id in models_to_test:
        print(f"\n🔍 Estimating wait time for: {model_id}")
        
        wait_time = await fallback_manager.estimate_wait_time(model_id)
        
        print(f"   📊 Total Wait Time: {wait_time.total_wait_time}")
        print(f"   🎯 Confidence: {wait_time.confidence}")
        
        if wait_time.download_time:
            print(f"   📥 Download Time: {wait_time.download_time}")
        
        if wait_time.queue_wait_time:
            print(f"   📋 Queue Wait Time: {wait_time.queue_wait_time}")
        
        if wait_time.queue_position > 0:
            print(f"   📍 Queue Position: {wait_time.queue_position}")
        
        print(f"   📝 Factors:")
        for factor in wait_time.factors:
            print(f"     - {factor}")


async def demo_queue_processing():
    """Demonstrate queue processing when models become available"""
    print("\n\n⚙️  Queue Processing Demo")
    print("=" * 50)
    
    availability_manager = DemoAvailabilityManager()
    fallback_manager = IntelligentFallbackManager(availability_manager)
    
    # Add some requests to queue
    print("📝 Adding requests to queue...")
    
    processed_callbacks = []
    
    async def demo_callback(request):
        processed_callbacks.append(request.request_id)
        print(f"   ✅ Processed request {request.request_id} for model {request.model_id}")
    
    # Queue requests for missing model
    req1 = GenerationRequirements(model_type="i2v-A14B", priority="high")
    req2 = GenerationRequirements(model_type="i2v-A14B", priority="normal")
    
    result1 = await fallback_manager.queue_request_for_downloading_model(
        "i2v-A14B", req1, callback=demo_callback
    )
    result2 = await fallback_manager.queue_request_for_downloading_model(
        "i2v-A14B", req2, callback=demo_callback
    )
    
    print(f"   Queued 2 requests for i2v-A14B (currently missing)")
    
    # Process queue while model is still missing
    print(f"\n🔄 Processing queue (model still missing)...")
    result = await fallback_manager.process_queue()
    print(f"   Processed: {result['processed_requests']}, Remaining: {result['remaining_requests']}")
    
    # Simulate model becoming available
    print(f"\n📥 Simulating model i2v-A14B becoming available...")
    availability_manager.model_states["i2v-A14B"]["status"] = "available"
    
    # Process queue again
    print(f"\n🔄 Processing queue (model now available)...")
    result = await fallback_manager.process_queue()
    print(f"   Processed: {result['processed_requests']}, Remaining: {result['remaining_requests']}")
    print(f"   Callbacks executed: {len(processed_callbacks)}")


async def main():
    """Run all demos"""
    print("🚀 Intelligent Fallback Manager Demo")
    print("=" * 60)
    print("This demo showcases the key features of the Intelligent Fallback Manager:")
    print("- Model compatibility scoring algorithms")
    print("- Alternative model suggestions based on requirements")
    print("- Fallback strategy decision engine with multiple options")
    print("- Request queuing for downloading models")
    print("- Estimated wait time calculations for model downloads")
    print("=" * 60)
    
    try:
        await demo_model_compatibility_scoring()
        await demo_fallback_strategies()
        await demo_request_queuing()
        await demo_wait_time_estimation()
        await demo_queue_processing()
        
        print("\n\n🎉 Demo completed successfully!")
        print("The Intelligent Fallback Manager provides comprehensive")
        print("fallback capabilities for enhanced model availability.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())