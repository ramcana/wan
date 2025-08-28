"""
Enhanced Error Recovery System Demo

This demo shows how to use the Enhanced Error Recovery System with various
failure scenarios and recovery strategies.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

# Import the enhanced error recovery system
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_error_recovery import (
    EnhancedErrorRecovery,
    EnhancedFailureType,
    RecoveryStrategy,
    ErrorSeverity,
    ErrorContext,
    create_enhanced_error_recovery
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedErrorRecoveryDemo:
    """Demo class for Enhanced Error Recovery System"""
    
    def __init__(self):
        self.recovery_system = create_enhanced_error_recovery()
        logger.info("Enhanced Error Recovery Demo initialized")
    
    async def demo_model_download_failure(self):
        """Demo model download failure recovery"""
        logger.info("\n=== Model Download Failure Demo ===")
        
        # Simulate a model download failure
        error = Exception("Failed to download model: Connection timeout after 30 seconds")
        context = {
            "model_id": "text-to-video-model-v2",
            "operation": "download",
            "user_parameters": {
                "quality": "high",
                "resolution": "1920x1080"
            },
            "system_state": {
                "available_storage_gb": 50,
                "network_speed_mbps": 10
            },
            "correlation_id": "demo_download_001"
        }
        
        # Categorize the error
        error_context = await self.recovery_system.categorize_error(error, context)
        logger.info(f"Error categorized as: {error_context.failure_type.value}")
        logger.info(f"Severity: {error_context.severity.value}")
        
        # Attempt recovery
        recovery_result = await self.recovery_system.handle_enhanced_failure(error_context)
        
        # Display results
        logger.info(f"Recovery successful: {recovery_result.success}")
        logger.info(f"Strategy used: {recovery_result.strategy_used.value}")
        logger.info(f"User message: {recovery_result.user_message}")
        
        if recovery_result.actionable_steps:
            logger.info("Actionable steps:")
            for step in recovery_result.actionable_steps:
                logger.info(f"  - {step}")
        
        return recovery_result
    
    async def demo_vram_exhaustion(self):
        """Demo VRAM exhaustion recovery"""
        logger.info("\n=== VRAM Exhaustion Demo ===")
        
        # Simulate VRAM exhaustion
        error = Exception("CUDA out of memory: tried to allocate 2.50GiB (GPU 0; 8.00GiB total capacity)")
        context = {
            "model_id": "large-video-model",
            "operation": "generation",
            "user_parameters": {
                "resolution": "1920x1080",
                "steps": 50,
                "num_frames": 32,
                "quality": "high"
            },
            "system_state": {
                "vram_total_gb": 8,
                "vram_used_gb": 7.5
            },
            "correlation_id": "demo_vram_001"
        }
        
        # Categorize and recover
        error_context = await self.recovery_system.categorize_error(error, context)
        logger.info(f"Error categorized as: {error_context.failure_type.value}")
        
        recovery_result = await self.recovery_system.handle_enhanced_failure(error_context)
        
        # Show parameter adjustments
        if recovery_result.success and "adjusted_parameters" in recovery_result.system_changes:
            original = recovery_result.system_changes.get("original_parameters", {})
            adjusted = recovery_result.system_changes.get("adjusted_parameters", {})
            
            logger.info("Parameter adjustments made:")
            for key in adjusted:
                if key in original and original[key] != adjusted[key]:
                    logger.info(f"  {key}: {original[key]} -> {adjusted[key]}")
        
        logger.info(f"Recovery result: {recovery_result.message}")
        return recovery_result
    
    async def demo_model_corruption(self):
        """Demo model corruption detection and repair"""
        logger.info("\n=== Model Corruption Demo ===")
        
        # Simulate model corruption
        error = Exception("Model integrity check failed: checksum mismatch detected in model files")
        context = {
            "model_id": "corrupted-model-v1",
            "operation": "load",
            "system_state": {
                "model_path": "/models/corrupted-model-v1",
                "expected_checksum": "abc123",
                "actual_checksum": "def456"
            },
            "correlation_id": "demo_corruption_001"
        }
        
        # Categorize and recover
        error_context = await self.recovery_system.categorize_error(error, context)
        logger.info(f"Error categorized as: {error_context.failure_type.value}")
        logger.info(f"Severity: {error_context.severity.value}")
        
        recovery_result = await self.recovery_system.handle_enhanced_failure(error_context)
        
        logger.info(f"Automatic repair attempted: {recovery_result.success}")
        logger.info(f"User message: {recovery_result.user_message}")
        
        return recovery_result
    
    async def demo_network_connectivity_loss(self):
        """Demo network connectivity loss recovery"""
        logger.info("\n=== Network Connectivity Loss Demo ===")
        
        # Simulate network loss
        error = Exception("Network unreachable: DNS resolution failed for model repository")
        context = {
            "operation": "download",
            "model_id": "remote-model",
            "system_state": {
                "network_status": "disconnected",
                "last_successful_connection": "2024-01-15 10:30:00"
            },
            "correlation_id": "demo_network_001"
        }
        
        # Categorize and recover
        error_context = await self.recovery_system.categorize_error(error, context)
        logger.info(f"Error categorized as: {error_context.failure_type.value}")
        
        recovery_result = await self.recovery_system.handle_enhanced_failure(error_context)
        
        logger.info(f"Graceful degradation: {recovery_result.success}")
        logger.info(f"Strategy: {recovery_result.strategy_used.value}")
        logger.info(f"User guidance: {recovery_result.user_message}")
        
        return recovery_result
    
    async def demo_invalid_parameters(self):
        """Demo invalid parameter handling"""
        logger.info("\n=== Invalid Parameters Demo ===")
        
        # Simulate invalid parameters
        error = Exception("Invalid resolution specified: '2560x1440' not supported")
        context = {
            "operation": "generation",
            "user_parameters": {
                "resolution": "2560x1440",  # Invalid
                "steps": -10,  # Invalid
                "num_frames": 100,  # Invalid
                "quality": "ultra"  # Invalid
            },
            "correlation_id": "demo_params_001"
        }
        
        # Categorize and recover
        error_context = await self.recovery_system.categorize_error(error, context)
        logger.info(f"Error categorized as: {error_context.failure_type.value}")
        
        recovery_result = await self.recovery_system.handle_enhanced_failure(error_context)
        
        # Show parameter corrections
        if recovery_result.success:
            adjustments = recovery_result.system_changes.get("adjustments_made", [])
            logger.info(f"Parameter corrections made: {', '.join(adjustments)}")
            
            corrected_params = recovery_result.system_changes.get("adjusted_parameters", {})
            logger.info("Corrected parameters:")
            for key, value in corrected_params.items():
                logger.info(f"  {key}: {value}")
        
        return recovery_result
    
    async def demo_user_intervention_required(self):
        """Demo scenario requiring user intervention"""
        logger.info("\n=== User Intervention Required Demo ===")
        
        # Simulate permission denied error
        error = Exception("Permission denied: cannot write to model directory")
        context = {
            "operation": "download",
            "model_id": "restricted-model",
            "system_state": {
                "user_permissions": "read-only",
                "target_directory": "/restricted/models/"
            },
            "correlation_id": "demo_permission_001"
        }
        
        # Categorize and recover
        error_context = await self.recovery_system.categorize_error(error, context)
        logger.info(f"Error categorized as: {error_context.failure_type.value}")
        logger.info(f"Severity: {error_context.severity.value}")
        
        recovery_result = await self.recovery_system.handle_enhanced_failure(error_context)
        
        logger.info(f"Requires user action: {recovery_result.requires_user_action}")
        logger.info(f"User message: {recovery_result.user_message}")
        
        if recovery_result.actionable_steps:
            logger.info("Required actions:")
            for step in recovery_result.actionable_steps:
                logger.info(f"  - {step}")
        
        return recovery_result
    
    async def demo_recovery_metrics(self):
        """Demo recovery metrics tracking"""
        logger.info("\n=== Recovery Metrics Demo ===")
        
        # Get current metrics
        metrics = await self.recovery_system.get_recovery_metrics()
        
        logger.info(f"Total recovery attempts: {metrics.total_attempts}")
        logger.info(f"Successful recoveries: {metrics.successful_recoveries}")
        logger.info(f"Failed recoveries: {metrics.failed_recoveries}")
        
        if metrics.successful_recoveries > 0:
            success_rate = (metrics.successful_recoveries / metrics.total_attempts) * 100
            logger.info(f"Overall success rate: {success_rate:.1f}%")
        
        if metrics.strategy_success_rates:
            logger.info("Strategy success rates:")
            for strategy, rate in metrics.strategy_success_rates.items():
                logger.info(f"  {strategy.value}: {rate:.2f}")
        
        if metrics.failure_type_frequencies:
            logger.info("Failure type frequencies:")
            for failure_type, count in metrics.failure_type_frequencies.items():
                logger.info(f"  {failure_type.value}: {count}")
        
        return metrics
    
    async def run_all_demos(self):
        """Run all demo scenarios"""
        logger.info("Starting Enhanced Error Recovery System Demo")
        logger.info("=" * 60)
        
        try:
            # Run individual demos
            await self.demo_model_download_failure()
            await self.demo_vram_exhaustion()
            await self.demo_model_corruption()
            await self.demo_network_connectivity_loss()
            await self.demo_invalid_parameters()
            await self.demo_user_intervention_required()
            
            # Show final metrics
            await self.demo_recovery_metrics()
            
            logger.info("\n" + "=" * 60)
            logger.info("Enhanced Error Recovery System Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed with error: {e}")
            raise


async def main():
    """Main demo function"""
    demo = EnhancedErrorRecoveryDemo()
    await demo.run_all_demos()


if __name__ == "__main__":
    asyncio.run(main())