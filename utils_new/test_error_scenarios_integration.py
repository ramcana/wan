"""
Integration Tests for Error Scenarios and Recovery Mechanisms
Tests various error conditions and automatic recovery strategies
"""

import pytest
import tempfile
import json
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Optional

# Mock dependencies
if 'torch' not in sys.modules:
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = True
    torch_mock.cuda.OutOfMemoryError = Exception  # Mock CUDA OOM error
    torch_mock.cuda.memory_allocated.return_value = 4096 * 1024 * 1024
    torch_mock.cuda.get_device_properties.return_value.total_memory = 12288 * 1024 * 1024
    torch_mock.cuda.empty_cache = Mock()
    sys.modules['torch'] = torch_mock

from utils import generate_video, generate_video_enhanced
from error_handler import ErrorCategory, ErrorSeverity, UserFriendlyError
from resource_manager import ResourceStatus, OptimizationLevel

class TestVRAMErrorScenarios:
    """Test VRAM-related error scenarios and recovery"""
    
    def test_vram_insufficient_with_automatic_recovery(self):
        """Test VRAM insufficient error with automatic parameter optimization"""
        with patch('utils.get_enhanced_generation_pipeline') as mock_get_pipeline:
            mock_pipeline = Mock()
            mock_get_pipeline.return_value = mock_pipeline
            
            # Mock generation that succeeds after retry with optimization
            mock_result = Mock()
            mock_result.success = True
            mock_result.output_path = "/tmp/vram_recovery_test.mp4"
            mock_result.generation_time = 48.9
            mock_result.retry_count = 2
            mock_result.context = Mock()
            mock_result.context.metadata = {
                "error_recovery_applied": True,
                "original_error": "CUDA out of memory: tried to allocate 2.50 GiB",
                "recovery_actions": [
                    "Reduced resolution from 1080p to 720p",
                    "Decreased steps from 50 to 35",
                    "Enabled CPU offloading for VAE",
                    "Applied gradient checkpointing",
                    "Cleared GPU cache"
                ],
                "optimization_results": {
                    "original_vram_requirement_mb": 11800,
                    "optimized_vram_requirement_mb": 7200,
                    "memory_savings_mb": 4600,
                    "performance_impact": "15% slower generation"
                },
                "final_parameters": {
                    "resolution": "720p",
                    "steps": 35,
                    "enable_cpu_offload": True,
                    "gradient_checkpointing": True
                }
            }
            
            with patch('asyncio.run') as mock_asyncio_run:
                mock_asyncio_run.return_value = mock_result
                
                result = generate_video_enhanced(
                    model_type="t2v-A14B",
                    prompt="A complex scene requiring high VRAM",
                    resolution="1080p",
                    steps=50,
                    guidance_scale=8.0
                )
                
                # Verify successful recovery
                assert result["success"] == True
                assert result["retry_count"] == 2
                assert result["metadata"]["error_recovery_applied"] == True
                assert "CUDA out of memory" in result["metadata"]["original_error"]
                assert len(result["metadata"]["recovery_actions"]) >= 4
                assert result["metadata"]["optimization_results"]["memory_savings_mb"] > 4000
    
    def test_vram_critical_with_aggressive_optimization(self):
        """Test critical VRAM situation requiring aggressive optimization"""
        with patch('utils.generate_video_legacy') as mock_legacy_gen:
            # First call fails with critical VRAM error
            # Second call succeeds with aggressive optimization
            call_count = 0
            def mock_generation_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                
                if call_count == 1:
                    return {
                        "success": False,
                        "error": "Critical VRAM shortage: Only 1.2GB available, need 8.5GB",
                        "recovery_suggestions": [
                            "Switch to CPU-only mode",
                            "Use int8 quantization",
                            "Reduce resolution to 480p",
                            "Use minimal steps (20-25)"
                        ]
                    }
                else:
                    return {
                        "success": True,
                        "output_path": "/tmp/vram_critical_recovery.mp4",
                        "generation_time": 125.3,  # Much slower due to aggressive optimization
                        "retry_count": 1,
                        "metadata": {
                            "aggressive_optimization_applied": True,
                            "optimization_level": "maximum",
                            "final_settings": {
                                "resolution": "480p",
                                "steps": 25,
                                "quantization": "int8",
                                "cpu_offload": "full",
                                "memory_efficient_attention": True
                            },
                            "performance_trade_offs": {
                                "quality_reduction": "moderate",
                                "speed_reduction": "significant",
                                "memory_usage_mb": 1800
                            }
                        }
                    }
            
            mock_legacy_gen.side_effect = mock_generation_side_effect
            
            result = generate_video(
                model_type="t2v-A14B",
                prompt="Test prompt for critical VRAM scenario",
                resolution="1080p",
                steps=50
            )
            
            # Verify aggressive optimization worked
            assert result["success"] == True
            assert result["retry_count"] == 1
            assert result["metadata"]["aggressive_optimization_applied"] == True
            assert result["metadata"]["final_settings"]["resolution"] == "480p"
            assert result["metadata"]["final_settings"]["steps"] == 25
            assert result["metadata"]["performance_trade_offs"]["memory_usage_mb"] < 2000
    
    def test_vram_fragmentation_error(self):
        """Test VRAM fragmentation error handling"""
        with patch('utils.get_enhanced_generation_pipeline') as mock_get_pipeline:
            mock_pipeline = Mock()
            mock_get_pipeline.return_value = mock_pipeline
            
            mock_result = Mock()
            mock_result.success = True
            mock_result.output_path = "/tmp/vram_fragmentation_fix.mp4"
            mock_result.generation_time = 52.1
            mock_result.retry_count = 1
            mock_result.context = Mock()
            mock_result.context.metadata = {
                "fragmentation_detected": True,
                "fragmentation_recovery": {
                    "cache_cleared": True,
                    "memory_defragmented": True,
                    "model_reloaded": True,
                    "available_after_cleanup_mb": 9200
                },
                "memory_management": {
                    "before_cleanup": {
                        "total_mb": 12288,
                        "allocated_mb": 8500,
                        "fragmented_mb": 2800,
                        "largest_free_block_mb": 800
                    },
                    "after_cleanup": {
                        "total_mb": 12288,
                        "allocated_mb": 3000,
                        "fragmented_mb": 200,
                        "largest_free_block_mb": 9200
                    }
                }
            }
            
            with patch('asyncio.run') as mock_asyncio_run:
                mock_asyncio_run.return_value = mock_result
                
                result = generate_video_enhanced(
                    model_type="t2v-A14B",
                    prompt="Test fragmentation recovery",
                    resolution="720p",
                    steps=50
                )
                
                assert result["success"] == True
                assert result["metadata"]["fragmentation_detected"] == True
                assert result["metadata"]["fragmentation_recovery"]["cache_cleared"] == True
                assert result["metadata"]["memory_management"]["after_cleanup"]["largest_free_block_mb"] > 8000

class TestModelLoadingErrorScenarios:
    """Test model loading error scenarios and recovery"""
    
    def test_model_not_found_with_download_recovery(self):
        """Test model not found error with automatic download"""
        with patch('utils.generate_video_legacy') as mock_legacy_gen:
            call_count = 0
            def mock_generation_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                
                if call_count == 1:
                    return {
                        "success": False,
                        "error": "Model not found: /models/t2v-A14B/model.safetensors",
                        "recovery_suggestions": [
                            "Download the model automatically",
                            "Check model directory permissions",
                            "Verify internet connection",
                            "Use alternative model if available"
                        ]
                    }
                else:
                    return {
                        "success": True,
                        "output_path": "/tmp/model_download_recovery.mp4",
                        "generation_time": 67.8,  # Longer due to download time
                        "retry_count": 1,
                        "metadata": {
                            "model_download_applied": True,
                            "download_info": {
                                "model_size_gb": 13.2,
                                "download_time_seconds": 245,
                                "download_speed_mbps": 45.2,
                                "verification_passed": True
                            },
                            "model_loading": {
                                "loading_time_seconds": 12.3,
                                "quantization_applied": "bf16",
                                "memory_usage_mb": 8400
                            }
                        }
                    }
            
            mock_legacy_gen.side_effect = mock_generation_side_effect
            
            result = generate_video(
                model_type="t2v-A14B",
                prompt="Test model download recovery",
                resolution="720p",
                steps=50
            )
            
            assert result["success"] == True
            assert result["retry_count"] == 1
            assert result["metadata"]["model_download_applied"] == True
            assert result["metadata"]["download_info"]["verification_passed"] == True
    
    def test_model_corruption_with_redownload(self):
        """Test corrupted model file with automatic redownload"""
        with patch('utils.generate_video_legacy') as mock_legacy_gen:
            mock_legacy_gen.return_value = {
                "success": False,
                "error": "Model file corrupted: Checksum mismatch detected",
                "recovery_suggestions": [
                    "Redownload the model file",
                    "Clear model cache",
                    "Verify file integrity",
                    "Check disk health"
                ]
            }
            
            result = generate_video(
                model_type="t2v-A14B",
                prompt="Test corruption handling",
                resolution="720p",
                steps=50
            )
            
            assert result["success"] == False
            assert "corrupted" in result["error"].lower()
            assert "redownload" in " ".join(result["recovery_suggestions"]).lower()
    
    def test_model_incompatibility_error(self):
        """Test model incompatibility with hardware"""
        with patch('utils.generate_video_legacy') as mock_legacy_gen:
            mock_legacy_gen.return_value = {
                "success": False,
                "error": "Model incompatible: Requires CUDA compute capability 8.0+, found 7.5",
                "recovery_suggestions": [
                    "Use CPU-only mode (significantly slower)",
                    "Try quantized model version",
                    "Upgrade GPU hardware",
                    "Use alternative model architecture"
                ]
            }
            
            result = generate_video(
                model_type="t2v-A14B",
                prompt="Test compatibility check",
                resolution="720p",
                steps=50
            )
            
            assert result["success"] == False
            assert "incompatible" in result["error"].lower()
            assert "compute capability" in result["error"]
            assert len(result["recovery_suggestions"]) >= 3

class TestGenerationPipelineErrorScenarios:
    """Test generation pipeline error scenarios"""
    
    def test_generation_timeout_handling(self):
        """Test generation timeout with recovery strategies"""
        with patch('utils.get_enhanced_generation_pipeline') as mock_get_pipeline:
            mock_pipeline = Mock()
            mock_get_pipeline.return_value = mock_pipeline
            
            mock_result = Mock()
            mock_result.success = False
            mock_result.error = Mock()
            mock_result.error.category = ErrorCategory.GENERATION_PIPELINE
            mock_result.error.message = "Generation timed out after 300 seconds"
            mock_result.error.recovery_suggestions = [
                "Reduce the number of inference steps to 30-40",
                "Use a lower resolution (720p instead of 1080p)",
                "Increase timeout limit in configuration",
                "Enable faster sampling methods",
                "Check system performance and background processes"
            ]
            mock_result.retry_count = 0
            
            with patch('asyncio.run') as mock_asyncio_run:
                mock_asyncio_run.return_value = mock_result
                
                result = generate_video_enhanced(
                    model_type="t2v-A14B",
                    prompt="A very complex scene that takes too long to generate",
                    resolution="1080p",
                    steps=100,
                    guidance_scale=9.0
                )
                
                assert result["success"] == False
                assert "timed out" in result["error"]
                assert len(result["recovery_suggestions"]) >= 4
                assert any("steps" in suggestion for suggestion in result["recovery_suggestions"])
                assert any("resolution" in suggestion for suggestion in result["recovery_suggestions"])
    
    def test_pipeline_crash_with_state_recovery(self):
        """Test pipeline crash with state recovery"""
        with patch('utils.generate_video_legacy') as mock_legacy_gen:
            call_count = 0
            def mock_generation_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                
                if call_count == 1:
                    return {
                        "success": False,
                        "error": "Pipeline crashed: Segmentation fault in diffusion step 23/50",
                        "recovery_suggestions": [
                            "Restart generation from last stable checkpoint",
                            "Reduce batch size to prevent memory issues",
                            "Enable error recovery mode",
                            "Check system stability"
                        ]
                    }
                else:
                    return {
                        "success": True,
                        "output_path": "/tmp/pipeline_crash_recovery.mp4",
                        "generation_time": 58.2,
                        "retry_count": 1,
                        "metadata": {
                            "crash_recovery_applied": True,
                            "recovery_details": {
                                "crash_point": "diffusion_step_23",
                                "recovery_method": "checkpoint_restart",
                                "state_preserved": True,
                                "progress_recovered": 0.46
                            },
                            "stability_measures": {
                                "batch_size_reduced": True,
                                "memory_monitoring": "enhanced",
                                "error_detection": "enabled",
                                "checkpoint_frequency": "every_10_steps"
                            }
                        }
                    }
            
            mock_legacy_gen.side_effect = mock_generation_side_effect
            
            result = generate_video(
                model_type="t2v-A14B",
                prompt="Test pipeline crash recovery",
                resolution="720p",
                steps=50
            )
            
            assert result["success"] == True
            assert result["retry_count"] == 1
            assert result["metadata"]["crash_recovery_applied"] == True
            assert result["metadata"]["recovery_details"]["progress_recovered"] > 0.4
    
    def test_numerical_instability_error(self):
        """Test numerical instability in generation"""
        with patch('utils.generate_video_legacy') as mock_legacy_gen:
            mock_legacy_gen.return_value = {
                "success": False,
                "error": "Numerical instability detected: NaN values in tensor computation",
                "recovery_suggestions": [
                    "Reduce guidance scale to 7.5 or lower",
                    "Use mixed precision training",
                    "Enable gradient clipping",
                    "Check input data for extreme values",
                    "Try different random seed"
                ]
            }
            
            result = generate_video(
                model_type="t2v-A14B",
                prompt="Test numerical stability",
                resolution="720p",
                steps=50,
                guidance_scale=15.0  # Very high guidance scale
            )
            
            assert result["success"] == False
            assert "numerical instability" in result["error"].lower() or "nan" in result["error"].lower()
            assert any("guidance scale" in suggestion.lower() for suggestion in result["recovery_suggestions"])

class TestFileSystemErrorScenarios:
    """Test file system related error scenarios"""
    
    def test_disk_full_error_handling(self):
        """Test disk full error during video generation"""
        with patch('utils.generate_video_legacy') as mock_legacy_gen:
            mock_legacy_gen.return_value = {
                "success": False,
                "error": "Disk full: Cannot save output video (need 2.1GB, have 0.8GB free)",
                "recovery_suggestions": [
                    "Free up disk space (need at least 3GB)",
                    "Change output directory to different drive",
                    "Reduce video quality to decrease file size",
                    "Enable temporary file cleanup",
                    "Use video compression"
                ]
            }
            
            result = generate_video(
                model_type="t2v-A14B",
                prompt="Test disk space handling",
                resolution="1080p",
                steps=50
            )
            
            assert result["success"] == False
            assert "disk full" in result["error"].lower()
            assert any("free up" in suggestion.lower() for suggestion in result["recovery_suggestions"])
            assert any("directory" in suggestion.lower() for suggestion in result["recovery_suggestions"])
    
    def test_permission_denied_error(self):
        """Test permission denied error for output directory"""
        with patch('utils.generate_video_legacy') as mock_legacy_gen:
            mock_legacy_gen.return_value = {
                "success": False,
                "error": "Permission denied: Cannot write to output directory '/outputs'",
                "recovery_suggestions": [
                    "Check directory permissions (need write access)",
                    "Run with administrator privileges",
                    "Change output directory to user-writable location",
                    "Create output directory if it doesn't exist"
                ]
            }
            
            result = generate_video(
                model_type="t2v-A14B",
                prompt="Test permission handling",
                resolution="720p",
                steps=50
            )
            
            assert result["success"] == False
            assert "permission denied" in result["error"].lower()
            assert any("permission" in suggestion.lower() for suggestion in result["recovery_suggestions"])
    
    def test_file_corruption_during_save(self):
        """Test file corruption during video save"""
        with patch('utils.generate_video_legacy') as mock_legacy_gen:
            call_count = 0
            def mock_generation_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                
                if call_count == 1:
                    return {
                        "success": False,
                        "error": "File corruption detected during video save: Invalid frame data",
                        "recovery_suggestions": [
                            "Retry with different output format",
                            "Check disk health",
                            "Use alternative codec",
                            "Enable error correction"
                        ]
                    }
                else:
                    return {
                        "success": True,
                        "output_path": "/tmp/file_corruption_recovery.mp4",
                        "generation_time": 47.3,
                        "retry_count": 1,
                        "metadata": {
                            "file_recovery_applied": True,
                            "recovery_actions": [
                                "Switched to H.264 codec",
                                "Enabled error correction",
                                "Added frame validation",
                                "Used alternative output path"
                            ],
                            "file_integrity": {
                                "checksum_verified": True,
                                "frame_count_correct": True,
                                "metadata_valid": True
                            }
                        }
                    }
            
            mock_legacy_gen.side_effect = mock_generation_side_effect
            
            result = generate_video(
                model_type="t2v-A14B",
                prompt="Test file corruption recovery",
                resolution="720p",
                steps=50
            )
            
            assert result["success"] == True
            assert result["retry_count"] == 1
            assert result["metadata"]["file_recovery_applied"] == True
            assert result["metadata"]["file_integrity"]["checksum_verified"] == True

class TestNetworkErrorScenarios:
    """Test network-related error scenarios"""
    
    def test_model_download_network_failure(self):
        """Test network failure during model download"""
        with patch('utils.generate_video_legacy') as mock_legacy_gen:
            mock_legacy_gen.return_value = {
                "success": False,
                "error": "Network error: Failed to download model (Connection timeout)",
                "recovery_suggestions": [
                    "Check internet connection",
                    "Retry download with different mirror",
                    "Use cached model if available",
                    "Download manually and place in models directory",
                    "Check firewall settings"
                ]
            }
            
            result = generate_video(
                model_type="t2v-A14B",
                prompt="Test network error handling",
                resolution="720p",
                steps=50
            )
            
            assert result["success"] == False
            assert "network error" in result["error"].lower()
            assert any("connection" in suggestion.lower() for suggestion in result["recovery_suggestions"])
    
    def test_huggingface_api_rate_limit(self):
        """Test Hugging Face API rate limit error"""
        with patch('utils.generate_video_legacy') as mock_legacy_gen:
            mock_legacy_gen.return_value = {
                "success": False,
                "error": "API rate limit exceeded: Too many requests to Hugging Face Hub",
                "recovery_suggestions": [
                    "Wait 60 seconds before retrying",
                    "Use authentication token for higher limits",
                    "Use local model cache if available",
                    "Try again during off-peak hours"
                ]
            }
            
            result = generate_video(
                model_type="t2v-A14B",
                prompt="Test rate limit handling",
                resolution="720p",
                steps=50
            )
            
            assert result["success"] == False
            assert "rate limit" in result["error"].lower()
            assert any("wait" in suggestion.lower() for suggestion in result["recovery_suggestions"])

class TestConcurrentErrorScenarios:
    """Test error scenarios in concurrent generation"""
    
    def test_resource_contention_error(self):
        """Test resource contention between concurrent generations"""
        with patch('utils.generate_video_legacy') as mock_legacy_gen:
            mock_legacy_gen.return_value = {
                "success": False,
                "error": "Resource contention: Another generation is using GPU resources",
                "recovery_suggestions": [
                    "Wait for current generation to complete",
                    "Use queue system for sequential processing",
                    "Reduce concurrent generation limit",
                    "Enable resource sharing mode"
                ]
            }
            
            result = generate_video(
                model_type="t2v-A14B",
                prompt="Test resource contention",
                resolution="720p",
                steps=50
            )
            
            assert result["success"] == False
            assert "contention" in result["error"].lower()
            assert any("queue" in suggestion.lower() for suggestion in result["recovery_suggestions"])
    
    def test_memory_leak_detection(self):
        """Test memory leak detection and recovery"""
        with patch('utils.get_enhanced_generation_pipeline') as mock_get_pipeline:
            mock_pipeline = Mock()
            mock_get_pipeline.return_value = mock_pipeline
            
            mock_result = Mock()
            mock_result.success = True
            mock_result.output_path = "/tmp/memory_leak_recovery.mp4"
            mock_result.generation_time = 55.7
            mock_result.retry_count = 1
            mock_result.context = Mock()
            mock_result.context.metadata = {
                "memory_leak_detected": True,
                "leak_recovery": {
                    "leak_source": "intermediate_tensors",
                    "memory_freed_mb": 3200,
                    "cleanup_actions": [
                        "Cleared intermediate tensor cache",
                        "Forced garbage collection",
                        "Reset CUDA memory allocator",
                        "Reloaded model with fresh state"
                    ]
                },
                "memory_monitoring": {
                    "before_cleanup_mb": 10800,
                    "after_cleanup_mb": 7600,
                    "leak_rate_mb_per_step": 45.2,
                    "monitoring_enabled": True
                }
            }
            
            with patch('asyncio.run') as mock_asyncio_run:
                mock_asyncio_run.return_value = mock_result
                
                result = generate_video_enhanced(
                    model_type="t2v-A14B",
                    prompt="Test memory leak detection",
                    resolution="720p",
                    steps=50
                )
                
                assert result["success"] == True
                assert result["metadata"]["memory_leak_detected"] == True
                assert result["metadata"]["leak_recovery"]["memory_freed_mb"] > 3000
                assert len(result["metadata"]["leak_recovery"]["cleanup_actions"]) >= 3

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-x"])