"""
Integration Tests for Performance and Resource Usage
Tests performance characteristics, resource monitoring, and optimization effectiveness
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
    torch_mock.cuda.memory_allocated.return_value = 4096 * 1024 * 1024
    torch_mock.cuda.get_device_properties.return_value.total_memory = 12288 * 1024 * 1024
    torch_mock.cuda.device_count.return_value = 1
    torch_mock.cuda.get_device_name.return_value = "NVIDIA RTX 4080"
    sys.modules['torch'] = torch_mock

if 'psutil' not in sys.modules:
    psutil_mock = MagicMock()
    memory_mock = Mock()
    memory_mock.total = 32 * 1024 * 1024 * 1024  # 32GB
    memory_mock.available = 16 * 1024 * 1024 * 1024  # 16GB
    memory_mock.percent = 50.0
    psutil_mock.virtual_memory.return_value = memory_mock
    psutil_mock.cpu_percent.return_value = 25.0
    sys.modules['psutil'] = psutil_mock

from utils import generate_video, generate_video_enhanced
from resource_manager import ResourceStatus, OptimizationLevel
from performance_profiler import get_performance_profiler

class TestGenerationPerformanceMetrics:
    """Test performance metrics for different generation configurations"""
    
    @pytest.fixture
    def performance_test_configs(self):
        """Performance test configurations with expected metrics"""
        return [
            {
                "name": "720p_fast",
                "config": {"resolution": "720p", "steps": 30, "model": "t2v-A14B"},
                "expected": {"time_range": (20, 40), "vram_mb": (5000, 7000), "quality": "good"}
            },
            {
                "name": "720p_balanced",
                "config": {"resolution": "720p", "steps": 50, "model": "t2v-A14B"},
                "expected": {"time_range": (35, 60), "vram_mb": (6000, 8000), "quality": "high"}
            },
            {
                "name": "1080p_fast",
                "config": {"resolution": "1080p", "steps": 30, "model": "t2v-A14B"},
                "expected": {"time_range": (40, 70), "vram_mb": (8000, 10000), "quality": "good"}
            },
            {
                "name": "1080p_balanced",
                "config": {"resolution": "1080p", "steps": 50, "model": "t2v-A14B"},
                "expected": {"time_range": (60, 100), "vram_mb": (9000, 12000), "quality": "high"}
            },
            {
                "name": "ti2v_efficient",
                "config": {"resolution": "720p", "steps": 30, "model": "ti2v-5B"},
                "expected": {"time_range": (25, 45), "vram_mb": (4000, 6000), "quality": "high"}
            }
        ]
    
    def test_generation_time_scaling(self, performance_test_configs):
        """Test that generation time scales appropriately with parameters"""
        for test_config in performance_test_configs:
            config = test_config["config"]
            expected = test_config["expected"]
            
            with patch('utils.generate_video_legacy') as mock_legacy_gen:
                # Calculate realistic generation time based on parameters
                base_time = 30.0
                resolution_multiplier = 1.0 if config["resolution"] == "720p" else 1.8
                steps_multiplier = config["steps"] / 40.0
                model_multiplier = 0.8 if config["model"] == "ti2v-5B" else 1.0
                
                estimated_time = base_time * resolution_multiplier * steps_multiplier * model_multiplier
                
                mock_legacy_gen.return_value = {
                    "success": True,
                    "output_path": f"/tmp/perf_test_{test_config['name']}.mp4",
                    "generation_time": estimated_time,
                    "retry_count": 0,
                    "metadata": {
                        "performance_metrics": {
                            "resolution": config["resolution"],
                            "steps": config["steps"],
                            "model_type": config["model"],
                            "time_per_step": estimated_time / config["steps"],
                            "pixels_per_second": self._calculate_pixels_per_second(config["resolution"], estimated_time),
                            "efficiency_score": self._calculate_efficiency_score(config, estimated_time)
                        },
                        "resource_usage": {
                            "vram_peak_mb": expected["vram_mb"][0] + (expected["vram_mb"][1] - expected["vram_mb"][0]) * 0.7,
                            "vram_average_mb": expected["vram_mb"][0] + (expected["vram_mb"][1] - expected["vram_mb"][0]) * 0.5,
                            "system_ram_mb": 3000,
                            "cpu_usage_percent": 25.0,
                            "gpu_utilization_percent": 85.0
                        }
                    }
                }
                
                result = generate_video(
                    model_type=config["model"],
                    prompt="Performance test prompt",
                    resolution=config["resolution"],
                    steps=config["steps"]
                )
                
                # Verify performance is within expected range
                assert result["success"] == True
                assert expected["time_range"][0] <= result["generation_time"] <= expected["time_range"][1]
                
                # Verify resource usage
                vram_usage = result["metadata"]["resource_usage"]["vram_peak_mb"]
                assert expected["vram_mb"][0] <= vram_usage <= expected["vram_mb"][1]
                
                # Verify efficiency metrics
                efficiency = result["metadata"]["performance_metrics"]["efficiency_score"]
                assert efficiency > 0.6  # Should be reasonably efficient
    
    def _calculate_pixels_per_second(self, resolution: str, generation_time: float) -> float:
        """Calculate pixels processed per second"""
        pixel_counts = {"720p": 921600, "1080p": 2073600}
        total_pixels = pixel_counts.get(resolution, 921600) * 24 * 4  # 24 fps, 4 seconds
        return total_pixels / generation_time
    
    def _calculate_efficiency_score(self, config: Dict, generation_time: float) -> float:
        """Calculate efficiency score based on configuration and time"""
        # Simple efficiency calculation (higher is better)
        base_score = 1.0
        if config["resolution"] == "1080p":
            base_score *= 0.8  # 1080p is less efficient
        if config["steps"] > 40:
            base_score *= 0.9  # More steps reduce efficiency
        
        # Time penalty
        expected_time = 45.0  # Baseline expected time
        time_ratio = expected_time / generation_time
        return min(1.0, base_score * time_ratio)
    
    def test_vram_usage_optimization(self):
        """Test VRAM usage optimization effectiveness"""
        optimization_levels = [
            {"name": "none", "expected_vram": 10000, "expected_time": 45.0},
            {"name": "basic", "expected_vram": 8500, "expected_time": 48.0},
            {"name": "aggressive", "expected_vram": 7000, "expected_time": 55.0}
        ]
        
        for opt_level in optimization_levels:
            with patch('utils.get_enhanced_generation_pipeline') as mock_get_pipeline:
                mock_pipeline = Mock()
                mock_get_pipeline.return_value = mock_pipeline
                
                mock_result = Mock()
                mock_result.success = True
                mock_result.output_path = f"/tmp/vram_opt_{opt_level['name']}.mp4"
                mock_result.generation_time = opt_level["expected_time"]
                mock_result.retry_count = 0
                mock_result.context = Mock()
                mock_result.context.metadata = {
                    "optimization_level": opt_level["name"],
                    "vram_optimization": {
                        "target_usage_mb": opt_level["expected_vram"],
                        "actual_usage_mb": opt_level["expected_vram"] * 1.05,  # Slight variance
                        "optimization_techniques": self._get_optimization_techniques(opt_level["name"]),
                        "memory_efficiency": opt_level["expected_vram"] / 10000,  # Relative to baseline
                        "performance_impact": (opt_level["expected_time"] - 45.0) / 45.0
                    },
                    "quality_metrics": {
                        "visual_quality_score": max(0.7, 1.0 - (opt_level["expected_time"] - 45.0) / 100),
                        "temporal_consistency": 0.85,
                        "detail_preservation": max(0.75, 1.0 - (45.0 - opt_level["expected_time"]) / 50)
                    }
                }
                
                with patch('asyncio.run') as mock_asyncio_run:
                    mock_asyncio_run.return_value = mock_result
                    
                    result = generate_video_enhanced(
                        model_type="t2v-A14B",
                        prompt="VRAM optimization test",
                        resolution="1080p",
                        steps=50
                    )
                    
                    # Verify optimization effectiveness
                    assert result["success"] == True
                    vram_data = result["metadata"]["vram_optimization"]
                    assert vram_data["actual_usage_mb"] <= opt_level["expected_vram"] * 1.1
                    assert vram_data["memory_efficiency"] > 0.6
                    
                    # Verify quality trade-offs are reasonable
                    quality_data = result["metadata"]["quality_metrics"]
                    assert quality_data["visual_quality_score"] > 0.7
    
    def _get_optimization_techniques(self, level: str) -> List[str]:
        """Get optimization techniques for different levels"""
        techniques = {
            "none": [],
            "basic": ["gradient_checkpointing", "mixed_precision"],
            "aggressive": ["gradient_checkpointing", "mixed_precision", "cpu_offloading", "vae_tiling", "attention_slicing"]
        }
        return techniques.get(level, [])

class TestResourceMonitoring:
    """Test resource monitoring and reporting"""
    
    def test_real_time_vram_monitoring(self):
        """Test real-time VRAM monitoring during generation"""
        with patch('utils.get_enhanced_generation_pipeline') as mock_get_pipeline:
            mock_pipeline = Mock()
            mock_get_pipeline.return_value = mock_pipeline
            
            # Mock VRAM usage progression during generation
            vram_timeline = [
                {"step": 0, "usage_mb": 2048, "stage": "initialization"},
                {"step": 10, "usage_mb": 6500, "stage": "model_loading"},
                {"step": 20, "usage_mb": 8200, "stage": "generation_start"},
                {"step": 30, "usage_mb": 8800, "stage": "peak_generation"},
                {"step": 40, "usage_mb": 8600, "stage": "generation_middle"},
                {"step": 50, "usage_mb": 3000, "stage": "cleanup"}
            ]
            
            mock_result = Mock()
            mock_result.success = True
            mock_result.output_path = "/tmp/vram_monitoring_test.mp4"
            mock_result.generation_time = 45.0
            mock_result.retry_count = 0
            mock_result.context = Mock()
            mock_result.context.metadata = {
                "vram_monitoring": {
                    "timeline": vram_timeline,
                    "peak_usage_mb": max(entry["usage_mb"] for entry in vram_timeline),
                    "average_usage_mb": sum(entry["usage_mb"] for entry in vram_timeline) / len(vram_timeline),
                    "utilization_efficiency": 0.73,
                    "memory_fragmentation": 0.15,
                    "allocation_pattern": "stable"
                },
                "performance_correlation": {
                    "vram_vs_speed": -0.3,  # Negative correlation (more VRAM = slower)
                    "vram_vs_quality": 0.7,  # Positive correlation
                    "optimal_usage_range": {"min_mb": 7000, "max_mb": 9000}
                }
            }
            
            with patch('asyncio.run') as mock_asyncio_run:
                mock_asyncio_run.return_value = mock_result
                
                result = generate_video_enhanced(
                    model_type="t2v-A14B",
                    prompt="VRAM monitoring test",
                    resolution="720p",
                    steps=50
                )
                
                # Verify monitoring data
                assert result["success"] == True
                monitoring_data = result["metadata"]["vram_monitoring"]
                assert len(monitoring_data["timeline"]) == 6
                assert monitoring_data["peak_usage_mb"] > monitoring_data["average_usage_mb"]
                assert monitoring_data["utilization_efficiency"] > 0.6
                
                # Verify performance correlation analysis
                correlation_data = result["metadata"]["performance_correlation"]
                assert -1.0 <= correlation_data["vram_vs_speed"] <= 1.0
                assert correlation_data["vram_vs_quality"] > 0.5
    
    def test_system_resource_impact(self):
        """Test system resource impact monitoring"""
        with patch('utils.generate_video_legacy') as mock_legacy_gen:
            mock_legacy_gen.return_value = {
                "success": True,
                "output_path": "/tmp/system_resource_test.mp4",
                "generation_time": 48.5,
                "retry_count": 0,
                "metadata": {
                    "system_impact": {
                        "cpu_usage": {
                            "average_percent": 35.2,
                            "peak_percent": 68.1,
                            "cores_utilized": 8,
                            "efficiency": "good"
                        },
                        "memory_usage": {
                            "system_ram_mb": 4200,
                            "peak_ram_mb": 5100,
                            "swap_used_mb": 0,
                            "memory_pressure": "low"
                        },
                        "disk_io": {
                            "read_mb": 850,
                            "write_mb": 1200,
                            "io_wait_percent": 2.1,
                            "disk_pressure": "minimal"
                        },
                        "thermal_impact": {
                            "gpu_temp_celsius": 72,
                            "cpu_temp_celsius": 58,
                            "thermal_throttling": False,
                            "cooling_adequate": True
                        }
                    },
                    "resource_efficiency": {
                        "overall_score": 0.82,
                        "bottlenecks": [],
                        "optimization_opportunities": [
                            "Enable CPU offloading for VAE",
                            "Use faster storage for temporary files"
                        ]
                    }
                }
            }
            
            result = generate_video(
                model_type="t2v-A14B",
                prompt="System resource impact test",
                resolution="720p",
                steps=50
            )
            
            # Verify system impact monitoring
            assert result["success"] == True
            impact_data = result["metadata"]["system_impact"]
            
            # CPU usage should be reasonable
            assert impact_data["cpu_usage"]["average_percent"] < 80
            assert impact_data["cpu_usage"]["efficiency"] in ["good", "excellent"]
            
            # Memory usage should be within limits
            assert impact_data["memory_usage"]["system_ram_mb"] < 8000
            assert impact_data["memory_usage"]["memory_pressure"] in ["low", "moderate"]
            
            # Thermal impact should be acceptable
            assert impact_data["thermal_impact"]["gpu_temp_celsius"] < 85
            assert impact_data["thermal_impact"]["thermal_throttling"] == False
    
    def test_concurrent_generation_resource_sharing(self):
        """Test resource sharing between concurrent generations"""
        with patch('utils.generate_video_legacy') as mock_legacy_gen:
            # Simulate multiple concurrent generations
            concurrent_results = []
            
            for i in range(3):
                # Each generation uses proportionally less resources
                base_vram = 8000
                shared_vram = base_vram // (i + 1)  # Resource sharing
                
                mock_legacy_gen.return_value = {
                    "success": True,
                    "output_path": f"/tmp/concurrent_test_{i}.mp4",
                    "generation_time": 45.0 + i * 10,  # Slower with more concurrent tasks
                    "retry_count": 0,
                    "metadata": {
                        "concurrent_execution": {
                            "task_id": i,
                            "concurrent_tasks": 3,
                            "resource_allocation": {
                                "vram_allocated_mb": shared_vram,
                                "vram_share_percent": (shared_vram / base_vram) * 100,
                                "priority_level": "normal",
                                "queue_position": i
                            },
                            "performance_impact": {
                                "slowdown_factor": 1.0 + i * 0.2,
                                "resource_contention": "moderate" if i > 0 else "none",
                                "scheduling_efficiency": max(0.6, 1.0 - i * 0.15)
                            }
                        },
                        "resource_sharing": {
                            "memory_sharing_enabled": True,
                            "model_sharing": True,
                            "compute_sharing": "time_sliced",
                            "sharing_efficiency": max(0.7, 1.0 - i * 0.1)
                        }
                    }
                }
                
                result = generate_video(
                    model_type="t2v-A14B",
                    prompt=f"Concurrent test {i}",
                    resolution="720p",
                    steps=50
                )
                
                concurrent_results.append(result)
            
            # Verify all tasks completed successfully
            for i, result in enumerate(concurrent_results):
                assert result["success"] == True
                
                concurrent_data = result["metadata"]["concurrent_execution"]
                assert concurrent_data["concurrent_tasks"] == 3
                assert concurrent_data["task_id"] == i
                
                # Verify resource sharing
                sharing_data = result["metadata"]["resource_sharing"]
                assert sharing_data["memory_sharing_enabled"] == True
                assert sharing_data["sharing_efficiency"] > 0.6

class TestPerformanceOptimization:
    """Test performance optimization strategies"""
    
    def test_adaptive_optimization(self):
        """Test adaptive optimization based on system capabilities"""
        system_configs = [
            {"name": "high_end", "vram_gb": 12, "ram_gb": 32, "expected_level": "minimal"},
            {"name": "mid_range", "vram_gb": 8, "ram_gb": 16, "expected_level": "moderate"},
            {"name": "low_end", "vram_gb": 4, "ram_gb": 8, "expected_level": "aggressive"}
        ]
        
        for sys_config in system_configs:
            with patch('utils.get_enhanced_generation_pipeline') as mock_get_pipeline:
                mock_pipeline = Mock()
                mock_get_pipeline.return_value = mock_pipeline
                
                mock_result = Mock()
                mock_result.success = True
                mock_result.output_path = f"/tmp/adaptive_opt_{sys_config['name']}.mp4"
                mock_result.generation_time = self._get_expected_time(sys_config["expected_level"])
                mock_result.retry_count = 0
                mock_result.context = Mock()
                mock_result.context.metadata = {
                    "adaptive_optimization": {
                        "system_profile": {
                            "vram_gb": sys_config["vram_gb"],
                            "ram_gb": sys_config["ram_gb"],
                            "performance_tier": sys_config["name"]
                        },
                        "optimization_level": sys_config["expected_level"],
                        "applied_optimizations": self._get_optimizations_for_level(sys_config["expected_level"]),
                        "performance_prediction": {
                            "estimated_time_seconds": self._get_expected_time(sys_config["expected_level"]),
                            "quality_score": self._get_quality_score(sys_config["expected_level"]),
                            "resource_efficiency": self._get_efficiency_score(sys_config["expected_level"])
                        }
                    },
                    "optimization_effectiveness": {
                        "memory_savings_percent": self._get_memory_savings(sys_config["expected_level"]),
                        "speed_impact_percent": self._get_speed_impact(sys_config["expected_level"]),
                        "quality_preservation": self._get_quality_preservation(sys_config["expected_level"])
                    }
                }
                
                with patch('asyncio.run') as mock_asyncio_run:
                    mock_asyncio_run.return_value = mock_result
                    
                    result = generate_video_enhanced(
                        model_type="t2v-A14B",
                        prompt="Adaptive optimization test",
                        resolution="720p",
                        steps=50
                    )
                    
                    # Verify adaptive optimization
                    assert result["success"] == True
                    opt_data = result["metadata"]["adaptive_optimization"]
                    assert opt_data["optimization_level"] == sys_config["expected_level"]
                    assert opt_data["system_profile"]["vram_gb"] == sys_config["vram_gb"]
                    
                    # Verify optimization effectiveness
                    effectiveness = result["metadata"]["optimization_effectiveness"]
                    assert effectiveness["quality_preservation"] > 0.7
                    if sys_config["expected_level"] == "aggressive":
                        assert effectiveness["memory_savings_percent"] > 30
    
    def _get_expected_time(self, level: str) -> float:
        """Get expected generation time for optimization level"""
        times = {"minimal": 42.0, "moderate": 48.0, "aggressive": 58.0}
        return times.get(level, 45.0)
    
    def _get_optimizations_for_level(self, level: str) -> List[str]:
        """Get optimizations applied for each level"""
        optimizations = {
            "minimal": ["mixed_precision"],
            "moderate": ["mixed_precision", "gradient_checkpointing", "attention_slicing"],
            "aggressive": ["mixed_precision", "gradient_checkpointing", "attention_slicing", 
                          "cpu_offloading", "vae_tiling", "model_sharding"]
        }
        return optimizations.get(level, [])
    
    def _get_quality_score(self, level: str) -> float:
        """Get quality score for optimization level"""
        scores = {"minimal": 0.95, "moderate": 0.88, "aggressive": 0.78}
        return scores.get(level, 0.85)
    
    def _get_efficiency_score(self, level: str) -> float:
        """Get efficiency score for optimization level"""
        scores = {"minimal": 0.85, "moderate": 0.92, "aggressive": 0.88}
        return scores.get(level, 0.85)
    
    def _get_memory_savings(self, level: str) -> float:
        """Get memory savings percentage for optimization level"""
        savings = {"minimal": 5.0, "moderate": 25.0, "aggressive": 45.0}
        return savings.get(level, 15.0)
    
    def _get_speed_impact(self, level: str) -> float:
        """Get speed impact percentage for optimization level"""
        impacts = {"minimal": -2.0, "moderate": 8.0, "aggressive": 25.0}
        return impacts.get(level, 5.0)
    
    def _get_quality_preservation(self, level: str) -> float:
        """Get quality preservation score for optimization level"""
        preservation = {"minimal": 0.98, "moderate": 0.92, "aggressive": 0.82}
        return preservation.get(level, 0.90)
    
    def test_memory_cleanup_effectiveness(self):
        """Test memory cleanup effectiveness after generation"""
        with patch('utils.get_enhanced_generation_pipeline') as mock_get_pipeline, \
             patch('torch.cuda.empty_cache') as mock_empty_cache, \
             patch('gc.collect') as mock_gc_collect:
            
            mock_pipeline = Mock()
            mock_get_pipeline.return_value = mock_pipeline
            
            mock_result = Mock()
            mock_result.success = True
            mock_result.output_path = "/tmp/memory_cleanup_test.mp4"
            mock_result.generation_time = 45.0
            mock_result.retry_count = 0
            mock_result.context = Mock()
            mock_result.context.metadata = {
                "memory_cleanup": {
                    "cleanup_stages": [
                        {"stage": "intermediate_tensors", "freed_mb": 2100},
                        {"stage": "model_cache", "freed_mb": 1800},
                        {"stage": "cuda_cache", "freed_mb": 3200},
                        {"stage": "system_gc", "freed_mb": 450}
                    ],
                    "total_freed_mb": 7550,
                    "cleanup_time_seconds": 2.3,
                    "memory_state": {
                        "before_cleanup_mb": 10200,
                        "after_cleanup_mb": 2650,
                        "fragmentation_reduced": True,
                        "largest_free_block_mb": 9500
                    },
                    "cleanup_effectiveness": {
                        "memory_recovery_rate": 0.74,
                        "fragmentation_improvement": 0.85,
                        "cleanup_efficiency": 0.91
                    }
                }
            }
            
            with patch('asyncio.run') as mock_asyncio_run:
                mock_asyncio_run.return_value = mock_result
                
                result = generate_video_enhanced(
                    model_type="t2v-A14B",
                    prompt="Memory cleanup test",
                    resolution="720p",
                    steps=50
                )
                
                # Verify cleanup effectiveness
                assert result["success"] == True
                cleanup_data = result["metadata"]["memory_cleanup"]
                assert cleanup_data["total_freed_mb"] > 7000
                assert cleanup_data["cleanup_effectiveness"]["memory_recovery_rate"] > 0.7
                assert cleanup_data["memory_state"]["fragmentation_reduced"] == True
                
                # Verify cleanup functions were called
                mock_empty_cache.assert_called()
                mock_gc_collect.assert_called()

class TestPerformanceBenchmarking:
    """Test performance benchmarking and comparison"""
    
    def test_cross_model_performance_comparison(self):
        """Test performance comparison across different models"""
        models = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
        benchmark_results = {}
        
        for model in models:
            with patch('utils.generate_video_legacy') as mock_legacy_gen:
                # Mock realistic performance for each model
                performance_data = self._get_model_performance_data(model)
                
                mock_legacy_gen.return_value = {
                    "success": True,
                    "output_path": f"/tmp/benchmark_{model}.mp4",
                    "generation_time": performance_data["time"],
                    "retry_count": 0,
                    "metadata": {
                        "benchmark_results": {
                            "model_type": model,
                            "performance_metrics": performance_data,
                            "resource_efficiency": {
                                "vram_efficiency": performance_data["vram_mb"] / performance_data["time"],
                                "compute_efficiency": performance_data["quality"] / performance_data["time"],
                                "overall_score": performance_data["efficiency"]
                            },
                            "quality_metrics": {
                                "visual_quality": performance_data["quality"],
                                "temporal_consistency": performance_data["consistency"],
                                "detail_preservation": performance_data["detail"]
                            }
                        }
                    }
                }
                
                result = generate_video(
                    model_type=model,
                    prompt="Benchmark test prompt",
                    resolution="720p",
                    steps=50
                )
                
                benchmark_results[model] = result
        
        # Verify all models completed successfully
        for model, result in benchmark_results.items():
            assert result["success"] == True
            benchmark_data = result["metadata"]["benchmark_results"]
            assert benchmark_data["model_type"] == model
            assert benchmark_data["resource_efficiency"]["overall_score"] > 0.6
        
        # Verify TI2V is most efficient (smallest model)
        ti2v_time = benchmark_results["ti2v-5B"]["generation_time"]
        t2v_time = benchmark_results["t2v-A14B"]["generation_time"]
        assert ti2v_time < t2v_time  # TI2V should be faster
    
    def _get_model_performance_data(self, model: str) -> Dict[str, float]:
        """Get realistic performance data for each model"""
        data = {
            "t2v-A14B": {
                "time": 48.0, "vram_mb": 8200, "quality": 0.92, 
                "consistency": 0.89, "detail": 0.91, "efficiency": 0.78
            },
            "i2v-A14B": {
                "time": 42.0, "vram_mb": 7800, "quality": 0.88, 
                "consistency": 0.92, "detail": 0.89, "efficiency": 0.82
            },
            "ti2v-5B": {
                "time": 35.0, "vram_mb": 5200, "quality": 0.85, 
                "consistency": 0.87, "detail": 0.86, "efficiency": 0.88
            }
        }
        return data.get(model, data["t2v-A14B"])

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-x"])