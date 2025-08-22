"""
Tests for OptimizationManager and resource management system.

Tests cover optimization strategies under different resource constraints,
chunked processing, memory estimation, and system resource analysis.
"""

import pytest
import torch
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from optimization_manager import (
    OptimizationManager, ChunkedProcessor, SystemResources, ModelRequirements,
    OptimizationPlan, OptimizationResult, get_gpu_memory_info, estimate_model_vram_usage
)


class TestOptimizationManager:
    """Test OptimizationManager functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.optimization_manager = OptimizationManager()
        
        # Mock system resources for testing
        self.mock_high_vram_system = SystemResources(
            total_vram_mb=24576,  # 24GB
            available_vram_mb=20480,  # 20GB available
            total_ram_mb=32768,
            available_ram_mb=16384,
            gpu_name="RTX 4090",
            gpu_compute_capability=(8, 9),
            cpu_cores=16,
            supports_mixed_precision=True,
            supports_cpu_offload=True
        )
        
        self.mock_low_vram_system = SystemResources(
            total_vram_mb=8192,  # 8GB
            available_vram_mb=6144,  # 6GB available
            total_ram_mb=16384,
            available_ram_mb=8192,
            gpu_name="RTX 3070",
            gpu_compute_capability=(8, 6),
            cpu_cores=8,
            supports_mixed_precision=True,
            supports_cpu_offload=True
        )
        
        self.mock_minimal_system = SystemResources(
            total_vram_mb=4096,  # 4GB
            available_vram_mb=3072,  # 3GB available
            total_ram_mb=8192,
            available_ram_mb=4096,
            gpu_name="GTX 1660",
            gpu_compute_capability=(7, 5),
            cpu_cores=4,
            supports_mixed_precision=True,
            supports_cpu_offload=True
        )
        
        # Mock model requirements
        self.mock_large_model = ModelRequirements(
            min_vram_mb=8192,
            recommended_vram_mb=12288,
            model_size_mb=10240,
            supports_mixed_precision=True,
            supports_cpu_offload=True,
            supports_chunked_processing=True,
            component_sizes={"transformer": 6144, "vae": 2048, "text_encoder": 1024}
        )
        
        self.mock_small_model = ModelRequirements(
            min_vram_mb=4096,
            recommended_vram_mb=6144,
            model_size_mb=5120,
            supports_mixed_precision=True,
            supports_cpu_offload=True,
            supports_chunked_processing=True,
            component_sizes={"transformer": 3072, "vae": 1024, "text_encoder": 512}
        )
    
    def test_config_loading(self):
        """Test configuration loading"""
        # Test default config
        manager = OptimizationManager()
        assert manager.config["vram_safety_margin_mb"] == 1024
        assert manager.config["max_chunk_size"] == 8
        
        # Test custom config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            custom_config = {
                "vram_safety_margin_mb": 2048,
                "max_chunk_size": 4,
                "performance_priority": "memory"
            }
            json.dump(custom_config, f)
            config_path = f.name
        
        try:
            manager = OptimizationManager(config_path)
            assert manager.config["vram_safety_margin_mb"] == 2048
            assert manager.config["max_chunk_size"] == 4
            assert manager.config["performance_priority"] == "memory"
        finally:
            Path(config_path).unlink()
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_count')
    def test_analyze_system_resources(self, mock_cpu_count, mock_virtual_memory,
                                    mock_memory_allocated, mock_device_props, mock_cuda_available):
        """Test system resource analysis"""
        # Mock CUDA availability
        mock_cuda_available.return_value = True
        
        # Mock GPU properties
        mock_props = Mock()
        mock_props.total_memory = 24 * 1024 * 1024 * 1024  # 24GB
        mock_props.name = "RTX 4090"
        mock_props.major = 8
        mock_props.minor = 9
        mock_device_props.return_value = mock_props
        
        # Mock memory usage
        mock_memory_allocated.return_value = 2 * 1024 * 1024 * 1024  # 2GB used
        
        # Mock system memory
        mock_memory = Mock()
        mock_memory.total = 32 * 1024 * 1024 * 1024  # 32GB
        mock_memory.available = 16 * 1024 * 1024 * 1024  # 16GB available
        mock_virtual_memory.return_value = mock_memory
        
        # Mock CPU count
        mock_cpu_count.return_value = 16
        
        # Test analysis
        resources = self.optimization_manager.analyze_system_resources()
        
        assert resources.total_vram_mb == 24576
        assert resources.available_vram_mb == 22528  # 24GB - 2GB used
        assert resources.total_ram_mb == 32768
        assert resources.available_ram_mb == 16384
        assert resources.gpu_name == "RTX 4090"
        assert resources.gpu_compute_capability == (8, 9)
        assert resources.cpu_cores == 16
        assert resources.supports_mixed_precision is True
        assert resources.supports_cpu_offload is True
    
    def test_recommend_optimizations_high_vram(self):
        """Test optimization recommendations for high VRAM system"""
        plan = self.optimization_manager.recommend_optimizations(
            self.mock_large_model, self.mock_high_vram_system
        )
        
        # Should not need optimizations
        assert plan.use_mixed_precision is False
        assert plan.enable_cpu_offload is False
        assert plan.chunk_frames is False
        assert "No optimization needed" in plan.optimization_steps[0]
    
    def test_recommend_optimizations_low_vram(self):
        """Test optimization recommendations for low VRAM system"""
        plan = self.optimization_manager.recommend_optimizations(
            self.mock_large_model, self.mock_low_vram_system
        )
        
        # Should recommend mixed precision
        assert plan.use_mixed_precision is True
        assert plan.precision_type in ["fp16", "bf16"]
        assert plan.estimated_vram_reduction > 0
        
        # May recommend CPU offload
        if plan.enable_cpu_offload:
            assert plan.offload_strategy in ["sequential", "model", "full"]
    
    def test_recommend_optimizations_minimal_system(self):
        """Test optimization recommendations for minimal system"""
        plan = self.optimization_manager.recommend_optimizations(
            self.mock_large_model, self.mock_minimal_system
        )
        
        # Should recommend aggressive optimizations
        assert plan.use_mixed_precision is True
        assert plan.enable_cpu_offload is True
        assert plan.chunk_frames is True
        assert plan.max_chunk_size <= 8
        assert len(plan.warnings) > 0  # Should have warnings about performance
    
    def test_apply_memory_optimizations(self):
        """Test applying memory optimizations to pipeline"""
        # Create mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.to = Mock(return_value=mock_pipeline)
        mock_pipeline.enable_model_cpu_offload = Mock()
        mock_pipeline.enable_sequential_cpu_offload = Mock()
        
        # Create optimization plan
        plan = OptimizationPlan(
            use_mixed_precision=True,
            precision_type="fp16",
            enable_cpu_offload=True,
            offload_strategy="sequential",
            chunk_frames=False,
            max_chunk_size=1,
            estimated_vram_reduction=0.4,
            estimated_performance_impact=0.1,
            optimization_steps=["Enable fp16", "Enable sequential offload"],
            warnings=[]
        )
        
        # Apply optimizations
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.memory_allocated', return_value=1024*1024*1024), \
             patch('torch.cuda.empty_cache'), \
             patch('gc.collect'):
            
            result = self.optimization_manager.apply_memory_optimizations(mock_pipeline, plan)
        
        # Verify optimizations were applied
        assert result.success is True
        assert "Mixed precision (fp16)" in result.applied_optimizations
        assert "Sequential CPU offloading" in result.applied_optimizations
        mock_pipeline.to.assert_called_once()
        mock_pipeline.enable_sequential_cpu_offload.assert_called_once()
    
    def test_enable_chunked_processing(self):
        """Test enabling chunked processing"""
        mock_pipeline = Mock()
        
        config = self.optimization_manager.enable_chunked_processing(mock_pipeline, chunk_size=4)
        
        assert config["enabled"] is True
        assert config["chunk_size"] == 4
        assert config["overlap_frames"] == 1
        assert config["memory_cleanup"] is True
    
    def test_estimate_memory_usage(self):
        """Test memory usage estimation"""
        generation_params = {
            "width": 512,
            "height": 512,
            "num_frames": 16,
            "batch_size": 1
        }
        
        estimate = self.optimization_manager.estimate_memory_usage(
            self.mock_large_model, generation_params
        )
        
        assert "base_model_mb" in estimate
        assert "intermediate_tensors_mb" in estimate
        assert "output_tensors_mb" in estimate
        assert "overhead_mb" in estimate
        assert "total_estimated_mb" in estimate
        assert estimate["base_model_mb"] == self.mock_large_model.model_size_mb
        assert estimate["total_estimated_mb"] > estimate["base_model_mb"]
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    @patch('torch.cuda.max_memory_allocated')
    @patch('torch.cuda.get_device_properties')
    @patch('psutil.virtual_memory')
    def test_monitor_memory_usage(self, mock_virtual_memory, mock_device_props,
                                 mock_max_allocated, mock_reserved, mock_allocated, mock_cuda_available):
        """Test memory usage monitoring"""
        # Mock CUDA
        mock_cuda_available.return_value = True
        mock_allocated.return_value = 2 * 1024 * 1024 * 1024  # 2GB
        mock_reserved.return_value = 3 * 1024 * 1024 * 1024   # 3GB
        mock_max_allocated.return_value = 4 * 1024 * 1024 * 1024  # 4GB
        
        mock_props = Mock()
        mock_props.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
        mock_device_props.return_value = mock_props
        
        # Mock system memory
        mock_memory = Mock()
        mock_memory.total = 16 * 1024 * 1024 * 1024
        mock_memory.available = 8 * 1024 * 1024 * 1024
        mock_memory.percent = 50.0
        mock_virtual_memory.return_value = mock_memory
        
        stats = self.optimization_manager.monitor_memory_usage()
        
        assert stats["gpu_allocated_mb"] == 2048
        assert stats["gpu_reserved_mb"] == 3072
        assert stats["gpu_max_allocated_mb"] == 4096
        assert stats["gpu_utilization_percent"] == 25.0  # 2GB / 8GB
        assert stats["system_available_mb"] == 8192
        assert stats["system_utilization_percent"] == 50.0
    
    def test_get_optimization_recommendations(self):
        """Test getting optimization recommendations"""
        generation_params = {
            "width": 1024,
            "height": 1024,
            "num_frames": 32
        }
        
        with patch.object(self.optimization_manager, 'analyze_system_resources',
                         return_value=self.mock_low_vram_system):
            recommendations = self.optimization_manager.get_optimization_recommendations(
                "/path/to/model", generation_params
            )
        
        assert len(recommendations) > 0
        assert any("mixed precision" in rec.lower() for rec in recommendations)
        assert any("resolution" in rec.lower() for rec in recommendations)
        assert any("frame count" in rec.lower() for rec in recommendations)


class TestChunkedProcessor:
    """Test ChunkedProcessor functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = ChunkedProcessor(chunk_size=4, overlap_frames=1)
    
    def test_calculate_chunks(self):
        """Test chunk calculation"""
        # Test normal chunking
        chunks = self.processor._calculate_chunks(10)
        expected = [(0, 5), (4, 9), (8, 10)]  # With overlap
        assert chunks == expected
        
        # Test single chunk
        chunks = self.processor._calculate_chunks(3)
        assert len(chunks) == 1
        assert chunks[0] == (0, 3)
        
        # Test exact chunk size
        chunks = self.processor._calculate_chunks(4)
        assert len(chunks) == 1
        assert chunks[0] == (0, 4)
    
    def test_generate_single_chunk(self):
        """Test single chunk generation"""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.frames = [torch.randn(3, 64, 64) for _ in range(4)]
        mock_pipeline.return_value = mock_result
        
        frames = self.processor._generate_single_chunk(
            mock_pipeline, "test prompt", 4, width=512, height=512
        )
        
        assert len(frames) == 4
        mock_pipeline.assert_called_once_with("test prompt", num_frames=4, width=512, height=512)
    
    @patch('torch.cuda.empty_cache')
    @patch('gc.collect')
    def test_process_chunked_generation(self, mock_gc, mock_empty_cache):
        """Test chunked generation process"""
        # Mock pipeline
        mock_pipeline = Mock()
        
        def mock_generation(prompt, **kwargs):
            num_frames = kwargs.get('num_frames', 4)
            result = Mock()
            result.frames = [torch.randn(3, 64, 64) for _ in range(num_frames)]
            return result
        
        mock_pipeline.side_effect = mock_generation
        
        # Test chunked processing
        frames = self.processor.process_chunked_generation(
            mock_pipeline, "test prompt", 10, width=512, height=512
        )
        
        # Should have generated frames (accounting for overlap removal)
        assert len(frames) > 0
        assert mock_pipeline.call_count > 1  # Multiple chunks
        assert mock_empty_cache.call_count > 0  # Memory cleanup
        assert mock_gc.call_count > 0
    
    def test_process_single_chunk_no_chunking(self):
        """Test processing when no chunking is needed"""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.frames = [torch.randn(3, 64, 64) for _ in range(3)]
        mock_pipeline.return_value = mock_result
        
        frames = self.processor.process_chunked_generation(
            mock_pipeline, "test prompt", 3
        )
        
        assert len(frames) == 3
        assert mock_pipeline.call_count == 1  # Single call, no chunking


class TestUtilityFunctions:
    """Test utility functions"""
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    def test_get_gpu_memory_info(self, mock_reserved, mock_allocated, mock_props, mock_cuda_available):
        """Test GPU memory info retrieval"""
        # Test with CUDA available
        mock_cuda_available.return_value = True
        mock_props.return_value.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
        mock_allocated.return_value = 2 * 1024 * 1024 * 1024  # 2GB
        mock_reserved.return_value = 3 * 1024 * 1024 * 1024   # 3GB
        
        info = get_gpu_memory_info()
        
        assert info["total"] == 8192
        assert info["allocated"] == 2048
        assert info["reserved"] == 3072
        assert info["free"] == 6144
        
        # Test without CUDA
        mock_cuda_available.return_value = False
        info = get_gpu_memory_info()
        
        assert all(value == 0 for value in info.values())
    
    def test_estimate_model_vram_usage(self):
        """Test model VRAM usage estimation"""
        # Test FP32
        usage_fp32 = estimate_model_vram_usage(1000, "fp32", 1)
        assert usage_fp32 == 1500  # 1000 + 50% overhead
        
        # Test FP16
        usage_fp16 = estimate_model_vram_usage(1000, "fp16", 1)
        assert usage_fp16 == 750  # 500 + 50% overhead
        
        # Test with batch size
        usage_batch = estimate_model_vram_usage(1000, "fp32", 2)
        assert usage_batch == 2000  # 1000 + 100% overhead (2x batch)


class TestIntegrationScenarios:
    """Integration tests for different resource constraint scenarios"""
    
    def setup_method(self):
        """Set up integration test fixtures"""
        self.optimization_manager = OptimizationManager()
    
    def test_high_memory_scenario(self):
        """Test optimization for high memory system"""
        # High-end system with plenty of VRAM
        system = SystemResources(
            total_vram_mb=24576, available_vram_mb=20480,
            total_ram_mb=65536, available_ram_mb=32768,
            gpu_name="RTX 4090", gpu_compute_capability=(8, 9),
            cpu_cores=16, supports_mixed_precision=True, supports_cpu_offload=True
        )
        
        # Large model
        model = ModelRequirements(
            min_vram_mb=8192, recommended_vram_mb=12288, model_size_mb=10240,
            supports_mixed_precision=True, supports_cpu_offload=True,
            supports_chunked_processing=True, component_sizes={}
        )
        
        plan = self.optimization_manager.recommend_optimizations(model, system)
        
        # Should not need optimizations
        assert not plan.use_mixed_precision
        assert not plan.enable_cpu_offload
        assert not plan.chunk_frames
        assert plan.estimated_vram_reduction == 0.0
    
    def test_medium_memory_scenario(self):
        """Test optimization for medium memory system"""
        # Mid-range system
        system = SystemResources(
            total_vram_mb=12288, available_vram_mb=10240,
            total_ram_mb=32768, available_ram_mb=16384,
            gpu_name="RTX 4070", gpu_compute_capability=(8, 9),
            cpu_cores=12, supports_mixed_precision=True, supports_cpu_offload=True
        )
        
        # Large model that barely fits
        model = ModelRequirements(
            min_vram_mb=8192, recommended_vram_mb=14336, model_size_mb=12288,
            supports_mixed_precision=True, supports_cpu_offload=True,
            supports_chunked_processing=True, component_sizes={}
        )
        
        plan = self.optimization_manager.recommend_optimizations(model, system)
        
        # Should recommend some optimizations
        assert plan.use_mixed_precision or plan.enable_cpu_offload
        assert plan.estimated_vram_reduction > 0
    
    def test_low_memory_scenario(self):
        """Test optimization for low memory system"""
        # Low-end system
        system = SystemResources(
            total_vram_mb=6144, available_vram_mb=4096,
            total_ram_mb=16384, available_ram_mb=8192,
            gpu_name="RTX 3060", gpu_compute_capability=(8, 6),
            cpu_cores=8, supports_mixed_precision=True, supports_cpu_offload=True
        )
        
        # Large model that doesn't fit
        model = ModelRequirements(
            min_vram_mb=8192, recommended_vram_mb=12288, model_size_mb=10240,
            supports_mixed_precision=True, supports_cpu_offload=True,
            supports_chunked_processing=True, component_sizes={}
        )
        
        plan = self.optimization_manager.recommend_optimizations(model, system)
        
        # Should recommend aggressive optimizations
        assert plan.use_mixed_precision
        assert plan.enable_cpu_offload
        # Chunking may or may not be needed depending on how effective other optimizations are
        assert plan.estimated_vram_reduction > 0.5
    
    def test_minimal_memory_scenario(self):
        """Test optimization for minimal memory system"""
        # Very low-end system
        system = SystemResources(
            total_vram_mb=4096, available_vram_mb=2048,
            total_ram_mb=8192, available_ram_mb=4096,
            gpu_name="GTX 1660", gpu_compute_capability=(7, 5),
            cpu_cores=4, supports_mixed_precision=True, supports_cpu_offload=True
        )
        
        # Model that definitely doesn't fit
        model = ModelRequirements(
            min_vram_mb=8192, recommended_vram_mb=12288, model_size_mb=10240,
            supports_mixed_precision=True, supports_cpu_offload=True,
            supports_chunked_processing=True, component_sizes={}
        )
        
        plan = self.optimization_manager.recommend_optimizations(model, system)
        
        # Should recommend maximum optimizations
        assert plan.use_mixed_precision
        assert plan.enable_cpu_offload
        assert plan.offload_strategy == "full"
        assert plan.chunk_frames
        assert plan.max_chunk_size <= 4
        assert len(plan.warnings) > 0
        assert "insufficient" in " ".join(plan.warnings).lower() or "performance" in " ".join(plan.warnings).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])