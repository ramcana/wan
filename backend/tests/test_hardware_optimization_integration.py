"""
Test hardware optimization integration with generation service
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.services.generation_service import GenerationService, VRAMMonitor


class TestHardwareOptimizationIntegration:
    """Test hardware optimization integration with generation service"""
    
    @pytest.fixture
    def mock_hardware_profile(self):
        """Mock hardware profile for testing"""
        from backend.core.services.wan22_system_optimizer import HardwareProfile
        return HardwareProfile(
            cpu_model="AMD Ryzen Threadripper PRO 5975WX",
            cpu_cores=32,
            cpu_threads=64,
            total_memory_gb=128.0,
            gpu_model="NVIDIA GeForce RTX 4080",
            vram_gb=16.0,
            cuda_version="12.1",
            driver_version="535.98",
            platform_info="Windows 11",
            detection_timestamp="2024-01-01T00:00:00"
        )
    
    @pytest.fixture
    def mock_system_optimizer(self, mock_hardware_profile):
        """Mock WAN22SystemOptimizer for testing"""
        optimizer = Mock()
        optimizer.get_hardware_profile.return_value = mock_hardware_profile
        optimizer.monitor_system_health.return_value = Mock(
            cpu_usage_percent=25.0,
            memory_usage_gb=32.0,
            vram_usage_mb=4096,
            vram_total_mb=16384,
            gpu_temperature=65.0,
            timestamp="2024-01-01T00:00:00"
        )
        optimizer.get_optimization_history.return_value = []
        return optimizer
    
    @pytest.fixture
    def generation_service(self):
        """Create generation service for testing"""
        return GenerationService()
    
    @pytest.mark.asyncio
    async def test_hardware_optimization_initialization(self, generation_service, mock_system_optimizer):
        """Test hardware optimization initialization"""
        # Mock system integration
        with patch('backend.services.generation_service.get_system_integration') as mock_get_integration:
            mock_integration = AsyncMock()
            mock_integration.get_wan22_system_optimizer.return_value = mock_system_optimizer
            mock_get_integration.return_value = mock_integration
            
            # Initialize hardware optimization
            await generation_service._initialize_hardware_optimization()
            
            # Verify initialization
            assert generation_service.wan22_system_optimizer is not None
            assert generation_service.hardware_profile is not None
            assert generation_service.hardware_profile.gpu_model == "NVIDIA GeForce RTX 4080"
            assert generation_service.hardware_profile.vram_gb == 16.0
    
    @pytest.mark.asyncio
    async def test_rtx_4080_optimizations(self, generation_service, mock_system_optimizer):
        """Test RTX 4080 specific optimizations"""
        generation_service.wan22_system_optimizer = mock_system_optimizer
        generation_service.hardware_profile = mock_system_optimizer.get_hardware_profile()
        
        # Apply hardware optimizations
        await generation_service._apply_hardware_optimizations_for_generation()
        
        # Verify RTX 4080 optimizations
        assert hasattr(generation_service, 'optimal_vram_usage_gb')
        assert generation_service.optimal_vram_usage_gb <= 14.0  # 85% of 16GB
        assert hasattr(generation_service, 'enable_tensor_cores')
        assert generation_service.enable_tensor_cores is True
        assert generation_service.optimization_applied is True
    
    @pytest.mark.asyncio
    async def test_threadripper_optimizations(self, generation_service, mock_system_optimizer):
        """Test Threadripper PRO optimizations"""
        generation_service.wan22_system_optimizer = mock_system_optimizer
        generation_service.hardware_profile = mock_system_optimizer.get_hardware_profile()
        
        # Apply hardware optimizations
        await generation_service._apply_hardware_optimizations_for_generation()
        
        # Verify Threadripper optimizations
        assert hasattr(generation_service, 'enable_cpu_multithreading')
        assert generation_service.enable_cpu_multithreading is True
        assert hasattr(generation_service, 'cpu_worker_threads')
        assert generation_service.cpu_worker_threads <= 16  # Max 16 threads
    
    @pytest.mark.asyncio
    async def test_high_memory_optimizations(self, generation_service, mock_system_optimizer):
        """Test high memory optimizations"""
        generation_service.wan22_system_optimizer = mock_system_optimizer
        generation_service.hardware_profile = mock_system_optimizer.get_hardware_profile()
        
        # Apply hardware optimizations
        await generation_service._apply_hardware_optimizations_for_generation()
        
        # Verify high memory optimizations
        assert hasattr(generation_service, 'enable_model_caching')
        assert generation_service.enable_model_caching is True
        assert hasattr(generation_service, 'max_cached_models')
        assert generation_service.max_cached_models == 3
    
    def test_vram_monitor_creation(self, mock_hardware_profile):
        """Test VRAM monitor creation and functionality"""
        vram_monitor = VRAMMonitor(
            total_vram_gb=16.0,
            optimal_usage_gb=13.6,  # 85% of 16GB
            system_optimizer=None
        )
        
        assert vram_monitor.total_vram_gb == 16.0
        assert vram_monitor.optimal_usage_gb == 13.6
        assert vram_monitor.warning_threshold == 0.9
        assert vram_monitor.critical_threshold == 0.95
    
    def test_vram_availability_check(self):
        """Test VRAM availability checking"""
        vram_monitor = VRAMMonitor(
            total_vram_gb=16.0,
            optimal_usage_gb=13.6,
            system_optimizer=None
        )
        
        # Mock current VRAM usage
        with patch.object(vram_monitor, 'get_current_vram_usage') as mock_usage:
            mock_usage.return_value = {
                "allocated_gb": 8.0,
                "usage_percent": 50.0
            }
            
            # Test sufficient VRAM
            available, message = vram_monitor.check_vram_availability(4.0)
            assert available is True
            assert "Sufficient VRAM available" in message
            
            # Test insufficient VRAM
            available, message = vram_monitor.check_vram_availability(8.0)
            assert available is False
            assert "Insufficient VRAM" in message
    
    def test_vram_optimization_suggestions(self):
        """Test VRAM optimization suggestions"""
        vram_monitor = VRAMMonitor(
            total_vram_gb=16.0,
            optimal_usage_gb=13.6,
            system_optimizer=None
        )
        
        # Test critical VRAM usage
        with patch.object(vram_monitor, 'get_current_vram_usage') as mock_usage:
            mock_usage.return_value = {
                "allocated_gb": 13.0,
                "usage_percent": 81.25,
                "optimal_usage_percent": 95.6  # Critical level
            }
            
            suggestions = vram_monitor.get_optimization_suggestions()
            assert len(suggestions) > 0
            assert any("offloading" in s.lower() for s in suggestions)
            assert any("tile size" in s.lower() for s in suggestions)
    
    def test_vram_requirements_estimation(self, generation_service):
        """Test VRAM requirements estimation"""
        # Test T2V model estimation
        vram_req = generation_service._estimate_vram_requirements("t2v", "1280x720")
        assert vram_req > 8.0  # Should be base requirement + safety margin
        
        # Test higher resolution
        vram_req_hd = generation_service._estimate_vram_requirements("t2v", "1920x1080")
        assert vram_req_hd > vram_req  # Higher resolution should require more VRAM
        
        # Test I2V model
        vram_req_i2v = generation_service._estimate_vram_requirements("i2v", "1280x720")
        assert vram_req_i2v > 9.0  # I2V has higher base requirement
    
    @pytest.mark.asyncio
    async def test_vram_optimizations_application(self, generation_service):
        """Test automatic VRAM optimizations application"""
        # Create mock VRAM monitor
        generation_service.vram_monitor = VRAMMonitor(16.0, 13.6)
        
        # Apply VRAM optimizations
        await generation_service._apply_vram_optimizations()
        
        # Verify optimizations were applied
        assert hasattr(generation_service, 'enable_model_offloading')
        assert generation_service.enable_model_offloading is True
        assert hasattr(generation_service, 'vae_tile_size')
        assert generation_service.vae_tile_size == 256
        assert hasattr(generation_service, 'enable_gradient_checkpointing')
        assert generation_service.enable_gradient_checkpointing is True
    
    def test_queue_status_with_hardware_optimization(self, generation_service, mock_system_optimizer):
        """Test queue status includes hardware optimization information"""
        # Set up hardware optimization
        generation_service.wan22_system_optimizer = mock_system_optimizer
        generation_service.hardware_profile = mock_system_optimizer.get_hardware_profile()
        generation_service.optimization_applied = True
        generation_service.vram_monitor = VRAMMonitor(16.0, 13.6)
        
        # Mock database session
        with patch('backend.services.generation_service.SessionLocal') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.query.return_value.count.return_value = 0
            mock_db.query.return_value.filter.return_value.count.return_value = 0
            
            status = generation_service.get_queue_status()
            
            # Verify hardware optimization information is included
            assert "hardware_optimization" in status
            hw_opt = status["hardware_optimization"]
            assert hw_opt["optimizer_available"] is True
            assert hw_opt["hardware_profile_loaded"] is True
            assert hw_opt["optimization_applied"] is True
            assert hw_opt["vram_monitoring_enabled"] is True
            assert hw_opt["gpu_model"] == "NVIDIA GeForce RTX 4080"
            assert hw_opt["vram_gb"] == 16.0
    
    def test_generation_stats_with_hardware_optimization(self, generation_service, mock_system_optimizer):
        """Test generation stats include hardware optimization information"""
        # Set up hardware optimization
        generation_service.wan22_system_optimizer = mock_system_optimizer
        generation_service.vram_monitor = VRAMMonitor(16.0, 13.6)
        
        # Mock VRAM usage
        with patch.object(generation_service.vram_monitor, 'get_current_vram_usage') as mock_usage:
            mock_usage.return_value = {
                "allocated_gb": 8.0,
                "usage_percent": 50.0
            }
            
            stats = generation_service.get_generation_stats()
            
            # Verify hardware optimization stats are included
            assert "system_health" in stats
            assert "vram_monitoring" in stats
            assert stats["components_status"]["hardware_optimizer"] is True
            assert stats["components_status"]["vram_monitor"] is True
            assert stats["vram_monitoring"]["current_usage_gb"] == 8.0
            assert stats["vram_monitoring"]["optimal_usage_gb"] == 13.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
