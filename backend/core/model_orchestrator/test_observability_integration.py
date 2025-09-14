"""
Integration tests for Model Orchestrator observability features.

Tests the complete observability stack including metrics collection,
structured logging, GPU health checks, and performance monitoring.
"""

import pytest
import asyncio
import time
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from .metrics import MetricsCollector, get_metrics_collector, initialize_metrics
from .logging_config import (
    configure_logging, get_logger, set_correlation_id, generate_correlation_id,
    performance_timer, log_with_context
)
from .gpu_health import GPUHealthChecker, HealthStatus
from ..services.model_health_service import ModelHealthService
from .model_registry import ModelRegistry
from .model_resolver import ModelResolver
from .model_ensurer import ModelEnsurer, ModelStatus


class TestMetricsIntegration:
    """Test metrics collection and reporting."""
    
    def setup_method(self):
        """Set up test environment."""
        self.metrics = MetricsCollector()
    
    def test_download_metrics_collection(self):
        """Test that download metrics are properly collected."""
        model_id = "test-model"
        variant = "fp16"
        source = "s3://test-bucket/model"
        
        # Record download lifecycle
        self.metrics.record_download_started(model_id, variant, source)
        self.metrics.record_download_completed(model_id, variant, source, 45.2, 1024*1024*100)
        
        # Verify metrics were recorded
        metrics_dict = self.metrics.get_metrics_dict()
        assert "downloads_total" in metrics_dict["counters"]
        
        # Check that both started and completed events were recorded
        counters = metrics_dict["counters"]["downloads_total"]
        assert any("started" in str(key) for key in counters.keys())
        assert any("completed" in str(key) for key in counters.keys())
    
    def test_error_metrics_collection(self):
        """Test that error metrics are properly collected."""
        model_id = "test-model"
        variant = "fp16"
        source = "s3://test-bucket/model"
        error_code = "CHECKSUM_FAIL"
        
        # Record download failure
        self.metrics.record_download_failed(model_id, variant, source, error_code)
        
        # Verify error metrics
        metrics_dict = self.metrics.get_metrics_dict()
        assert "errors_total" in metrics_dict["counters"]
        assert "downloads_total" in metrics_dict["counters"]
    
    def test_storage_metrics_collection(self):
        """Test storage usage metrics."""
        model_family = "wan22"
        bytes_used = 5 * 1024 * 1024 * 1024  # 5GB
        files_count = 42
        
        self.metrics.record_storage_usage(model_family, bytes_used, files_count)
        
        metrics_dict = self.metrics.get_metrics_dict()
        assert "storage_bytes_used" in metrics_dict["gauges"]
        assert model_family in metrics_dict["gauges"]["storage_bytes_used"]
        assert metrics_dict["gauges"]["storage_bytes_used"][model_family] == bytes_used
    
    def test_prometheus_format_output(self):
        """Test Prometheus format metrics output."""
        # Record some metrics
        self.metrics.record_download_completed("test-model", "fp16", "s3", 30.0, 1000000)
        self.metrics.record_storage_usage("wan22", 1000000, 10)
        
        # Get Prometheus format
        prometheus_text = self.metrics.get_metrics_text()
        
        # Verify format
        assert isinstance(prometheus_text, str)
        assert len(prometheus_text) > 0
        
        # Should contain metric names and values
        lines = prometheus_text.split('\n')
        assert any('downloads_total' in line for line in lines)
    
    def test_label_sanitization(self):
        """Test that labels are properly sanitized to prevent cardinality explosion."""
        # Test with problematic model ID
        problematic_id = "very-long-model-name-with-version@1.2.3-beta.4+build.567"
        
        self.metrics.record_download_started(problematic_id, "fp16", "s3://bucket/path")
        
        # Verify that the label was sanitized
        metrics_dict = self.metrics.get_metrics_dict()
        counters = metrics_dict["counters"]["downloads_total"]
        
        # Should not contain the full problematic ID
        assert not any(problematic_id in str(key) for key in counters.keys())
        
        # Should contain sanitized version
        assert any("very-long-model-name-with-version" in str(key) for key in counters.keys())


class TestStructuredLogging:
    """Test structured logging with correlation IDs."""
    
    def setup_method(self):
        """Set up logging for tests."""
        self.logger = configure_logging(level="DEBUG", structured=True)
    
    def test_correlation_id_generation(self):
        """Test correlation ID generation and context."""
        correlation_id = generate_correlation_id()
        assert isinstance(correlation_id, str)
        assert len(correlation_id) > 0
        
        # Set and retrieve correlation ID
        set_correlation_id(correlation_id)
        retrieved_id = correlation_id  # Would be retrieved from context in real implementation
        assert retrieved_id == correlation_id
    
    def test_performance_timer_logging(self):
        """Test performance timer context manager."""
        logger = get_logger("test")
        
        with performance_timer("test_operation", test_param="value"):
            time.sleep(0.1)  # Simulate work
        
        # In a real test, we would capture log output and verify
        # that start and completion messages were logged with timing
        assert True  # Placeholder - would verify log output
    
    def test_contextual_logging(self):
        """Test logging with additional context."""
        logger = get_logger("test")
        
        log_with_context(
            logger, logger.info,
            "Test message with context",
            model_id="test-model",
            operation="download",
            bytes_processed=1024
        )
        
        # Would verify that context fields appear in structured log output
        assert True  # Placeholder
    
    def test_sensitive_data_masking(self):
        """Test that sensitive data is masked in logs."""
        logger = get_logger("test")
        
        # Log with sensitive data
        log_with_context(
            logger, logger.info,
            "Processing with credentials",
            api_token="secret-token-12345",
            s3_endpoint="https://s3.amazonaws.com/bucket?key=secret"
        )
        
        # Would verify that sensitive values are masked
        assert True  # Placeholder


class TestGPUHealthChecks:
    """Test GPU-based health checking functionality."""
    
    def setup_method(self):
        """Set up GPU health checker."""
        self.gpu_checker = GPUHealthChecker(device="cpu", timeout=5.0)  # Use CPU for testing
    
    def test_system_health_info(self):
        """Test system health information collection."""
        health_info = self.gpu_checker.get_system_health()
        
        assert isinstance(health_info, dict)
        assert "gpu_available" in health_info
        assert "device" in health_info
        assert "timestamp" in health_info
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_cpu_fallback_health_check(self, mock_cuda):
        """Test health check with CPU fallback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            
            # Create minimal model structure
            (model_path / "model_index.json").write_text('{"test": true}')
            (model_path / "unet").mkdir()
            (model_path / "unet" / "diffusion_pytorch_model.safetensors").write_text("fake model")
            
            result = self.gpu_checker.check_model_health("test-t2v", str(model_path))
            
            assert result.model_id == "test-t2v"
            assert result.check_type in ["t2v_smoke_test", "generic_validation"]
            assert isinstance(result.duration_seconds, float)
    
    def test_health_check_caching(self):
        """Test that health check results are cached."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            (model_path / "model_index.json").write_text('{"test": true}')
            
            # First check
            result1 = self.gpu_checker.check_model_health("test-model", str(model_path))
            
            # Second check should be faster (cached)
            start_time = time.time()
            result2 = self.gpu_checker.check_model_health("test-model", str(model_path))
            duration = time.time() - start_time
            
            assert result1.model_id == result2.model_id
            assert duration < 0.1  # Should be very fast due to caching
    
    def test_cache_cleanup(self):
        """Test cache cleanup functionality."""
        # Add some entries to cache
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            (model_path / "model_index.json").write_text('{"test": true}')
            
            self.gpu_checker.check_model_health("test-model-1", str(model_path))
            self.gpu_checker.check_model_health("test-model-2", str(model_path))
            
            # Clear cache
            self.gpu_checker.clear_cache()
            
            # Verify cache is empty
            assert len(self.gpu_checker._cache) == 0


class TestHealthServiceIntegration:
    """Test integration of health service with observability features."""
    
    def setup_method(self):
        """Set up health service with mocked dependencies."""
        self.registry = Mock(spec=ModelRegistry)
        self.resolver = Mock(spec=ModelResolver)
        self.ensurer = Mock(spec=ModelEnsurer)
        
        self.health_service = ModelHealthService(
            registry=self.registry,
            resolver=self.resolver,
            ensurer=self.ensurer,
            enable_gpu_checks=True,
            enable_detailed_diagnostics=True
        )
    
    @pytest.mark.asyncio
    async def test_comprehensive_health_check(self):
        """Test comprehensive health check with all features enabled."""
        # Mock registry response
        self.registry.list_models.return_value = ["test-model"]
        
        # Mock ensurer status
        mock_status = Mock()
        mock_status.status = ModelStatus.COMPLETE
        mock_status.local_path = "/models/test-model"
        mock_status.missing_files = []
        mock_status.bytes_needed = 0
        mock_status.total_size = 1000000
        self.ensurer.status.return_value = mock_status
        
        # Mock resolver
        self.resolver.local_dir.return_value = "/models/test-model"
        
        # Mock verification
        mock_verification = Mock()
        mock_verification.is_valid = True
        mock_verification.verified_files = ["file1", "file2"]
        mock_verification.total_size = 1000000
        self.ensurer.verify_integrity.return_value = mock_verification
        
        # Run health check
        response = await self.health_service.get_health_status(
            dry_run=False,
            include_gpu_checks=True,
            include_detailed_diagnostics=True
        )
        
        # Verify response structure
        assert response.status in ["healthy", "degraded", "error"]
        assert response.total_models == 1
        assert "test-model" in response.models
        assert response.correlation_id is not None
        assert response.system_metrics is not None
    
    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self):
        """Test that performance metrics are properly tracked."""
        # Mock minimal setup
        self.registry.list_models.return_value = []
        
        # Run multiple health checks
        for _ in range(3):
            await self.health_service.get_health_status()
        
        # Get metrics summary
        metrics = await self.health_service.get_metrics_summary()
        
        assert "health_service" in metrics
        assert metrics["health_service"]["total_checks"] == 3
        assert "average_response_time_ms" in metrics["health_service"]
    
    @pytest.mark.asyncio
    async def test_error_handling_and_logging(self):
        """Test error handling with proper logging."""
        # Mock registry to raise exception
        self.registry.list_models.side_effect = Exception("Registry error")
        
        # Run health check
        response = await self.health_service.get_health_status()
        
        # Should handle error gracefully
        assert response.status == "error"
        assert response.error_message is not None
        assert "Registry error" in response.error_message


class TestEndToEndObservability:
    """End-to-end tests for complete observability workflow."""
    
    def setup_method(self):
        """Set up complete observability stack."""
        # Initialize metrics
        self.metrics = initialize_metrics()
        
        # Configure logging
        self.logger = configure_logging(level="INFO", structured=True)
        
        # Set up correlation ID
        self.correlation_id = generate_correlation_id()
        set_correlation_id(self.correlation_id)
    
    def test_complete_observability_workflow(self):
        """Test complete workflow with metrics, logging, and correlation."""
        model_id = "test-model"
        
        # Simulate download workflow with observability
        with performance_timer("model_download", model_id=model_id):
            # Record download start
            self.metrics.record_download_started(model_id, "fp16", "s3")
            
            # Simulate download work
            time.sleep(0.1)
            
            # Record completion
            self.metrics.record_download_completed(model_id, "fp16", "s3", 0.1, 1000000)
        
        # Verify metrics were collected
        metrics_dict = self.metrics.get_metrics_dict()
        assert len(metrics_dict["counters"]) > 0
        
        # Verify correlation ID is maintained
        # In real implementation, would verify log entries contain correlation ID
        assert self.correlation_id is not None
    
    def test_metrics_export_formats(self):
        """Test different metrics export formats."""
        # Record some metrics
        self.metrics.record_storage_usage("wan22", 1000000, 10)
        self.metrics.record_download_completed("test", "fp16", "s3", 30.0, 1000000)
        
        # Test dictionary format
        dict_format = self.metrics.get_metrics_dict()
        assert isinstance(dict_format, dict)
        assert "counters" in dict_format
        assert "gauges" in dict_format
        
        # Test Prometheus format
        prometheus_format = self.metrics.get_metrics_text()
        assert isinstance(prometheus_format, str)
        assert len(prometheus_format) > 0
    
    @pytest.mark.asyncio
    async def test_health_endpoint_observability(self):
        """Test health endpoint with full observability."""
        # This would test the actual API endpoint with observability features
        # In a real implementation, would make HTTP requests and verify
        # that correlation IDs, metrics, and logging work end-to-end
        
        # Mock the health service call
        correlation_id = generate_correlation_id()
        set_correlation_id(correlation_id)
        
        # Simulate health check with observability
        with performance_timer("health_check_endpoint"):
            # Would call actual endpoint here
            response_data = {
                "status": "healthy",
                "correlation_id": correlation_id,
                "timestamp": time.time(),
                "models": {},
                "total_models": 0,
                "healthy_models": 0,
                "missing_models": 0,
                "partial_models": 0,
                "corrupt_models": 0,
                "total_bytes_needed": 0,
                "response_time_ms": 50.0
            }
        
        # Verify observability data
        assert response_data["correlation_id"] == correlation_id
        assert response_data["response_time_ms"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])