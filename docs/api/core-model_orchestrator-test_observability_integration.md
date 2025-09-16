---
title: core.model_orchestrator.test_observability_integration
category: api
tags: [api, core]
---

# core.model_orchestrator.test_observability_integration

Integration tests for Model Orchestrator observability features.

Tests the complete observability stack including metrics collection,
structured logging, GPU health checks, and performance monitoring.

## Classes

### TestMetricsIntegration

Test metrics collection and reporting.

#### Methods

##### setup_method(self: Any)

Set up test environment.

##### test_download_metrics_collection(self: Any)

Test that download metrics are properly collected.

##### test_error_metrics_collection(self: Any)

Test that error metrics are properly collected.

##### test_storage_metrics_collection(self: Any)

Test storage usage metrics.

##### test_prometheus_format_output(self: Any)

Test Prometheus format metrics output.

##### test_label_sanitization(self: Any)

Test that labels are properly sanitized to prevent cardinality explosion.

### TestStructuredLogging

Test structured logging with correlation IDs.

#### Methods

##### setup_method(self: Any)

Set up logging for tests.

##### test_correlation_id_generation(self: Any)

Test correlation ID generation and context.

##### test_performance_timer_logging(self: Any)

Test performance timer context manager.

##### test_contextual_logging(self: Any)

Test logging with additional context.

##### test_sensitive_data_masking(self: Any)

Test that sensitive data is masked in logs.

### TestGPUHealthChecks

Test GPU-based health checking functionality.

#### Methods

##### setup_method(self: Any)

Set up GPU health checker.

##### test_system_health_info(self: Any)

Test system health information collection.

##### test_cpu_fallback_health_check(self: Any, mock_cuda: Any)

Test health check with CPU fallback.

##### test_health_check_caching(self: Any)

Test that health check results are cached.

##### test_cache_cleanup(self: Any)

Test cache cleanup functionality.

### TestHealthServiceIntegration

Test integration of health service with observability features.

#### Methods

##### setup_method(self: Any)

Set up health service with mocked dependencies.

### TestEndToEndObservability

End-to-end tests for complete observability workflow.

#### Methods

##### setup_method(self: Any)

Set up complete observability stack.

##### test_complete_observability_workflow(self: Any)

Test complete workflow with metrics, logging, and correlation.

##### test_metrics_export_formats(self: Any)

Test different metrics export formats.

