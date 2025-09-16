---
title: services.model_health_service
category: api
tags: [api, services]
---

# services.model_health_service

Model Health Service - Enhanced health monitoring with observability features.

Provides comprehensive model health checking including GPU validation,
performance metrics, and detailed diagnostics with structured logging.

## Classes

### ModelHealthInfo

Enhanced health information for a single model.

### OrchestratorHealthResponse

Enhanced orchestrator health response with observability metrics.

### ModelHealthService

Enhanced service for providing comprehensive model health monitoring.

#### Methods

##### __init__(self: Any, registry: ModelRegistry, resolver: ModelResolver, ensurer: ModelEnsurer, timeout_ms: float, enable_gpu_checks: bool, enable_detailed_diagnostics: bool)



##### to_dict(self: Any, response: OrchestratorHealthResponse) -> <ast.Subscript object at 0x000001942A247CA0>

Convert health response to dictionary for JSON serialization.

