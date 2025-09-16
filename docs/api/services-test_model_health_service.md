---
title: services.test_model_health_service
category: api
tags: [api, services]
---

# services.test_model_health_service

Tests for Model Health Service

## Classes

### TestModelHealthService

Test cases for ModelHealthService.

#### Methods

##### mock_registry(self: Any)

Mock model registry.

##### mock_resolver(self: Any)

Mock model resolver.

##### mock_ensurer(self: Any)

Mock model ensurer.

##### health_service(self: Any, mock_registry: Any, mock_resolver: Any, mock_ensurer: Any)

Create health service with mocked dependencies.

##### test_to_dict_conversion(self: Any, health_service: Any)

Test conversion of health response to dictionary.

##### test_global_service_initialization(self: Any, mock_registry: Any, mock_resolver: Any, mock_ensurer: Any)

Test global service initialization and retrieval.

### TestModelHealthServiceIntegration

Integration tests for ModelHealthService.

