---
title: api.v1.endpoints.test_models_health
category: api
tags: [api, api]
---

# api.v1.endpoints.test_models_health

Tests for Model Health API endpoints

## Classes

### TestModelsHealthEndpoints

Test cases for model health API endpoints.

#### Methods

##### test_get_models_health_success(self: Any, mock_get_service: Any, client: Any, mock_health_service: Any)

Test successful health check for all models.

##### test_get_models_health_degraded_state(self: Any, mock_get_service: Any, client: Any, mock_health_service: Any)

Test health check with models in degraded state.

##### test_get_models_health_service_not_initialized(self: Any, mock_get_service: Any, client: Any)

Test health check when service is not initialized.

##### test_get_models_health_service_error(self: Any, mock_get_service: Any, client: Any, mock_health_service: Any)

Test health check when service raises an error.

##### test_get_models_health_dry_run_parameter(self: Any, mock_get_service: Any, client: Any, mock_health_service: Any)

Test that dry_run parameter is properly passed.

##### test_get_model_health_individual_success(self: Any, mock_get_service: Any, client: Any, mock_health_service: Any)

Test successful health check for individual model.

##### test_get_model_health_individual_missing(self: Any, mock_get_service: Any, client: Any, mock_health_service: Any)

Test health check for missing individual model.

##### test_get_model_health_individual_error(self: Any, mock_get_service: Any, client: Any, mock_health_service: Any)

Test individual model health check with error.

##### test_get_model_health_individual_service_not_initialized(self: Any, mock_get_service: Any, client: Any)

Test individual model health when service is not initialized.

##### test_get_model_health_individual_service_error(self: Any, mock_get_service: Any, client: Any, mock_health_service: Any)

Test individual model health when service raises an error.

##### test_health_endpoint_response_time_requirement(self: Any, client: Any)

Test that health endpoint meets response time requirements.

##### test_health_endpoint_supports_json_output(self: Any, client: Any)

Test that health endpoint returns proper JSON structure.

