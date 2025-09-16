---
title: core.model_orchestrator.tests.test_end_to_end_workflows
category: api
tags: [api, core]
---

# core.model_orchestrator.tests.test_end_to_end_workflows

End-to-end integration tests for complete Model Orchestrator workflows.

Tests complete user journeys from model request to ready-to-use model paths,
including all storage backends, error recovery, and concurrent access scenarios.

## Classes

### TestEndToEndWorkflows

Test complete workflows from user request to model availability.

#### Methods

##### temp_models_root(self: Any)

Create temporary models root directory.

##### sample_manifest(self: Any, temp_models_root: Any)

Create sample manifest file for testing.

##### orchestrator_components(self: Any, temp_models_root: Any, sample_manifest: Any)

Set up complete orchestrator component stack.

##### test_complete_model_download_workflow(self: Any, orchestrator_components: Any, temp_models_root: Any)

Test complete workflow: request → download → verify → ready path.

##### test_concurrent_model_access_workflow(self: Any, orchestrator_components: Any)

Test concurrent access: multiple processes requesting same model.

##### test_source_failover_workflow(self: Any, orchestrator_components: Any)

Test failover: primary source fails, secondary succeeds.

##### test_disk_space_management_workflow(self: Any, orchestrator_components: Any, temp_models_root: Any)

Test disk space management: quota exceeded triggers GC.

##### test_integrity_verification_workflow(self: Any, orchestrator_components: Any)

Test integrity verification: checksum failure triggers re-download.

##### test_resume_interrupted_download_workflow(self: Any, orchestrator_components: Any, temp_models_root: Any)

Test resume capability: interrupted download resumes correctly.

##### test_health_monitoring_workflow(self: Any, orchestrator_components: Any)

Test health monitoring: status checks without side effects.

##### test_garbage_collection_workflow(self: Any, orchestrator_components: Any, temp_models_root: Any)

Test garbage collection: LRU cleanup when quota exceeded.

### TestCrossWorkflowIntegration

Test integration between different workflow components.

#### Methods

##### test_cli_to_api_integration(self: Any, orchestrator_components: Any)

Test CLI operations integrate with API operations.

##### test_pipeline_integration_workflow(self: Any, orchestrator_components: Any)

Test integration with WAN pipeline loader.

##### test_metrics_and_monitoring_integration(self: Any, orchestrator_components: Any)

Test metrics collection during complete workflows.

