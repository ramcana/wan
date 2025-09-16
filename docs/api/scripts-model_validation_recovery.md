---
title: scripts.model_validation_recovery
category: api
tags: [api, scripts]
---

# scripts.model_validation_recovery

WAN Model Validation and Recovery Script
Validates model integrity and provides recovery mechanisms for corrupted models.

## Classes

### ModelValidationRecovery

Handles model validation and recovery operations

#### Methods

##### __init__(self: Any, models_dir: str)



##### validate_model_structure(self: Any, model_type: str) -> <ast.Subscript object at 0x000001942CE11C30>

Validate basic model file structure

##### validate_model_functionality(self: Any, model_type: str) -> <ast.Subscript object at 0x000001942CCF6FB0>

Validate model can be loaded and used

##### create_model_backup(self: Any, model_type: str) -> bool

Create backup of model before recovery attempts

##### recover_from_backup(self: Any, model_type: str, backup_name: <ast.Subscript object at 0x0000019427F0C070>) -> bool

Recover model from backup

##### repair_model(self: Any, model_type: str) -> <ast.Subscript object at 0x0000019427BBDA80>

Attempt to repair a corrupted model

##### full_model_recovery(self: Any, model_type: str) -> bool

Complete model recovery process

##### validate_all_models(self: Any) -> <ast.Subscript object at 0x0000019429CB5180>

Validate all available models

##### generate_validation_report(self: Any, results: <ast.Subscript object at 0x0000019429CB6200>) -> str

Generate human-readable validation report

