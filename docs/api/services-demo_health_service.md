---
title: services.demo_health_service
category: api
tags: [api, services]
---

# services.demo_health_service

Demo script showing how to use the Model Health Service.

This demonstrates the basic usage of the health monitoring endpoint
for the Model Orchestrator.

## Classes

### MockModelRegistry

Mock model registry for demo purposes.

#### Methods

##### list_models(self: Any)



### MockModelResolver

Mock model resolver for demo purposes.

#### Methods

##### __init__(self: Any, models_root: Any)



##### local_dir(self: Any, model_id: Any, variant: Any)



### MockModelEnsurer

Mock model ensurer for demo purposes.

#### Methods

##### __init__(self: Any, models_root: Any)



##### status(self: Any, model_id: Any, variant: Any)

Mock status method that simulates different model states.

