---
title: tests.test_lora_integration_simple
category: api
tags: [api, tests]
---

# tests.test_lora_integration_simple

Simple LoRA integration test without external dependencies

## Classes

### MockLoRATracker



#### Methods

##### __init__(self: Any)



##### track_lora(self: Any, task_id: str, lora_name: str, strength: float, path: str)

Track applied LoRA for a task

##### cleanup_lora(self: Any, task_id: str)

Clean up LoRA for completed task

##### get_applied_loras(self: Any)

Get currently applied LoRAs

