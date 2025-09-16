---
title: core.model_orchestrator.garbage_collector
category: api
tags: [api, core]
---

# core.model_orchestrator.garbage_collector

Garbage Collector - Disk space management with LRU-based cleanup.

## Classes

### GCTrigger

Reasons for triggering garbage collection.

### ModelInfo

Information about a model for garbage collection.

### GCConfig

Configuration for garbage collection.

### GCResult

Result of garbage collection operation.

### DiskUsage

Disk usage information.

### GarbageCollector

Manages disk space through configurable retention policies.

#### Methods

##### __init__(self: Any, registry: ModelRegistry, resolver: ModelResolver, config: <ast.Subscript object at 0x000001942CB41A50>, component_deduplicator: Any)



##### collect(self: Any, dry_run: bool, trigger: GCTrigger) -> GCResult

Perform garbage collection with configurable policies.

##### pin_model(self: Any, model_id: str, variant: <ast.Subscript object at 0x000001942CB43160>) -> None

Pin a model to protect it from garbage collection.

##### unpin_model(self: Any, model_id: str, variant: <ast.Subscript object at 0x00000194278F5C30>) -> None

Unpin a model to allow garbage collection.

##### is_pinned(self: Any, model_id: str, variant: <ast.Subscript object at 0x00000194278F6680>) -> bool

Check if a model is pinned.

##### get_disk_usage(self: Any) -> DiskUsage

Get current disk usage information.

##### estimate_reclaimable_space(self: Any) -> int

Estimate how much space could be reclaimed by garbage collection.

##### should_trigger_gc(self: Any) -> <ast.Subscript object at 0x00000194278F5000>

Check if garbage collection should be triggered automatically.

##### _perform_collection(self: Any, dry_run: bool, trigger: GCTrigger) -> GCResult

Perform the actual garbage collection.

##### _discover_models(self: Any) -> <ast.Subscript object at 0x0000019427ADB250>

Discover all models in the models directory.

##### _analyze_model_directory(self: Any, model_dir: Path) -> <ast.Subscript object at 0x0000019427ADA590>

Analyze a model directory to extract information.

##### _select_removal_candidates(self: Any, models: <ast.Subscript object at 0x0000019427ADAB30>, dry_run: bool) -> <ast.Subscript object at 0x00000194281044C0>

Select models for removal based on configured policies.

##### _auto_pin_recent_models(self: Any, models: <ast.Subscript object at 0x0000019428104820>) -> None

Automatically pin recently accessed models.

##### _remove_model(self: Any, model: ModelInfo) -> None

Remove a model from disk.

##### _calculate_directory_size(self: Any, directory: Path) -> int

Calculate the total size of a directory.

##### _get_last_access_time(self: Any, model_dir: Path) -> float

Get the last access time for a model directory.

##### _get_verification_time(self: Any, model_dir: Path) -> <ast.Subscript object at 0x0000019427A95A20>

Get the verification time from the verification file.

##### _get_model_key(self: Any, model_id: str, variant: <ast.Subscript object at 0x0000019427A96770>) -> str

Get a unique key for a model.

##### _load_pinned_models(self: Any) -> <ast.Subscript object at 0x000001942CC89C90>

Load pinned models from disk.

##### _save_pinned_models(self: Any) -> None

Save pinned models to disk.

## Constants

### MANUAL

Type: `str`

Value: `manual`

### QUOTA_EXCEEDED

Type: `str`

Value: `quota_exceeded`

### LOW_DISK_SPACE

Type: `str`

Value: `low_disk_space`

### SCHEDULED

Type: `str`

Value: `scheduled`

