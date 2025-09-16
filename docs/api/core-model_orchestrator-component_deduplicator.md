---
title: core.model_orchestrator.component_deduplicator
category: api
tags: [api, core]
---

# core.model_orchestrator.component_deduplicator

Component Deduplication System - Content-addressed storage for shared model components.

This module implements a deduplication system that identifies common files across
models and creates hardlinks/symlinks to save disk space while maintaining
reference tracking to prevent premature deletion.

## Classes

### ComponentInfo

Information about a shared component.

### DeduplicationResult

Result of a deduplication operation.

### ComponentDeduplicator

Manages component deduplication with content-addressed storage.

Features:
- Content-addressed storage for shared components
- Hardlink/symlink creation based on platform capabilities
- Reference tracking to prevent premature deletion
- Cross-platform compatibility (Windows junctions, Unix hardlinks/symlinks)

#### Methods

##### __init__(self: Any, models_root: str)

Initialize the component deduplicator.

Args:
    models_root: Base directory for model storage

##### _check_hardlink_support(self: Any) -> bool

Check if the filesystem supports hardlinks.

##### _check_symlink_support(self: Any) -> bool

Check if the filesystem supports symbolic links.

##### _load_component_metadata(self: Any) -> None

Load component metadata from disk.

##### _save_component_metadata(self: Any) -> None

Save component metadata to disk.

##### _calculate_file_hash(self: Any, file_path: Path) -> str

Calculate SHA256 hash of a file.

##### _identify_component_type(self: Any, file_path: str) -> <ast.Subscript object at 0x00000194340BA170>

Identify the component type based on file path patterns.

##### _get_component_key(self: Any, component_type: str, content_hash: str) -> str

Generate a unique key for a component.

##### _create_shared_component_path(self: Any, component_key: str, original_path: str) -> Path

Create the path for a shared component.

##### _create_link(self: Any, source_path: Path, target_path: Path) -> bool

Create a link from target to source, choosing the best method available.

Args:
    source_path: Path to the original file (in shared storage)
    target_path: Path where the link should be created (in model directory)
    
Returns:
    True if link was created successfully

##### deduplicate_model(self: Any, model_id: str, model_path: Path) -> DeduplicationResult

Deduplicate components within a single model.

Args:
    model_id: Identifier for the model
    model_path: Path to the model directory
    
Returns:
    DeduplicationResult with statistics

##### deduplicate_across_models(self: Any, model_paths: <ast.Subscript object at 0x000001942FC8A5F0>) -> DeduplicationResult

Deduplicate components across multiple models.

Args:
    model_paths: Dictionary mapping model_id to model directory path
    
Returns:
    DeduplicationResult with statistics

##### add_model_reference(self: Any, model_id: str, model_path: Path) -> None

Add references for a model that uses shared components.

Args:
    model_id: Identifier for the model
    model_path: Path to the model directory

##### remove_model_reference(self: Any, model_id: str) -> <ast.Subscript object at 0x000001943409DB40>

Remove references for a model and return list of components that can be cleaned up.

Args:
    model_id: Identifier for the model being removed
    
Returns:
    List of component keys that have no remaining references

##### cleanup_orphaned_components(self: Any, component_keys: <ast.Subscript object at 0x000001943409DC90>) -> int

Clean up components that have no remaining references.

Args:
    component_keys: List of component keys to clean up
    
Returns:
    Number of bytes reclaimed

##### get_component_stats(self: Any) -> <ast.Subscript object at 0x0000019434094670>

Get statistics about shared components.

