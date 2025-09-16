---
title: core.model_orchestrator.model_resolver
category: api
tags: [api, core]
---

# core.model_orchestrator.model_resolver

Model path resolution with cross-platform support and atomic operations.

This module provides deterministic path resolution for WAN2.2 models with support
for variants, temporary directories, and cross-platform compatibility including
Windows long paths and WSL scenarios.

## Classes

### PathIssue

Represents a path validation issue.

#### Methods

##### __init__(self: Any, issue_type: str, message: str, suggestion: <ast.Subscript object at 0x000001942F386DA0>)



##### __repr__(self: Any) -> str



### ModelResolver

Provides deterministic path resolution for WAN2.2 models with cross-platform support.

Handles:
- Deterministic path generation from MODELS_ROOT
- Model variant support in path resolution
- Temporary directory strategy for atomic operations
- Windows long path scenarios
- Cross-platform compatibility (Windows, WSL, Unix)

#### Methods

##### __init__(self: Any, models_root: str)

Initialize the ModelResolver with a models root directory.

Args:
    models_root: Base directory for all model storage
    
Raises:
    ModelOrchestratorError: If models_root is invalid or inaccessible

##### _detect_wsl(self: Any) -> bool

Detect if running under Windows Subsystem for Linux.

##### local_dir(self: Any, model_id: str, variant: <ast.Subscript object at 0x000001942F3841F0>) -> str

Get the local directory path for a model.

Args:
    model_id: Canonical model ID (e.g., "t2v-A14B@2.2.0")
    variant: Optional variant (e.g., "fp16", "bf16")
    
Returns:
    Absolute path to the model directory
    
Raises:
    ModelOrchestratorError: If model_id is invalid or path issues exist

##### temp_dir(self: Any, model_id: str, variant: <ast.Subscript object at 0x000001942F38C580>) -> str

Get a temporary directory path for atomic downloads.

Uses pattern: {MODELS_ROOT}/.tmp/{model}@{variant}.{uuid}.partial
Ensures temp and final paths are on the same filesystem for atomic rename.

Args:
    model_id: Canonical model ID
    variant: Optional variant
    
Returns:
    Absolute path to temporary directory
    
Raises:
    ModelOrchestratorError: If path cannot be created

##### validate_path_constraints(self: Any, path: str) -> <ast.Subscript object at 0x000001942FC43BB0>

Validate path against platform-specific constraints.

Args:
    path: Path to validate
    
Returns:
    List of PathIssue objects describing any problems

##### ensure_directory_exists(self: Any, path: str) -> None

Ensure a directory exists, creating it if necessary.

Args:
    path: Directory path to create
    
Raises:
    ModelOrchestratorError: If directory cannot be created

##### _normalize_model_id(self: Any, model_id: str) -> str

Normalize model ID to a consistent format.

Args:
    model_id: Raw model ID
    
Returns:
    Normalized model ID safe for filesystem use

##### _path_on_windows_drive(self: Any, path: str) -> bool

Check if a WSL path is on a Windows drive (e.g., /mnt/c/).

Args:
    path: Path to check
    
Returns:
    True if path is on Windows drive in WSL

##### _get_invalid_path_chars(self: Any) -> set

Get set of characters invalid in paths for current platform.

Returns:
    Set of invalid characters

##### get_models_root(self: Any) -> str

Get the configured models root directory.

Returns:
    Absolute path to models root

##### is_same_filesystem(self: Any, path1: str, path2: str) -> bool

Check if two paths are on the same filesystem for atomic operations.

Args:
    path1: First path
    path2: Second path
    
Returns:
    True if paths are on same filesystem

## Constants

### WINDOWS_RESERVED_NAMES

Type: `unknown`

### MAX_PATH_WINDOWS

Type: `int`

Value: `260`

### MAX_PATH_WINDOWS_EXTENDED

Type: `int`

Value: `32767`

### MAX_PATH_UNIX

Type: `int`

Value: `4096`

