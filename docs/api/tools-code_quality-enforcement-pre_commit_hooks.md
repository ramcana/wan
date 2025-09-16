---
title: tools.code_quality.enforcement.pre_commit_hooks
category: api
tags: [api, tools]
---

# tools.code_quality.enforcement.pre_commit_hooks

Pre-commit hook management for automated quality enforcement.

## Classes

### PreCommitHookManager

Manages pre-commit hooks for code quality enforcement.

#### Methods

##### __init__(self: Any, project_root: Path)

Initialize hook manager.

##### install_hooks(self: Any, config: <ast.Subscript object at 0x0000019431A3B490>) -> bool

Install pre-commit hooks for quality enforcement.

Args:
    config: Optional configuration for hooks

Returns:
    True if installation successful

##### uninstall_hooks(self: Any) -> bool

Uninstall pre-commit hooks.

##### run_hooks(self: Any, files: <ast.Subscript object at 0x0000019431A39930>) -> <ast.Subscript object at 0x0000019431A383D0>

Run pre-commit hooks manually.

Args:
    files: Optional list of files to check

Returns:
    Results of hook execution

##### update_hooks(self: Any) -> bool

Update pre-commit hooks to latest versions.

##### validate_config(self: Any) -> <ast.Subscript object at 0x0000019431A536A0>

Validate pre-commit configuration.

##### _get_default_config(self: Any) -> <ast.Subscript object at 0x0000019431A52320>

Get default pre-commit configuration.

##### _write_pre_commit_config(self: Any, config: <ast.Subscript object at 0x0000019431A52170>) -> None

Write pre-commit configuration to file.

##### _is_pre_commit_available(self: Any) -> bool

Check if pre-commit is available.

##### _install_manual_hooks(self: Any) -> bool

Install manual Git hooks when pre-commit is not available.

##### _remove_manual_hooks(self: Any) -> bool

Remove manual Git hooks.

##### _run_manual_hooks(self: Any, files: <ast.Subscript object at 0x0000019431A50220>) -> <ast.Subscript object at 0x000001942F8422C0>

Run manual quality checks.

##### get_hook_status(self: Any) -> <ast.Subscript object at 0x000001942FC73B50>

Get status of pre-commit hooks.

