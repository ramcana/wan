---
title: core.model_orchestrator.credential_cli
category: api
tags: [api, core]
---

# core.model_orchestrator.credential_cli

Command-line interface for secure credential management.

This module provides CLI commands for managing credentials securely,
including storing, retrieving, and rotating credentials.

## Classes

### CredentialCLI

Command-line interface for credential management.

#### Methods

##### __init__(self: Any, config: <ast.Subscript object at 0x000001942C5C2440>)

Initialize credential CLI.

Args:
    config: Credential configuration

##### store_credential(self: Any, key: str, value: <ast.Subscript object at 0x000001942C5C1F60>, interactive: bool) -> bool

Store a credential securely.

Args:
    key: Credential key
    value: Credential value (if None and interactive, will prompt)
    interactive: Whether to prompt for value if not provided
    
Returns:
    True if stored successfully, False otherwise

##### get_credential(self: Any, key: str, show_value: bool) -> <ast.Subscript object at 0x00000194283E2080>

Retrieve a credential.

Args:
    key: Credential key
    show_value: Whether to show the actual value (dangerous!)
    
Returns:
    Credential value if found, None otherwise

##### delete_credential(self: Any, key: str, confirm: bool) -> bool

Delete a credential.

Args:
    key: Credential key
    confirm: Whether to ask for confirmation
    
Returns:
    True if deleted successfully, False otherwise

##### list_credentials(self: Any, show_values: bool) -> None

List all stored credentials.

Args:
    show_values: Whether to show credential values (dangerous!)

##### setup_aws_credentials(self: Any, interactive: bool) -> bool

Set up AWS credentials interactively.

Args:
    interactive: Whether to prompt for values
    
Returns:
    True if setup successful, False otherwise

##### setup_hf_credentials(self: Any, interactive: bool) -> bool

Set up HuggingFace credentials interactively.

Args:
    interactive: Whether to prompt for values
    
Returns:
    True if setup successful, False otherwise

##### test_credentials(self: Any, source_type: str) -> bool

Test credentials for a specific source type.

Args:
    source_type: Source type (s3, hf)
    
Returns:
    True if credentials work, False otherwise

##### export_config(self: Any, output_file: str, include_values: bool) -> bool

Export credential configuration to file.

Args:
    output_file: Output file path
    include_values: Whether to include credential values (dangerous!)
    
Returns:
    True if exported successfully, False otherwise

