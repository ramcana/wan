---
title: scripts.startup_manager.cli
category: api
tags: [api, scripts]
---

# scripts.startup_manager.cli

Interactive CLI interface for the WAN22 Server Startup Manager.

This module provides a rich, user-friendly command-line interface with:
- Progress bars and spinners
- Interactive prompts with clear options
- Verbose/quiet modes
- Colored output and structured display

## Classes

### VerbosityLevel

Verbosity levels for CLI output

### CLIOptions

CLI configuration options

### InteractiveCLI

Rich-based interactive CLI for startup management

#### Methods

##### __init__(self: Any, options: CLIOptions)



##### _setup_styles(self: Any)

Setup console styles and themes

##### display_banner(self: Any)

Display the startup manager banner

##### create_progress_context(self: Any, description: str, total: <ast.Subscript object at 0x000001942F8394B0>)

Create a progress context manager

##### show_spinner(self: Any, description: str)

Show a spinner for long-running operations

##### print_status(self: Any, message: str, status: str)

Print a status message with appropriate styling

##### print_verbose(self: Any, message: str)

Print message only in verbose mode

##### print_debug(self: Any, message: str)

Print message only in debug mode

##### confirm_action(self: Any, message: str, default: bool) -> bool

Get user confirmation for actions

##### prompt_choice(self: Any, message: str, choices: <ast.Subscript object at 0x00000194319452A0>, default: str) -> str

Prompt user for a choice from a list

##### prompt_number(self: Any, message: str, default: int, min_val: int, max_val: int) -> int

Prompt user for a number

##### display_table(self: Any, title: str, headers: <ast.Subscript object at 0x0000019431922FB0>, rows: <ast.Subscript object at 0x0000019431922EF0>)

Display a formatted table

##### display_key_value_pairs(self: Any, title: str, pairs: <ast.Subscript object at 0x0000019431917070>)

Display key-value pairs in a formatted way

##### display_section_header(self: Any, title: str)

Display a section header

##### display_summary_panel(self: Any, title: str, content: str, style: str)

Display a summary panel with content

### _DummyProgress

Dummy progress context for quiet mode

#### Methods

##### __enter__(self: Any)



##### __exit__(self: Any)



##### add_task(self: Any, description: str, total: int)



##### update(self: Any, task_id: int, advance: int)



##### start(self: Any)



##### stop(self: Any)



## Constants

### QUIET

Type: `str`

Value: `quiet`

### NORMAL

Type: `str`

Value: `normal`

### VERBOSE

Type: `str`

Value: `verbose`

### DEBUG

Type: `str`

Value: `debug`

