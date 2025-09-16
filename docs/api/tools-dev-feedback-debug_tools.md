---
title: tools.dev-feedback.debug_tools
category: api
tags: [api, tools]
---

# tools.dev-feedback.debug_tools

Debug Tools with Comprehensive Logging

This module provides debugging tools with comprehensive logging and error reporting.

## Classes

### LogEntry

Log entry structure

### ErrorPattern

Error pattern for analysis

### DebugSession

Debug session information

### DebugLogHandler

Custom log handler for debug tools

#### Methods

##### __init__(self: Any, debug_tools: DebugTools)



##### emit(self: Any, record: Any)



##### format_exception(self: Any, record: Any) -> str

Format exception information

### DebugTools

Comprehensive debugging tools

#### Methods

##### __init__(self: Any, project_root: <ast.Subscript object at 0x0000019433D59AB0>)



##### _load_error_patterns(self: Any) -> <ast.Subscript object at 0x0000019430323DC0>

Load common error patterns for analysis

##### enable_debug_logging(self: Any, level: str)

Enable comprehensive debug logging

##### disable_debug_logging(self: Any)

Disable debug logging

##### start_debug_session(self: Any, session_id: <ast.Subscript object at 0x000001942F92B430>) -> str

Start a new debug session

##### end_debug_session(self: Any) -> <ast.Subscript object at 0x00000194344A0100>

End the current debug session

##### add_log_entry(self: Any, log_entry: LogEntry)

Add log entry to current session

##### analyze_logs(self: Any, session_id: <ast.Subscript object at 0x00000194344A2140>) -> <ast.Subscript object at 0x00000194318AAC80>

Analyze logs for patterns and issues

##### profile_function(self: Any, func: Callable) -> Callable

Decorator to profile function execution time

##### get_performance_report(self: Any) -> <ast.Subscript object at 0x00000194318A90F0>

Get performance analysis report

##### export_debug_report(self: Any, output_file: Path, session_id: <ast.Subscript object at 0x00000194318AA710>)

Export comprehensive debug report

##### _get_system_info(self: Any) -> <ast.Subscript object at 0x0000019431A9EB90>

Get system information for debug report

##### _get_installed_packages(self: Any) -> <ast.Subscript object at 0x0000019431A9D600>

Get list of installed Python packages

##### clear_logs(self: Any)

Clear accumulated log entries

##### get_recent_errors(self: Any, count: int) -> <ast.Subscript object at 0x0000019431A9F370>

Get recent error log entries

##### search_logs(self: Any, query: str, level: <ast.Subscript object at 0x0000019431A9C250>) -> <ast.Subscript object at 0x0000019431A9C9A0>

Search log entries by message content

