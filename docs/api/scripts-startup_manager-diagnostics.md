---
title: scripts.startup_manager.diagnostics
category: api
tags: [api, scripts]
---

# scripts.startup_manager.diagnostics

System diagnostics and troubleshooting features for the startup manager.

This module provides comprehensive system information collection, diagnostic mode,
and log analysis tools to help identify and resolve common startup issues.

## Classes

### SystemInfo

System information data structure.

### DiagnosticResult

Diagnostic check result.

### LogAnalysisResult

Log analysis result.

### SystemDiagnostics

System diagnostics and information collection.

Provides comprehensive system information gathering for troubleshooting
and diagnostic purposes.

#### Methods

##### __init__(self: Any)



##### collect_system_info(self: Any) -> SystemInfo

Collect comprehensive system information.

Returns:
    SystemInfo object with all collected information

##### _get_os_info(self: Any) -> <ast.Subscript object at 0x0000019432D99C00>

Get operating system information.

##### _get_python_info(self: Any) -> <ast.Subscript object at 0x0000019432D99540>

Get Python environment information.

##### _get_hardware_info(self: Any) -> <ast.Subscript object at 0x0000019432D99F60>

Get hardware information.

##### _get_network_info(self: Any) -> <ast.Subscript object at 0x0000019432D9ACB0>

Get network interface information.

##### _get_environment_info(self: Any) -> <ast.Subscript object at 0x0000019434631F00>

Get relevant environment variables.

##### _get_package_info(self: Any) -> <ast.Subscript object at 0x0000019434632B60>

Get list of installed Python packages.

##### run_diagnostic_checks(self: Any) -> <ast.Subscript object at 0x00000194343D8280>

Run comprehensive diagnostic checks.

Returns:
    List of diagnostic results

##### _check_python_version(self: Any) -> DiagnosticResult

Check Python version compatibility.

##### _check_virtual_environment(self: Any) -> DiagnosticResult

Check if virtual environment is active.

##### _check_required_packages(self: Any) -> DiagnosticResult

Check if required packages are installed.

##### _check_port_availability(self: Any) -> DiagnosticResult

Check if default ports are available.

##### _check_disk_space(self: Any) -> DiagnosticResult

Check available disk space.

##### _check_memory_usage(self: Any) -> DiagnosticResult

Check system memory usage.

##### _check_network_connectivity(self: Any) -> DiagnosticResult

Check basic network connectivity.

##### _check_file_permissions(self: Any) -> DiagnosticResult

Check file system permissions.

##### _check_node_environment(self: Any) -> DiagnosticResult

Check Node.js environment.

### LogAnalyzer

Log analysis tools for identifying common issues and patterns.

Analyzes startup logs to identify common problems and provide
suggestions for resolution.

#### Methods

##### __init__(self: Any)



##### analyze_logs(self: Any, log_dir: <ast.Subscript object at 0x0000019434040970>) -> LogAnalysisResult

Analyze log files for common issues and patterns.

Args:
    log_dir: Directory containing log files
    
Returns:
    LogAnalysisResult with analysis findings

##### _categorize_error(self: Any, message: str, error_patterns_found: <ast.Subscript object at 0x000001942F02AF80>)

Categorize error message by pattern.

##### _process_error_patterns(self: Any, error_patterns_found: <ast.Subscript object at 0x0000019434671630>) -> <ast.Subscript object at 0x0000019434671BA0>

Process error patterns into common errors list.

##### _get_error_description(self: Any, category: str) -> str

Get description for error category.

##### _get_error_solutions(self: Any, category: str) -> <ast.Subscript object at 0x00000194346722C0>

Get solutions for error category.

##### _process_performance_data(self: Any, performance_data: <ast.Subscript object at 0x0000019434672B00>) -> <ast.Subscript object at 0x0000019434671D50>

Process performance metrics.

##### _generate_suggestions(self: Any, common_errors: <ast.Subscript object at 0x0000019434673940>, error_count: int, warning_count: int) -> <ast.Subscript object at 0x000001943448C670>

Generate suggestions based on analysis.

### DiagnosticMode

Diagnostic mode that captures detailed startup process information.

Provides comprehensive diagnostic information collection during
startup process for troubleshooting purposes.

#### Methods

##### __init__(self: Any)



##### run_full_diagnostics(self: Any, log_dir: <ast.Subscript object at 0x000001943448D510>) -> <ast.Subscript object at 0x000001943448CB80>

Run comprehensive diagnostics including system info, checks, and log analysis.

Args:
    log_dir: Directory containing log files to analyze
    
Returns:
    Dictionary containing all diagnostic information

##### _generate_summary(self: Any, diagnostic_results: <ast.Subscript object at 0x000001943448E8C0>, log_analysis: LogAnalysisResult) -> <ast.Subscript object at 0x000001943448D1B0>

Generate diagnostic summary.

##### _get_top_recommendations(self: Any, diagnostic_results: <ast.Subscript object at 0x000001943448EF20>, log_analysis: LogAnalysisResult) -> <ast.Subscript object at 0x000001942F8FFBB0>

Get top recommendations based on diagnostics.

##### save_diagnostic_report(self: Any, diagnostic_data: <ast.Subscript object at 0x000001942F8FEAA0>, output_file: <ast.Subscript object at 0x000001942F8FFA30>) -> Path

Save diagnostic report to file.

Args:
    diagnostic_data: Diagnostic data to save
    output_file: Output file path (optional)
    
Returns:
    Path to saved report file

