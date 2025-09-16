---
title: tools.codebase-cleanup.naming_standardizer
category: api
tags: [api, tools]
---

# tools.codebase-cleanup.naming_standardizer



## Classes

### NamingViolation

Represents a naming convention violation

### InconsistentPattern

Represents an inconsistent naming pattern across the codebase

### OrganizationSuggestion

Represents a file organization suggestion

### NamingReport

Report of naming convention analysis

### NamingStandardizer

Comprehensive naming standardization system that:
- Analyzes naming conventions across the codebase
- Identifies inconsistent patterns
- Provides safe refactoring capabilities
- Suggests file organization improvements

#### Methods

##### __init__(self: Any, root_path: str, backup_dir: str)



##### analyze_naming_conventions(self: Any) -> NamingReport

Perform comprehensive naming convention analysis

Returns:
    NamingReport with all findings and recommendations

##### _get_files_to_analyze(self: Any) -> <ast.Subscript object at 0x0000019432DE67D0>

Get list of files to analyze for naming conventions

##### _find_naming_violations(self: Any, files: <ast.Subscript object at 0x0000019432DE6680>) -> <ast.Subscript object at 0x0000019432DE5FF0>

Find naming convention violations in files

##### _check_file_naming(self: Any, file_path: Path) -> <ast.Subscript object at 0x0000019432DE4610>

Check file and directory naming conventions

##### _check_code_element_naming(self: Any, file_path: Path) -> <ast.Subscript object at 0x0000019432E37D60>

Check naming conventions in code files

##### _check_python_naming(self: Any, file_path: Path) -> <ast.Subscript object at 0x0000019432E5D9C0>

Check Python naming conventions (PEP 8)

##### _check_javascript_naming(self: Any, file_path: Path) -> <ast.Subscript object at 0x0000019432DAFD60>

Check JavaScript/TypeScript naming conventions

##### _is_valid_name(self: Any, name: str, convention: str) -> bool

Check if name follows the specified convention

##### _detect_convention(self: Any, name: str) -> str

Detect the naming convention used by a name

##### _convert_to_convention(self: Any, name: str, target_convention: str) -> str

Convert name to target naming convention

##### _split_name_into_words(self: Any, name: str) -> <ast.Subscript object at 0x00000194318F7970>

Split a name into constituent words

##### _find_inconsistent_patterns(self: Any, files: <ast.Subscript object at 0x00000194318F7A90>) -> <ast.Subscript object at 0x000001943448C4C0>

Find inconsistent naming patterns across the codebase

##### _generate_organization_suggestions(self: Any, files: <ast.Subscript object at 0x000001943448F490>) -> <ast.Subscript object at 0x000001943448D9F0>

Generate file organization suggestions

##### _detect_file_purpose(self: Any, filename: str) -> str

Detect the purpose of a file based on its name

##### _suggest_directory_for_purpose(self: Any, purpose: str) -> str

Suggest appropriate directory name for a file purpose

##### _create_convention_summary(self: Any, violations: <ast.Subscript object at 0x0000019431AB9990>) -> <ast.Subscript object at 0x0000019431AB85B0>

Create summary of naming conventions found

##### _generate_recommendations(self: Any, violations: <ast.Subscript object at 0x0000019431AB8C40>, inconsistent_patterns: <ast.Subscript object at 0x0000019431AB9540>, organization_suggestions: <ast.Subscript object at 0x0000019431ABB7C0>) -> <ast.Subscript object at 0x0000019431ABAA10>

Generate recommendations for naming improvements

##### apply_naming_fixes(self: Any, report: NamingReport, target_convention: str) -> <ast.Subscript object at 0x0000019434633A60>

Apply naming fixes based on the report

Args:
    report: NamingReport from analysis
    target_convention: Target naming convention to apply
    
Returns:
    Dict mapping operation to result message

##### _apply_naming_fixes_batch(self: Any, violations: <ast.Subscript object at 0x0000019434633F70>) -> int

Apply a batch of naming fixes

##### _apply_file_naming_fixes(self: Any, file_path: str, violations: <ast.Subscript object at 0x0000019434630CA0>) -> bool

Apply naming fixes to a single file

##### create_backup(self: Any, files_to_backup: <ast.Subscript object at 0x0000019434631570>) -> str

Create backup of files before applying fixes

##### save_report(self: Any, report: NamingReport, output_path: str) -> None

Save naming analysis report to file

