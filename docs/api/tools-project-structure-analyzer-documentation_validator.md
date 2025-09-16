---
title: tools.project-structure-analyzer.documentation_validator
category: api
tags: [api, tools]
---

# tools.project-structure-analyzer.documentation_validator

Documentation Validation and Maintenance System

Implements documentation link checking, freshness validation,
completeness analysis, and accessibility checking.

## Classes

### LinkValidationResult

Result of link validation.

### DocumentationIssue

Represents a documentation issue.

### DocumentationMetrics

Metrics about documentation quality.

### DocumentationValidationReport

Complete documentation validation report.

### DocumentationValidator

Validates and maintains project documentation.

#### Methods

##### __init__(self: Any, project_root: str, docs_dirs: <ast.Subscript object at 0x000001942F8FA350>)

Initialize the documentation validator.

##### validate_all(self: Any, structure: <ast.Subscript object at 0x000001942F8F9240>, relationships: <ast.Subscript object at 0x000001942F873370>) -> DocumentationValidationReport

Perform complete documentation validation.

##### _find_documentation_files(self: Any) -> <ast.Subscript object at 0x000001942F870040>

Find all documentation files in the project.

##### _validate_all_links(self: Any, doc_files: <ast.Subscript object at 0x000001942F89BEB0>) -> <ast.Subscript object at 0x000001942F89B790>

Validate all links in documentation files.

##### _validate_links_in_file(self: Any, file_path: Path) -> <ast.Subscript object at 0x000001942F8FD090>

Validate all links in a single file.

##### _validate_single_link(self: Any, link: str, source_file: Path, pattern_type: str) -> LinkValidationResult

Validate a single link.

##### _validate_anchor_link(self: Any, anchor: str, source_file: Path) -> <ast.Subscript object at 0x000001942FDCE980>

Validate an anchor link within a file.

##### _validate_internal_link(self: Any, link: str, source_file: Path) -> <ast.Subscript object at 0x000001942FE3B820>

Validate an internal file link.

##### _check_documentation_freshness(self: Any, doc_files: <ast.Subscript object at 0x000001942FE3B610>) -> <ast.Subscript object at 0x000001942F3F0B50>

Check if documentation files are up to date.

##### _determine_file_criticality(self: Any, file_path: Path) -> str

Determine how critical a documentation file is.

##### _check_documentation_completeness(self: Any, doc_files: <ast.Subscript object at 0x000001942F3F2C80>, structure: ProjectStructure, relationships: ComponentRelationshipMap) -> <ast.Subscript object at 0x000001942F3F1C90>

Check if all components have adequate documentation.

##### _component_needs_documentation(self: Any, component: ComponentInfo) -> bool

Determine if a component needs documentation.

##### _get_alternative_doc_names(self: Any, doc_name: str) -> <ast.Subscript object at 0x0000019431889C00>

Get alternative names for a documentation file.

##### _check_accessibility(self: Any, doc_files: <ast.Subscript object at 0x0000019431889F60>) -> <ast.Subscript object at 0x000001943191A5C0>

Check documentation accessibility and searchability.

##### _calculate_metrics(self: Any, doc_files: <ast.Subscript object at 0x000001943191A3B0>, link_results: <ast.Subscript object at 0x000001943191A290>, issues: <ast.Subscript object at 0x000001943191A1D0>, structure: <ast.Subscript object at 0x000001943191A110>) -> DocumentationMetrics

Calculate documentation quality metrics.

##### _generate_recommendations(self: Any, metrics: DocumentationMetrics, issues: <ast.Subscript object at 0x000001942FBCCB50>) -> <ast.Subscript object at 0x000001942F8E1090>

Generate recommendations for improving documentation.

##### generate_maintenance_plan(self: Any, report: DocumentationValidationReport) -> <ast.Subscript object at 0x000001942F8E1AE0>

Generate a maintenance plan based on validation results.

##### save_report(self: Any, report: DocumentationValidationReport, output_path: str)

Save validation report to file.

##### generate_summary_report(self: Any, report: DocumentationValidationReport) -> str

Generate a human-readable summary report.

