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

##### __init__(self: Any, project_root: str, docs_dirs: <ast.Subscript object at 0x0000019428DAA350>)

Initialize the documentation validator.

##### validate_all(self: Any, structure: <ast.Subscript object at 0x0000019428D87430>, relationships: <ast.Subscript object at 0x0000019428D87370>) -> DocumentationValidationReport

Perform complete documentation validation.

##### _find_documentation_files(self: Any) -> <ast.Subscript object at 0x0000019428D84040>

Find all documentation files in the project.

##### _validate_all_links(self: Any, doc_files: <ast.Subscript object at 0x0000019428D07EB0>) -> <ast.Subscript object at 0x0000019428CFE770>

Validate all links in documentation files.

##### _validate_links_in_file(self: Any, file_path: Path) -> <ast.Subscript object at 0x0000019428CFD090>

Validate all links in a single file.

##### _validate_single_link(self: Any, link: str, source_file: Path, pattern_type: str) -> LinkValidationResult

Validate a single link.

##### _validate_anchor_link(self: Any, anchor: str, source_file: Path) -> <ast.Subscript object at 0x0000019428D26980>

Validate an anchor link within a file.

##### _validate_internal_link(self: Any, link: str, source_file: Path) -> <ast.Subscript object at 0x0000019428D83820>

Validate an internal file link.

##### _check_documentation_freshness(self: Any, doc_files: <ast.Subscript object at 0x0000019428D83610>) -> <ast.Subscript object at 0x000001942CB727A0>

Check if documentation files are up to date.

##### _determine_file_criticality(self: Any, file_path: Path) -> str

Determine how critical a documentation file is.

##### _check_documentation_completeness(self: Any, doc_files: <ast.Subscript object at 0x000001942CB71C90>, structure: ProjectStructure, relationships: ComponentRelationshipMap) -> <ast.Subscript object at 0x000001942CB75870>

Check if all components have adequate documentation.

##### _component_needs_documentation(self: Any, component: ComponentInfo) -> bool

Determine if a component needs documentation.

##### _get_alternative_doc_names(self: Any, doc_name: str) -> <ast.Subscript object at 0x000001942CCF55A0>

Get alternative names for a documentation file.

##### _check_accessibility(self: Any, doc_files: <ast.Subscript object at 0x000001942CCF5480>) -> <ast.Subscript object at 0x0000019428509C60>

Check documentation accessibility and searchability.

##### _calculate_metrics(self: Any, doc_files: <ast.Subscript object at 0x0000019428509DE0>, link_results: <ast.Subscript object at 0x0000019428509EA0>, issues: <ast.Subscript object at 0x0000019428509F60>, structure: <ast.Subscript object at 0x000001942850A020>) -> DocumentationMetrics

Calculate documentation quality metrics.

##### _generate_recommendations(self: Any, metrics: DocumentationMetrics, issues: <ast.Subscript object at 0x000001942850A590>) -> <ast.Subscript object at 0x0000019427BB6B90>

Generate recommendations for improving documentation.

##### generate_maintenance_plan(self: Any, report: DocumentationValidationReport) -> <ast.Subscript object at 0x0000019427BB6530>

Generate a maintenance plan based on validation results.

##### save_report(self: Any, report: DocumentationValidationReport, output_path: str)

Save validation report to file.

##### generate_summary_report(self: Any, report: DocumentationValidationReport) -> str

Generate a human-readable summary report.

