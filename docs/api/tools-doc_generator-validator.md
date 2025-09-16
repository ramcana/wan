---
title: tools.doc_generator.validator
category: api
tags: [api, tools]
---

# tools.doc_generator.validator

Documentation Validator

Validates documentation for broken links, content quality,
freshness, and compliance with style guidelines.

## Classes

### ValidationIssue

Represents a validation issue

### ValidationReport

Complete validation report

### LinkCheckResult

Result of link checking

### DocumentationValidator

Comprehensive documentation validator

#### Methods

##### __init__(self: Any, docs_root: Path, config: <ast.Subscript object at 0x000001942C660A30>)



##### _default_config(self: Any) -> <ast.Subscript object at 0x000001942C62B9A0>

Default validation configuration

##### _load_style_rules(self: Any) -> <ast.Subscript object at 0x000001942C62B340>

Load style validation rules

##### validate_all(self: Any) -> ValidationReport

Run comprehensive validation on all documentation

##### _should_skip_file(self: Any, file_path: Path) -> bool

Check if file should be skipped

##### _validate_file(self: Any, file_path: Path)

Validate a single documentation file

##### _validate_metadata(self: Any, content: str, file_path: str)

Validate frontmatter metadata

##### _validate_style(self: Any, content: str, file_path: str)

Validate content style and formatting

##### _validate_links(self: Any, content: str, file_path: str, full_path: Path)

Validate all links in the document

##### _validate_anchor_link(self: Any, anchor: str, content: str, file_path: str, line_num: int)

Validate anchor links within the document

##### _validate_internal_link(self: Any, link_url: str, file_path: str, full_path: Path, line_num: int)

Validate internal documentation links

##### _validate_external_link(self: Any, url: str, file_path: str, line_num: int)

Validate external links

##### _check_external_link(self: Any, url: str) -> LinkCheckResult

Check if external link is accessible

##### _validate_freshness(self: Any, content: str, file_path: str, full_path: Path)

Validate document freshness

##### _extract_metadata(self: Any, content: str) -> <ast.Subscript object at 0x0000019428400A60>

Extract YAML frontmatter metadata

##### _generate_summary(self: Any) -> <ast.Subscript object at 0x000001942C686E00>

Generate validation summary statistics

##### check_links_only(self: Any, external_only: bool) -> <ast.Subscript object at 0x0000019428504B20>

Quick link-only validation

##### generate_report_html(self: Any, report: ValidationReport, output_path: Path)

Generate HTML validation report

##### save_report_json(self: Any, report: ValidationReport, output_path: Path)

Save validation report as JSON

