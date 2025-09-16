---
title: tools.code_quality.enforcement.enforcement_cli
category: api
tags: [api, tools]
---

# tools.code_quality.enforcement.enforcement_cli

CLI for automated quality enforcement system.

## Classes

### EnforcementCLI

Command-line interface for quality enforcement.

#### Methods

##### __init__(self: Any, project_root: <ast.Subscript object at 0x000001942F3339A0>)

Initialize enforcement CLI.

##### setup_hooks(self: Any, config_file: <ast.Subscript object at 0x000001942F3333A0>) -> bool

Set up pre-commit hooks.

##### setup_ci(self: Any, platform: str) -> bool

Set up CI/CD integration.

##### run_checks(self: Any, files: <ast.Subscript object at 0x000001942F313220>) -> bool

Run quality checks.

##### status(self: Any) -> None

Show enforcement system status.

##### create_dashboard(self: Any) -> bool

Create quality metrics dashboard.

##### generate_report(self: Any, format_type: str) -> None

Generate quality report.

##### _generate_html_report(self: Any, results: <ast.Subscript object at 0x000001942F307550>) -> str

Generate HTML quality report.

