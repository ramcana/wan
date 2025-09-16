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

##### __init__(self: Any, project_root: <ast.Subscript object at 0x0000019427BBA1A0>)

Initialize enforcement CLI.

##### setup_hooks(self: Any, config_file: <ast.Subscript object at 0x0000019427BB9BA0>) -> bool

Set up pre-commit hooks.

##### setup_ci(self: Any, platform: str) -> bool

Set up CI/CD integration.

##### run_checks(self: Any, files: <ast.Subscript object at 0x0000019427BB80A0>) -> bool

Run quality checks.

##### status(self: Any) -> None

Show enforcement system status.

##### create_dashboard(self: Any) -> bool

Create quality metrics dashboard.

##### generate_report(self: Any, format_type: str) -> None

Generate quality report.

##### _generate_html_report(self: Any, results: <ast.Subscript object at 0x0000019427BBD4B0>) -> str

Generate HTML quality report.

