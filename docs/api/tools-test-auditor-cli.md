---
title: tools.test-auditor.cli
category: api
tags: [api, tools]
---

# tools.test-auditor.cli

Test Auditor CLI

Command-line interface for the comprehensive test suite auditor.
Provides various commands for analyzing and reporting on test suite health.

## Classes

### TestAuditorCLI

Command-line interface for test auditor

#### Methods

##### __init__(self: Any)



##### run(self: Any, args: <ast.Subscript object at 0x0000019427B2FF40>) -> int

Run the CLI with given arguments

##### _create_parser(self: Any) -> argparse.ArgumentParser

Create argument parser

##### _audit_command(self: Any, args: Any) -> int

Run comprehensive audit

##### _summary_command(self: Any, args: Any) -> int

Show audit summary

##### _issues_command(self: Any, args: Any) -> int

Show test issues

##### _performance_command(self: Any, args: Any) -> int

Show performance analysis

##### _files_command(self: Any, args: Any) -> int

Show file analysis

##### _load_or_run_audit(self: Any, report_file: <ast.Subscript object at 0x000001942A2E6FB0>) -> TestSuiteAuditReport

Load existing report or run new audit

##### _print_summary(self: Any, report: TestSuiteAuditReport)

Print audit summary

##### _save_json_report(self: Any, report: TestSuiteAuditReport, output_file: Path)

Save report as JSON

##### _save_text_report(self: Any, report: TestSuiteAuditReport, output_file: Path)

Save report as text

##### _save_html_report(self: Any, report: TestSuiteAuditReport, output_file: Path)

Save report as HTML

