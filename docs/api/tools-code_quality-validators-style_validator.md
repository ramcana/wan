---
title: tools.code_quality.validators.style_validator
category: api
tags: [api, tools]
---

# tools.code_quality.validators.style_validator

Style validation using flake8 and custom rules.

## Classes

### StyleValidator

Validates code style using flake8 and custom rules.

#### Methods

##### __init__(self: Any, config: QualityConfig)

Initialize validator with configuration.

##### validate_style(self: Any, file_path: Path, content: str) -> <ast.Subscript object at 0x0000019428AC19C0>

Validate code style in the given file.

##### _run_flake8_check(self: Any, file_path: Path) -> <ast.Subscript object at 0x0000019428AC3460>

Run flake8 style checker on the file.

##### _parse_flake8_output(self: Any, file_path: Path, line: str) -> QualityIssue

Parse flake8 output line into QualityIssue.

##### _get_severity_from_code(self: Any, code: str) -> QualitySeverity

Determine severity based on flake8 error code.

##### _is_auto_fixable(self: Any, code: str) -> bool

Check if error code represents an auto-fixable issue.

##### _get_suggestion_for_code(self: Any, code: str) -> str

Get suggestion for fixing specific error code.

##### _check_custom_style_rules(self: Any, file_path: Path, content: str) -> <ast.Subscript object at 0x000001942A189E70>

Check custom style rules not covered by flake8.

##### _has_hardcoded_strings(self: Any, line: str) -> bool

Check if line contains hardcoded strings that should be constants.

