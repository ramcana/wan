---
title: tools.code_quality.formatters.code_formatter
category: api
tags: [api, tools]
---

# tools.code_quality.formatters.code_formatter

Code formatting checker and fixer.

## Classes

### CodeFormatter

Handles code formatting checking and fixing.

#### Methods

##### __init__(self: Any, config: QualityConfig)

Initialize formatter with configuration.

##### check_formatting(self: Any, file_path: Path, content: str) -> <ast.Subscript object at 0x000001942F1DADA0>

Check formatting issues in the given file.

##### fix_formatting(self: Any, file_path: Path) -> bool

Fix formatting issues in the given file.

##### fix_imports(self: Any, file_path: Path) -> bool

Fix import issues in the given file.

##### _check_black_formatting(self: Any, file_path: Path, content: str) -> <ast.Subscript object at 0x000001942F1CA560>

Check formatting with black.

##### _check_isort_formatting(self: Any, file_path: Path, content: str) -> <ast.Subscript object at 0x000001942F1C9360>

Check import sorting with isort.

##### _check_basic_formatting(self: Any, file_path: Path, content: str) -> <ast.Subscript object at 0x000001942F1B6BF0>

Check basic formatting rules.

##### _run_black(self: Any, file_path: Path) -> None

Run black formatter on file.

##### _run_isort(self: Any, file_path: Path) -> None

Run isort on file.

