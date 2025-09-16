---
title: tools.health-checker.pre_commit_import_validation
category: api
tags: [api, tools]
---

# tools.health-checker.pre_commit_import_validation

Pre-commit hook for import path validation.
Validates that import statements are correct and follow project structure.

## Classes

### ImportVisitor

AST visitor to collect import statements.

#### Methods

##### __init__(self: Any)



##### visit_Import(self: Any, node: Any)



##### visit_ImportFrom(self: Any, node: Any)



