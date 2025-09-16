---
title: tools.health-checker.checkers.documentation_health_checker
category: api
tags: [api, tools]
---

# tools.health-checker.checkers.documentation_health_checker

Documentation health checker

## Classes

### DocumentationHealthChecker

Checks the health of project documentation

#### Methods

##### __init__(self: Any, config: HealthConfig)



##### check_health(self: Any) -> ComponentHealth

Check documentation health

##### _discover_documentation(self: Any) -> <ast.Subscript object at 0x000001942FA99A20>

Discover documentation files

##### _check_essential_documentation(self: Any) -> <ast.Subscript object at 0x000001942FA9A5F0>

Check for essential documentation files

##### _check_broken_links(self: Any, doc_files: <ast.Subscript object at 0x000001942FA99D80>) -> <ast.Subscript object at 0x000001942FA9AD70>

Check for broken links in documentation

##### _is_broken_link(self: Any, url: str, source_file: Path) -> bool

Check if a link is broken

##### _check_documentation_freshness(self: Any, doc_files: <ast.Subscript object at 0x000001942F6C5B10>) -> <ast.Subscript object at 0x000001942F6C5720>

Check for potentially outdated documentation

##### _check_scattered_documentation(self: Any) -> <ast.Subscript object at 0x000001942F6C7280>

Check for documentation files outside the docs directory

##### _calculate_documentation_score(self: Any, metrics: <ast.Subscript object at 0x000001942F6C4D60>, issues: <ast.Subscript object at 0x000001942F6C65C0>) -> float

Calculate documentation health score

##### _determine_status(self: Any, score: float) -> str

Determine health status from score

