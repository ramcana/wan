---
title: tools.code_quality.analyzers.complexity_analyzer
category: api
tags: [api, tools]
---

# tools.code_quality.analyzers.complexity_analyzer

Code complexity analysis and recommendations.

## Classes

### ComplexityAnalyzer

Analyzes code complexity and provides refactoring recommendations.

#### Methods

##### __init__(self: Any, config: QualityConfig)

Initialize analyzer with configuration.

##### analyze_complexity(self: Any, file_path: Path, tree: ast.AST) -> <ast.Subscript object at 0x000001943451CF10>

Analyze complexity in the given AST.

Returns:
    Tuple of (issues, metrics)

##### _analyze_module_complexity(self: Any, file_path: Path, tree: ast.AST) -> <ast.Subscript object at 0x0000019432E65E70>

Analyze module-level complexity.

##### _analyze_function_complexity(self: Any, file_path: Path, node: ast.FunctionDef) -> <ast.Subscript object at 0x000001942F02B3A0>

Analyze complexity of a specific function.

##### _analyze_class_complexity(self: Any, file_path: Path, node: ast.ClassDef) -> <ast.Subscript object at 0x0000019434138E50>

Analyze complexity of a specific class.

##### _calculate_cyclomatic_complexity(self: Any, node: ast.AST) -> int

Calculate cyclomatic complexity for a function or method.

##### _calculate_max_nesting_depth(self: Any, node: ast.AST) -> int

Calculate maximum nesting depth in a function.

##### _calculate_maintainability_index(self: Any, file_path: Path, tree: ast.AST, avg_complexity: float) -> float

Calculate maintainability index (0-100, higher is better).

##### _calculate_halstead_metrics(self: Any, tree: ast.AST) -> <ast.Subscript object at 0x000001943193A080>

Calculate simplified Halstead operators and operands count.

##### _get_complexity_reduction_suggestion(self: Any, node: ast.FunctionDef, complexity: int) -> str

Get specific suggestion for reducing function complexity.

