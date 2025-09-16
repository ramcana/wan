---
title: tools.project-structure-analyzer.complexity_analyzer
category: api
tags: [api, tools]
---

# tools.project-structure-analyzer.complexity_analyzer



## Classes

### FileComplexity

Complexity metrics for a single file.

### ComponentComplexity

Complexity metrics for a component/directory.

### ProjectComplexityReport

Complete project complexity analysis.

### ProjectComplexityAnalyzer

Analyzes project complexity and identifies documentation needs.

#### Methods

##### __init__(self: Any, root_path: str)

Initialize analyzer with project root path.

##### analyze(self: Any) -> ProjectComplexityReport

Perform complete project complexity analysis.

##### _should_ignore_dir(self: Any, dirname: str) -> bool

Check if directory should be ignored.

##### _analyze_file_complexity(self: Any, file_path: Path) -> <ast.Subscript object at 0x00000194280C9600>

Analyze complexity of a single Python file.

##### _calculate_cyclomatic_complexity(self: Any, tree: ast.AST) -> int

Calculate cyclomatic complexity of AST.

##### _calculate_cognitive_complexity(self: Any, tree: ast.AST) -> int

Calculate cognitive complexity (more human-focused than cyclomatic).

##### _count_functions(self: Any, tree: ast.AST) -> int

Count function definitions in AST.

##### _count_classes(self: Any, tree: ast.AST) -> int

Count class definitions in AST.

##### _count_imports(self: Any, tree: ast.AST) -> int

Count import statements in AST.

##### _count_docstring_lines(self: Any, tree: ast.AST, content: str) -> int

Count lines in docstrings.

##### _calculate_complexity_score(self: Any, lines: int, cyclomatic: int, cognitive: int, functions: int, classes: int, doc_ratio: float) -> int

Calculate overall complexity score for a file.

##### _needs_documentation(self: Any, complexity_score: int, doc_ratio: float, functions: int, classes: int) -> bool

Determine if a file needs better documentation.

##### _identify_file_issues(self: Any, lines: int, cyclomatic: int, functions: int, classes: int, doc_ratio: float) -> <ast.Subscript object at 0x000001942CAFA800>

Identify specific issues with a file.

##### _analyze_component_complexity(self: Any, component_path: str, files: <ast.Subscript object at 0x000001942CAFA6E0>) -> ComponentComplexity

Analyze complexity of a component (directory).

##### _generate_component_recommendations(self: Any, files: <ast.Subscript object at 0x000001942CB77730>, doc_coverage: float, avg_complexity: float) -> <ast.Subscript object at 0x000001942CB76380>

Generate recommendations for improving a component.

##### _identify_high_priority_areas(self: Any, components: <ast.Subscript object at 0x000001942CBC1990>) -> <ast.Subscript object at 0x000001942CBC10F0>

Identify components that need immediate attention.

##### _identify_documentation_gaps(self: Any, components: <ast.Subscript object at 0x000001942CBC0F40>) -> <ast.Subscript object at 0x000001942CBC09D0>

Identify areas with poor documentation.

##### _identify_complexity_hotspots(self: Any, components: <ast.Subscript object at 0x000001942CBC07C0>) -> <ast.Subscript object at 0x000001942850CFD0>

Identify the most complex areas of the codebase.

##### _generate_recommendations(self: Any, components: <ast.Subscript object at 0x000001942850D570>) -> <ast.Subscript object at 0x00000194280C0790>

Generate overall project recommendations.

##### save_analysis(self: Any, analysis: ProjectComplexityReport, output_path: str)

Save analysis results to JSON file.

##### generate_summary_report(self: Any, analysis: ProjectComplexityReport) -> str

Generate a human-readable summary report.

