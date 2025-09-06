import pytest
"""
Project Complexity Analyzer

Analyzes project complexity and identifies areas needing documentation
based on various complexity metrics and patterns.
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import json


@dataclass
class FileComplexity:
    """Complexity metrics for a single file."""
    path: str
    lines_of_code: int
    cyclomatic_complexity: int
    cognitive_complexity: int
    function_count: int
    class_count: int
    import_count: int
    documentation_ratio: float  # Ratio of comment/docstring lines to code lines
    complexity_score: int
    needs_documentation: bool
    issues: List[str]


@dataclass
class ComponentComplexity:
    """Complexity metrics for a component/directory."""
    name: str
    path: str
    total_files: int
    total_lines: int
    average_complexity: float
    max_complexity: int
    files: List[FileComplexity]
    documentation_coverage: float
    complexity_score: int
    priority_level: str  # 'high', 'medium', 'low'
    recommendations: List[str]


@dataclass
class ProjectComplexityReport:
    """Complete project complexity analysis."""
    total_files: int
    total_lines: int
    average_complexity: float
    components: List[ComponentComplexity]
    high_priority_areas: List[str]
    documentation_gaps: List[str]
    complexity_hotspots: List[str]
    recommendations: List[str]


class ProjectComplexityAnalyzer:
    """Analyzes project complexity and identifies documentation needs."""
    
    def __init__(self, root_path: str):
        """Initialize analyzer with project root path."""
        self.root_path = Path(root_path).resolve()
        
    def analyze(self) -> ProjectComplexityReport:
        """Perform complete project complexity analysis."""
        print("Analyzing project complexity...")
        
        components = []
        all_files = []
        
        # Analyze each component
        for root, dirs, files in os.walk(self.root_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if not self._should_ignore_dir(d)]
            
            current_path = Path(root)
            relative_path = current_path.relative_to(self.root_path)
            
            # Skip root directory
            if current_path == self.root_path:
                continue
            
            # Only analyze directories with Python files
            python_files = [f for f in files if f.endswith('.py')]
            if not python_files:
                continue
            
            # Analyze files in this component
            file_complexities = []
            for py_file in python_files:
                file_path = current_path / py_file
                complexity = self._analyze_file_complexity(file_path)
                if complexity:
                    file_complexities.append(complexity)
                    all_files.append(complexity)
            
            if file_complexities:
                component_complexity = self._analyze_component_complexity(
                    str(relative_path), file_complexities
                )
                components.append(component_complexity)
        
        # Generate overall analysis
        total_files = len(all_files)
        total_lines = sum(f.lines_of_code for f in all_files)
        average_complexity = sum(f.complexity_score for f in all_files) / total_files if total_files > 0 else 0
        
        # Identify high priority areas
        high_priority = self._identify_high_priority_areas(components)
        documentation_gaps = self._identify_documentation_gaps(components)
        complexity_hotspots = self._identify_complexity_hotspots(components)
        recommendations = self._generate_recommendations(components)
        
        return ProjectComplexityReport(
            total_files=total_files,
            total_lines=total_lines,
            average_complexity=average_complexity,
            components=components,
            high_priority_areas=high_priority,
            documentation_gaps=documentation_gaps,
            complexity_hotspots=complexity_hotspots,
            recommendations=recommendations
        )
    
    def _should_ignore_dir(self, dirname: str) -> bool:
        """Check if directory should be ignored."""
        ignore_patterns = {
            '__pycache__', '.git', '.pytest_cache', 'node_modules',
            '.vscode', '.idea', 'venv', 'env', '.env', 'build', 'dist'
        }
        return dirname in ignore_patterns or dirname.startswith('.')
    
    def _analyze_file_complexity(self, file_path: Path) -> Optional[FileComplexity]:
        """Analyze complexity of a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return None
            
            # Count lines of code (excluding empty lines and comments)
            lines = content.split('\n')
            code_lines = 0
            comment_lines = 0
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                elif stripped.startswith('#'):
                    comment_lines += 1
                else:
                    code_lines += 1
            
            # Calculate metrics
            cyclomatic_complexity = self._calculate_cyclomatic_complexity(tree)
            cognitive_complexity = self._calculate_cognitive_complexity(tree)
            function_count = self._count_functions(tree)
            class_count = self._count_classes(tree)
            import_count = self._count_imports(tree)
            
            # Calculate documentation ratio
            docstring_lines = self._count_docstring_lines(tree, content)
            total_doc_lines = comment_lines + docstring_lines
            doc_ratio = total_doc_lines / max(code_lines, 1)
            
            # Calculate overall complexity score
            complexity_score = self._calculate_complexity_score(
                code_lines, cyclomatic_complexity, cognitive_complexity,
                function_count, class_count, doc_ratio
            )
            
            # Determine if needs documentation
            needs_documentation = self._needs_documentation(
                complexity_score, doc_ratio, function_count, class_count
            )
            
            # Identify issues
            issues = self._identify_file_issues(
                code_lines, cyclomatic_complexity, function_count, class_count, doc_ratio
            )
            
            relative_path = str(file_path.relative_to(self.root_path))
            
            return FileComplexity(
                path=relative_path,
                lines_of_code=code_lines,
                cyclomatic_complexity=cyclomatic_complexity,
                cognitive_complexity=cognitive_complexity,
                function_count=function_count,
                class_count=class_count,
                import_count=import_count,
                documentation_ratio=doc_ratio,
                complexity_score=complexity_score,
                needs_documentation=needs_documentation,
                issues=issues
            )
        
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of AST."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            # Decision points that increase complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.comprehension):
                complexity += 1
        
        return complexity
    
    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity (more human-focused than cyclomatic)."""
        complexity = 0
        nesting_level = 0
        
        def visit_node(node, level=0):
            nonlocal complexity, nesting_level
            
            # Increment for control structures
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1 + level
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1 + level
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.Lambda):
                complexity += 1
            
            # Increase nesting for certain structures
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
                for child in ast.iter_child_nodes(node):
                    visit_node(child, level + 1)
            else:
                for child in ast.iter_child_nodes(node):
                    visit_node(child, level)
        
        visit_node(tree)
        return complexity
    
    def _count_functions(self, tree: ast.AST) -> int:
        """Count function definitions in AST."""
        return len([node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))])
    
    def _count_classes(self, tree: ast.AST) -> int:
        """Count class definitions in AST."""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
    
    def _count_imports(self, tree: ast.AST) -> int:
        """Count import statements in AST."""
        imports = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports += len(node.names)
            elif isinstance(node, ast.ImportFrom):
                imports += len(node.names)
        return imports
    
    def _count_docstring_lines(self, tree: ast.AST, content: str) -> int:
        """Count lines in docstrings."""
        docstring_lines = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(node)
                if docstring:
                    docstring_lines += len(docstring.split('\n'))
        
        return docstring_lines
    
    def _calculate_complexity_score(self, lines: int, cyclomatic: int, cognitive: int,
                                  functions: int, classes: int, doc_ratio: float) -> int:
        """Calculate overall complexity score for a file."""
        score = 0
        
        # Lines of code contribution
        if lines > 500:
            score += 20
        elif lines > 200:
            score += 10
        elif lines > 100:
            score += 5
        
        # Cyclomatic complexity contribution
        if cyclomatic > 20:
            score += 15
        elif cyclomatic > 10:
            score += 10
        elif cyclomatic > 5:
            score += 5
        
        # Cognitive complexity contribution
        if cognitive > 15:
            score += 15
        elif cognitive > 10:
            score += 10
        elif cognitive > 5:
            score += 5
        
        # Function/class count contribution
        if functions > 20:
            score += 10
        elif functions > 10:
            score += 5
        
        if classes > 5:
            score += 10
        elif classes > 2:
            score += 5
        
        # Documentation penalty
        if doc_ratio < 0.1:
            score += 15
        elif doc_ratio < 0.2:
            score += 10
        elif doc_ratio < 0.3:
            score += 5
        
        return score
    
    def _needs_documentation(self, complexity_score: int, doc_ratio: float,
                           functions: int, classes: int) -> bool:
        """Determine if a file needs better documentation."""
        # High complexity files need documentation
        if complexity_score > 30:
            return True
        
        # Files with many functions/classes need documentation
        if functions > 5 or classes > 2:
            return True
        
        # Files with low documentation ratio need documentation
        if doc_ratio < 0.2:
            return True
        
        return False
    
    def _identify_file_issues(self, lines: int, cyclomatic: int, functions: int,
                            classes: int, doc_ratio: float) -> List[str]:
        """Identify specific issues with a file."""
        issues = []
        
        if lines > 500:
            issues.append("File is very long (>500 lines)")
        elif lines > 200:
            issues.append("File is long (>200 lines)")
        
        if cyclomatic > 20:
            issues.append("Very high cyclomatic complexity")
        elif cyclomatic > 10:
            issues.append("High cyclomatic complexity")
        
        if functions > 20:
            issues.append("Too many functions in one file")
        
        if classes > 5:
            issues.append("Too many classes in one file")
        
        if doc_ratio < 0.1:
            issues.append("Very poor documentation")
        elif doc_ratio < 0.2:
            issues.append("Poor documentation")
        
        return issues
    
    def _analyze_component_complexity(self, component_path: str,
                                    files: List[FileComplexity]) -> ComponentComplexity:
        """Analyze complexity of a component (directory)."""
        total_files = len(files)
        total_lines = sum(f.lines_of_code for f in files)
        
        # Calculate average complexity
        avg_complexity = sum(f.complexity_score for f in files) / total_files if total_files > 0 else 0
        max_complexity = max(f.complexity_score for f in files) if files else 0
        
        # Calculate documentation coverage
        documented_files = len([f for f in files if f.documentation_ratio > 0.2])
        doc_coverage = documented_files / total_files if total_files > 0 else 0
        
        # Calculate component complexity score
        complexity_score = int(avg_complexity + (max_complexity * 0.3))
        
        # Determine priority level
        if complexity_score > 40 or doc_coverage < 0.3:
            priority = 'high'
        elif complexity_score > 25 or doc_coverage < 0.5:
            priority = 'medium'
        else:
            priority = 'low'
        
        # Generate recommendations
        recommendations = self._generate_component_recommendations(files, doc_coverage, avg_complexity)
        
        component_name = Path(component_path).name
        
        return ComponentComplexity(
            name=component_name,
            path=component_path,
            total_files=total_files,
            total_lines=total_lines,
            average_complexity=avg_complexity,
            max_complexity=max_complexity,
            files=files,
            documentation_coverage=doc_coverage,
            complexity_score=complexity_score,
            priority_level=priority,
            recommendations=recommendations
        )
    
    def _generate_component_recommendations(self, files: List[FileComplexity],
                                         doc_coverage: float, avg_complexity: float) -> List[str]:
        """Generate recommendations for improving a component."""
        recommendations = []
        
        # Documentation recommendations
        if doc_coverage < 0.3:
            recommendations.append("Add comprehensive documentation and docstrings")
        elif doc_coverage < 0.5:
            recommendations.append("Improve documentation coverage")
        
        # Complexity recommendations
        if avg_complexity > 30:
            recommendations.append("Refactor complex functions to reduce complexity")
        
        # File-specific recommendations
        large_files = [f for f in files if f.lines_of_code > 300]
        if large_files:
            recommendations.append(f"Split large files: {', '.join(f.path for f in large_files[:3])}")
        
        complex_files = [f for f in files if f.complexity_score > 40]
        if complex_files:
            recommendations.append(f"Simplify complex files: {', '.join(f.path for f in complex_files[:3])}")
        
        return recommendations
    
    def _identify_high_priority_areas(self, components: List[ComponentComplexity]) -> List[str]:
        """Identify components that need immediate attention."""
        high_priority = []
        
        for component in components:
            if component.priority_level == 'high':
                high_priority.append(component.name)
        
        # Sort by complexity score
        high_priority.sort(key=lambda name: next(
            c.complexity_score for c in components if c.name == name
        ), reverse=True)
        
        return high_priority
    
    def _identify_documentation_gaps(self, components: List[ComponentComplexity]) -> List[str]:
        """Identify areas with poor documentation."""
        gaps = []
        
        for component in components:
            if component.documentation_coverage < 0.3:
                gaps.append(f"{component.name} ({component.documentation_coverage:.1%} coverage)")
        
        return gaps
    
    def _identify_complexity_hotspots(self, components: List[ComponentComplexity]) -> List[str]:
        """Identify the most complex areas of the codebase."""
        hotspots = []
        
        # Sort components by complexity
        sorted_components = sorted(components, key=lambda c: c.complexity_score, reverse=True)
        
        for component in sorted_components[:10]:  # Top 10 most complex
            if component.complexity_score > 25:
                hotspots.append(f"{component.name} (score: {component.complexity_score})")
        
        return hotspots
    
    def _generate_recommendations(self, components: List[ComponentComplexity]) -> List[str]:
        """Generate overall project recommendations."""
        recommendations = []
        
        # Overall documentation
        total_components = len(components)
        poorly_documented = len([c for c in components if c.documentation_coverage < 0.3])
        
        if poorly_documented > total_components * 0.5:
            recommendations.append("Implement project-wide documentation standards")
        
        # Complexity issues
        high_complexity = len([c for c in components if c.complexity_score > 30])
        if high_complexity > total_components * 0.3:
            recommendations.append("Establish code complexity guidelines and refactoring plan")
        
        # Specific areas
        api_components = [c for c in components if 'api' in c.name.lower()]
        if api_components and any(c.priority_level == 'high' for c in api_components):
            recommendations.append("Prioritize API documentation and simplification")
        
        core_components = [c for c in components if 'core' in c.name.lower()]
        if core_components and any(c.priority_level == 'high' for c in core_components):
            recommendations.append("Focus on core component documentation and architecture clarity")
        
        return recommendations
    
    def save_analysis(self, analysis: ProjectComplexityReport, output_path: str):
        """Save analysis results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict for JSON serialization
        analysis_dict = asdict(analysis)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Complexity analysis saved to: {output_file}")
    
    def generate_summary_report(self, analysis: ProjectComplexityReport) -> str:
        """Generate a human-readable summary report."""
        report = []
        report.append("# Project Complexity Analysis Report")
        report.append("")
        
        # Overview
        report.append("## Overview")
        report.append("")
        report.append(f"- **Total Files Analyzed:** {analysis.total_files:,}")
        report.append(f"- **Total Lines of Code:** {analysis.total_lines:,}")
        report.append(f"- **Average Complexity Score:** {analysis.average_complexity:.1f}")
        report.append(f"- **Components Analyzed:** {len(analysis.components)}")
        report.append("")
        
        # High Priority Areas
        if analysis.high_priority_areas:
            report.append("## 🚨 High Priority Areas")
            report.append("")
            report.append("These components need immediate attention:")
            report.append("")
            for area in analysis.high_priority_areas:
                component = next(c for c in analysis.components if c.name == area)
                report.append(f"- **{area}** (Score: {component.complexity_score}, Doc: {component.documentation_coverage:.1%})")
            report.append("")
        
        # Documentation Gaps
        if analysis.documentation_gaps:
            report.append("## 📚 Documentation Gaps")
            report.append("")
            report.append("Components with poor documentation coverage:")
            report.append("")
            for gap in analysis.documentation_gaps:
                report.append(f"- {gap}")
            report.append("")
        
        # Complexity Hotspots
        if analysis.complexity_hotspots:
            report.append("## 🔥 Complexity Hotspots")
            report.append("")
            report.append("Most complex components:")
            report.append("")
            for hotspot in analysis.complexity_hotspots:
                report.append(f"- {hotspot}")
            report.append("")
        
        # Recommendations
        if analysis.recommendations:
            report.append("## 💡 Recommendations")
            report.append("")
            for i, rec in enumerate(analysis.recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        # Component Details
        report.append("## Component Details")
        report.append("")
        
        # Sort by priority and complexity
        sorted_components = sorted(
            analysis.components,
            key=lambda c: (c.priority_level == 'high', c.complexity_score),
            reverse=True
        )
        
        for component in sorted_components[:15]:  # Top 15
            report.append(f"### {component.name}")
            report.append("")
            report.append(f"- **Path:** {component.path}")
            report.append(f"- **Priority:** {component.priority_level.upper()}")
            report.append(f"- **Files:** {component.total_files}")
            report.append(f"- **Lines of Code:** {component.total_lines:,}")
            report.append(f"- **Complexity Score:** {component.complexity_score}")
            report.append(f"- **Documentation Coverage:** {component.documentation_coverage:.1%}")
            
            if component.recommendations:
                report.append("")
                report.append("**Recommendations:**")
                for rec in component.recommendations:
                    report.append(f"- {rec}")
            
            # Show most complex files
            complex_files = sorted(component.files, key=lambda f: f.complexity_score, reverse=True)[:3]
            if complex_files:
                report.append("")
                report.append("**Most Complex Files:**")
                for file in complex_files:
                    report.append(f"- {file.path} (Score: {file.complexity_score})")
            
            report.append("")
        
        return "\n".join(report)