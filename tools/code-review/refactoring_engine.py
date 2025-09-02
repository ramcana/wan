"""
Refactoring Recommendation Engine

This module provides intelligent refactoring recommendations based on code analysis
and quality metrics to help improve code maintainability and performance.
"""

import ast
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from enum import Enum
import json
from datetime import datetime


class RefactoringType(Enum):
    """Types of refactoring recommendations"""
    EXTRACT_METHOD = "extract_method"
    EXTRACT_CLASS = "extract_class"
    RENAME_VARIABLE = "rename_variable"
    SIMPLIFY_CONDITIONAL = "simplify_conditional"
    REMOVE_DUPLICATION = "remove_duplication"
    OPTIMIZE_IMPORTS = "optimize_imports"
    ADD_TYPE_HINTS = "add_type_hints"
    IMPROVE_NAMING = "improve_naming"
    REDUCE_COMPLEXITY = "reduce_complexity"
    OPTIMIZE_PERFORMANCE = "optimize_performance"


@dataclass
class RefactoringPattern:
    """Represents a refactoring pattern"""
    pattern_id: str
    name: str
    description: str
    detection_rules: List[str]
    refactoring_steps: List[str]
    benefits: List[str]
    complexity: str  # low, medium, high
    impact: str  # low, medium, high


@dataclass
class RefactoringSuggestion:
    """Represents a specific refactoring suggestion"""
    file_path: str
    start_line: int
    end_line: int
    refactoring_type: RefactoringType
    pattern_id: str
    title: str
    description: str
    current_code: str
    suggested_code: str
    benefits: List[str]
    effort_estimate: str
    confidence: float
    priority: int
    related_suggestions: List[str] = None


class RefactoringEngine:
    """Main refactoring recommendation engine"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.patterns = self._load_refactoring_patterns()
        self.suggestions: List[RefactoringSuggestion] = []
    
    def _load_refactoring_patterns(self) -> List[RefactoringPattern]:
        """Load refactoring patterns from configuration"""
        return [
            RefactoringPattern(
                pattern_id="long_method",
                name="Extract Method",
                description="Break down long methods into smaller, focused methods",
                detection_rules=["method_length > 30", "complexity > 8"],
                refactoring_steps=[
                    "Identify logical blocks within the method",
                    "Extract each block into a separate method",
                    "Replace original code with method calls",
                    "Add appropriate parameters and return values"
                ],
                benefits=[
                    "Improved readability",
                    "Better testability",
                    "Easier maintenance",
                    "Code reusability"
                ],
                complexity="medium",
                impact="high"
            ),
            RefactoringPattern(
                pattern_id="duplicate_code",
                name="Remove Code Duplication",
                description="Extract common code into reusable functions or classes",
                detection_rules=["duplicate_lines > 5", "similarity > 0.8"],
                refactoring_steps=[
                    "Identify duplicated code blocks",
                    "Extract common functionality",
                    "Create shared function or method",
                    "Replace duplicated code with calls to shared function"
                ],
                benefits=[
                    "Reduced maintenance burden",
                    "Consistent behavior",
                    "Smaller codebase",
                    "Single point of change"
                ],
                complexity="medium",
                impact="high"
            ),
            RefactoringPattern(
                pattern_id="complex_conditional",
                name="Simplify Complex Conditionals",
                description="Break down complex conditional statements",
                detection_rules=["nested_conditions > 3", "boolean_complexity > 5"],
                refactoring_steps=[
                    "Extract complex conditions into well-named variables",
                    "Use guard clauses to reduce nesting",
                    "Consider using polymorphism for complex type checking",
                    "Extract condition logic into separate methods"
                ],
                benefits=[
                    "Improved readability",
                    "Easier debugging",
                    "Better testability",
                    "Reduced cognitive load"
                ],
                complexity="low",
                impact="medium"
            ),
            RefactoringPattern(
                pattern_id="large_class",
                name="Extract Class",
                description="Break down large classes with multiple responsibilities",
                detection_rules=["class_length > 200", "method_count > 15"],
                refactoring_steps=[
                    "Identify distinct responsibilities",
                    "Group related methods and attributes",
                    "Extract each responsibility into a separate class",
                    "Update references and dependencies"
                ],
                benefits=[
                    "Single Responsibility Principle",
                    "Better organization",
                    "Improved testability",
                    "Easier to understand"
                ],
                complexity="high",
                impact="high"
            ),
            RefactoringPattern(
                pattern_id="poor_naming",
                name="Improve Naming",
                description="Use more descriptive and meaningful names",
                detection_rules=["short_names", "unclear_abbreviations", "misleading_names"],
                refactoring_steps=[
                    "Identify poorly named variables, functions, and classes",
                    "Choose descriptive, intention-revealing names",
                    "Update all references consistently",
                    "Ensure names follow project conventions"
                ],
                benefits=[
                    "Self-documenting code",
                    "Reduced need for comments",
                    "Better code comprehension",
                    "Easier maintenance"
                ],
                complexity="low",
                impact="medium"
            )
        ]
    
    def analyze_file(self, file_path: str) -> List[RefactoringSuggestion]:
        """Analyze a file and generate refactoring suggestions"""
        if not file_path.endswith('.py'):
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            lines = content.split('\n')
            
            suggestions = []
            suggestions.extend(self._analyze_methods(file_path, tree, lines))
            suggestions.extend(self._analyze_classes(file_path, tree, lines))
            suggestions.extend(self._analyze_conditionals(file_path, tree, lines))
            suggestions.extend(self._analyze_naming(file_path, tree, lines))
            suggestions.extend(self._analyze_imports(file_path, tree, lines))
            
            return suggestions
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return []
    
    def _analyze_methods(self, file_path: str, tree: ast.AST, lines: List[str]) -> List[RefactoringSuggestion]:
        """Analyze methods for refactoring opportunities"""
        suggestions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                method_length = self._get_method_length(node)
                complexity = self._calculate_complexity(node)
                
                # Long method refactoring
                if method_length > 30:
                    suggestions.append(RefactoringSuggestion(
                        file_path=file_path,
                        start_line=node.lineno,
                        end_line=getattr(node, 'end_lineno', node.lineno + method_length),
                        refactoring_type=RefactoringType.EXTRACT_METHOD,
                        pattern_id="long_method",
                        title=f"Extract methods from long function '{node.name}'",
                        description=f"Function '{node.name}' is {method_length} lines long. Consider breaking it into smaller, focused methods.",
                        current_code=self._get_node_code(node, lines),
                        suggested_code=self._generate_extract_method_suggestion(node, lines),
                        benefits=[
                            "Improved readability",
                            "Better testability",
                            "Easier maintenance"
                        ],
                        effort_estimate="medium",
                        confidence=0.8,
                        priority=2 if method_length > 50 else 3
                    ))
                
                # High complexity refactoring
                if complexity > 10:
                    suggestions.append(RefactoringSuggestion(
                        file_path=file_path,
                        start_line=node.lineno,
                        end_line=getattr(node, 'end_lineno', node.lineno + method_length),
                        refactoring_type=RefactoringType.REDUCE_COMPLEXITY,
                        pattern_id="complex_method",
                        title=f"Reduce complexity of function '{node.name}'",
                        description=f"Function '{node.name}' has high cyclomatic complexity ({complexity}). Consider simplifying the logic.",
                        current_code=self._get_node_code(node, lines),
                        suggested_code=self._generate_complexity_reduction_suggestion(node, lines),
                        benefits=[
                            "Easier to understand",
                            "Fewer bugs",
                            "Better testability"
                        ],
                        effort_estimate="high",
                        confidence=0.7,
                        priority=1 if complexity > 15 else 2
                    ))
        
        return suggestions
    
    def _analyze_classes(self, file_path: str, tree: ast.AST, lines: List[str]) -> List[RefactoringSuggestion]:
        """Analyze classes for refactoring opportunities"""
        suggestions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_length = self._get_class_length(node)
                method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                
                # Large class refactoring
                if class_length > 200 or method_count > 15:
                    suggestions.append(RefactoringSuggestion(
                        file_path=file_path,
                        start_line=node.lineno,
                        end_line=getattr(node, 'end_lineno', node.lineno + class_length),
                        refactoring_type=RefactoringType.EXTRACT_CLASS,
                        pattern_id="large_class",
                        title=f"Extract classes from large class '{node.name}'",
                        description=f"Class '{node.name}' is large ({class_length} lines, {method_count} methods). Consider extracting separate responsibilities.",
                        current_code=self._get_node_code(node, lines),
                        suggested_code=self._generate_extract_class_suggestion(node, lines),
                        benefits=[
                            "Single Responsibility Principle",
                            "Better organization",
                            "Improved testability"
                        ],
                        effort_estimate="high",
                        confidence=0.6,
                        priority=2
                    ))
        
        return suggestions
    
    def _analyze_conditionals(self, file_path: str, tree: ast.AST, lines: List[str]) -> List[RefactoringSuggestion]:
        """Analyze conditional statements for refactoring opportunities"""
        suggestions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                nesting_level = self._get_nesting_level(node)
                boolean_complexity = self._get_boolean_complexity(node.test)
                
                if nesting_level > 3 or boolean_complexity > 3:
                    suggestions.append(RefactoringSuggestion(
                        file_path=file_path,
                        start_line=node.lineno,
                        end_line=getattr(node, 'end_lineno', node.lineno + 5),
                        refactoring_type=RefactoringType.SIMPLIFY_CONDITIONAL,
                        pattern_id="complex_conditional",
                        title="Simplify complex conditional statement",
                        description=f"Complex conditional with nesting level {nesting_level} and boolean complexity {boolean_complexity}",
                        current_code=self._get_node_code(node, lines),
                        suggested_code=self._generate_conditional_simplification(node, lines),
                        benefits=[
                            "Improved readability",
                            "Easier debugging",
                            "Better testability"
                        ],
                        effort_estimate="low",
                        confidence=0.8,
                        priority=3
                    ))
        
        return suggestions
    
    def _analyze_naming(self, file_path: str, tree: ast.AST, lines: List[str]) -> List[RefactoringSuggestion]:
        """Analyze naming for improvement opportunities"""
        suggestions = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if self._is_poor_name(node.name):
                    suggestions.append(RefactoringSuggestion(
                        file_path=file_path,
                        start_line=node.lineno,
                        end_line=node.lineno,
                        refactoring_type=RefactoringType.IMPROVE_NAMING,
                        pattern_id="poor_naming",
                        title=f"Improve name of {'function' if isinstance(node, ast.FunctionDef) else 'class'} '{node.name}'",
                        description=f"The name '{node.name}' could be more descriptive and meaningful",
                        current_code=f"{'def' if isinstance(node, ast.FunctionDef) else 'class'} {node.name}",
                        suggested_code=f"# Consider a more descriptive name like:\n# {'def' if isinstance(node, ast.FunctionDef) else 'class'} {self._suggest_better_name(node.name)}",
                        benefits=[
                            "Self-documenting code",
                            "Better code comprehension",
                            "Easier maintenance"
                        ],
                        effort_estimate="low",
                        confidence=0.6,
                        priority=4
                    ))
        
        return suggestions
    
    def _analyze_imports(self, file_path: str, tree: ast.AST, lines: List[str]) -> List[RefactoringSuggestion]:
        """Analyze imports for optimization opportunities"""
        suggestions = []
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(node)
        
        if len(imports) > 20:
            suggestions.append(RefactoringSuggestion(
                file_path=file_path,
                start_line=1,
                end_line=len([n for n in imports]),
                refactoring_type=RefactoringType.OPTIMIZE_IMPORTS,
                pattern_id="import_optimization",
                title="Optimize import statements",
                description=f"File has {len(imports)} import statements. Consider organizing and optimizing them.",
                current_code="# Current imports...",
                suggested_code="# Organize imports by: standard library, third-party, local imports",
                benefits=[
                    "Better organization",
                    "Faster import times",
                    "Cleaner code structure"
                ],
                effort_estimate="low",
                confidence=0.9,
                priority=4
            ))
        
        return suggestions
    
    def _get_method_length(self, node: ast.FunctionDef) -> int:
        """Calculate method length in lines"""
        if hasattr(node, 'end_lineno'):
            return node.end_lineno - node.lineno + 1
        return 20  # Estimate
    
    def _get_class_length(self, node: ast.ClassDef) -> int:
        """Calculate class length in lines"""
        if hasattr(node, 'end_lineno'):
            return node.end_lineno - node.lineno + 1
        return 50  # Estimate
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
        return complexity
    
    def _get_nesting_level(self, node: ast.If) -> int:
        """Get nesting level of conditional"""
        level = 0
        current = node
        while current:
            if isinstance(current, ast.If):
                level += 1
            current = getattr(current, 'orelse', None)
            if current and len(current) == 1 and isinstance(current[0], ast.If):
                current = current[0]
            else:
                break
        return level
    
    def _get_boolean_complexity(self, node: ast.AST) -> int:
        """Calculate boolean expression complexity"""
        complexity = 0
        for child in ast.walk(node):
            if isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.Compare):
                complexity += len(child.comparators)
        return max(1, complexity)
    
    def _is_poor_name(self, name: str) -> bool:
        """Check if a name is poorly chosen"""
        poor_patterns = [
            r'^[a-z]$',  # Single letter
            r'^[a-z]{1,2}$',  # Very short
            r'.*\d+$',  # Ends with number
            r'^(data|info|obj|item|thing|stuff)$',  # Generic names
        ]
        
        for pattern in poor_patterns:
            if re.match(pattern, name):
                return True
        return False
    
    def _suggest_better_name(self, name: str) -> str:
        """Suggest a better name"""
        suggestions = {
            'data': 'user_data',
            'info': 'user_info',
            'obj': 'user_object',
            'item': 'list_item',
            'thing': 'entity',
            'stuff': 'items'
        }
        return suggestions.get(name, f"descriptive_{name}")
    
    def _get_node_code(self, node: ast.AST, lines: List[str]) -> str:
        """Extract code for a given AST node"""
        start_line = node.lineno - 1
        end_line = getattr(node, 'end_lineno', start_line + 5) - 1
        return '\n'.join(lines[start_line:end_line + 1])
    
    def _generate_extract_method_suggestion(self, node: ast.FunctionDef, lines: List[str]) -> str:
        """Generate extract method refactoring suggestion"""
        return f"""# Consider extracting logical blocks into separate methods:

def {node.name}(self, ...):
    \"\"\"Main method with extracted logic\"\"\"
    self._validate_input()
    result = self._process_data()
    self._handle_result(result)
    return result

def _validate_input(self):
    \"\"\"Validate input parameters\"\"\"
    # Extracted validation logic
    pass

def _process_data(self):
    \"\"\"Process the main data\"\"\"
    # Extracted processing logic
    pass

def _handle_result(self, result):
    \"\"\"Handle the processing result\"\"\"
    # Extracted result handling logic
    pass"""
    
    def _generate_complexity_reduction_suggestion(self, node: ast.FunctionDef, lines: List[str]) -> str:
        """Generate complexity reduction suggestion"""
        return f"""# Consider simplifying complex logic:

def {node.name}(self, ...):
    \"\"\"Simplified method with reduced complexity\"\"\"
    # Use guard clauses to reduce nesting
    if not self._is_valid_input():
        return None
    
    # Extract complex conditions
    should_process = self._should_process_data()
    if should_process:
        return self._process_data()
    
    return self._handle_alternative_case()

def _is_valid_input(self) -> bool:
    \"\"\"Check if input is valid\"\"\"
    # Extracted validation logic
    pass

def _should_process_data(self) -> bool:
    \"\"\"Determine if data should be processed\"\"\"
    # Extracted condition logic
    pass"""
    
    def _generate_extract_class_suggestion(self, node: ast.ClassDef, lines: List[str]) -> str:
        """Generate extract class refactoring suggestion"""
        return f"""# Consider extracting responsibilities into separate classes:

class {node.name}:
    \"\"\"Main class with core responsibility\"\"\"
    def __init__(self):
        self.data_manager = {node.name}DataManager()
        self.validator = {node.name}Validator()
        self.processor = {node.name}Processor()

class {node.name}DataManager:
    \"\"\"Handles data management operations\"\"\"
    pass

class {node.name}Validator:
    \"\"\"Handles validation operations\"\"\"
    pass

class {node.name}Processor:
    \"\"\"Handles processing operations\"\"\"
    pass"""
    
    def _generate_conditional_simplification(self, node: ast.If, lines: List[str]) -> str:
        """Generate conditional simplification suggestion"""
        return f"""# Simplify complex conditional:

# Before: Complex nested conditions
# After: Use guard clauses and extracted conditions

def method_name(self):
    # Use guard clauses
    if not condition1:
        return early_return_value
    
    if not condition2:
        return another_early_return
    
    # Main logic with reduced nesting
    if self._is_special_case():
        return self._handle_special_case()
    
    return self._handle_normal_case()

def _is_special_case(self) -> bool:
    \"\"\"Check for special case conditions\"\"\"
    return complex_condition_1 and complex_condition_2"""
    
    def generate_suggestions_report(self, output_path: str = "refactoring_suggestions.json"):
        """Generate refactoring suggestions report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_suggestions": len(self.suggestions),
            "suggestions_by_type": {},
            "suggestions_by_priority": {},
            "suggestions": []
        }
        
        # Group by type and priority
        for suggestion in self.suggestions:
            type_key = suggestion.refactoring_type.value
            report["suggestions_by_type"][type_key] = report["suggestions_by_type"].get(type_key, 0) + 1
            
            priority_key = f"priority_{suggestion.priority}"
            report["suggestions_by_priority"][priority_key] = report["suggestions_by_priority"].get(priority_key, 0) + 1
            
            report["suggestions"].append({
                "file_path": suggestion.file_path,
                "start_line": suggestion.start_line,
                "end_line": suggestion.end_line,
                "type": suggestion.refactoring_type.value,
                "pattern_id": suggestion.pattern_id,
                "title": suggestion.title,
                "description": suggestion.description,
                "benefits": suggestion.benefits,
                "effort_estimate": suggestion.effort_estimate,
                "confidence": suggestion.confidence,
                "priority": suggestion.priority
            })
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report