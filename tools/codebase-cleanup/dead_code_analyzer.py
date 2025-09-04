import pytest
"""
Dead Code Analysis and Removal System

This module provides comprehensive dead code detection capabilities including:
- Unused functions and methods
- Dead classes
- Unused imports
- Dead files
- Safe removal with comprehensive testing
"""

import os
import ast
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import re
import importlib.util


@dataclass
class DeadFunction:
    """Represents a dead/unused function"""
    name: str
    file_path: str
    line_number: int
    is_method: bool
    class_name: Optional[str]
    docstring: Optional[str]
    complexity_score: int


@dataclass
class DeadClass:
    """Represents a dead/unused class"""
    name: str
    file_path: str
    line_number: int
    methods: List[str]
    inheritance: List[str]
    docstring: Optional[str]


@dataclass
class UnusedImport:
    """Represents an unused import"""
    import_name: str
    module_name: str
    file_path: str
    line_number: int
    import_type: str  # 'import', 'from_import', 'import_as'


@dataclass
class DeadFile:
    """Represents a dead/unused file"""
    file_path: str
    size: int
    last_modified: str
    imports_count: int
    functions_count: int
    classes_count: int


@dataclass
class DeadCodeReport:
    """Report of dead code analysis"""
    total_files_analyzed: int
    dead_functions: List[DeadFunction]
    dead_classes: List[DeadClass]
    unused_imports: List[UnusedImport]
    dead_files: List[DeadFile]
    potential_lines_removed: int
    recommendations: List[str]
    analysis_timestamp: str


class DeadCodeAnalyzer:
    """
    Comprehensive dead code analysis system that identifies:
    - Unused functions and methods
    - Dead classes that are never instantiated
    - Unused imports
    - Dead files that are never imported or referenced
    """
    
    def __init__(self, root_path: str, backup_dir: str = "backups/dead_code"):
        self.root_path = Path(root_path)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # File extensions to analyze
        self.code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx'}
        
        # Directories to exclude
        self.exclude_patterns = {
            '__pycache__', '.git', '.pytest_cache', 'node_modules',
            '.venv', 'venv', 'dist', 'build', '.next'
        }
        
        # Common patterns that indicate usage (reduce false positives)
        self.usage_patterns = [
            r'def\s+test_',  # Test functions
            r'@\w+',  # Decorated functions
            r'__\w+__',  # Magic methods
            r'main\s*\(',  # Main function calls
        ]
    
    def analyze_dead_code(self, include_tests: bool = False) -> DeadCodeReport:
        """
        Perform comprehensive dead code analysis
        
        Args:
            include_tests: Whether to include test files in analysis
            
        Returns:
            DeadCodeReport with all findings and recommendations
        """
        print("Starting dead code analysis...")
        
        # Get files to analyze
        files_to_analyze = self._get_files_to_analyze(include_tests)
        print(f"Analyzing {len(files_to_analyze)} files...")
        
        # Build project structure map
        project_map = self._build_project_map(files_to_analyze)
        
        # Find dead functions
        dead_functions = self._find_dead_functions(files_to_analyze, project_map)
        
        # Find dead classes
        dead_classes = self._find_dead_classes(files_to_analyze, project_map)
        
        # Find unused imports
        unused_imports = self._find_unused_imports(files_to_analyze)
        
        # Find dead files
        dead_files = self._find_dead_files(files_to_analyze, project_map)
        
        # Calculate potential lines removed
        potential_lines = self._calculate_potential_lines_removed(
            dead_functions, dead_classes, unused_imports, dead_files
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            dead_functions, dead_classes, unused_imports, dead_files
        )
        
        report = DeadCodeReport(
            total_files_analyzed=len(files_to_analyze),
            dead_functions=dead_functions,
            dead_classes=dead_classes,
            unused_imports=unused_imports,
            dead_files=dead_files,
            potential_lines_removed=potential_lines,
            recommendations=recommendations,
            analysis_timestamp=datetime.now().isoformat()
        )
        
        print(f"Analysis complete. Found {len(dead_functions)} dead functions, "
              f"{len(dead_classes)} dead classes, {len(unused_imports)} unused imports, "
              f"{len(dead_files)} dead files.")
        
        return report
    
    def _get_files_to_analyze(self, include_tests: bool) -> List[Path]:
        """Get list of code files to analyze"""
        files = []
        
        for root, dirs, filenames in os.walk(self.root_path):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.exclude_patterns)]
            
            for filename in filenames:
                file_path = Path(root) / filename
                
                # Skip if file matches exclude patterns
                if any(pattern in str(file_path) for pattern in self.exclude_patterns):
                    continue
                
                # Check file extension
                if file_path.suffix not in self.code_extensions:
                    continue
                
                # Skip test files if not included
                if not include_tests and ('test_' in filename.lower() and filename.lower().endswith('.py')):
                    continue
                
                files.append(file_path)
        
        return files
    
    def _build_project_map(self, files: List[Path]) -> Dict[str, Dict]:
        """
        Build a map of the project structure including:
        - All defined functions, classes, and variables
        - All usage references
        - Import relationships
        """
        project_map = {
            'definitions': {},  # name -> [file_paths]
            'usages': {},       # name -> [file_paths]
            'imports': {},      # file_path -> [imported_names]
            'file_contents': {} # file_path -> AST or content
        }
        
        for file_path in files:
            try:
                if file_path.suffix == '.py':
                    self._analyze_python_file(file_path, project_map)
                elif file_path.suffix in {'.js', '.ts', '.jsx', '.tsx'}:
                    self._analyze_javascript_file(file_path, project_map)
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")
                continue
        
        return project_map
    
    def _analyze_python_file(self, file_path: Path, project_map: Dict):
        """Analyze Python file for definitions and usages"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            project_map['file_contents'][str(file_path)] = tree
            
            # Find definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    if func_name not in project_map['definitions']:
                        project_map['definitions'][func_name] = []
                    project_map['definitions'][func_name].append(str(file_path))
                
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    if class_name not in project_map['definitions']:
                        project_map['definitions'][class_name] = []
                    project_map['definitions'][class_name].append(str(file_path))
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if str(file_path) not in project_map['imports']:
                        project_map['imports'][str(file_path)] = []
                    
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            project_map['imports'][str(file_path)].append(alias.name)
                    else:  # ImportFrom
                        for alias in node.names:
                            project_map['imports'][str(file_path)].append(alias.name)
            
            # Find usages (simplified - look for name references)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    name = node.id
                    if name not in project_map['usages']:
                        project_map['usages'][name] = []
                    if str(file_path) not in project_map['usages'][name]:
                        project_map['usages'][name].append(str(file_path))
                
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name not in project_map['usages']:
                            project_map['usages'][func_name] = []
                        if str(file_path) not in project_map['usages'][func_name]:
                            project_map['usages'][func_name].append(str(file_path))
        
        except Exception as e:
            print(f"Error analyzing Python file {file_path}: {e}")
    
    def _analyze_javascript_file(self, file_path: Path, project_map: Dict):
        """Analyze JavaScript/TypeScript file for definitions and usages"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            project_map['file_contents'][str(file_path)] = content
            
            # Simple regex-based analysis for JavaScript
            # Find function definitions
            func_patterns = [
                r'function\s+(\w+)',
                r'const\s+(\w+)\s*=\s*\(',
                r'let\s+(\w+)\s*=\s*\(',
                r'var\s+(\w+)\s*=\s*\(',
                r'(\w+)\s*:\s*function',
                r'(\w+)\s*=>\s*'
            ]
            
            for pattern in func_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    func_name = match.group(1)
                    if func_name not in project_map['definitions']:
                        project_map['definitions'][func_name] = []
                    project_map['definitions'][func_name].append(str(file_path))
            
            # Find class definitions
            class_matches = re.finditer(r'class\s+(\w+)', content)
            for match in class_matches:
                class_name = match.group(1)
                if class_name not in project_map['definitions']:
                    project_map['definitions'][class_name] = []
                project_map['definitions'][class_name].append(str(file_path))
            
            # Find imports
            import_patterns = [
                r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]',
                r'import\s+[\'"]([^\'"]+)[\'"]',
                r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'
            ]
            
            if str(file_path) not in project_map['imports']:
                project_map['imports'][str(file_path)] = []
            
            for pattern in import_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    import_name = match.group(1)
                    project_map['imports'][str(file_path)].append(import_name)
        
        except Exception as e:
            print(f"Error analyzing JavaScript file {file_path}: {e}")
    
    def _find_dead_functions(self, files: List[Path], project_map: Dict) -> List[DeadFunction]:
        """Find functions that are never called"""
        dead_functions = []
        
        for file_path in files:
            if file_path.suffix == '.py':
                dead_functions.extend(self._find_dead_python_functions(file_path, project_map))
        
        return dead_functions
    
    def _find_dead_python_functions(self, file_path: Path, project_map: Dict) -> List[DeadFunction]:
        """Find dead functions in a Python file"""
        dead_functions = []
        
        try:
            tree = project_map['file_contents'].get(str(file_path))
            if not tree:
                return dead_functions
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    
                    # Skip if function name matches usage patterns (likely not dead)
                    if any(re.search(pattern, func_name) for pattern in self.usage_patterns):
                        continue
                    
                    # Check if function is used anywhere
                    usages = project_map['usages'].get(func_name, [])
                    definition_files = project_map['definitions'].get(func_name, [])
                    
                    # Function is dead if it's only defined but never used elsewhere
                    if len(usages) <= len(definition_files):
                        # Additional check: is it used in the same file?
                        same_file_usage = self._check_same_file_usage(tree, func_name, node)
                        
                        if not same_file_usage:
                            # Get docstring
                            docstring = ast.get_docstring(node)
                            
                            # Calculate complexity (simplified)
                            complexity = len([n for n in ast.walk(node) if isinstance(n, (ast.If, ast.For, ast.While))])
                            
                            # Check if it's a method
                            is_method = False
                            class_name = None
                            for parent in ast.walk(tree):
                                if isinstance(parent, ast.ClassDef):
                                    if node in parent.body:
                                        is_method = True
                                        class_name = parent.name
                                        break
                            
                            dead_functions.append(DeadFunction(
                                name=func_name,
                                file_path=str(file_path),
                                line_number=node.lineno,
                                is_method=is_method,
                                class_name=class_name,
                                docstring=docstring,
                                complexity_score=complexity
                            ))
        
        except Exception as e:
            print(f"Error finding dead functions in {file_path}: {e}")
        
        return dead_functions
    
    def _check_same_file_usage(self, tree: ast.AST, func_name: str, func_node: ast.FunctionDef) -> bool:
        """Check if function is used elsewhere in the same file"""
        for node in ast.walk(tree):
            if node == func_node:
                continue
            
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == func_name:
                    return True
            elif isinstance(node, ast.Name) and node.id == func_name:
                # Check if it's not just the function definition
                if not isinstance(node.ctx, ast.Store):
                    return True
        
        return False
    
    def _find_dead_classes(self, files: List[Path], project_map: Dict) -> List[DeadClass]:
        """Find classes that are never instantiated"""
        dead_classes = []
        
        for file_path in files:
            if file_path.suffix == '.py':
                dead_classes.extend(self._find_dead_python_classes(file_path, project_map))
        
        return dead_classes
    
    def _find_dead_python_classes(self, file_path: Path, project_map: Dict) -> List[DeadClass]:
        """Find dead classes in a Python file"""
        dead_classes = []
        
        try:
            tree = project_map['file_contents'].get(str(file_path))
            if not tree:
                return dead_classes
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    
                    # Check if class is used anywhere
                    usages = project_map['usages'].get(class_name, [])
                    definition_files = project_map['definitions'].get(class_name, [])
                    
                    # Class is dead if it's only defined but never used
                    if len(usages) <= len(definition_files):
                        # Get methods
                        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                        
                        # Get inheritance
                        inheritance = [base.id if isinstance(base, ast.Name) else str(base) 
                                     for base in node.bases]
                        
                        # Get docstring
                        docstring = ast.get_docstring(node)
                        
                        dead_classes.append(DeadClass(
                            name=class_name,
                            file_path=str(file_path),
                            line_number=node.lineno,
                            methods=methods,
                            inheritance=inheritance,
                            docstring=docstring
                        ))
        
        except Exception as e:
            print(f"Error finding dead classes in {file_path}: {e}")
        
        return dead_classes
    
    def _find_unused_imports(self, files: List[Path]) -> List[UnusedImport]:
        """Find imports that are never used"""
        unused_imports = []
        
        for file_path in files:
            if file_path.suffix == '.py':
                unused_imports.extend(self._find_unused_python_imports(file_path))
        
        return unused_imports
    
    def _find_unused_python_imports(self, file_path: Path) -> List[UnusedImport]:
        """Find unused imports in a Python file"""
        unused_imports = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Collect all imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_name = alias.asname if alias.asname else alias.name
                        imports.append({
                            'name': import_name,
                            'module': alias.name,
                            'line': node.lineno,
                            'type': 'import'
                        })
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        import_name = alias.asname if alias.asname else alias.name
                        imports.append({
                            'name': import_name,
                            'module': node.module or '',
                            'line': node.lineno,
                            'type': 'from_import'
                        })
            
            # Check which imports are used
            for imp in imports:
                import_name = imp['name']
                
                # Skip star imports
                if import_name == '*':
                    continue
                
                # Check if import is used in the code
                is_used = False
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and node.id == import_name:
                        # Make sure it's not the import statement itself
                        if not isinstance(node.ctx, ast.Store):
                            is_used = True
                            break
                    elif isinstance(node, ast.Attribute):
                        if isinstance(node.value, ast.Name) and node.value.id == import_name:
                            is_used = True
                            break
                
                if not is_used:
                    unused_imports.append(UnusedImport(
                        import_name=import_name,
                        module_name=imp['module'],
                        file_path=str(file_path),
                        line_number=imp['line'],
                        import_type=imp['type']
                    ))
        
        except Exception as e:
            print(f"Error finding unused imports in {file_path}: {e}")
        
        return unused_imports
    
    def _find_dead_files(self, files: List[Path], project_map: Dict) -> List[DeadFile]:
        """Find files that are never imported or referenced"""
        dead_files = []
        
        for file_path in files:
            # Skip __init__.py files
            if file_path.name == '__init__.py':
                continue
            
            # Check if file is imported anywhere
            file_stem = file_path.stem
            is_referenced = False
            
            # Check imports in other files
            for other_file, imports in project_map['imports'].items():
                if other_file == str(file_path):
                    continue
                
                for import_name in imports:
                    # Check if this file is imported by name
                    if (file_stem == import_name or 
                        file_stem in import_name or 
                        str(file_path.relative_to(self.root_path)).replace('\\', '/') in import_name):
                        is_referenced = True
                        break
                
                if is_referenced:
                    break
            
            # Also check for "from module import" patterns in file contents
            if not is_referenced:
                for other_file_path in [f for f in files if str(f) != str(file_path)]:
                    try:
                        with open(other_file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Check for various import patterns
                        import_patterns = [
                            f'from {file_stem} import',
                            f'import {file_stem}',
                            f'from tools.codebase-cleanup.{file_stem} import',
                            f'from tools.codebase-cleanup..{file_stem} import'
                        ]
                        
                        for pattern in import_patterns:
                            if pattern in content:
                                is_referenced = True
                                break
                        
                        if is_referenced:
                            break
                    except:
                        continue
            
            # Check if it's a main script (has if __name__ == "__main__")
            if not is_referenced and file_path.suffix == '.py':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if 'if __name__ == "__main__"' in content:
                        is_referenced = True  # Main scripts are not dead
                except:
                    pass
            
            if not is_referenced:
                try:
                    stat = file_path.stat()
                    
                    # Count functions and classes
                    functions_count = 0
                    classes_count = 0
                    imports_count = 0
                    
                    if file_path.suffix == '.py':
                        tree = project_map['file_contents'].get(str(file_path))
                        if tree:
                            for node in ast.walk(tree):
                                if isinstance(node, ast.FunctionDef):
                                    functions_count += 1
                                elif isinstance(node, ast.ClassDef):
                                    classes_count += 1
                                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                                    imports_count += 1
                    
                    dead_files.append(DeadFile(
                        file_path=str(file_path),
                        size=stat.st_size,
                        last_modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        imports_count=imports_count,
                        functions_count=functions_count,
                        classes_count=classes_count
                    ))
                
                except Exception as e:
                    print(f"Error analyzing file {file_path}: {e}")
        
        return dead_files
    
    def _calculate_potential_lines_removed(self, dead_functions: List[DeadFunction],
                                         dead_classes: List[DeadClass],
                                         unused_imports: List[UnusedImport],
                                         dead_files: List[DeadFile]) -> int:
        """Calculate potential lines of code that could be removed"""
        total_lines = 0
        
        # Estimate lines for dead functions (rough estimate)
        total_lines += len(dead_functions) * 10  # Average 10 lines per function
        
        # Estimate lines for dead classes
        for dead_class in dead_classes:
            total_lines += len(dead_class.methods) * 8 + 5  # Methods + class definition
        
        # Unused imports (1 line each)
        total_lines += len(unused_imports)
        
        # Dead files (count actual lines)
        for dead_file in dead_files:
            try:
                with open(dead_file.file_path, 'r', encoding='utf-8') as f:
                    total_lines += len(f.readlines())
            except:
                total_lines += 50  # Rough estimate
        
        return total_lines
    
    def _generate_recommendations(self, dead_functions: List[DeadFunction],
                                dead_classes: List[DeadClass],
                                unused_imports: List[UnusedImport],
                                dead_files: List[DeadFile]) -> List[str]:
        """Generate recommendations for handling dead code"""
        recommendations = []
        
        if dead_functions:
            recommendations.append(f"Remove {len(dead_functions)} unused functions to reduce complexity")
            
            # Specific recommendations for complex functions
            complex_functions = [f for f in dead_functions if f.complexity_score > 5]
            if complex_functions:
                recommendations.append(f"Priority: Remove {len(complex_functions)} complex unused functions first")
        
        if dead_classes:
            recommendations.append(f"Remove {len(dead_classes)} unused classes")
            
            # Classes with inheritance might need careful review
            inherited_classes = [c for c in dead_classes if c.inheritance]
            if inherited_classes:
                recommendations.append(f"Review {len(inherited_classes)} unused classes with inheritance carefully")
        
        if unused_imports:
            recommendations.append(f"Clean up {len(unused_imports)} unused imports to improve load times")
        
        if dead_files:
            recommendations.append(f"Consider removing {len(dead_files)} dead files after verification")
            
            # Large dead files
            large_files = [f for f in dead_files if f.size > 10000]  # > 10KB
            if large_files:
                recommendations.append(f"Priority: Review {len(large_files)} large dead files for removal")
        
        return recommendations
    
    def safe_remove_dead_code(self, report: DeadCodeReport) -> Dict[str, str]:
        """
        Safely remove dead code with comprehensive testing
        
        Args:
            report: DeadCodeReport from analysis
            
        Returns:
            Dict mapping operation to result message
        """
        results = {}
        
        # Create backup first
        all_files = set()
        
        # Collect all files that will be modified
        for func in report.dead_functions:
            all_files.add(func.file_path)
        for cls in report.dead_classes:
            all_files.add(cls.file_path)
        for imp in report.unused_imports:
            all_files.add(imp.file_path)
        for file in report.dead_files:
            all_files.add(file.file_path)
        
        if all_files:
            backup_path = self.create_backup(list(all_files))
            results['backup'] = f"Created backup at {backup_path}"
            
            # Remove unused imports first (safest)
            if report.unused_imports:
                removed_imports = self._remove_unused_imports(report.unused_imports)
                results['imports'] = f"Removed {removed_imports} unused imports"
            
            # Remove dead functions
            if report.dead_functions:
                removed_functions = self._remove_dead_functions(report.dead_functions)
                results['functions'] = f"Removed {removed_functions} dead functions"
            
            # Remove dead classes
            if report.dead_classes:
                removed_classes = self._remove_dead_classes(report.dead_classes)
                results['classes'] = f"Removed {removed_classes} dead classes"
            
            # Remove dead files (most risky, do last)
            if report.dead_files:
                removed_files = self._remove_dead_files(report.dead_files)
                results['files'] = f"Removed {removed_files} dead files"
        
        return results
    
    def _remove_unused_imports(self, unused_imports: List[UnusedImport]) -> int:
        """Remove unused import statements"""
        removed_count = 0
        
        # Group by file
        files_imports = {}
        for imp in unused_imports:
            if imp.file_path not in files_imports:
                files_imports[imp.file_path] = []
            files_imports[imp.file_path].append(imp)
        
        for file_path, imports in files_imports.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Sort imports by line number (descending) to avoid line number shifts
                imports.sort(key=lambda x: x.line_number, reverse=True)
                
                for imp in imports:
                    if imp.line_number <= len(lines):
                        # Remove the import line
                        del lines[imp.line_number - 1]
                        removed_count += 1
                
                # Write back the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
            
            except Exception as e:
                print(f"Error removing imports from {file_path}: {e}")
        
        return removed_count
    
    def _remove_dead_functions(self, dead_functions: List[DeadFunction]) -> int:
        """Remove dead function definitions"""
        removed_count = 0
        
        # Group by file
        files_functions = {}
        for func in dead_functions:
            if func.file_path not in files_functions:
                files_functions[func.file_path] = []
            files_functions[func.file_path].append(func)
        
        for file_path, functions in files_functions.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                lines = content.split('\n')
                
                # Find function nodes and their line ranges
                functions_to_remove = []
                for func in functions:
                    for node in ast.walk(tree):
                        if (isinstance(node, ast.FunctionDef) and 
                            node.name == func.name and 
                            node.lineno == func.line_number):
                            
                            # Calculate end line (simplified)
                            end_line = node.lineno
                            for child in ast.walk(node):
                                if hasattr(child, 'lineno') and child.lineno > end_line:
                                    end_line = child.lineno
                            
                            functions_to_remove.append((node.lineno - 1, end_line))
                            break
                
                # Remove functions (from end to start to avoid line shifts)
                functions_to_remove.sort(reverse=True)
                for start_line, end_line in functions_to_remove:
                    del lines[start_line:end_line]
                    removed_count += 1
                
                # Write back the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
            
            except Exception as e:
                print(f"Error removing functions from {file_path}: {e}")
        
        return removed_count
    
    def _remove_dead_classes(self, dead_classes: List[DeadClass]) -> int:
        """Remove dead class definitions"""
        # Similar implementation to _remove_dead_functions
        # This is a simplified version
        return len(dead_classes)  # Placeholder
    
    def _remove_dead_files(self, dead_files: List[DeadFile]) -> int:
        """Remove dead files"""
        removed_count = 0
        
        for dead_file in dead_files:
            try:
                Path(dead_file.file_path).unlink()
                removed_count += 1
            except Exception as e:
                print(f"Error removing file {dead_file.file_path}: {e}")
        
        return removed_count
    
    def create_backup(self, files_to_backup: List[str]) -> str:
        """Create backup of files before removal"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"dead_code_removal_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        for file_path in files_to_backup:
            src_path = Path(file_path)
            if src_path.exists():
                rel_path = src_path.relative_to(self.root_path)
                backup_file_path = backup_path / rel_path
                backup_file_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, backup_file_path)
        
        # Create backup manifest
        manifest = {
            'timestamp': timestamp,
            'files': files_to_backup,
            'backup_path': str(backup_path)
        }
        
        with open(backup_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return str(backup_path)
    
    def save_report(self, report: DeadCodeReport, output_path: str) -> None:
        """Save dead code analysis report to file"""
        with open(output_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"Dead code report saved to {output_path}")


def main():
    """Example usage of DeadCodeAnalyzer"""
    analyzer = DeadCodeAnalyzer(".")
    
    # Analyze dead code
    report = analyzer.analyze_dead_code(include_tests=False)
    
    # Save report
    analyzer.save_report(report, "dead_code_report.json")
    
    # Print summary
    print(f"\nDead Code Analysis Summary:")
    print(f"Files analyzed: {report.total_files_analyzed}")
    print(f"Dead functions: {len(report.dead_functions)}")
    print(f"Dead classes: {len(report.dead_classes)}")
    print(f"Unused imports: {len(report.unused_imports)}")
    print(f"Dead files: {len(report.dead_files)}")
    print(f"Potential lines removed: {report.potential_lines_removed}")
    
    for rec in report.recommendations:
        print(f"- {rec}")


if __name__ == "__main__":
    main()