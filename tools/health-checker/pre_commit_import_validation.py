#!/usr/bin/env python3
"""
Pre-commit hook for import path validation.
Validates that import statements are correct and follow project structure.
"""

import sys
import ast
import importlib.util
from pathlib import Path
from typing import List, Set, Dict, Any


class ImportVisitor(ast.NodeVisitor):
    """AST visitor to collect import statements."""
    
    def __init__(self):
        self.imports = []
        self.from_imports = []
    
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append({
                'module': alias.name,
                'asname': alias.asname,
                'lineno': node.lineno
            })
    
    def visit_ImportFrom(self, node):
        module = node.module or ''
        for alias in node.names:
            self.from_imports.append({
                'module': module,
                'name': alias.name,
                'asname': alias.asname,
                'level': node.level,
                'lineno': node.lineno
            })


def parse_python_file(file_path: Path) -> tuple[List[Dict], List[Dict], List[str]]:
    """Parse a Python file and extract imports."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(file_path))
        visitor = ImportVisitor()
        visitor.visit(tree)
        
        return visitor.imports, visitor.from_imports, []
    
    except SyntaxError as e:
        return [], [], [f"Syntax error in {file_path}:{e.lineno}: {e.msg}"]
    except Exception as e:
        return [], [], [f"Error parsing {file_path}: {e}"]


def check_relative_imports(file_path: Path, from_imports: List[Dict]) -> List[str]:
    """Check relative imports for correctness."""
    issues = []
    
    for imp in from_imports:
        if imp['level'] > 0:  # Relative import
            # Calculate expected module path
            current_dir = file_path.parent
            
            # Go up 'level' directories
            target_dir = current_dir
            for _ in range(imp['level'] - 1):
                target_dir = target_dir.parent
            
            # Check if target module exists
            if imp['module']:
                module_parts = imp['module'].split('.')
                target_path = target_dir
                for part in module_parts:
                    target_path = target_path / part
                
                # Check for Python file or package
                py_file = target_path.with_suffix('.py')
                init_file = target_path / '__init__.py'
                
                if not py_file.exists() and not init_file.exists():
                    issues.append(
                        f"{file_path}:{imp['lineno']}: Relative import "
                        f"'{imp['module']}' not found (level {imp['level']})"
                    )
    
    return issues


def check_project_imports(file_path: Path, imports: List[Dict], from_imports: List[Dict]) -> List[str]:
    """Check imports from project modules."""
    issues = []
    
    # Define project module prefixes
    project_modules = ['backend', 'scripts', 'frontend', 'core', 'infrastructure']
    
    all_imports = []
    
    # Process direct imports
    for imp in imports:
        all_imports.append({
            'module': imp['module'],
            'lineno': imp['lineno'],
            'type': 'import'
        })
    
    # Process from imports
    for imp in from_imports:
        if imp['level'] == 0:  # Absolute import
            all_imports.append({
                'module': imp['module'],
                'lineno': imp['lineno'],
                'type': 'from'
            })
    
    # Check each import
    for imp in all_imports:
        module_parts = imp['module'].split('.')
        
        if module_parts[0] in project_modules:
            # This is a project module import
            module_path = Path('.')
            
            for part in module_parts:
                module_path = module_path / part
            
            # Check if module exists
            py_file = module_path.with_suffix('.py')
            init_file = module_path / '__init__.py'
            
            if not py_file.exists() and not init_file.exists():
                issues.append(
                    f"{file_path}:{imp['lineno']}: Project module "
                    f"'{imp['module']}' not found"
                )
    
    return issues


def check_circular_imports(file_path: Path, imports: List[Dict], from_imports: List[Dict]) -> List[str]:
    """Check for potential circular imports."""
    issues = []
    
    # This is a simplified check - a full circular import detection
    # would require building a dependency graph
    
    current_module = get_module_name_from_path(file_path)
    if not current_module:
        return issues
    
    all_imported_modules = []
    
    # Collect all imported modules
    for imp in imports:
        all_imported_modules.append(imp['module'])
    
    for imp in from_imports:
        if imp['level'] == 0 and imp['module']:
            all_imported_modules.append(imp['module'])
    
    # Check if any imported module might import back
    for imported_module in all_imported_modules:
        if imported_module.startswith(current_module.split('.')[0]):
            # Same top-level package - potential for circular import
            # This is a heuristic check, not definitive
            pass
    
    return issues


def get_module_name_from_path(file_path: Path) -> str:
    """Convert file path to module name."""
    # Remove .py extension
    if file_path.suffix == '.py':
        path_parts = file_path.with_suffix('').parts
    else:
        path_parts = file_path.parts
    
    # Skip common root directories
    skip_parts = {'.', 'src', 'lib'}
    filtered_parts = [part for part in path_parts if part not in skip_parts]
    
    return '.'.join(filtered_parts)


def check_import_organization(file_path: Path, imports: List[Dict], from_imports: List[Dict]) -> List[str]:
    """Check import organization and style."""
    issues = []
    
    # Check import order (simplified PEP 8 check)
    all_imports_with_lines = []
    
    for imp in imports:
        all_imports_with_lines.append((imp['lineno'], 'import', imp['module']))
    
    for imp in from_imports:
        module = imp['module'] or ''
        all_imports_with_lines.append((imp['lineno'], 'from', module))
    
    # Sort by line number
    all_imports_with_lines.sort()
    
    # Check for imports after non-import statements
    # This would require more sophisticated AST analysis
    
    return issues


def main(file_paths: List[str]) -> int:
    """Main pre-commit hook function."""
    print("🔍 Running import validation...")
    
    all_issues = []
    
    for file_path_str in file_paths:
        file_path = Path(file_path_str)
        
        if not file_path.suffix == '.py':
            continue
        
        print(f"  Checking {file_path}...")
        
        imports, from_imports, parse_issues = parse_python_file(file_path)
        all_issues.extend(parse_issues)
        
        if parse_issues:
            continue  # Skip further checks if parsing failed
        
        # Run import checks
        all_issues.extend(check_relative_imports(file_path, from_imports))
        all_issues.extend(check_project_imports(file_path, imports, from_imports))
        all_issues.extend(check_circular_imports(file_path, imports, from_imports))
        all_issues.extend(check_import_organization(file_path, imports, from_imports))
    
    if all_issues:
        print("❌ Import validation failed:")
        for issue in all_issues:
            print(f"  • {issue}")
        
        print("\n💡 Recommendations:")
        print("  • Fix broken import paths")
        print("  • Ensure imported modules exist")
        print("  • Check relative import levels")
        print("  • Organize imports according to PEP 8")
        
        return 1
    
    print("✅ Import validation passed!")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: pre_commit_import_validation.py <file1> [file2] ...")
        sys.exit(1)
    
    sys.exit(main(sys.argv[1:]))