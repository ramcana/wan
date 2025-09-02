"""
Component Relationship Analyzer

Analyzes dependencies and relationships between project components
by examining imports, references, and file interactions.
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import json


@dataclass
class ImportInfo:
    """Information about an import statement."""
    module: str
    alias: Optional[str]
    from_module: Optional[str]
    is_relative: bool
    line_number: int


@dataclass
class ComponentDependency:
    """Represents a dependency between components."""
    source_component: str
    target_component: str
    dependency_type: str  # 'import', 'config', 'file_reference', 'api_call'
    strength: int  # 1-10, how strong the dependency is
    files_involved: List[str]
    details: Optional[str] = None


@dataclass
class ComponentInfo:
    """Information about a project component."""
    name: str
    path: str
    component_type: str  # 'package', 'module', 'directory', 'service'
    files: List[str]
    dependencies: List[ComponentDependency]
    dependents: List[ComponentDependency]
    purpose: Optional[str] = None
    complexity_score: int = 0


@dataclass
class ComponentRelationshipMap:
    """Complete map of component relationships."""
    components: List[ComponentInfo]
    dependencies: List[ComponentDependency]
    circular_dependencies: List[List[str]]
    isolated_components: List[str]
    critical_components: List[str]  # Components with many dependencies
    entry_points: List[str]


class ComponentRelationshipAnalyzer:
    """Analyzes relationships and dependencies between project components."""
    
    def __init__(self, root_path: str):
        """Initialize analyzer with project root path."""
        self.root_path = Path(root_path).resolve()
        self.components = {}
        self.dependencies = []
        
    def analyze(self) -> ComponentRelationshipMap:
        """Perform complete component relationship analysis."""
        print("Analyzing component relationships...")
        
        # Step 1: Identify components
        self._identify_components()
        
        # Step 2: Analyze Python imports
        self._analyze_python_imports()
        
        # Step 3: Analyze configuration references
        self._analyze_config_references()
        
        # Step 4: Analyze file references
        self._analyze_file_references()
        
        # Step 5: Analyze API calls and service interactions
        self._analyze_api_interactions()
        
        # Step 6: Calculate component metrics
        self._calculate_component_metrics()
        
        # Step 7: Detect circular dependencies
        circular_deps = self._detect_circular_dependencies()
        
        # Step 8: Identify critical and isolated components
        critical_components = self._identify_critical_components()
        isolated_components = self._identify_isolated_components()
        entry_points = self._identify_entry_points()
        
        return ComponentRelationshipMap(
            components=list(self.components.values()),
            dependencies=self.dependencies,
            circular_dependencies=circular_deps,
            isolated_components=isolated_components,
            critical_components=critical_components,
            entry_points=entry_points
        )
    
    def _identify_components(self):
        """Identify major components in the project."""
        print("Identifying project components...")
        
        # Walk through project structure
        for root, dirs, files in os.walk(self.root_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if not self._should_ignore_dir(d)]
            
            current_path = Path(root)
            relative_path = current_path.relative_to(self.root_path)
            
            # Skip root directory itself
            if current_path == self.root_path:
                continue
            
            # Determine if this is a significant component
            if self._is_significant_component(current_path, files):
                component_name = str(relative_path).replace(os.sep, '.')
                component_type = self._determine_component_type(current_path, files)
                purpose = self._determine_component_purpose(current_path.name, files)
                
                # Get all Python files in this component
                python_files = []
                for file in files:
                    if file.endswith('.py'):
                        file_path = str((current_path / file).relative_to(self.root_path))
                        python_files.append(file_path)
                
                self.components[component_name] = ComponentInfo(
                    name=component_name,
                    path=str(relative_path),
                    component_type=component_type,
                    files=python_files,
                    dependencies=[],
                    dependents=[],
                    purpose=purpose
                )
    
    def _should_ignore_dir(self, dirname: str) -> bool:
        """Check if directory should be ignored."""
        ignore_patterns = {
            '__pycache__', '.git', '.pytest_cache', 'node_modules',
            '.vscode', '.idea', 'venv', 'env', '.env', 'build', 'dist'
        }
        return dirname in ignore_patterns or dirname.startswith('.')
    
    def _is_significant_component(self, path: Path, files: List[str]) -> bool:
        """Determine if a directory represents a significant component."""
        # Has Python files
        python_files = [f for f in files if f.endswith('.py')]
        if len(python_files) >= 2:  # At least 2 Python files
            return True
        
        # Is a Python package
        if '__init__.py' in files:
            return True
        
        # Has significant configuration or documentation
        important_files = [f for f in files if f.lower() in [
            'readme.md', 'config.json', 'config.yaml', 'requirements.txt'
        ]]
        if important_files and python_files:
            return True
        
        return False
    
    def _determine_component_type(self, path: Path, files: List[str]) -> str:
        """Determine the type of component."""
        if '__init__.py' in files:
            return 'package'
        elif any(f.endswith('.py') for f in files):
            return 'module'
        elif any(f.startswith('api') or 'endpoint' in f for f in files):
            return 'service'
        else:
            return 'directory'
    
    def _determine_component_purpose(self, name: str, files: List[str]) -> Optional[str]:
        """Determine the purpose of a component."""
        name_lower = name.lower()
        
        purposes = {
            'api': 'API endpoints and handlers',
            'core': 'Core business logic',
            'models': 'Data models and schemas',
            'services': 'Business services',
            'utils': 'Utility functions',
            'config': 'Configuration management',
            'tests': 'Test suite',
            'scripts': 'Automation scripts',
            'tools': 'Development tools',
            'frontend': 'Frontend application',
            'backend': 'Backend application',
            'websocket': 'WebSocket handlers',
            'monitoring': 'System monitoring',
            'examples': 'Example code',
            'docs': 'Documentation'
        }
        
        return purposes.get(name_lower)
    
    def _analyze_python_imports(self):
        """Analyze Python import statements to find dependencies."""
        print("Analyzing Python imports...")
        
        for component in self.components.values():
            for file_path in component.files:
                full_path = self.root_path / file_path
                try:
                    imports = self._extract_imports_from_file(full_path)
                    self._process_imports(component.name, file_path, imports)
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
                    continue
    
    def _extract_imports_from_file(self, file_path: Path) -> List[ImportInfo]:
        """Extract import information from a Python file."""
        imports = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(ImportInfo(
                            module=alias.name,
                            alias=alias.asname,
                            from_module=None,
                            is_relative=False,
                            line_number=node.lineno
                        ))
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    is_relative = node.level > 0
                    
                    for alias in node.names:
                        imports.append(ImportInfo(
                            module=alias.name,
                            alias=alias.asname,
                            from_module=module,
                            is_relative=is_relative,
                            line_number=node.lineno
                        ))
        
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"Could not parse {file_path}: {e}")
        
        return imports
    
    def _process_imports(self, source_component: str, file_path: str, imports: List[ImportInfo]):
        """Process imports and create dependency relationships."""
        for import_info in imports:
            target_component = self._resolve_import_to_component(import_info)
            
            if target_component and target_component != source_component:
                # Check if dependency already exists
                existing_dep = None
                for dep in self.dependencies:
                    if (dep.source_component == source_component and 
                        dep.target_component == target_component and
                        dep.dependency_type == 'import'):
                        existing_dep = dep
                        break
                
                if existing_dep:
                    # Update existing dependency
                    if file_path not in existing_dep.files_involved:
                        existing_dep.files_involved.append(file_path)
                        existing_dep.strength += 1
                else:
                    # Create new dependency
                    dependency = ComponentDependency(
                        source_component=source_component,
                        target_component=target_component,
                        dependency_type='import',
                        strength=1,
                        files_involved=[file_path],
                        details=f"Imports {import_info.module}"
                    )
                    self.dependencies.append(dependency)
    
    def _resolve_import_to_component(self, import_info: ImportInfo) -> Optional[str]:
        """Resolve an import to a component name."""
        if import_info.is_relative:
            # Handle relative imports - more complex logic needed
            return None
        
        # Get the full module path
        if import_info.from_module:
            full_module = import_info.from_module
        else:
            full_module = import_info.module
        
        # Try to match to existing components
        for component_name in self.components.keys():
            component_path = component_name.replace('.', os.sep)
            
            # Direct match
            if full_module == component_name:
                return component_name
            
            # Partial match (import is from a submodule)
            if full_module.startswith(component_name + '.'):
                return component_name
            
            # Path-based match
            if full_module.replace('.', os.sep).startswith(component_path):
                return component_name
        
        return None
    
    def _analyze_config_references(self):
        """Analyze configuration file references."""
        print("Analyzing configuration references...")
        
        # Find configuration files
        config_files = []
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if self._is_config_file(file):
                    config_files.append(Path(root) / file)
        
        # Analyze references to config files
        for component in self.components.values():
            for file_path in component.files:
                full_path = self.root_path / file_path
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Look for config file references
                    for config_file in config_files:
                        config_name = config_file.name
                        if config_name in content or str(config_file.relative_to(self.root_path)) in content:
                            # Create config dependency
                            dependency = ComponentDependency(
                                source_component=component.name,
                                target_component='config',
                                dependency_type='config',
                                strength=2,
                                files_involved=[file_path],
                                details=f"References {config_name}"
                            )
                            self.dependencies.append(dependency)
                            break
                
                except Exception as e:
                    continue
    
    def _is_config_file(self, filename: str) -> bool:
        """Check if a file is a configuration file."""
        config_extensions = {'.json', '.yaml', '.yml', '.ini', '.cfg', '.conf', '.toml'}
        config_names = {'config', 'settings', 'setup'}
        
        name_lower = filename.lower()
        extension = Path(filename).suffix.lower()
        
        return (extension in config_extensions or 
                any(config_name in name_lower for config_name in config_names))
    
    def _analyze_file_references(self):
        """Analyze file path references in code."""
        print("Analyzing file references...")
        
        # Pattern to match file paths
        file_path_pattern = re.compile(r'["\']([^"\']+\.(py|json|yaml|yml|txt|md|csv))["\']')
        
        for component in self.components.values():
            for file_path in component.files:
                full_path = self.root_path / file_path
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    matches = file_path_pattern.findall(content)
                    for match in matches:
                        referenced_file = match[0]
                        target_component = self._find_component_for_file(referenced_file)
                        
                        if target_component and target_component != component.name:
                            dependency = ComponentDependency(
                                source_component=component.name,
                                target_component=target_component,
                                dependency_type='file_reference',
                                strength=1,
                                files_involved=[file_path],
                                details=f"References {referenced_file}"
                            )
                            self.dependencies.append(dependency)
                
                except Exception as e:
                    continue
    
    def _find_component_for_file(self, file_path: str) -> Optional[str]:
        """Find which component a file belongs to."""
        for component_name, component in self.components.items():
            if file_path in component.files:
                return component_name
            
            # Check if file is in component directory
            component_path = component.path
            if file_path.startswith(component_path):
                return component_name
        
        return None
    
    def _analyze_api_interactions(self):
        """Analyze API calls and service interactions."""
        print("Analyzing API interactions...")
        
        # Patterns for API calls
        api_patterns = [
            re.compile(r'requests\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']'),
            re.compile(r'fetch\s*\(\s*["\']([^"\']+)["\']'),
            re.compile(r'@app\.route\s*\(\s*["\']([^"\']+)["\']'),
            re.compile(r'@router\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']')
        ]
        
        for component in self.components.values():
            for file_path in component.files:
                full_path = self.root_path / file_path
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern in api_patterns:
                        matches = pattern.findall(content)
                        for match in matches:
                            if isinstance(match, tuple):
                                endpoint = match[-1]  # Last element is usually the endpoint
                            else:
                                endpoint = match
                            
                            # Try to find which component handles this endpoint
                            target_component = self._find_api_handler_component(endpoint)
                            
                            if target_component and target_component != component.name:
                                dependency = ComponentDependency(
                                    source_component=component.name,
                                    target_component=target_component,
                                    dependency_type='api_call',
                                    strength=3,
                                    files_involved=[file_path],
                                    details=f"Calls API endpoint {endpoint}"
                                )
                                self.dependencies.append(dependency)
                
                except Exception as e:
                    continue
    
    def _find_api_handler_component(self, endpoint: str) -> Optional[str]:
        """Find which component handles an API endpoint."""
        # Look for route definitions in API components
        for component_name, component in self.components.items():
            if 'api' in component_name.lower():
                for file_path in component.files:
                    full_path = self.root_path / file_path
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if endpoint in content:
                            return component_name
                    except Exception:
                        continue
        
        return None
    
    def _calculate_component_metrics(self):
        """Calculate complexity and importance metrics for components."""
        print("Calculating component metrics...")
        
        for component in self.components.values():
            # Calculate complexity score
            complexity = 0
            
            # File count contributes to complexity
            complexity += len(component.files) * 2
            
            # Dependencies contribute to complexity
            deps_out = len([d for d in self.dependencies if d.source_component == component.name])
            deps_in = len([d for d in self.dependencies if d.target_component == component.name])
            complexity += deps_out * 3 + deps_in * 2
            
            # API components are more complex
            if 'api' in component.name.lower():
                complexity += 10
            
            # Core components are more complex
            if 'core' in component.name.lower():
                complexity += 15
            
            component.complexity_score = complexity
            
            # Update dependency lists
            component.dependencies = [d for d in self.dependencies if d.source_component == component.name]
            component.dependents = [d for d in self.dependencies if d.target_component == component.name]
    
    def _detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies between components."""
        print("Detecting circular dependencies...")
        
        # Build adjacency list
        graph = defaultdict(set)
        for dep in self.dependencies:
            graph[dep.source_component].add(dep.target_component)
        
        # Find cycles using DFS
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph[node]:
                dfs(neighbor, path.copy())
            
            rec_stack.remove(node)
        
        for component in self.components.keys():
            if component not in visited:
                dfs(component, [])
        
        return cycles
    
    def _identify_critical_components(self) -> List[str]:
        """Identify components that many others depend on."""
        dependency_count = defaultdict(int)
        
        for dep in self.dependencies:
            dependency_count[dep.target_component] += dep.strength
        
        # Sort by dependency count
        sorted_components = sorted(dependency_count.items(), key=lambda x: x[1], reverse=True)
        
        # Return top components with significant dependencies
        critical = []
        for component, count in sorted_components:
            if count >= 5:  # Threshold for being "critical"
                critical.append(component)
        
        return critical[:10]  # Top 10 critical components
    
    def _identify_isolated_components(self) -> List[str]:
        """Identify components with no or very few dependencies."""
        all_components = set(self.components.keys())
        connected_components = set()
        
        for dep in self.dependencies:
            connected_components.add(dep.source_component)
            connected_components.add(dep.target_component)
        
        isolated = []
        for component in all_components:
            if component not in connected_components:
                isolated.append(component)
            else:
                # Check if it has very few connections
                deps_out = len([d for d in self.dependencies if d.source_component == component])
                deps_in = len([d for d in self.dependencies if d.target_component == component])
                
                if deps_out + deps_in <= 1:
                    isolated.append(component)
        
        return isolated
    
    def _identify_entry_points(self) -> List[str]:
        """Identify components that serve as entry points."""
        entry_points = []
        
        for component in self.components.values():
            # Check for main files
            for file_path in component.files:
                filename = Path(file_path).name.lower()
                if filename in ['main.py', 'app.py', 'start.py', '__main__.py']:
                    entry_points.append(component.name)
                    break
            
            # Check for components with no incoming dependencies but outgoing ones
            deps_in = len([d for d in self.dependencies if d.target_component == component.name])
            deps_out = len([d for d in self.dependencies if d.source_component == component.name])
            
            if deps_in == 0 and deps_out > 0:
                entry_points.append(component.name)
        
        return list(set(entry_points))  # Remove duplicates
    
    def save_analysis(self, analysis: ComponentRelationshipMap, output_path: str):
        """Save analysis results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict for JSON serialization
        analysis_dict = asdict(analysis)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Component analysis saved to: {output_file}")
    
    def generate_summary_report(self, analysis: ComponentRelationshipMap) -> str:
        """Generate a human-readable summary report."""
        report = []
        report.append("# Component Relationship Analysis Report")
        report.append("")
        
        # Overview
        report.append("## Overview")
        report.append("")
        report.append(f"- **Total Components:** {len(analysis.components)}")
        report.append(f"- **Total Dependencies:** {len(analysis.dependencies)}")
        report.append(f"- **Circular Dependencies:** {len(analysis.circular_dependencies)}")
        report.append(f"- **Critical Components:** {len(analysis.critical_components)}")
        report.append(f"- **Isolated Components:** {len(analysis.isolated_components)}")
        report.append(f"- **Entry Points:** {len(analysis.entry_points)}")
        report.append("")
        
        # Critical Components
        if analysis.critical_components:
            report.append("## Critical Components")
            report.append("")
            report.append("These components are heavily depended upon by others:")
            report.append("")
            for component in analysis.critical_components:
                comp_info = next((c for c in analysis.components if c.name == component), None)
                if comp_info:
                    deps_in = len(comp_info.dependents)
                    report.append(f"- **{component}** ({deps_in} dependencies)")
                    if comp_info.purpose:
                        report.append(f"  - Purpose: {comp_info.purpose}")
            report.append("")
        
        # Entry Points
        if analysis.entry_points:
            report.append("## Entry Points")
            report.append("")
            report.append("These components serve as application entry points:")
            report.append("")
            for entry_point in analysis.entry_points:
                comp_info = next((c for c in analysis.components if c.name == entry_point), None)
                if comp_info:
                    report.append(f"- **{entry_point}**")
                    if comp_info.purpose:
                        report.append(f"  - Purpose: {comp_info.purpose}")
                    report.append(f"  - Files: {len(comp_info.files)}")
            report.append("")
        
        # Circular Dependencies
        if analysis.circular_dependencies:
            report.append("## Circular Dependencies")
            report.append("")
            report.append("⚠️ These circular dependencies should be resolved:")
            report.append("")
            for i, cycle in enumerate(analysis.circular_dependencies, 1):
                report.append(f"**Cycle {i}:** {' → '.join(cycle)}")
            report.append("")
        
        # Component Details
        report.append("## Component Details")
        report.append("")
        
        # Sort components by complexity
        sorted_components = sorted(analysis.components, key=lambda c: c.complexity_score, reverse=True)
        
        for component in sorted_components[:15]:  # Top 15 most complex
            report.append(f"### {component.name}")
            report.append("")
            if component.purpose:
                report.append(f"**Purpose:** {component.purpose}")
            report.append(f"**Type:** {component.component_type}")
            report.append(f"**Files:** {len(component.files)}")
            report.append(f"**Complexity Score:** {component.complexity_score}")
            report.append(f"**Dependencies:** {len(component.dependencies)} out, {len(component.dependents)} in")
            
            if component.dependencies:
                report.append("")
                report.append("**Depends on:**")
                for dep in component.dependencies[:5]:  # Top 5 dependencies
                    report.append(f"- {dep.target_component} ({dep.dependency_type})")
            
            report.append("")
        
        return "\n".join(report)