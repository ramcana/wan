import pytest
"""
Documentation Generator

Consolidates scattered documentation, generates API docs from code annotations,
and provides migration tools for organizing documentation.
"""

import os
import re
import ast
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import yaml


@dataclass
class DocumentationPage:
    """Represents a documentation page with metadata"""
    path: Path
    title: str
    content: str
    metadata: Dict[str, Any]
    links: List[str]
    backlinks: List[str]
    last_modified: datetime
    category: str
    tags: List[str]


@dataclass
class APIDocumentation:
    """Represents API documentation extracted from code"""
    module_name: str
    classes: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    constants: List[Dict[str, Any]]
    docstring: str


@dataclass
class MigrationReport:
    """Report of documentation migration process"""
    total_files: int
    migrated_files: int
    failed_files: List[str]
    duplicate_files: List[str]
    broken_links: List[str]
    warnings: List[str]


class DocumentationGenerator:
    """
    Main class for consolidating existing documentation and generating API docs
    """
    
    def __init__(self, source_dirs: List[Path], output_dir: Path):
        self.source_dirs = [Path(d) for d in source_dirs]
        self.output_dir = Path(output_dir)
        self.pages: List[DocumentationPage] = []
        self.api_docs: List[APIDocumentation] = []
        self.link_graph: Dict[str, List[str]] = {}
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Documentation categories mapping
        self.category_mapping = {
            'user': ['user-guide', 'installation', 'configuration', 'troubleshooting'],
            'developer': ['developer-guide', 'api', 'architecture', 'contributing'],
            'deployment': ['deployment', 'production', 'monitoring', 'operations'],
            'reference': ['api-reference', 'changelog', 'migration']
        }
    
    def discover_documentation_files(self) -> List[Path]:
        """
        Discover all documentation files in source directories
        """
        doc_files = []
        doc_extensions = {'.md', '.rst', '.txt'}
        
        for source_dir in self.source_dirs:
            if not source_dir.exists():
                continue
                
            for file_path in source_dir.rglob('*'):
                if (file_path.is_file() and 
                    file_path.suffix.lower() in doc_extensions and
                    not self._should_exclude_file(file_path)):
                    doc_files.append(file_path)
        
        return doc_files
    
    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded from documentation"""
        exclude_patterns = [
            'node_modules', 'venv', '__pycache__', '.git',
            '.pytest_cache', 'dist', 'build'
        ]
        
        return any(pattern in str(file_path) for pattern in exclude_patterns)
    
    def parse_documentation_file(self, file_path: Path) -> Optional[DocumentationPage]:
        """
        Parse a documentation file and extract metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract title from first heading or filename
            title = self._extract_title(content, file_path)
            
            # Extract metadata from frontmatter if present
            metadata = self._extract_frontmatter(content)
            
            # Extract links
            links = self._extract_links(content)
            
            # Determine category
            category = self._determine_category(file_path, metadata)
            
            # Extract tags
            tags = self._extract_tags(content, metadata)
            
            return DocumentationPage(
                path=file_path,
                title=title,
                content=content,
                metadata=metadata,
                links=links,
                backlinks=[],  # Will be populated later
                last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
                category=category,
                tags=tags
            )
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def _extract_title(self, content: str, file_path: Path) -> str:
        """Extract title from content or filename"""
        # Try to find first heading
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
            elif line.startswith('=') and len(line) > 3:  # RST style
                prev_line_idx = lines.index(line) - 1
                if prev_line_idx >= 0:
                    return lines[prev_line_idx].strip()
        
        # Fallback to filename
        return file_path.stem.replace('_', ' ').replace('-', ' ').title()
    
    def _extract_frontmatter(self, content: str) -> Dict[str, Any]:
        """Extract YAML frontmatter from content"""
        if not content.startswith('---'):
            return {}
        
        try:
            end_marker = content.find('---', 3)
            if end_marker == -1:
                return {}
            
            frontmatter = content[3:end_marker].strip()
            return yaml.safe_load(frontmatter) or {}
        except Exception:
            return {}
    
    def _extract_links(self, content: str) -> List[str]:
        """Extract markdown links from content"""
        # Markdown link pattern: [text](url)
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.findall(link_pattern, content)
        return [url for _, url in matches]
    
    def _determine_category(self, file_path: Path, metadata: Dict[str, Any]) -> str:
        """Determine documentation category"""
        if 'category' in metadata:
            return metadata['category']
        
        path_str = str(file_path).lower()
        
        for category, keywords in self.category_mapping.items():
            if any(keyword in path_str for keyword in keywords):
                return category
        
        return 'reference'  # Default category
    
    def _extract_tags(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Extract tags from content and metadata"""
        tags = []
        
        # From metadata
        if 'tags' in metadata:
            if isinstance(metadata['tags'], list):
                tags.extend(metadata['tags'])
            elif isinstance(metadata['tags'], str):
                tags.extend([tag.strip() for tag in metadata['tags'].split(',')])
        
        # From content (look for common patterns)
        content_lower = content.lower()
        tag_keywords = {
            'api': ['api', 'endpoint', 'rest'],
            'configuration': ['config', 'settings', 'environment'],
            'installation': ['install', 'setup', 'requirements'],
            'troubleshooting': ['troubleshoot', 'debug', 'error', 'fix'],
            'performance': ['performance', 'optimization', 'benchmark'],
            'security': ['security', 'authentication', 'authorization']
        }
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                tags.append(tag)
        
        return list(set(tags))  # Remove duplicates
    
    def generate_api_docs(self, code_dirs: List[Path]) -> List[APIDocumentation]:
        """
        Generate API documentation from code annotations
        """
        api_docs = []
        
        for code_dir in code_dirs:
            if not code_dir.exists():
                continue
                
            for py_file in code_dir.rglob('*.py'):
                if self._should_exclude_file(py_file):
                    continue
                
                try:
                    api_doc = self._parse_python_file(py_file)
                    if api_doc:
                        api_docs.append(api_doc)
                except Exception as e:
                    print(f"Error parsing Python file {py_file}: {e}")
        
        self.api_docs = api_docs
        return api_docs
    
    def _parse_python_file(self, file_path: Path) -> Optional[APIDocumentation]:
        """Parse Python file and extract API documentation"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            module_name = self._get_module_name(file_path)
            classes = []
            functions = []
            constants = []
            module_docstring = ast.get_docstring(tree) or ""
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._extract_class_info(node)
                    if class_info:
                        classes.append(class_info)
                
                elif isinstance(node, ast.FunctionDef):
                    # Only top-level functions
                    if isinstance(node.parent if hasattr(node, 'parent') else None, ast.Module):
                        func_info = self._extract_function_info(node)
                        if func_info:
                            functions.append(func_info)
                
                elif isinstance(node, ast.Assign):
                    const_info = self._extract_constant_info(node)
                    if const_info:
                        constants.append(const_info)
            
            # Add parent references for proper function detection
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    child.parent = node
            
            return APIDocumentation(
                module_name=module_name,
                classes=classes,
                functions=functions,
                constants=constants,
                docstring=module_docstring
            )
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def _get_module_name(self, file_path: Path) -> str:
        """Get module name from file path"""
        # Convert file path to module name
        parts = file_path.parts
        if 'backend' in parts:
            start_idx = parts.index('backend') + 1
        elif 'frontend' in parts:
            start_idx = parts.index('frontend') + 1
        else:
            start_idx = 0
        
        module_parts = parts[start_idx:]
        if module_parts and module_parts[-1].endswith('.py'):
            module_parts = module_parts[:-1] + (module_parts[-1][:-3],)
        
        return '.'.join(module_parts)
    
    def _extract_class_info(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Extract class information"""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._extract_function_info(item)
                if method_info:
                    methods.append(method_info)
        
        return {
            'name': node.name,
            'docstring': ast.get_docstring(node) or "",
            'methods': methods,
            'bases': [self._get_name(base) for base in node.bases],
            'decorators': [self._get_name(dec) for dec in node.decorator_list]
        }
    
    def _extract_function_info(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract function information"""
        args = []
        for arg in node.args.args:
            arg_info = {'name': arg.arg}
            if arg.annotation:
                arg_info['type'] = self._get_name(arg.annotation)
            args.append(arg_info)
        
        return_type = None
        if node.returns:
            return_type = self._get_name(node.returns)
        
        return {
            'name': node.name,
            'docstring': ast.get_docstring(node) or "",
            'args': args,
            'return_type': return_type,
            'decorators': [self._get_name(dec) for dec in node.decorator_list],
            'is_async': isinstance(node, ast.AsyncFunctionDef)
        }
    
    def _extract_constant_info(self, node: ast.Assign) -> Optional[Dict[str, Any]]:
        """Extract constant information"""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.isupper():  # Convention for constants
                value = None
                try:
                    if isinstance(node.value, (ast.Str, ast.Constant)):
                        value = node.value.s if hasattr(node.value, 's') else node.value.value
                    elif isinstance(node.value, ast.Num):
                        value = node.value.n
                except:
                    pass
                
                return {
                    'name': name,
                    'value': value,
                    'type': type(value).__name__ if value is not None else 'unknown'
                }
        return None
    
    def _get_name(self, node: ast.AST) -> str:
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        else:
            return str(node)
    
    def consolidate_existing_docs(self) -> MigrationReport:
        """
        Consolidate existing documentation into unified structure
        """
        doc_files = self.discover_documentation_files()
        
        migrated_files = 0
        failed_files = []
        duplicate_files = []
        
        for file_path in doc_files:
            try:
                page = self.parse_documentation_file(file_path)
                if page:
                    # Determine target location
                    target_path = self._get_target_path(page)
                    
                    # Check for duplicates
                    if target_path.exists():
                        duplicate_files.append(str(file_path))
                        continue
                    
                    # Create target directory
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file with updated content
                    self._migrate_file(page, target_path)
                    
                    self.pages.append(page)
                    migrated_files += 1
                    
            except Exception as e:
                failed_files.append(f"{file_path}: {e}")
        
        # Build link graph
        self._build_link_graph()
        
        return MigrationReport(
            total_files=len(doc_files),
            migrated_files=migrated_files,
            failed_files=failed_files,
            duplicate_files=duplicate_files,
            broken_links=self._find_broken_links(),
            warnings=[]
        )
    
    def _get_target_path(self, page: DocumentationPage) -> Path:
        """Determine target path for migrated documentation"""
        category = page.category
        filename = page.path.name
        
        # Clean up filename
        if filename.startswith('TASK_') or filename.startswith('WAN22_'):
            # Convert task files to more readable names
            clean_name = self._clean_filename(filename)
        else:
            clean_name = filename
        
        category_dir = self.output_dir / category
        return category_dir / clean_name
    
    def _clean_filename(self, filename: str) -> str:
        """Clean up filename for better organization"""
        # Remove common prefixes
        name = filename
        prefixes = ['TASK_', 'WAN22_', 'ENHANCED_', 'IMPLEMENTATION_', 'SUMMARY_']
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
        
        # Convert to lowercase and replace underscores
        name = name.lower().replace('_', '-')
        
        return name
    
    def _migrate_file(self, page: DocumentationPage, target_path: Path):
        """Migrate file to target location with updated content"""
        # Add metadata header if not present
        content = page.content
        
        if not content.startswith('---'):
            metadata = {
                'title': page.title,
                'category': page.category,
                'tags': page.tags,
                'last_updated': page.last_modified.isoformat(),
                'original_path': str(page.path)
            }
            
            frontmatter = yaml.dump(metadata, default_flow_style=False)
            content = f"---\n{frontmatter}---\n\n{content}"
        
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _build_link_graph(self):
        """Build link graph for cross-references"""
        for page in self.pages:
            page_path = str(page.path)
            self.link_graph[page_path] = page.links
            
            # Build backlinks
            for link in page.links:
                for other_page in self.pages:
                    if link in str(other_page.path) or link in other_page.title:
                        other_page.backlinks.append(page_path)
    
    def _find_broken_links(self) -> List[str]:
        """Find broken links in documentation"""
        broken_links = []
        
        for page in self.pages:
            for link in page.links:
                if link.startswith('http'):
                    continue  # Skip external links for now
                
                # Check if link points to existing file
                link_path = page.path.parent / link
                if not link_path.exists():
                    broken_links.append(f"{page.path}: {link}")
        
        return broken_links
    
    def generate_index(self) -> str:
        """Generate main documentation index"""
        categories = {}
        for page in self.pages:
            if page.category not in categories:
                categories[page.category] = []
            categories[page.category].append(page)
        
        # Sort pages within categories
        for category in categories:
            categories[category].sort(key=lambda p: p.title)
        
        # Generate index content
        index_content = """---
title: Documentation Index
category: reference
tags: [index, navigation]
---

# WAN22 Video Generation System Documentation

Welcome to the comprehensive documentation for the WAN22 Video Generation System.

## Quick Navigation

"""
        
        for category, pages in categories.items():
            index_content += f"\n### {category.title()}\n\n"
            for page in pages:
                relative_path = os.path.relpath(page.path, self.output_dir)
                index_content += f"- [{page.title}]({relative_path})\n"
        
        index_content += """
## API Documentation

Auto-generated API documentation is available in the [API Reference](api/) section.

## Search

Use the search functionality to quickly find specific topics, functions, or configuration options.

## Contributing

See the [Developer Guide](developer-guide/) for information on contributing to this documentation.
"""
        
        return index_content
    
    def generate_api_index(self) -> str:
        """Generate API documentation index"""
        if not self.api_docs:
            return "# API Documentation\n\nNo API documentation available."
        
        content = """---
title: API Reference
category: reference
tags: [api, reference]
---

# API Reference

Auto-generated API documentation for the WAN22 Video Generation System.

## Modules

"""
        
        for api_doc in self.api_docs:
            content += f"\n### {api_doc.module_name}\n\n"
            if api_doc.docstring:
                content += f"{api_doc.docstring}\n\n"
            
            if api_doc.classes:
                content += "#### Classes\n\n"
                for cls in api_doc.classes:
                    content += f"- **{cls['name']}**: {cls['docstring'][:100]}...\n"
            
            if api_doc.functions:
                content += "\n#### Functions\n\n"
                for func in api_doc.functions:
                    content += f"- **{func['name']}**: {func['docstring'][:100]}...\n"
        
        return content
    
    def save_consolidated_docs(self):
        """Save all consolidated documentation"""
        # Save main index
        index_path = self.output_dir / 'index.md'
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_index())
        
        # Save API index
        api_dir = self.output_dir / 'api'
        api_dir.mkdir(exist_ok=True)
        api_index_path = api_dir / 'index.md'
        with open(api_index_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_api_index())
        
        # Generate detailed API docs
        for api_doc in self.api_docs:
            self._generate_detailed_api_doc(api_doc, api_dir)
    
    def _generate_detailed_api_doc(self, api_doc: APIDocumentation, api_dir: Path):
        """Generate detailed API documentation for a module"""
        filename = api_doc.module_name.replace('.', '-') + '.md'
        file_path = api_dir / filename
        
        content = f"""---
title: {api_doc.module_name}
category: api
tags: [api, {api_doc.module_name.split('.')[0]}]
---

# {api_doc.module_name}

{api_doc.docstring}

"""
        
        # Classes
        if api_doc.classes:
            content += "## Classes\n\n"
            for cls in api_doc.classes:
                content += f"### {cls['name']}\n\n"
                content += f"{cls['docstring']}\n\n"
                
                if cls['methods']:
                    content += "#### Methods\n\n"
                    for method in cls['methods']:
                        args_str = ', '.join([f"{arg['name']}: {arg.get('type', 'Any')}" for arg in method['args']])
                        return_str = f" -> {method['return_type']}" if method['return_type'] else ""
                        content += f"##### {method['name']}({args_str}){return_str}\n\n"
                        content += f"{method['docstring']}\n\n"
        
        # Functions
        if api_doc.functions:
            content += "## Functions\n\n"
            for func in api_doc.functions:
                args_str = ', '.join([f"{arg['name']}: {arg.get('type', 'Any')}" for arg in func['args']])
                return_str = f" -> {func['return_type']}" if func['return_type'] else ""
                content += f"### {func['name']}({args_str}){return_str}\n\n"
                content += f"{func['docstring']}\n\n"
        
        # Constants
        if api_doc.constants:
            content += "## Constants\n\n"
            for const in api_doc.constants:
                content += f"### {const['name']}\n\n"
                content += f"Type: `{const['type']}`\n\n"
                if const['value'] is not None:
                    content += f"Value: `{const['value']}`\n\n"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)


def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate consolidated documentation')
    parser.add_argument('--source-dirs', nargs='+', default=['.'], 
                       help='Source directories to scan for documentation')
    parser.add_argument('--output-dir', default='docs', 
                       help='Output directory for consolidated documentation')
    parser.add_argument('--code-dirs', nargs='+', default=['backend', 'frontend/src'], 
                       help='Code directories to scan for API documentation')
    parser.add_argument('--generate-api', action='store_true', 
                       help='Generate API documentation from code')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = DocumentationGenerator(args.source_dirs, args.output_dir)
    
    # Consolidate existing documentation
    print("Consolidating existing documentation...")
    report = generator.consolidate_existing_docs()
    
    print(f"Migration Report:")
    print(f"  Total files: {report.total_files}")
    print(f"  Migrated: {report.migrated_files}")
    print(f"  Failed: {len(report.failed_files)}")
    print(f"  Duplicates: {len(report.duplicate_files)}")
    print(f"  Broken links: {len(report.broken_links)}")
    
    # Generate API documentation if requested
    if args.generate_api:
        print("Generating API documentation...")
        api_docs = generator.generate_api_docs([Path(d) for d in args.code_dirs])
        print(f"Generated API docs for {len(api_docs)} modules")
    
    # Save consolidated documentation
    print("Saving consolidated documentation...")
    generator.save_consolidated_docs()
    
    print(f"Documentation consolidated in: {args.output_dir}")


if __name__ == '__main__':
    main()
