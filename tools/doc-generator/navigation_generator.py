"""
Navigation Generator

Automatically generates navigation menus and site structure
for WAN22 documentation based on file organization and metadata.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import re


@dataclass
class NavigationItem:
    """Navigation menu item"""
    title: str
    path: Optional[str] = None
    url: Optional[str] = None
    children: Optional[List['NavigationItem']] = None
    category: Optional[str] = None
    order: int = 999
    icon: Optional[str] = None
    description: Optional[str] = None


@dataclass
class NavigationConfig:
    """Configuration for navigation generation"""
    auto_generate: bool = True
    show_categories: bool = True
    show_tags: bool = False
    max_depth: int = 3
    sort_by: str = "order"  # order, title, date
    include_index_pages: bool = True
    custom_order: Dict[str, int] = None


class NavigationGenerator:
    """
    Generates navigation structure for documentation
    """
    
    def __init__(self, docs_root: Path, config: NavigationConfig = None):
        self.docs_root = Path(docs_root)
        self.config = config or NavigationConfig()
        
        # Default category order and icons
        self.category_config = {
            'user-guide': {
                'title': 'User Guide',
                'order': 1,
                'icon': '📚',
                'description': 'Guides for end users'
            },
            'developer-guide': {
                'title': 'Developer Guide',
                'order': 2,
                'icon': '🔧',
                'description': 'Technical documentation for developers'
            },
            'api': {
                'title': 'API Reference',
                'order': 3,
                'icon': '🔌',
                'description': 'API documentation and reference'
            },
            'deployment': {
                'title': 'Deployment',
                'order': 4,
                'icon': '🚀',
                'description': 'Deployment and operations'
            },
            'reference': {
                'title': 'Reference',
                'order': 5,
                'icon': '📖',
                'description': 'Technical reference materials'
            }
        }
        
        # Page order within categories
        self.page_order = {
            'index.md': 0,
            'installation.md': 1,
            'quick-start.md': 2,
            'configuration.md': 3,
            'troubleshooting.md': 4,
            'architecture.md': 1,
            'contributing.md': 2,
            'testing.md': 3
        }
    
    def generate_navigation(self) -> NavigationItem:
        """
        Generate complete navigation structure
        """
        # Discover all documentation files
        pages = self._discover_pages()
        
        # Organize by category
        categories = self._organize_by_category(pages)
        
        # Build navigation tree
        root = NavigationItem(
            title="Documentation",
            children=[]
        )
        
        # Add home page
        home_page = self._find_home_page(pages)
        if home_page:
            root.children.append(NavigationItem(
                title="Home",
                path=home_page['path'],
                url=home_page['url'],
                order=0,
                icon="🏠"
            ))
        
        # Add categories
        for category_name, category_pages in categories.items():
            category_item = self._build_category_navigation(category_name, category_pages)
            if category_item:
                root.children.append(category_item)
        
        # Sort children
        if root.children:
            root.children.sort(key=lambda x: x.order)
        
        return root
    
    def _discover_pages(self) -> List[Dict[str, Any]]:
        """Discover all documentation pages"""
        pages = []
        
        for md_file in self.docs_root.rglob('*.md'):
            if self._should_skip_file(md_file):
                continue
            
            try:
                page_info = self._extract_page_info(md_file)
                if page_info:
                    pages.append(page_info)
            except Exception as e:
                print(f"Warning: Could not process {md_file}: {e}")
        
        return pages
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_patterns = [
            'templates',
            '.metadata',
            '.search',
            'node_modules',
            '.git'
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _extract_page_info(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract page information for navigation"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata
            metadata = self._parse_frontmatter(content)
            
            # Get basic info
            relative_path = str(file_path.relative_to(self.docs_root))
            title = metadata.get('title', self._extract_title_from_content(content))
            category = metadata.get('category', self._infer_category(file_path))
            
            # Determine order
            order = self._determine_order(file_path, metadata)
            
            return {
                'title': title,
                'path': relative_path,
                'url': relative_path.replace('.md', '.html'),
                'category': category,
                'order': order,
                'tags': metadata.get('tags', []),
                'status': metadata.get('status', 'published'),
                'description': metadata.get('description', ''),
                'last_updated': metadata.get('last_updated', ''),
                'file_path': file_path
            }
            
        except Exception as e:
            print(f"Error extracting info from {file_path}: {e}")
            return None
    
    def _parse_frontmatter(self, content: str) -> Dict[str, Any]:
        """Parse YAML frontmatter"""
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
    
    def _extract_title_from_content(self, content: str) -> str:
        """Extract title from first heading"""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        return "Untitled"
    
    def _infer_category(self, file_path: Path) -> str:
        """Infer category from file path"""
        path_parts = file_path.parts
        
        for part in path_parts:
            if part in self.category_config:
                return part
        
        # Default category
        return 'reference'
    
    def _determine_order(self, file_path: Path, metadata: Dict[str, Any]) -> int:
        """Determine page order"""
        # Check metadata first
        if 'order' in metadata:
            return metadata['order']
        
        # Check predefined order
        filename = file_path.name
        if filename in self.page_order:
            return self.page_order[filename]
        
        # Default order based on filename
        if filename == 'index.md':
            return 0
        elif 'install' in filename.lower():
            return 1
        elif 'quick' in filename.lower() or 'start' in filename.lower():
            return 2
        elif 'config' in filename.lower():
            return 3
        elif 'troubleshoot' in filename.lower():
            return 99
        else:
            return 50
    
    def _organize_by_category(self, pages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Organize pages by category"""
        categories = {}
        
        for page in pages:
            category = page['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(page)
        
        # Sort pages within each category
        for category_pages in categories.values():
            category_pages.sort(key=lambda p: (p['order'], p['title']))
        
        return categories
    
    def _find_home_page(self, pages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the home/index page"""
        for page in pages:
            if page['path'] == 'index.md':
                return page
        return None
    
    def _build_category_navigation(self, category_name: str, pages: List[Dict[str, Any]]) -> Optional[NavigationItem]:
        """Build navigation for a category"""
        if not pages:
            return None
        
        # Get category configuration
        category_config = self.category_config.get(category_name, {})
        
        category_item = NavigationItem(
            title=category_config.get('title', category_name.replace('-', ' ').title()),
            category=category_name,
            order=category_config.get('order', 999),
            icon=category_config.get('icon'),
            description=category_config.get('description'),
            children=[]
        )
        
        # Build hierarchical structure
        category_item.children = self._build_hierarchical_structure(pages, category_name)
        
        return category_item
    
    def _build_hierarchical_structure(self, pages: List[Dict[str, Any]], category: str) -> List[NavigationItem]:
        """Build hierarchical navigation structure"""
        # Group pages by directory structure
        structure = {}
        
        for page in pages:
            if page['status'] != 'published' and page['status'] != 'approved':
                continue  # Skip draft pages
            
            path_parts = Path(page['path']).parts
            
            # Remove category from path if it's the first part
            if path_parts and path_parts[0] == category:
                path_parts = path_parts[1:]
            
            # Build nested structure
            current_level = structure
            for i, part in enumerate(path_parts[:-1]):  # Exclude filename
                if part not in current_level:
                    current_level[part] = {'_pages': [], '_children': {}}
                current_level = current_level[part]['_children']
            
            # Add page to appropriate level
            if path_parts:
                parent_key = path_parts[-2] if len(path_parts) > 1 else '_root'
                if parent_key == '_root':
                    if '_root' not in structure:
                        structure['_root'] = {'_pages': [], '_children': {}}
                    structure['_root']['_pages'].append(page)
                else:
                    # Find the parent directory
                    current_level = structure
                    for part in path_parts[:-2]:
                        if part in current_level:
                            current_level = current_level[part]['_children']
                    
                    if parent_key not in current_level:
                        current_level[parent_key] = {'_pages': [], '_children': {}}
                    current_level[parent_key]['_pages'].append(page)
        
        # Convert structure to NavigationItems
        return self._convert_structure_to_navigation(structure)
    
    def _convert_structure_to_navigation(self, structure: Dict[str, Any]) -> List[NavigationItem]:
        """Convert nested structure to NavigationItem list"""
        nav_items = []
        
        # Handle root pages first
        if '_root' in structure:
            for page in structure['_root']['_pages']:
                nav_items.append(NavigationItem(
                    title=page['title'],
                    path=page['path'],
                    url=page['url'],
                    order=page['order'],
                    description=page['description']
                ))
        
        # Handle subdirectories
        for dir_name, dir_data in structure.items():
            if dir_name == '_root':
                continue
            
            # Create directory item
            dir_item = NavigationItem(
                title=self._format_directory_title(dir_name),
                order=self._get_directory_order(dir_name),
                children=[]
            )
            
            # Add pages in this directory
            for page in dir_data['_pages']:
                dir_item.children.append(NavigationItem(
                    title=page['title'],
                    path=page['path'],
                    url=page['url'],
                    order=page['order'],
                    description=page['description']
                ))
            
            # Add subdirectories recursively
            if dir_data['_children']:
                subdir_items = self._convert_structure_to_navigation(dir_data['_children'])
                dir_item.children.extend(subdir_items)
            
            # Sort children
            if dir_item.children:
                dir_item.children.sort(key=lambda x: (x.order, x.title))
                nav_items.append(dir_item)
        
        # Sort items
        nav_items.sort(key=lambda x: (x.order, x.title))
        return nav_items
    
    def _format_directory_title(self, dir_name: str) -> str:
        """Format directory name as title"""
        # Convert kebab-case to title case
        title = dir_name.replace('-', ' ').replace('_', ' ')
        return ' '.join(word.capitalize() for word in title.split())
    
    def _get_directory_order(self, dir_name: str) -> int:
        """Get order for directory"""
        directory_order = {
            'installation': 1,
            'configuration': 2,
            'troubleshooting': 99,
            'reference': 90,
            'examples': 80
        }
        
        return directory_order.get(dir_name.lower(), 50)
    
    def generate_mkdocs_nav(self, root: NavigationItem) -> List[Any]:
        """Generate MkDocs navigation format"""
        nav = []
        
        for child in root.children or []:
            nav_item = self._convert_to_mkdocs_format(child)
            if nav_item:
                nav.append(nav_item)
        
        return nav
    
    def _convert_to_mkdocs_format(self, item: NavigationItem) -> Any:
        """Convert NavigationItem to MkDocs format"""
        if item.path:
            # Leaf item with path
            return {item.title: item.path}
        elif item.children:
            # Parent item with children
            children = []
            for child in item.children:
                child_item = self._convert_to_mkdocs_format(child)
                if child_item:
                    children.append(child_item)
            
            if children:
                return {item.title: children}
        
        return None
    
    def generate_sidebar_json(self, root: NavigationItem) -> Dict[str, Any]:
        """Generate sidebar JSON for web interface"""
        return {
            'title': root.title,
            'items': [self._convert_to_sidebar_format(child) for child in root.children or []]
        }
    
    def _convert_to_sidebar_format(self, item: NavigationItem) -> Dict[str, Any]:
        """Convert NavigationItem to sidebar JSON format"""
        sidebar_item = {
            'title': item.title,
            'order': item.order
        }
        
        if item.path:
            sidebar_item['path'] = item.path
            sidebar_item['url'] = item.url
        
        if item.icon:
            sidebar_item['icon'] = item.icon
        
        if item.description:
            sidebar_item['description'] = item.description
        
        if item.children:
            sidebar_item['children'] = [
                self._convert_to_sidebar_format(child) 
                for child in item.children
            ]
        
        return sidebar_item
    
    def generate_breadcrumb_data(self, current_path: str, root: NavigationItem) -> List[Dict[str, str]]:
        """Generate breadcrumb data for a given path"""
        breadcrumbs = [{'title': 'Home', 'url': '/'}]
        
        # Find the path in navigation tree
        path_items = self._find_path_in_tree(current_path, root)
        
        for item in path_items:
            breadcrumbs.append({
                'title': item.title,
                'url': item.url or '#'
            })
        
        return breadcrumbs
    
    def _find_path_in_tree(self, target_path: str, root: NavigationItem) -> List[NavigationItem]:
        """Find path to target in navigation tree"""
        def search_recursive(item: NavigationItem, path: List[NavigationItem]) -> Optional[List[NavigationItem]]:
            current_path = path + [item]
            
            if item.path == target_path:
                return current_path
            
            if item.children:
                for child in item.children:
                    result = search_recursive(child, current_path)
                    if result:
                        return result
            
            return None
        
        if root.children:
            for child in root.children:
                result = search_recursive(child, [])
                if result:
                    return result
        
        return []
    
    def save_navigation_files(self, root: NavigationItem, output_dir: Path):
        """Save navigation files in various formats"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save MkDocs navigation
        mkdocs_nav = self.generate_mkdocs_nav(root)
        with open(output_dir / 'mkdocs_nav.yaml', 'w', encoding='utf-8') as f:
            yaml.dump({'nav': mkdocs_nav}, f, default_flow_style=False)
        
        # Save sidebar JSON
        sidebar_json = self.generate_sidebar_json(root)
        with open(output_dir / 'sidebar.json', 'w', encoding='utf-8') as f:
            json.dump(sidebar_json, f, indent=2, ensure_ascii=False)
        
        # Save full navigation data
        nav_data = asdict(root)
        with open(output_dir / 'navigation.json', 'w', encoding='utf-8') as f:
            json.dump(nav_data, f, indent=2, ensure_ascii=False)
        
        print(f"Navigation files saved to {output_dir}")


def main():
    """CLI interface for navigation generator"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate documentation navigation')
    parser.add_argument('--docs-dir', default='docs', help='Documentation directory')
    parser.add_argument('--output-dir', default='docs/.metadata', help='Output directory')
    parser.add_argument('--format', choices=['mkdocs', 'json', 'all'], default='all',
                       help='Output format')
    
    args = parser.parse_args()
    
    config = NavigationConfig()
    generator = NavigationGenerator(Path(args.docs_dir), config)
    
    print("Generating navigation structure...")
    root = generator.generate_navigation()
    
    if args.format in ['mkdocs', 'all']:
        mkdocs_nav = generator.generate_mkdocs_nav(root)
        print("MkDocs Navigation:")
        print(yaml.dump({'nav': mkdocs_nav}, default_flow_style=False))
    
    if args.format in ['json', 'all']:
        sidebar_json = generator.generate_sidebar_json(root)
        print("Sidebar JSON:")
        print(json.dumps(sidebar_json, indent=2))
    
    # Save files
    generator.save_navigation_files(root, Path(args.output_dir))


if __name__ == '__main__':
    main()