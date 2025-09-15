"""
Documentation Metadata Manager

Manages documentation metadata, cross-references, and relationships
between documentation pages.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import re


@dataclass
class DocumentMetadata:
    """Metadata for a documentation page"""
    title: str
    category: str
    tags: List[str]
    last_updated: str
    author: Optional[str] = None
    reviewers: List[str] = None
    status: str = "draft"
    difficulty: Optional[str] = None
    estimated_time: Optional[str] = None
    prerequisites: List[str] = None
    related_pages: List[str] = None
    api_version: Optional[str] = None


@dataclass
class CrossReference:
    """Cross-reference between documentation pages"""
    source_page: str
    target_page: str
    reference_type: str  # link, mention, related, prerequisite
    context: str  # surrounding text or description


@dataclass
class DocumentationIndex:
    """Complete documentation index with metadata and relationships"""
    pages: Dict[str, DocumentMetadata]
    cross_references: List[CrossReference]
    categories: Dict[str, List[str]]
    tags: Dict[str, List[str]]
    last_updated: str


class MetadataManager:
    """
    Manages documentation metadata and cross-references
    """
    
    def __init__(self, docs_root: Path):
        self.docs_root = Path(docs_root)
        self.metadata_file = self.docs_root / '.metadata' / 'index.json'
        self.pages: Dict[str, DocumentMetadata] = {}
        self.cross_references: List[CrossReference] = []
        
        # Ensure metadata directory exists
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
    
    def scan_documentation(self) -> DocumentationIndex:
        """
        Scan all documentation files and extract metadata
        """
        self.pages = {}
        self.cross_references = []
        
        # Find all markdown files
        for md_file in self.docs_root.rglob('*.md'):
            if self._should_skip_file(md_file):
                continue
            
            try:
                metadata = self._extract_metadata(md_file)
                if metadata:
                    relative_path = str(md_file.relative_to(self.docs_root))
                    self.pages[relative_path] = metadata
                    
                    # Extract cross-references
                    refs = self._extract_cross_references(md_file, relative_path)
                    self.cross_references.extend(refs)
                    
            except Exception as e:
                print(f"Error processing {md_file}: {e}")
        
        return self._build_index()
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_patterns = [
            '.metadata',
            'templates',
            'node_modules',
            '.git'
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _extract_metadata(self, file_path: Path) -> Optional[DocumentMetadata]:
        """Extract metadata from a documentation file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract frontmatter
            frontmatter = self._parse_frontmatter(content)
            if not frontmatter:
                # Generate basic metadata if no frontmatter
                frontmatter = self._generate_basic_metadata(file_path)
            
            # Validate required fields
            required_fields = ['title', 'category']
            for field in required_fields:
                if field not in frontmatter:
                    if field == 'title':
                        frontmatter['title'] = self._generate_title(file_path)
                    elif field == 'category':
                        frontmatter['category'] = self._infer_category(file_path)
            
            # Ensure tags is a list
            if 'tags' in frontmatter:
                if isinstance(frontmatter['tags'], str):
                    frontmatter['tags'] = [tag.strip() for tag in frontmatter['tags'].split(',')]
            else:
                frontmatter['tags'] = []
            
            # Set default values
            frontmatter.setdefault('last_updated', datetime.now().strftime('%Y-%m-%d'))
            frontmatter.setdefault('status', 'draft')
            frontmatter.setdefault('reviewers', [])
            frontmatter.setdefault('prerequisites', [])
            frontmatter.setdefault('related_pages', [])
            
            return DocumentMetadata(**frontmatter)
            
        except Exception as e:
            print(f"Error extracting metadata from {file_path}: {e}")
            return None
    
    def _parse_frontmatter(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse YAML frontmatter from content"""
        if not content.startswith('---'):
            return None
        
        try:
            end_marker = content.find('---', 3)
            if end_marker == -1:
                return None
            
            frontmatter_text = content[3:end_marker].strip()
            return yaml.safe_load(frontmatter_text) or {}
            
        except Exception:
            return None
    
    def _generate_basic_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Generate basic metadata for files without frontmatter"""
        return {
            'title': self._generate_title(file_path),
            'category': self._infer_category(file_path),
            'tags': self._infer_tags(file_path),
            'last_updated': datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d'),
            'status': 'needs_review'
        }
    
    def _generate_title(self, file_path: Path) -> str:
        """Generate title from filename"""
        title = file_path.stem
        
        # Remove common prefixes
        prefixes = ['TASK_', 'WAN22_', 'ENHANCED_']
        for prefix in prefixes:
            if title.upper().startswith(prefix):
                title = title[len(prefix):]
        
        # Convert to readable format
        title = title.replace('_', ' ').replace('-', ' ')
        title = ' '.join(word.capitalize() for word in title.split())
        
        return title
    
    def _infer_category(self, file_path: Path) -> str:
        """Infer category from file path"""
        path_parts = file_path.parts
        
        category_mapping = {
            'user-guide': 'user-guide',
            'developer-guide': 'developer-guide',
            'deployment': 'deployment',
            'api': 'api',
            'reference': 'reference'
        }
        
        for part in path_parts:
            if part in category_mapping:
                return category_mapping[part]
        
        # Infer from filename
        filename_lower = file_path.name.lower()
        if any(word in filename_lower for word in ['user', 'guide', 'tutorial']):
            return 'user-guide'
        elif any(word in filename_lower for word in ['api', 'reference']):
            return 'api'
        elif any(word in filename_lower for word in ['deploy', 'install', 'setup']):
            return 'deployment'
        elif any(word in filename_lower for word in ['dev', 'develop', 'contrib']):
            return 'developer-guide'
        else:
            return 'reference'
    
    def _infer_tags(self, file_path: Path) -> List[str]:
        """Infer tags from filename and path"""
        tags = []
        
        filename_lower = file_path.name.lower()
        path_lower = str(file_path).lower()
        
        tag_keywords = {
            'api': ['api', 'endpoint', 'rest'],
            'configuration': ['config', 'settings', 'environment'],
            'installation': ['install', 'setup', 'requirements'],
            'troubleshooting': ['troubleshoot', 'debug', 'error', 'fix'],
            'performance': ['performance', 'optimization', 'benchmark'],
            'testing': ['test', 'validation', 'spec'],
            'ui': ['ui', 'interface', 'frontend', 'gradio'],
            'backend': ['backend', 'server', 'api'],
            'model': ['model', 'ai', 'generation', 'pipeline'],
            'startup': ['startup', 'launch', 'boot'],
            'monitoring': ['monitor', 'health', 'metrics']
        }
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in filename_lower or keyword in path_lower for keyword in keywords):
                tags.append(tag)
        
        return tags
    
    def _extract_cross_references(self, file_path: Path, relative_path: str) -> List[CrossReference]:
        """Extract cross-references from a documentation file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            references = []
            
            # Extract markdown links
            link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
            for match in re.finditer(link_pattern, content):
                link_text = match.group(1)
                link_url = match.group(2)
                
                # Skip external links
                if link_url.startswith(('http://', 'https://', 'mailto:')):
                    continue
                
                # Normalize relative paths
                if link_url.startswith('../'):
                    # Resolve relative path
                    current_dir = Path(relative_path).parent
                    target_path = (current_dir / link_url).resolve()
                    link_url = str(target_path)
                elif link_url.startswith('./'):
                    current_dir = Path(relative_path).parent
                    target_path = current_dir / link_url[2:]
                    link_url = str(target_path)
                
                references.append(CrossReference(
                    source_page=relative_path,
                    target_page=link_url,
                    reference_type='link',
                    context=link_text
                ))
            
            # Extract mentions (e.g., "see Configuration Guide")
            mention_pattern = r'(?:see|refer to|check)\s+([A-Z][A-Za-z\s]+(?:Guide|Manual|Reference|Documentation))'
            for match in re.finditer(mention_pattern, content, re.IGNORECASE):
                mentioned_doc = match.group(1)
                references.append(CrossReference(
                    source_page=relative_path,
                    target_page=mentioned_doc,
                    reference_type='mention',
                    context=match.group(0)
                ))
            
            return references
            
        except Exception as e:
            print(f"Error extracting cross-references from {file_path}: {e}")
            return []
    
    def _build_index(self) -> DocumentationIndex:
        """Build complete documentation index"""
        categories = {}
        tags = {}
        
        # Group by categories
        for page_path, metadata in self.pages.items():
            category = metadata.category
            if category not in categories:
                categories[category] = []
            categories[category].append(page_path)
        
        # Group by tags
        for page_path, metadata in self.pages.items():
            for tag in metadata.tags:
                if tag not in tags:
                    tags[tag] = []
                tags[tag].append(page_path)
        
        return DocumentationIndex(
            pages=self.pages,
            cross_references=self.cross_references,
            categories=categories,
            tags=tags,
            last_updated=datetime.now().isoformat()
        )
    
    def save_index(self, index: DocumentationIndex):
        """Save documentation index to file"""
        # Convert to serializable format
        index_data = {
            'pages': {path: asdict(metadata) for path, metadata in index.pages.items()},
            'cross_references': [asdict(ref) for ref in index.cross_references],
            'categories': index.categories,
            'tags': index.tags,
            'last_updated': index.last_updated
        }
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    def load_index(self) -> Optional[DocumentationIndex]:
        """Load documentation index from file"""
        if not self.metadata_file.exists():
            return None
        
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            # Convert back to objects
            pages = {}
            for path, metadata_dict in index_data['pages'].items():
                pages[path] = DocumentMetadata(**metadata_dict)
            
            cross_references = []
            for ref_dict in index_data['cross_references']:
                cross_references.append(CrossReference(**ref_dict))
            
            return DocumentationIndex(
                pages=pages,
                cross_references=cross_references,
                categories=index_data['categories'],
                tags=index_data['tags'],
                last_updated=index_data['last_updated']
            )
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return None
    
    def update_page_metadata(self, page_path: str, metadata: DocumentMetadata):
        """Update metadata for a specific page"""
        self.pages[page_path] = metadata
        
        # Update the actual file
        file_path = self.docs_root / page_path
        if file_path.exists():
            self._update_file_frontmatter(file_path, metadata)
    
    def _update_file_frontmatter(self, file_path: Path, metadata: DocumentMetadata):
        """Update frontmatter in a documentation file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Convert metadata to dict
            metadata_dict = asdict(metadata)
            
            # Remove None values
            metadata_dict = {k: v for k, v in metadata_dict.items() if v is not None}
            
            # Generate new frontmatter
            new_frontmatter = yaml.dump(metadata_dict, default_flow_style=False)
            
            # Replace or add frontmatter
            if content.startswith('---'):
                end_marker = content.find('---', 3)
                if end_marker != -1:
                    # Replace existing frontmatter
                    new_content = f"---\n{new_frontmatter}---\n{content[end_marker + 3:]}"
                else:
                    # Malformed frontmatter, add new one
                    new_content = f"---\n{new_frontmatter}---\n\n{content}"
            else:
                # Add frontmatter to file without it
                new_content = f"---\n{new_frontmatter}---\n\n{content}"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
        except Exception as e:
            print(f"Error updating frontmatter in {file_path}: {e}")
    
    def find_broken_references(self) -> List[CrossReference]:
        """Find cross-references that point to non-existent pages"""
        broken_refs = []
        
        for ref in self.cross_references:
            if ref.reference_type == 'link':
                # Check if target file exists
                target_path = self.docs_root / ref.target_page
                if not target_path.exists():
                    broken_refs.append(ref)
        
        return broken_refs
    
    def suggest_related_pages(self, page_path: str) -> List[str]:
        """Suggest related pages based on tags and content"""
        if page_path not in self.pages:
            return []
        
        current_page = self.pages[page_path]
        suggestions = []
        
        # Find pages with similar tags
        for other_path, other_metadata in self.pages.items():
            if other_path == page_path:
                continue
            
            # Calculate tag similarity
            common_tags = set(current_page.tags) & set(other_metadata.tags)
            if common_tags:
                score = len(common_tags) / max(len(current_page.tags), len(other_metadata.tags))
                suggestions.append((other_path, score))
        
        # Sort by similarity score
        suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return [path for path, score in suggestions[:5]]
    
    def generate_navigation_menu(self) -> Dict[str, Any]:
        """Generate navigation menu structure"""
        menu = {}
        
        for category, pages in self.categories.items():
            menu[category] = []
            
            # Sort pages by title
            sorted_pages = sorted(pages, key=lambda p: self.pages[p].title)
            
            for page_path in sorted_pages:
                metadata = self.pages[page_path]
                menu[category].append({
                    'title': metadata.title,
                    'path': page_path,
                    'status': metadata.status
                })
        
        return menu


def main():
    """CLI interface for metadata manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage documentation metadata')
    parser.add_argument('--docs-root', default='docs', help='Documentation root directory')
    parser.add_argument('--scan', action='store_true', help='Scan and update metadata')
    parser.add_argument('--check-refs', action='store_true', help='Check for broken references')
    parser.add_argument('--generate-nav', action='store_true', help='Generate navigation menu')
    
    args = parser.parse_args()
    
    manager = MetadataManager(args.docs_root)
    
    if args.scan:
        print("Scanning documentation...")
        index = manager.scan_documentation()
        manager.save_index(index)
        print(f"Scanned {len(index.pages)} pages")
        print(f"Found {len(index.cross_references)} cross-references")
    
    if args.check_refs:
        print("Checking for broken references...")
        index = manager.load_index()
        if index:
            broken_refs = manager.find_broken_references()
            if broken_refs:
                print(f"Found {len(broken_refs)} broken references:")
                for ref in broken_refs:
                    print(f"  {ref.source_page} -> {ref.target_page}")
            else:
                print("No broken references found")
    
    if args.generate_nav:
        print("Generating navigation menu...")
        index = manager.load_index()
        if index:
            nav_menu = manager.generate_navigation_menu()
            nav_file = Path(args.docs_root) / '.metadata' / 'navigation.json'
            with open(nav_file, 'w') as f:
                json.dump(nav_menu, f, indent=2)
            print(f"Navigation menu saved to {nav_file}")


if __name__ == '__main__':
    main()
