"""
Documentation Migration Tool

Specialized tool for migrating scattered documentation files
to a unified structure with proper categorization and metadata.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yaml
import json


@dataclass
class MigrationRule:
    """Rule for migrating documentation files"""
    pattern: str
    target_category: str
    target_name: Optional[str] = None
    add_metadata: Optional[Dict] = None


class DocumentationMigrator:
    """
    Tool for migrating scattered documentation to unified structure
    """
    
    def __init__(self, source_root: Path, target_root: Path):
        self.source_root = Path(source_root)
        self.target_root = Path(target_root)
        
        # Predefined migration rules
        self.migration_rules = [
            # User guides
            MigrationRule("*USER_GUIDE*", "user-guide", "user-guide.md"),
            MigrationRule("*INSTALLATION*", "user-guide", "installation.md"),
            MigrationRule("*TROUBLESHOOTING*", "user-guide", "troubleshooting.md"),
            MigrationRule("*GETTING_STARTED*", "user-guide", "getting-started.md"),
            
            # Developer guides
            MigrationRule("*DEVELOPER_GUIDE*", "developer-guide", "developer-guide.md"),
            MigrationRule("*CONTRIBUTING*", "developer-guide", "contributing.md"),
            MigrationRule("*ARCHITECTURE*", "developer-guide", "architecture.md"),
            MigrationRule("*API*", "developer-guide", "api-reference.md"),
            
            # Deployment guides
            MigrationRule("*DEPLOYMENT*", "deployment", "deployment-guide.md"),
            MigrationRule("*PRODUCTION*", "deployment", "production-setup.md"),
            MigrationRule("*MONITORING*", "deployment", "monitoring.md"),
            
            # Task summaries -> reference
            MigrationRule("TASK_*_SUMMARY*", "reference/task-summaries"),
            MigrationRule("*IMPLEMENTATION_SUMMARY*", "reference/implementation-summaries"),
            
            # WAN22 specific docs -> user-guide
            MigrationRule("WAN22_*_GUIDE*", "user-guide"),
            MigrationRule("WAN22_*_USER_*", "user-guide"),
            
            # Performance and optimization -> reference
            MigrationRule("*PERFORMANCE*", "reference/performance"),
            MigrationRule("*OPTIMIZATION*", "reference/performance"),
            MigrationRule("*BENCHMARK*", "reference/performance"),
            
            # Error handling and fixes -> reference
            MigrationRule("*ERROR*", "reference/troubleshooting"),
            MigrationRule("*FIX*", "reference/troubleshooting"),
            MigrationRule("*RECOVERY*", "reference/troubleshooting"),
            
            # Configuration -> reference
            MigrationRule("*CONFIG*", "reference/configuration"),
            MigrationRule("*SETTINGS*", "reference/configuration"),
        ]
    
    def discover_scattered_docs(self) -> List[Path]:
        """
        Discover all documentation files that need migration
        """
        doc_files = []
        doc_extensions = {'.md', '.rst', '.txt'}
        
        # Skip already organized docs directory
        exclude_dirs = {'docs', 'node_modules', 'venv', '__pycache__', '.git'}
        
        for file_path in self.source_root.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix.lower() in doc_extensions and
                not any(exclude_dir in file_path.parts for exclude_dir in exclude_dirs)):
                
                # Check if it's actually documentation
                if self._is_documentation_file(file_path):
                    doc_files.append(file_path)
        
        return doc_files
    
    def _is_documentation_file(self, file_path: Path) -> bool:
        """
        Determine if a file is documentation based on name and content
        """
        filename = file_path.name.upper()
        
        # Common documentation file patterns
        doc_patterns = [
            'README', 'CHANGELOG', 'GUIDE', 'MANUAL', 'DOCS',
            'INSTALLATION', 'SETUP', 'CONFIGURATION', 'TROUBLESHOOTING',
            'API', 'REFERENCE', 'TUTORIAL', 'EXAMPLES',
            'SUMMARY', 'REPORT', 'STATUS', 'IMPLEMENTATION'
        ]
        
        # Check filename patterns
        if any(pattern in filename for pattern in doc_patterns):
            return True
        
        # Check file content for documentation indicators
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(1000)  # Read first 1000 chars
                
            content_lower = content.lower()
            doc_indicators = [
                '# ', '## ', '### ',  # Markdown headers
                'documentation', 'guide', 'manual', 'tutorial',
                'installation', 'setup', 'configuration',
                'api reference', 'troubleshooting'
            ]
            
            return any(indicator in content_lower for indicator in doc_indicators)
            
        except Exception:
            return False
    
    def categorize_file(self, file_path: Path) -> Tuple[str, str]:
        """
        Categorize a file and determine its target location
        """
        filename = file_path.name
        
        # Apply migration rules
        for rule in self.migration_rules:
            if self._matches_pattern(filename, rule.pattern):
                target_name = rule.target_name or self._generate_target_name(filename)
                return rule.target_category, target_name
        
        # Default categorization
        return self._default_categorization(file_path)
    
    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches pattern (simple glob-like matching)"""
        import fnmatch
        return fnmatch.fnmatch(filename.upper(), pattern.upper())
    
    def _generate_target_name(self, filename: str) -> str:
        """Generate a clean target filename"""
        name = filename
        
        # Remove common prefixes
        prefixes = ['TASK_', 'WAN22_', 'ENHANCED_', 'IMPLEMENTATION_', 'SUMMARY_']
        for prefix in prefixes:
            if name.upper().startswith(prefix):
                name = name[len(prefix):]
        
        # Clean up the name
        name = name.lower()
        name = name.replace('_', '-')
        
        # Remove redundant suffixes
        suffixes = ['-summary', '-implementation', '-guide', '-status']
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        
        # Ensure .md extension
        if not name.endswith('.md'):
            name = name.rsplit('.', 1)[0] + '.md'
        
        return name
    
    def _default_categorization(self, file_path: Path) -> Tuple[str, str]:
        """Default categorization for files that don't match rules"""
        filename = file_path.name.lower()
        
        if 'readme' in filename:
            return 'reference', 'readme.md'
        elif 'changelog' in filename:
            return 'reference', 'changelog.md'
        elif any(word in filename for word in ['test', 'spec']):
            return 'developer-guide', self._generate_target_name(file_path.name)
        else:
            return 'reference/misc', self._generate_target_name(file_path.name)
    
    def migrate_file(self, source_path: Path, dry_run: bool = False) -> Dict:
        """
        Migrate a single file to the target structure
        """
        category, target_name = self.categorize_file(source_path)
        target_dir = self.target_root / category
        target_path = target_dir / target_name
        
        migration_info = {
            'source': str(source_path),
            'target': str(target_path),
            'category': category,
            'action': 'migrate'
        }
        
        if dry_run:
            migration_info['action'] = 'would_migrate'
            return migration_info
        
        try:
            # Create target directory
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Read source content
            with open(source_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add metadata if not present
            if not content.startswith('---'):
                metadata = self._generate_metadata(source_path, category)
                frontmatter = yaml.dump(metadata, default_flow_style=False)
                content = f"---\n{frontmatter}---\n\n{content}"
            
            # Handle duplicate files
            if target_path.exists():
                # Create backup name
                counter = 1
                base_name = target_path.stem
                while target_path.exists():
                    target_path = target_dir / f"{base_name}-{counter}.md"
                    counter += 1
                migration_info['target'] = str(target_path)
                migration_info['action'] = 'migrate_renamed'
            
            # Write to target
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            migration_info['status'] = 'success'
            
        except Exception as e:
            migration_info['status'] = 'error'
            migration_info['error'] = str(e)
        
        return migration_info
    
    def _generate_metadata(self, source_path: Path, category: str) -> Dict:
        """Generate metadata for migrated file"""
        return {
            'title': self._extract_title_from_filename(source_path.name),
            'category': category,
            'original_path': str(source_path.relative_to(self.source_root)),
            'migrated_date': str(Path(source_path).stat().st_mtime),
            'tags': self._generate_tags(source_path, category)
        }
    
    def _extract_title_from_filename(self, filename: str) -> str:
        """Extract a readable title from filename"""
        title = filename.rsplit('.', 1)[0]  # Remove extension
        
        # Remove prefixes
        prefixes = ['TASK_', 'WAN22_', 'ENHANCED_', 'IMPLEMENTATION_', 'SUMMARY_']
        for prefix in prefixes:
            if title.upper().startswith(prefix):
                title = title[len(prefix):]
        
        # Convert to readable format
        title = title.replace('_', ' ').replace('-', ' ')
        title = ' '.join(word.capitalize() for word in title.split())
        
        return title
    
    def _generate_tags(self, source_path: Path, category: str) -> List[str]:
        """Generate tags based on filename and category"""
        tags = [category]
        
        filename_lower = source_path.name.lower()
        
        tag_keywords = {
            'api': ['api', 'endpoint', 'rest'],
            'configuration': ['config', 'settings', 'environment'],
            'installation': ['install', 'setup', 'requirements'],
            'troubleshooting': ['troubleshoot', 'debug', 'error', 'fix'],
            'performance': ['performance', 'optimization', 'benchmark'],
            'testing': ['test', 'validation', 'spec'],
            'ui': ['ui', 'interface', 'frontend', 'gradio'],
            'backend': ['backend', 'server', 'api'],
            'model': ['model', 'ai', 'generation', 'pipeline']
        }
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in filename_lower for keyword in keywords):
                tags.append(tag)
        
        return list(set(tags))  # Remove duplicates
    
    def migrate_all(self, dry_run: bool = False) -> Dict:
        """
        Migrate all discovered documentation files
        """
        scattered_docs = self.discover_scattered_docs()
        
        results = {
            'total_files': len(scattered_docs),
            'successful_migrations': 0,
            'failed_migrations': 0,
            'renamed_files': 0,
            'migrations': [],
            'errors': []
        }
        
        for doc_file in scattered_docs:
            migration_result = self.migrate_file(doc_file, dry_run)
            results['migrations'].append(migration_result)
            
            if migration_result.get('status') == 'success':
                results['successful_migrations'] += 1
                if migration_result.get('action') == 'migrate_renamed':
                    results['renamed_files'] += 1
            elif migration_result.get('status') == 'error':
                results['failed_migrations'] += 1
                results['errors'].append(migration_result)
        
        return results
    
    def generate_migration_report(self, results: Dict) -> str:
        """Generate a human-readable migration report"""
        report = f"""
# Documentation Migration Report

## Summary
- Total files discovered: {results['total_files']}
- Successful migrations: {results['successful_migrations']}
- Failed migrations: {results['failed_migrations']}
- Files renamed due to conflicts: {results['renamed_files']}

## Migration Details

### Successful Migrations
"""
        
        for migration in results['migrations']:
            if migration.get('status') == 'success':
                report += f"- `{migration['source']}` → `{migration['target']}`\n"
        
        if results['errors']:
            report += "\n### Failed Migrations\n"
            for error in results['errors']:
                report += f"- `{error['source']}`: {error.get('error', 'Unknown error')}\n"
        
        report += f"""
## Next Steps

1. Review the migrated documentation in the target directory
2. Update any internal links that may have broken during migration
3. Set up documentation validation to check for broken links
4. Configure the documentation server for easy access
"""
        
        return report


def main():
    """CLI interface for migration tool"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate scattered documentation to unified structure')
    parser.add_argument('--source', default='.', help='Source root directory')
    parser.add_argument('--target', default='docs', help='Target documentation directory')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without doing it')
    parser.add_argument('--report', help='Save migration report to file')
    
    args = parser.parse_args()
    
    migrator = DocumentationMigrator(args.source, args.target)
    
    print(f"Discovering scattered documentation in {args.source}...")
    results = migrator.migrate_all(dry_run=args.dry_run)
    
    print(f"\nMigration {'simulation' if args.dry_run else 'completed'}:")
    print(f"  Total files: {results['total_files']}")
    print(f"  Successful: {results['successful_migrations']}")
    print(f"  Failed: {results['failed_migrations']}")
    print(f"  Renamed: {results['renamed_files']}")
    
    if args.report:
        report = migrator.generate_migration_report(results)
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"\nMigration report saved to: {args.report}")


if __name__ == '__main__':
    main()