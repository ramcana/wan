#!/usr/bin/env python3
"""
Pre-commit hook for documentation link checking.
Validates internal links in markdown files.
"""

import sys
import re
from pathlib import Path
from typing import List, Set, Tuple


def extract_markdown_links(content: str) -> List[Tuple[str, int]]:
    """Extract markdown links from content with line numbers."""
    links = []
    
    # Pattern for markdown links: [text](url)
    link_pattern = r'\[([^\]]*)\]\(([^)]+)\)'
    
    lines = content.split('\n')
    for line_num, line in enumerate(lines, 1):
        matches = re.finditer(link_pattern, line)
        for match in matches:
            link_url = match.group(2)
            links.append((link_url, line_num))
    
    return links


def extract_reference_links(content: str) -> List[Tuple[str, int]]:
    """Extract reference-style links from content."""
    links = []
    
    # Pattern for reference links: [text]: url
    ref_pattern = r'^\s*\[([^\]]+)\]:\s*(.+)$'
    
    lines = content.split('\n')
    for line_num, line in enumerate(lines, 1):
        match = re.match(ref_pattern, line)
        if match:
            link_url = match.group(2).strip()
            links.append((link_url, line_num))
    
    return links


def is_internal_link(url: str) -> bool:
    """Check if a URL is an internal link."""
    # Skip external URLs
    if url.startswith(('http://', 'https://', 'ftp://', 'mailto:')):
        return False
    
    # Skip anchors without files
    if url.startswith('#'):
        return False
    
    # Skip data URLs
    if url.startswith('data:'):
        return False
    
    return True


def resolve_link_path(link_url: str, current_file: Path) -> Path:
    """Resolve a link URL to an absolute path."""
    # Remove anchor fragments
    if '#' in link_url:
        link_url = link_url.split('#')[0]
    
    # Skip empty links (pure anchors)
    if not link_url:
        return None
    
    # Resolve relative to current file's directory
    if link_url.startswith('/'):
        # Absolute path from project root
        return Path('.') / link_url.lstrip('/')
    else:
        # Relative path
        return (current_file.parent / link_url).resolve()


def check_file_links(file_path: Path) -> List[str]:
    """Check all links in a markdown file."""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return [f"Could not read {file_path}: {e}"]
    
    # Extract all links
    all_links = []
    all_links.extend(extract_markdown_links(content))
    all_links.extend(extract_reference_links(content))
    
    for link_url, line_num in all_links:
        if not is_internal_link(link_url):
            continue
        
        target_path = resolve_link_path(link_url, file_path)
        if target_path is None:
            continue
        
        # Check if target exists
        if not target_path.exists():
            issues.append(
                f"{file_path}:{line_num}: Broken link to '{link_url}' "
                f"(resolved to {target_path})"
            )
        
        # Check if linking to a directory without index file
        elif target_path.is_dir():
            index_files = ['index.md', 'README.md', 'index.html']
            has_index = any((target_path / index_file).exists() for index_file in index_files)
            
            if not has_index:
                issues.append(
                    f"{file_path}:{line_num}: Link to directory '{link_url}' "
                    f"without index file"
                )
    
    return issues


def check_documentation_structure() -> List[str]:
    """Check overall documentation structure."""
    issues = []
    
    # Check for main documentation entry point
    main_docs = [Path('docs/index.md'), Path('docs/README.md'), Path('README.md')]
    if not any(doc.exists() for doc in main_docs):
        issues.append("No main documentation entry point found (docs/index.md, docs/README.md, or README.md)")
    
    # Check for essential documentation sections
    essential_docs = [
        'docs/user-guide/installation.md',
        'docs/developer-guide/index.md',
        'docs/api/index.md'
    ]
    
    for doc_path in essential_docs:
        if not Path(doc_path).exists():
            issues.append(f"Missing essential documentation: {doc_path}")
    
    return issues


def check_markdown_syntax(file_path: Path) -> List[str]:
    """Basic markdown syntax checking."""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return [f"Could not read {file_path}: {e}"]
    
    lines = content.split('\n')
    
    for line_num, line in enumerate(lines, 1):
        # Check for unmatched brackets
        open_brackets = line.count('[')
        close_brackets = line.count(']')
        open_parens = line.count('(')
        close_parens = line.count(')')
        
        # Simple check for unmatched brackets in links
        if '[' in line and ']' in line and '(' in line and ')' in line:
            # More sophisticated check would be needed for complex cases
            pass
        
        # Check for common markdown issues
        if line.strip().startswith('#') and not line.startswith('# ') and len(line.strip()) > 1:
            if not line.startswith('##') and not line.startswith('###'):
                issues.append(f"{file_path}:{line_num}: Header should have space after #")
    
    return issues


def main(file_paths: List[str]) -> int:
    """Main pre-commit hook function."""
    print("🔍 Running documentation link check...")
    
    all_issues = []
    
    # Check documentation structure
    all_issues.extend(check_documentation_structure())
    
    # Check each file
    for file_path_str in file_paths:
        file_path = Path(file_path_str)
        print(f"  Checking {file_path}...")
        
        # Check markdown syntax
        all_issues.extend(check_markdown_syntax(file_path))
        
        # Check links
        all_issues.extend(check_file_links(file_path))
    
    if all_issues:
        print("❌ Documentation link check failed:")
        for issue in all_issues:
            print(f"  • {issue}")
        
        print("\n💡 Recommendations:")
        print("  • Fix broken internal links")
        print("  • Add missing documentation files")
        print("  • Ensure directories have index files")
        print("  • Fix markdown syntax issues")
        
        return 1
    
    print("✅ Documentation link check passed!")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: pre_commit_link_check.py <file1> [file2] ...")
        sys.exit(1)
    
    sys.exit(main(sys.argv[1:]))
