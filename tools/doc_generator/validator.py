"""
Documentation Validator

Validates documentation for broken links, content quality,
freshness, and compliance with style guidelines.
"""

import os
import re
import json
import yaml
import requests
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
import concurrent.futures
import time


@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    severity: str  # error, warning, info
    category: str  # link, content, style, freshness
    message: str
    file_path: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report"""
    total_files: int
    issues: List[ValidationIssue]
    summary: Dict[str, int]
    execution_time: float
    timestamp: str


@dataclass
class LinkCheckResult:
    """Result of link checking"""
    url: str
    status_code: Optional[int]
    is_valid: bool
    error_message: Optional[str]
    response_time: Optional[float]


class DocumentationValidator:
    """
    Comprehensive documentation validator
    """
    
    def __init__(self, docs_root: Path, config: Optional[Dict[str, Any]] = None):
        self.docs_root = Path(docs_root)
        self.config = config or self._default_config()
        self.issues: List[ValidationIssue] = []
        
        # Cache for external link checks
        self.link_cache: Dict[str, LinkCheckResult] = {}
        
        # Style rules
        self.style_rules = self._load_style_rules()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default validation configuration"""
        return {
            'check_links': True,
            'check_external_links': True,
            'check_style': True,
            'check_freshness': True,
            'check_metadata': True,
            'external_link_timeout': 10,
            'freshness_threshold_days': 90,
            'max_line_length': 120,
            'required_metadata_fields': ['title', 'category', 'last_updated'],
            'allowed_categories': ['user-guide', 'developer-guide', 'api', 'deployment', 'reference'],
            'skip_patterns': ['templates', '.metadata', 'node_modules', '.git']
        }
    
    def _load_style_rules(self) -> Dict[str, Any]:
        """Load style validation rules"""
        return {
            'heading_patterns': {
                'no_trailing_punctuation': r'^#+\s+.*[.!?]$',
                'proper_capitalization': r'^#+\s+[A-Z]',
                'no_multiple_h1': r'^#\s+',
            },
            'link_patterns': {
                'markdown_link': r'\[([^\]]+)\]\(([^)]+)\)',
                'bare_url': r'https?://[^\s]+',
            },
            'content_rules': {
                'max_line_length': 120,
                'no_trailing_whitespace': r'\s+$',
                'consistent_list_markers': r'^[\s]*[-*+]\s+',
            }
        }
    
    def validate_all(self) -> ValidationReport:
        """
        Run comprehensive validation on all documentation
        """
        start_time = time.time()
        self.issues = []
        
        print("Starting documentation validation...")
        
        # Discover all markdown files
        md_files = list(self.docs_root.rglob('*.md'))
        md_files = [f for f in md_files if not self._should_skip_file(f)]
        
        print(f"Validating {len(md_files)} files...")
        
        # Validate each file
        for md_file in md_files:
            try:
                self._validate_file(md_file)
            except Exception as e:
                self.issues.append(ValidationIssue(
                    severity='error',
                    category='validation',
                    message=f"Validation failed: {e}",
                    file_path=str(md_file.relative_to(self.docs_root))
                ))
        
        # Generate summary
        summary = self._generate_summary()
        
        execution_time = time.time() - start_time
        
        report = ValidationReport(
            total_files=len(md_files),
            issues=self.issues,
            summary=summary,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
        
        print(f"Validation complete in {execution_time:.2f}s")
        print(f"Found {len(self.issues)} issues")
        
        return report
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        return any(pattern in str(file_path) for pattern in self.config['skip_patterns'])
    
    def _validate_file(self, file_path: Path):
        """Validate a single documentation file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            relative_path = str(file_path.relative_to(self.docs_root))
            
            # Run all validation checks
            if self.config['check_metadata']:
                self._validate_metadata(content, relative_path)
            
            if self.config['check_style']:
                self._validate_style(content, relative_path)
            
            if self.config['check_links']:
                self._validate_links(content, relative_path, file_path)
            
            if self.config['check_freshness']:
                self._validate_freshness(content, relative_path, file_path)
            
        except Exception as e:
            self.issues.append(ValidationIssue(
                severity='error',
                category='file',
                message=f"Could not read file: {e}",
                file_path=str(file_path.relative_to(self.docs_root))
            ))
    
    def _validate_metadata(self, content: str, file_path: str):
        """Validate frontmatter metadata"""
        metadata = self._extract_metadata(content)
        
        if not metadata:
            self.issues.append(ValidationIssue(
                severity='warning',
                category='metadata',
                message="No frontmatter metadata found",
                file_path=file_path,
                suggestion="Add YAML frontmatter with title, category, and last_updated"
            ))
            return
        
        # Check required fields
        for field in self.config['required_metadata_fields']:
            if field not in metadata:
                self.issues.append(ValidationIssue(
                    severity='error',
                    category='metadata',
                    message=f"Missing required metadata field: {field}",
                    file_path=file_path,
                    suggestion=f"Add '{field}' to frontmatter"
                ))
        
        # Validate category
        if 'category' in metadata:
            category = metadata['category']
            if category not in self.config['allowed_categories']:
                self.issues.append(ValidationIssue(
                    severity='warning',
                    category='metadata',
                    message=f"Unknown category: {category}",
                    file_path=file_path,
                    suggestion=f"Use one of: {', '.join(self.config['allowed_categories'])}"
                ))
        
        # Validate date format
        if 'last_updated' in metadata:
            try:
                datetime.strptime(metadata['last_updated'], '%Y-%m-%d')
            except ValueError:
                self.issues.append(ValidationIssue(
                    severity='error',
                    category='metadata',
                    message="Invalid date format in last_updated",
                    file_path=file_path,
                    suggestion="Use YYYY-MM-DD format"
                ))
        
        # Validate tags
        if 'tags' in metadata:
            tags = metadata['tags']
            if not isinstance(tags, list):
                self.issues.append(ValidationIssue(
                    severity='warning',
                    category='metadata',
                    message="Tags should be a list",
                    file_path=file_path,
                    suggestion="Use YAML list format: [tag1, tag2]"
                ))
    
    def _validate_style(self, content: str, file_path: str):
        """Validate content style and formatting"""
        lines = content.split('\n')
        
        h1_count = 0
        in_frontmatter = False
        
        for line_num, line in enumerate(lines, 1):
            # Skip frontmatter
            if line.strip() == '---':
                in_frontmatter = not in_frontmatter
                continue
            if in_frontmatter:
                continue
            
            # Check line length
            if len(line) > self.config['max_line_length']:
                self.issues.append(ValidationIssue(
                    severity='warning',
                    category='style',
                    message=f"Line too long ({len(line)} > {self.config['max_line_length']})",
                    file_path=file_path,
                    line_number=line_num,
                    suggestion="Break long lines or use shorter sentences"
                ))
            
            # Check trailing whitespace
            if re.search(self.style_rules['content_rules']['no_trailing_whitespace'], line):
                self.issues.append(ValidationIssue(
                    severity='info',
                    category='style',
                    message="Trailing whitespace",
                    file_path=file_path,
                    line_number=line_num,
                    suggestion="Remove trailing spaces"
                ))
            
            # Check headings
            if line.startswith('#'):
                # Count H1 headings
                if line.startswith('# '):
                    h1_count += 1
                
                # Check heading punctuation
                if re.search(self.style_rules['heading_patterns']['no_trailing_punctuation'], line):
                    self.issues.append(ValidationIssue(
                        severity='warning',
                        category='style',
                        message="Heading should not end with punctuation",
                        file_path=file_path,
                        line_number=line_num,
                        suggestion="Remove trailing punctuation from heading"
                    ))
                
                # Check heading capitalization
                if not re.search(self.style_rules['heading_patterns']['proper_capitalization'], line):
                    self.issues.append(ValidationIssue(
                        severity='info',
                        category='style',
                        message="Heading should start with capital letter",
                        file_path=file_path,
                        line_number=line_num,
                        suggestion="Capitalize first word of heading"
                    ))
        
        # Check for multiple H1 headings
        if h1_count > 1:
            self.issues.append(ValidationIssue(
                severity='warning',
                category='style',
                message=f"Multiple H1 headings found ({h1_count})",
                file_path=file_path,
                suggestion="Use only one H1 heading per page"
            ))
        
        # Check for bare URLs
        bare_urls = re.findall(self.style_rules['link_patterns']['bare_url'], content)
        if bare_urls:
            self.issues.append(ValidationIssue(
                severity='info',
                category='style',
                message=f"Found {len(bare_urls)} bare URLs",
                file_path=file_path,
                suggestion="Convert bare URLs to markdown links with descriptive text"
            ))
    
    def _validate_links(self, content: str, file_path: str, full_path: Path):
        """Validate all links in the document"""
        # Find all markdown links
        link_pattern = self.style_rules['link_patterns']['markdown_link']
        links = re.finditer(link_pattern, content)
        
        for match in links:
            link_text = match.group(1)
            link_url = match.group(2)
            line_num = content[:match.start()].count('\n') + 1
            
            # Skip anchor links
            if link_url.startswith('#'):
                self._validate_anchor_link(link_url, content, file_path, line_num)
                continue
            
            # Check external links
            if link_url.startswith(('http://', 'https://')):
                if self.config['check_external_links']:
                    self._validate_external_link(link_url, file_path, line_num)
                continue
            
            # Check internal links
            self._validate_internal_link(link_url, file_path, full_path, line_num)
    
    def _validate_anchor_link(self, anchor: str, content: str, file_path: str, line_num: int):
        """Validate anchor links within the document"""
        # Extract anchor name (remove #)
        anchor_name = anchor[1:].lower().replace(' ', '-')
        
        # Find all headings in the document
        headings = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        heading_anchors = []
        
        for heading in headings:
            # Convert heading to anchor format
            heading_anchor = re.sub(r'[^\w\s-]', '', heading.lower())
            heading_anchor = re.sub(r'[-\s]+', '-', heading_anchor)
            heading_anchors.append(heading_anchor)
        
        if anchor_name not in heading_anchors:
            self.issues.append(ValidationIssue(
                severity='error',
                category='link',
                message=f"Broken anchor link: {anchor}",
                file_path=file_path,
                line_number=line_num,
                suggestion=f"Available anchors: {', '.join(heading_anchors[:5])}"
            ))
    
    def _validate_internal_link(self, link_url: str, file_path: str, full_path: Path, line_num: int):
        """Validate internal documentation links"""
        # Resolve relative path
        if link_url.startswith('../'):
            # Relative to current file
            target_path = full_path.parent / link_url
        elif link_url.startswith('./'):
            # Relative to current directory
            target_path = full_path.parent / link_url[2:]
        else:
            # Relative to docs root
            target_path = self.docs_root / link_url
        
        try:
            target_path = target_path.resolve()
        except Exception:
            self.issues.append(ValidationIssue(
                severity='error',
                category='link',
                message=f"Invalid link path: {link_url}",
                file_path=file_path,
                line_number=line_num,
                suggestion="Check link syntax and path"
            ))
            return
        
        # Check if target exists
        if not target_path.exists():
            self.issues.append(ValidationIssue(
                severity='error',
                category='link',
                message=f"Broken internal link: {link_url}",
                file_path=file_path,
                line_number=line_num,
                suggestion="Verify the target file exists"
            ))
        
        # Check if target is within docs directory
        try:
            target_path.relative_to(self.docs_root)
        except ValueError:
            self.issues.append(ValidationIssue(
                severity='warning',
                category='link',
                message=f"Link points outside docs directory: {link_url}",
                file_path=file_path,
                line_number=line_num,
                suggestion="Use relative paths within docs directory"
            ))
    
    def _validate_external_link(self, url: str, file_path: str, line_num: int):
        """Validate external links"""
        # Check cache first
        if url in self.link_cache:
            result = self.link_cache[url]
        else:
            result = self._check_external_link(url)
            self.link_cache[url] = result
        
        if not result.is_valid:
            severity = 'error' if result.status_code and result.status_code >= 400 else 'warning'
            self.issues.append(ValidationIssue(
                severity=severity,
                category='link',
                message=f"External link issue: {result.error_message or f'HTTP {result.status_code}'}",
                file_path=file_path,
                line_number=line_num,
                suggestion="Verify the URL is correct and accessible"
            ))
    
    def _check_external_link(self, url: str) -> LinkCheckResult:
        """Check if external link is accessible"""
        try:
            start_time = time.time()
            response = requests.head(
                url, 
                timeout=self.config['external_link_timeout'],
                allow_redirects=True,
                headers={'User-Agent': 'WAN22-Documentation-Validator/1.0'}
            )
            response_time = time.time() - start_time
            
            return LinkCheckResult(
                url=url,
                status_code=response.status_code,
                is_valid=response.status_code < 400,
                error_message=None,
                response_time=response_time
            )
            
        except requests.exceptions.Timeout:
            return LinkCheckResult(
                url=url,
                status_code=None,
                is_valid=False,
                error_message="Request timeout",
                response_time=None
            )
        except requests.exceptions.ConnectionError:
            return LinkCheckResult(
                url=url,
                status_code=None,
                is_valid=False,
                error_message="Connection error",
                response_time=None
            )
        except Exception as e:
            return LinkCheckResult(
                url=url,
                status_code=None,
                is_valid=False,
                error_message=str(e),
                response_time=None
            )
    
    def _validate_freshness(self, content: str, file_path: str, full_path: Path):
        """Validate document freshness"""
        metadata = self._extract_metadata(content)
        
        # Check last_updated field
        if 'last_updated' in metadata:
            try:
                last_updated = datetime.strptime(metadata['last_updated'], '%Y-%m-%d')
                days_old = (datetime.now() - last_updated).days
                
                if days_old > self.config['freshness_threshold_days']:
                    self.issues.append(ValidationIssue(
                        severity='warning',
                        category='freshness',
                        message=f"Document is {days_old} days old",
                        file_path=file_path,
                        suggestion="Review and update the document if needed"
                    ))
            except ValueError:
                pass  # Already handled in metadata validation
        
        # Check file modification time
        try:
            file_mtime = datetime.fromtimestamp(full_path.stat().st_mtime)
            days_since_modified = (datetime.now() - file_mtime).days
            
            if days_since_modified > self.config['freshness_threshold_days']:
                self.issues.append(ValidationIssue(
                    severity='info',
                    category='freshness',
                    message=f"File not modified for {days_since_modified} days",
                    file_path=file_path,
                    suggestion="Consider reviewing for accuracy"
                ))
        except Exception:
            pass
    
    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract YAML frontmatter metadata"""
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
    
    def _generate_summary(self) -> Dict[str, int]:
        """Generate validation summary statistics"""
        summary = {
            'total_issues': len(self.issues),
            'errors': 0,
            'warnings': 0,
            'info': 0
        }
        
        # Count by severity
        for issue in self.issues:
            summary[issue.severity + 's'] = summary.get(issue.severity + 's', 0) + 1
        
        # Count by category
        categories = {}
        for issue in self.issues:
            categories[issue.category] = categories.get(issue.category, 0) + 1
        
        summary['by_category'] = categories
        
        return summary
    
    def check_links_only(self, external_only: bool = False) -> List[ValidationIssue]:
        """Quick link-only validation"""
        print("Checking links...")
        
        self.issues = []
        md_files = [f for f in self.docs_root.rglob('*.md') if not self._should_skip_file(f)]
        
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                relative_path = str(md_file.relative_to(self.docs_root))
                
                if external_only:
                    # Only check external links
                    external_links = re.findall(r'https?://[^\s\)]+', content)
                    for url in set(external_links):  # Remove duplicates
                        result = self._check_external_link(url)
                        if not result.is_valid:
                            self.issues.append(ValidationIssue(
                                severity='error',
                                category='link',
                                message=f"External link failed: {result.error_message}",
                                file_path=relative_path
                            ))
                else:
                    self._validate_links(content, relative_path, md_file)
                    
            except Exception as e:
                self.issues.append(ValidationIssue(
                    severity='error',
                    category='validation',
                    message=f"Error checking links: {e}",
                    file_path=str(md_file.relative_to(self.docs_root))
                ))
        
        print(f"Link check complete. Found {len(self.issues)} issues.")
        return self.issues
    
    def generate_report_html(self, report: ValidationReport, output_path: Path):
        """Generate HTML validation report"""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Documentation Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .summary { background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .issue { margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }
        .error { border-left-color: #d32f2f; background: #ffebee; }
        .warning { border-left-color: #f57c00; background: #fff3e0; }
        .info { border-left-color: #1976d2; background: #e3f2fd; }
        .file-path { font-family: monospace; color: #666; }
        .suggestion { font-style: italic; color: #555; margin-top: 5px; }
        .stats { display: flex; gap: 20px; }
        .stat { text-align: center; }
    </style>
</head>
<body>
    <h1>Documentation Validation Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <div class="stats">
            <div class="stat">
                <h3>{total_files}</h3>
                <p>Files Checked</p>
            </div>
            <div class="stat">
                <h3>{total_issues}</h3>
                <p>Total Issues</p>
            </div>
            <div class="stat">
                <h3>{errors}</h3>
                <p>Errors</p>
            </div>
            <div class="stat">
                <h3>{warnings}</h3>
                <p>Warnings</p>
            </div>
        </div>
        <p><strong>Execution Time:</strong> {execution_time:.2f} seconds</p>
        <p><strong>Generated:</strong> {timestamp}</p>
    </div>
    
    <h2>Issues</h2>
    {issues_html}
</body>
</html>
        """
        
        # Generate issues HTML
        issues_html = ""
        for issue in report.issues:
            location = f"{issue.file_path}"
            if issue.line_number:
                location += f":{issue.line_number}"
            
            suggestion_html = ""
            if issue.suggestion:
                suggestion_html = f'<div class="suggestion">💡 {issue.suggestion}</div>'
            
            issues_html += f"""
            <div class="issue {issue.severity}">
                <strong>{issue.severity.upper()}</strong> - {issue.category}
                <div class="file-path">{location}</div>
                <div>{issue.message}</div>
                {suggestion_html}
            </div>
            """
        
        # Fill template
        html_content = html_template.format(
            total_files=report.total_files,
            total_issues=report.summary['total_issues'],
            errors=report.summary.get('errors', 0),
            warnings=report.summary.get('warnings', 0),
            execution_time=report.execution_time,
            timestamp=report.timestamp,
            issues_html=issues_html
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report saved to {output_path}")
    
    def save_report_json(self, report: ValidationReport, output_path: Path):
        """Save validation report as JSON"""
        report_data = asdict(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"JSON report saved to {output_path}")


def main():
    """CLI interface for documentation validator"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate WAN22 documentation')
    parser.add_argument('--docs-dir', default='docs', help='Documentation directory')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--output', help='Output report file path')
    parser.add_argument('--format', choices=['json', 'html'], default='json', 
                       help='Report format')
    parser.add_argument('--check-links-only', action='store_true', 
                       help='Only check links')
    parser.add_argument('--external-links-only', action='store_true', 
                       help='Only check external links')
    parser.add_argument('--no-external-links', action='store_true', 
                       help='Skip external link checking')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Override config based on arguments
    if not config:
        config = {}
    
    if args.no_external_links:
        config['check_external_links'] = False
    
    # Initialize validator
    validator = DocumentationValidator(Path(args.docs_dir), config)
    
    # Run validation
    if args.check_links_only or args.external_links_only:
        issues = validator.check_links_only(args.external_links_only)
        
        # Create minimal report
        report = ValidationReport(
            total_files=0,
            issues=issues,
            summary={'total_issues': len(issues)},
            execution_time=0,
            timestamp=datetime.now().isoformat()
        )
    else:
        report = validator.validate_all()
    
    # Print summary
    print(f"\nValidation Summary:")
    print(f"  Total Issues: {report.summary['total_issues']}")
    if 'errors' in report.summary:
        print(f"  Errors: {report.summary['errors']}")
    if 'warnings' in report.summary:
        print(f"  Warnings: {report.summary['warnings']}")
    
    # Save report
    if args.output:
        output_path = Path(args.output)
        if args.format == 'html':
            validator.generate_report_html(report, output_path)
        else:
            validator.save_report_json(report, output_path)
    
    # Exit with error code if there are errors
    error_count = report.summary.get('errors', 0)
    exit_code = 1 if error_count > 0 else 0
    
    if error_count > 0:
        print(f"\n❌ Validation failed with {error_count} errors")
    else:
        print(f"\n✅ Validation passed")
    
    exit(exit_code)


if __name__ == '__main__':
    main()
