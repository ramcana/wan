"""
Documentation Validation and Maintenance System

Implements documentation link checking, freshness validation,
completeness analysis, and accessibility checking.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from urllib.parse import urlparse
import hashlib

from structure_analyzer import ProjectStructure, FileInfo
from component_analyzer import ComponentRelationshipMap, ComponentInfo


@dataclass
class LinkValidationResult:
    """Result of link validation."""
    link: str
    is_valid: bool
    error_message: Optional[str] = None
    link_type: str = 'unknown'  # 'internal', 'external', 'anchor'
    source_file: str = ''


@dataclass
class DocumentationIssue:
    """Represents a documentation issue."""
    file_path: str
    issue_type: str  # 'broken_link', 'outdated', 'missing', 'accessibility'
    severity: str  # 'high', 'medium', 'low'
    description: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class DocumentationMetrics:
    """Metrics about documentation quality."""
    total_files: int
    total_links: int
    broken_links: int
    outdated_files: int
    missing_components: int
    accessibility_issues: int
    coverage_percentage: float
    freshness_score: float  # 0-100


@dataclass
class DocumentationValidationReport:
    """Complete documentation validation report."""
    project_path: str
    validation_time: datetime
    metrics: DocumentationMetrics
    issues: List[DocumentationIssue]
    link_results: List[LinkValidationResult]
    recommendations: List[str]


class DocumentationValidator:
    """Validates and maintains project documentation."""
    
    def __init__(self, project_root: str, docs_dirs: Optional[List[str]] = None):
        """Initialize the documentation validator."""
        self.project_root = Path(project_root).resolve()
        self.docs_dirs = docs_dirs or ['docs', 'documentation', 'README.md']
        self.validation_time = datetime.now()
        
        # Patterns for different types of links
        self.link_patterns = {
            'markdown_link': re.compile(r'\[([^\]]+)\]\(([^)]+)\)'),
            'reference_link': re.compile(r'\[([^\]]+)\]:\s*(.+)'),
            'html_link': re.compile(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>'),
            'image_link': re.compile(r'!\[([^\]]*)\]\(([^)]+)\)'),
            'code_reference': re.compile(r'`([^`]+)`'),
            'file_reference': re.compile(r'["\']([^"\']+\.(py|md|json|yaml|yml|txt))["\']')
        }
        
        # File extensions to consider as documentation
        self.doc_extensions = {'.md', '.rst', '.txt', '.html', '.adoc'}
        
        # Freshness thresholds (in days)
        self.freshness_thresholds = {
            'critical': 30,  # Critical docs should be updated within 30 days
            'important': 90,  # Important docs within 90 days
            'normal': 180    # Normal docs within 180 days
        }
    
    def validate_all(self, structure: Optional[ProjectStructure] = None,
                    relationships: Optional[ComponentRelationshipMap] = None) -> DocumentationValidationReport:
        """Perform complete documentation validation."""
        print("Validating project documentation...")
        
        # Find all documentation files
        doc_files = self._find_documentation_files()
        print(f"Found {len(doc_files)} documentation files")
        
        # Validate links
        print("Validating links...")
        link_results = self._validate_all_links(doc_files)
        
        # Check freshness
        print("Checking documentation freshness...")
        freshness_issues = self._check_documentation_freshness(doc_files)
        
        # Check completeness
        print("Checking documentation completeness...")
        completeness_issues = []
        if structure and relationships:
            completeness_issues = self._check_documentation_completeness(
                doc_files, structure, relationships
            )
        
        # Check accessibility
        print("Checking accessibility...")
        accessibility_issues = self._check_accessibility(doc_files)
        
        # Combine all issues
        all_issues = freshness_issues + completeness_issues + accessibility_issues
        
        # Calculate metrics
        metrics = self._calculate_metrics(doc_files, link_results, all_issues, structure)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, all_issues)
        
        return DocumentationValidationReport(
            project_path=str(self.project_root),
            validation_time=self.validation_time,
            metrics=metrics,
            issues=all_issues,
            link_results=link_results,
            recommendations=recommendations
        )
    
    def _find_documentation_files(self) -> List[Path]:
        """Find all documentation files in the project."""
        doc_files = []
        
        # Search in specified documentation directories
        for docs_dir in self.docs_dirs:
            docs_path = self.project_root / docs_dir
            if docs_path.exists():
                if docs_path.is_file():
                    # Single file (like README.md)
                    if docs_path.suffix.lower() in self.doc_extensions:
                        doc_files.append(docs_path)
                else:
                    # Directory
                    for root, dirs, files in os.walk(docs_path):
                        # Skip hidden directories
                        dirs[:] = [d for d in dirs if not d.startswith('.')]
                        
                        for file in files:
                            file_path = Path(root) / file
                            if file_path.suffix.lower() in self.doc_extensions:
                                doc_files.append(file_path)
        
        # Also search for README files in component directories
        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden and common ignore directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and 
                      d not in {'__pycache__', 'node_modules', 'venv', 'env'}]
            
            for file in files:
                if file.lower().startswith('readme'):
                    file_path = Path(root) / file
                    if file_path.suffix.lower() in self.doc_extensions:
                        doc_files.append(file_path)
        
        return list(set(doc_files))  # Remove duplicates
    
    def _validate_all_links(self, doc_files: List[Path]) -> List[LinkValidationResult]:
        """Validate all links in documentation files."""
        all_results = []
        
        for doc_file in doc_files:
            try:
                results = self._validate_links_in_file(doc_file)
                all_results.extend(results)
            except Exception as e:
                print(f"Error validating links in {doc_file}: {e}")
                continue
        
        return all_results
    
    def _validate_links_in_file(self, file_path: Path) -> List[LinkValidationResult]:
        """Validate all links in a single file."""
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return [LinkValidationResult(
                link=str(file_path),
                is_valid=False,
                error_message=f"Could not read file: {e}",
                source_file=str(file_path)
            )]
        
        # Extract all links using different patterns
        for pattern_name, pattern in self.link_patterns.items():
            matches = pattern.findall(content)
            
            for match in matches:
                if isinstance(match, tuple):
                    # Most patterns return tuples (text, link)
                    if len(match) >= 2:
                        link = match[1] if pattern_name != 'code_reference' else match[0]
                    else:
                        link = match[0]
                else:
                    link = match
                
                # Skip empty links or anchors without content
                if not link or link.startswith('#'):
                    continue
                
                # Validate the link
                result = self._validate_single_link(link, file_path, pattern_name)
                results.append(result)
        
        return results
    
    def _validate_single_link(self, link: str, source_file: Path, pattern_type: str) -> LinkValidationResult:
        """Validate a single link."""
        link = link.strip()
        
        # Determine link type
        if link.startswith('http://') or link.startswith('https://'):
            link_type = 'external'
            # For external links, we'll just check if they're well-formed
            # (actual HTTP checking would require network requests)
            parsed = urlparse(link)
            is_valid = bool(parsed.netloc and parsed.scheme)
            error_msg = None if is_valid else "Malformed URL"
            
        elif link.startswith('#'):
            link_type = 'anchor'
            # For anchor links, check if the anchor exists in the same file
            is_valid, error_msg = self._validate_anchor_link(link, source_file)
            
        else:
            link_type = 'internal'
            # For internal links, check if the file exists
            is_valid, error_msg = self._validate_internal_link(link, source_file)
        
        return LinkValidationResult(
            link=link,
            is_valid=is_valid,
            error_message=error_msg,
            link_type=link_type,
            source_file=str(source_file.relative_to(self.project_root))
        )
    
    def _validate_anchor_link(self, anchor: str, source_file: Path) -> Tuple[bool, Optional[str]]:
        """Validate an anchor link within a file."""
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Convert anchor to expected header format
            anchor_text = anchor[1:]  # Remove #
            
            # Look for headers that match this anchor
            header_patterns = [
                re.compile(rf'^#+\s+.*{re.escape(anchor_text)}.*$', re.MULTILINE | re.IGNORECASE),
                re.compile(rf'<h[1-6][^>]*id=["\']?{re.escape(anchor_text)}["\']?[^>]*>', re.IGNORECASE),
                re.compile(rf'<a[^>]*name=["\']?{re.escape(anchor_text)}["\']?[^>]*>', re.IGNORECASE)
            ]
            
            for pattern in header_patterns:
                if pattern.search(content):
                    return True, None
            
            return False, f"Anchor '{anchor}' not found in file"
            
        except Exception as e:
            return False, f"Error reading file for anchor validation: {e}"
    
    def _validate_internal_link(self, link: str, source_file: Path) -> Tuple[bool, Optional[str]]:
        """Validate an internal file link."""
        # Handle relative paths
        if link.startswith('./') or link.startswith('../') or not link.startswith('/'):
            # Relative to source file
            target_path = (source_file.parent / link).resolve()
        else:
            # Absolute path from project root
            target_path = (self.project_root / link.lstrip('/')).resolve()
        
        # Check if target exists
        if target_path.exists():
            return True, None
        else:
            return False, f"File not found: {target_path}"    

    def _check_documentation_freshness(self, doc_files: List[Path]) -> List[DocumentationIssue]:
        """Check if documentation files are up to date."""
        issues = []
        
        for doc_file in doc_files:
            try:
                # Get file modification time
                mod_time = datetime.fromtimestamp(doc_file.stat().st_mtime)
                age_days = (self.validation_time - mod_time).days
                
                # Determine criticality based on file location and name
                criticality = self._determine_file_criticality(doc_file)
                threshold = self.freshness_thresholds[criticality]
                
                if age_days > threshold:
                    severity = 'high' if criticality == 'critical' else 'medium' if criticality == 'important' else 'low'
                    
                    issues.append(DocumentationIssue(
                        file_path=str(doc_file.relative_to(self.project_root)),
                        issue_type='outdated',
                        severity=severity,
                        description=f"File is {age_days} days old (threshold: {threshold} days)",
                        suggestion=f"Review and update this {criticality} documentation"
                    ))
            
            except Exception as e:
                issues.append(DocumentationIssue(
                    file_path=str(doc_file.relative_to(self.project_root)),
                    issue_type='outdated',
                    severity='medium',
                    description=f"Could not check file age: {e}",
                    suggestion="Verify file accessibility and permissions"
                ))
        
        return issues
    
    def _determine_file_criticality(self, file_path: Path) -> str:
        """Determine how critical a documentation file is."""
        file_name = file_path.name.lower()
        path_str = str(file_path).lower()
        
        # Critical files
        critical_patterns = [
            'readme', 'getting_started', 'installation', 'setup',
            'quick_start', 'onboarding', 'api', 'deployment'
        ]
        
        for pattern in critical_patterns:
            if pattern in file_name or pattern in path_str:
                return 'critical'
        
        # Important files
        important_patterns = [
            'guide', 'tutorial', 'architecture', 'design',
            'troubleshooting', 'faq', 'configuration'
        ]
        
        for pattern in important_patterns:
            if pattern in file_name or pattern in path_str:
                return 'important'
        
        return 'normal'
    
    def _check_documentation_completeness(self, doc_files: List[Path],
                                        structure: ProjectStructure,
                                        relationships: ComponentRelationshipMap) -> List[DocumentationIssue]:
        """Check if all components have adequate documentation."""
        issues = []
        
        # Get list of documented components
        documented_components = set()
        for doc_file in doc_files:
            # Extract component names mentioned in documentation
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                
                for component in relationships.components:
                    component_name = component.name.lower()
                    if component_name in content:
                        documented_components.add(component.name)
            
            except Exception:
                continue
        
        # Check for undocumented components
        for component in relationships.components:
            if component.name not in documented_components:
                # Determine if this component needs documentation
                needs_docs = self._component_needs_documentation(component)
                
                if needs_docs:
                    severity = 'high' if component.name in relationships.critical_components else 'medium'
                    
                    issues.append(DocumentationIssue(
                        file_path=component.path,
                        issue_type='missing',
                        severity=severity,
                        description=f"Component '{component.name}' lacks documentation",
                        suggestion=f"Create documentation for this {component.component_type}"
                    ))
        
        # Check for missing essential documentation
        essential_docs = [
            'README.md', 'INSTALLATION.md', 'GETTING_STARTED.md',
            'API.md', 'DEPLOYMENT.md', 'TROUBLESHOOTING.md'
        ]
        
        existing_doc_names = {doc.name.lower() for doc in doc_files}
        
        for essential_doc in essential_docs:
            if essential_doc.lower() not in existing_doc_names:
                # Check if content exists under different names
                alt_names = self._get_alternative_doc_names(essential_doc)
                if not any(alt_name in existing_doc_names for alt_name in alt_names):
                    issues.append(DocumentationIssue(
                        file_path='/',
                        issue_type='missing',
                        severity='high',
                        description=f"Missing essential documentation: {essential_doc}",
                        suggestion=f"Create {essential_doc} with appropriate content"
                    ))
        
        return issues
    
    def _component_needs_documentation(self, component: ComponentInfo) -> bool:
        """Determine if a component needs documentation."""
        # Components with many files need documentation
        if len(component.files) > 5:
            return True
        
        # Critical components need documentation
        if component.complexity_score > 30:
            return True
        
        # Public APIs need documentation
        if 'api' in component.name.lower():
            return True
        
        # Core components need documentation
        if 'core' in component.name.lower():
            return True
        
        return False
    
    def _get_alternative_doc_names(self, doc_name: str) -> List[str]:
        """Get alternative names for a documentation file."""
        base_name = doc_name.lower().replace('.md', '')
        
        alternatives = {
            'readme': ['readme.txt', 'readme.rst', 'index.md'],
            'installation': ['install.md', 'setup.md', 'getting_started.md'],
            'getting_started': ['quickstart.md', 'quick_start.md', 'start.md'],
            'api': ['api_reference.md', 'api_docs.md', 'reference.md'],
            'deployment': ['deploy.md', 'production.md', 'hosting.md'],
            'troubleshooting': ['faq.md', 'issues.md', 'problems.md']
        }
        
        return alternatives.get(base_name, [])
    
    def _check_accessibility(self, doc_files: List[Path]) -> List[DocumentationIssue]:
        """Check documentation accessibility and searchability."""
        issues = []
        
        for doc_file in doc_files:
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_issues = []
                
                # Check for proper heading structure
                headings = re.findall(r'^(#+)\s+(.+)$', content, re.MULTILINE)
                if headings:
                    heading_levels = [len(h[0]) for h in headings]
                    
                    # Check for skipped heading levels
                    for i in range(1, len(heading_levels)):
                        if heading_levels[i] > heading_levels[i-1] + 1:
                            file_issues.append("Skipped heading levels (affects navigation)")
                
                # Check for missing table of contents in long documents
                lines = content.split('\n')
                if len(lines) > 100 and 'table of contents' not in content.lower():
                    file_issues.append("Long document missing table of contents")
                
                # Check for alt text in images
                images = re.findall(r'!\[([^\]]*)\]\([^)]+\)', content)
                empty_alt_text = sum(1 for alt_text in images if not alt_text.strip())
                if empty_alt_text > 0:
                    file_issues.append(f"{empty_alt_text} images missing alt text")
                
                # Check for code blocks without language specification
                code_blocks = re.findall(r'```(\w*)\n', content)
                unspecified_blocks = sum(1 for lang in code_blocks if not lang)
                if unspecified_blocks > 0:
                    file_issues.append(f"{unspecified_blocks} code blocks missing language specification")
                
                # Check for very long lines (readability)
                long_lines = sum(1 for line in lines if len(line) > 120)
                if long_lines > len(lines) * 0.1:  # More than 10% of lines are too long
                    file_issues.append("Many lines exceed 120 characters (affects readability)")
                
                # Create issues for each problem found
                for issue_desc in file_issues:
                    issues.append(DocumentationIssue(
                        file_path=str(doc_file.relative_to(self.project_root)),
                        issue_type='accessibility',
                        severity='low',
                        description=issue_desc,
                        suggestion="Improve document structure and accessibility"
                    ))
            
            except Exception as e:
                issues.append(DocumentationIssue(
                    file_path=str(doc_file.relative_to(self.project_root)),
                    issue_type='accessibility',
                    severity='medium',
                    description=f"Could not analyze file: {e}",
                    suggestion="Check file encoding and accessibility"
                ))
        
        return issues
    
    def _calculate_metrics(self, doc_files: List[Path], link_results: List[LinkValidationResult],
                         issues: List[DocumentationIssue], structure: Optional[ProjectStructure]) -> DocumentationMetrics:
        """Calculate documentation quality metrics."""
        
        total_links = len(link_results)
        broken_links = sum(1 for result in link_results if not result.is_valid)
        
        outdated_files = len([issue for issue in issues if issue.issue_type == 'outdated'])
        missing_components = len([issue for issue in issues if issue.issue_type == 'missing'])
        accessibility_issues = len([issue for issue in issues if issue.issue_type == 'accessibility'])
        
        # Calculate coverage percentage
        if structure:
            # Rough estimate: assume we need docs for main components
            total_components = len(structure.main_components)
            documented_components = total_components - missing_components
            coverage_percentage = (documented_components / total_components * 100) if total_components > 0 else 0
        else:
            coverage_percentage = 0
        
        # Calculate freshness score (0-100)
        if doc_files:
            fresh_files = len(doc_files) - outdated_files
            freshness_score = (fresh_files / len(doc_files) * 100)
        else:
            freshness_score = 0
        
        return DocumentationMetrics(
            total_files=len(doc_files),
            total_links=total_links,
            broken_links=broken_links,
            outdated_files=outdated_files,
            missing_components=missing_components,
            accessibility_issues=accessibility_issues,
            coverage_percentage=coverage_percentage,
            freshness_score=freshness_score
        )
    
    def _generate_recommendations(self, metrics: DocumentationMetrics,
                                issues: List[DocumentationIssue]) -> List[str]:
        """Generate recommendations for improving documentation."""
        recommendations = []
        
        # Link issues
        if metrics.broken_links > 0:
            recommendations.append(f"Fix {metrics.broken_links} broken links to improve navigation")
        
        # Freshness issues
        if metrics.freshness_score < 70:
            recommendations.append("Update outdated documentation to maintain relevance")
        
        # Coverage issues
        if metrics.coverage_percentage < 80:
            recommendations.append("Increase documentation coverage for better developer onboarding")
        
        # Missing components
        if metrics.missing_components > 0:
            recommendations.append("Document missing components, especially critical ones")
        
        # Accessibility issues
        if metrics.accessibility_issues > 0:
            recommendations.append("Improve documentation accessibility and structure")
        
        # High-severity issues
        high_severity_issues = [issue for issue in issues if issue.severity == 'high']
        if len(high_severity_issues) > 5:
            recommendations.append("Address high-severity documentation issues first")
        
        # General recommendations based on metrics
        if metrics.total_files < 5:
            recommendations.append("Consider creating more comprehensive documentation")
        
        if metrics.total_links > 50 and metrics.broken_links / metrics.total_links > 0.1:
            recommendations.append("Implement automated link checking in CI/CD pipeline")
        
        return recommendations
    
    def generate_maintenance_plan(self, report: DocumentationValidationReport) -> Dict[str, List[str]]:
        """Generate a maintenance plan based on validation results."""
        plan = {
            'immediate': [],  # High priority, do now
            'short_term': [],  # Medium priority, do within 2 weeks
            'long_term': []   # Low priority, do within 3 months
        }
        
        # Categorize issues by priority
        for issue in report.issues:
            task = f"{issue.file_path}: {issue.description}"
            
            if issue.severity == 'high':
                plan['immediate'].append(task)
            elif issue.severity == 'medium':
                plan['short_term'].append(task)
            else:
                plan['long_term'].append(task)
        
        # Add broken link fixes to immediate if many
        broken_links = [r for r in report.link_results if not r.is_valid]
        if len(broken_links) > 10:
            plan['immediate'].append(f"Fix {len(broken_links)} broken links")
        elif len(broken_links) > 0:
            plan['short_term'].append(f"Fix {len(broken_links)} broken links")
        
        return plan
    
    def save_report(self, report: DocumentationValidationReport, output_path: str):
        """Save validation report to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict for JSON serialization
        report_dict = asdict(report)
        
        # Convert datetime to string
        report_dict['validation_time'] = report.validation_time.isoformat()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Documentation validation report saved to: {output_file}")
    
    def generate_summary_report(self, report: DocumentationValidationReport) -> str:
        """Generate a human-readable summary report."""
        lines = []
        lines.append("# Documentation Validation Report")
        lines.append("")
        lines.append(f"**Validation Date:** {report.validation_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Project:** {report.project_path}")
        lines.append("")
        
        # Metrics Overview
        lines.append("## Metrics Overview")
        lines.append("")
        lines.append(f"- **Total Documentation Files:** {report.metrics.total_files}")
        lines.append(f"- **Total Links Checked:** {report.metrics.total_links}")
        lines.append(f"- **Broken Links:** {report.metrics.broken_links}")
        lines.append(f"- **Outdated Files:** {report.metrics.outdated_files}")
        lines.append(f"- **Missing Components:** {report.metrics.missing_components}")
        lines.append(f"- **Accessibility Issues:** {report.metrics.accessibility_issues}")
        lines.append(f"- **Coverage:** {report.metrics.coverage_percentage:.1f}%")
        lines.append(f"- **Freshness Score:** {report.metrics.freshness_score:.1f}%")
        lines.append("")
        
        # Issues Summary
        if report.issues:
            lines.append("## Issues Summary")
            lines.append("")
            
            # Group issues by type
            issues_by_type = {}
            for issue in report.issues:
                if issue.issue_type not in issues_by_type:
                    issues_by_type[issue.issue_type] = []
                issues_by_type[issue.issue_type].append(issue)
            
            for issue_type, issues in issues_by_type.items():
                lines.append(f"### {issue_type.replace('_', ' ').title()} Issues ({len(issues)})")
                lines.append("")
                
                # Show top 10 issues of each type
                for issue in issues[:10]:
                    severity_icon = "🔴" if issue.severity == 'high' else "🟡" if issue.severity == 'medium' else "🟢"
                    lines.append(f"- {severity_icon} **{issue.file_path}**: {issue.description}")
                    if issue.suggestion:
                        lines.append(f"  - *Suggestion: {issue.suggestion}*")
                
                if len(issues) > 10:
                    lines.append(f"- ... and {len(issues) - 10} more issues")
                
                lines.append("")
        
        # Broken Links
        broken_links = [r for r in report.link_results if not r.is_valid]
        if broken_links:
            lines.append("## Broken Links")
            lines.append("")
            
            # Group by source file
            links_by_file = {}
            for link in broken_links:
                if link.source_file not in links_by_file:
                    links_by_file[link.source_file] = []
                links_by_file[link.source_file].append(link)
            
            for source_file, links in list(links_by_file.items())[:10]:  # Top 10 files
                lines.append(f"### {source_file}")
                lines.append("")
                for link in links[:5]:  # Top 5 links per file
                    lines.append(f"- `{link.link}` ({link.link_type})")
                    if link.error_message:
                        lines.append(f"  - Error: {link.error_message}")
                
                if len(links) > 5:
                    lines.append(f"- ... and {len(links) - 5} more broken links")
                lines.append("")
        
        # Recommendations
        if report.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        
        return "\n".join(lines)
