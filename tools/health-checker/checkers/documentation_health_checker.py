"""
Documentation health checker
"""

import re
import requests
from pathlib import Path
from typing import List, Dict, Any, Set
import logging
from urllib.parse import urljoin, urlparse

try:
    from ..health_models import ComponentHealth, HealthIssue, HealthCategory, Severity, HealthConfig
except ImportError:
    # Fallback imports - these will be defined in health_checker.py
    pass


class DocumentationHealthChecker:
    """Checks the health of project documentation"""
    
    def __init__(self, config: HealthConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def check_health(self) -> ComponentHealth:
        """Check documentation health"""
        issues = []
        metrics = {}
        
        # Check if docs directory exists
        if not self.config.docs_directory.exists():
            issues.append(HealthIssue(
                severity=Severity.HIGH,
                category=HealthCategory.DOCUMENTATION,
                title="Documentation Directory Missing",
                description=f"Documentation directory {self.config.docs_directory} does not exist",
                affected_components=["documentation"],
                remediation_steps=[
                    f"Create documentation directory: mkdir {self.config.docs_directory}",
                    "Add basic documentation structure",
                    "Create index.md file"
                ]
            ))
            return ComponentHealth(
                component_name="documentation",
                category=HealthCategory.DOCUMENTATION,
                score=20.0,  # Some points for having scattered docs
                status="critical",
                issues=issues,
                metrics=metrics
            )
        
        # Discover documentation files
        doc_files = self._discover_documentation()
        metrics["total_doc_files"] = len(doc_files)
        
        if len(doc_files) == 0:
            issues.append(HealthIssue(
                severity=Severity.HIGH,
                category=HealthCategory.DOCUMENTATION,
                title="No Documentation Files Found",
                description="No documentation files were found",
                affected_components=["documentation"],
                remediation_steps=[
                    "Create basic documentation files",
                    "Add README.md files",
                    "Document API endpoints"
                ]
            ))
        
        # Check for essential documentation
        essential_docs = self._check_essential_documentation()
        metrics.update(essential_docs)
        
        missing_essential = essential_docs.get("missing_essential", [])
        if missing_essential:
            issues.append(HealthIssue(
                severity=Severity.MEDIUM,
                category=HealthCategory.DOCUMENTATION,
                title="Missing Essential Documentation",
                description=f"Missing: {', '.join(missing_essential)}",
                affected_components=["documentation"],
                remediation_steps=[
                    f"Create {doc}" for doc in missing_essential
                ]
            ))
        
        # Check for broken links
        broken_links = self._check_broken_links(doc_files)
        metrics["broken_links"] = len(broken_links)
        
        if broken_links:
            issues.append(HealthIssue(
                severity=Severity.MEDIUM,
                category=HealthCategory.DOCUMENTATION,
                title="Broken Links Found",
                description=f"Found {len(broken_links)} broken links in documentation",
                affected_components=["documentation"],
                remediation_steps=[
                    "Fix or remove broken links",
                    "Update outdated URLs",
                    "Use relative links where possible"
                ],
                metadata={"broken_links": broken_links[:10]}  # Show first 10
            ))
        
        # Check documentation freshness
        outdated_docs = self._check_documentation_freshness(doc_files)
        metrics["outdated_docs"] = len(outdated_docs)
        
        if outdated_docs:
            issues.append(HealthIssue(
                severity=Severity.LOW,
                category=HealthCategory.DOCUMENTATION,
                title="Outdated Documentation",
                description=f"Found {len(outdated_docs)} potentially outdated documents",
                affected_components=["documentation"],
                remediation_steps=[
                    "Review and update outdated documentation",
                    "Add last-updated dates to documents",
                    "Set up documentation review schedule"
                ]
            ))
        
        # Check for scattered documentation
        scattered_docs = self._check_scattered_documentation()
        metrics["scattered_docs"] = len(scattered_docs)
        
        if scattered_docs:
            issues.append(HealthIssue(
                severity=Severity.MEDIUM,
                category=HealthCategory.DOCUMENTATION,
                title="Scattered Documentation",
                description=f"Found {len(scattered_docs)} documentation files outside docs directory",
                affected_components=["documentation"],
                remediation_steps=[
                    "Move scattered documentation to docs directory",
                    "Create proper documentation structure",
                    "Update links and references"
                ],
                metadata={"scattered_files": scattered_docs[:10]}
            ))
        
        # Calculate score
        score = self._calculate_documentation_score(metrics, issues)
        status = self._determine_status(score)
        
        return ComponentHealth(
            component_name="documentation",
            category=HealthCategory.DOCUMENTATION,
            score=score,
            status=status,
            issues=issues,
            metrics=metrics
        )
    
    def _discover_documentation(self) -> List[Path]:
        """Discover documentation files"""
        doc_files = []
        
        try:
            # Common documentation file patterns
            patterns = ["*.md", "*.rst", "*.txt"]
            
            for pattern in patterns:
                doc_files.extend(list(self.config.docs_directory.rglob(pattern)))
        except Exception as e:
            self.logger.warning(f"Failed to discover documentation: {e}")
        
        return doc_files
    
    def _check_essential_documentation(self) -> Dict[str, Any]:
        """Check for essential documentation files"""
        essential_files = [
            "README.md",
            "index.md", 
            "installation.md",
            "user-guide/index.md",
            "developer-guide/index.md",
            "api/index.md"
        ]
        
        existing = []
        missing = []
        
        for file_path in essential_files:
            full_path = self.config.docs_directory / file_path
            if full_path.exists():
                existing.append(file_path)
            else:
                # Also check in project root for README
                if file_path == "README.md" and (self.config.project_root / "README.md").exists():
                    existing.append("README.md (in root)")
                else:
                    missing.append(file_path)
        
        return {
            "essential_existing": existing,
            "missing_essential": missing,
            "essential_coverage": len(existing) / len(essential_files) * 100
        }
    
    def _check_broken_links(self, doc_files: List[Path]) -> List[Dict[str, str]]:
        """Check for broken links in documentation"""
        broken_links = []
        
        # URL pattern to find links
        url_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        
        for doc_file in doc_files:
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find all markdown links
                links = url_pattern.findall(content)
                
                for link_text, url in links:
                    if self._is_broken_link(url, doc_file):
                        broken_links.append({
                            "file": str(doc_file.relative_to(self.config.project_root)),
                            "link_text": link_text,
                            "url": url
                        })
            except Exception as e:
                self.logger.warning(f"Failed to check links in {doc_file}: {e}")
        
        return broken_links
    
    def _is_broken_link(self, url: str, source_file: Path) -> bool:
        """Check if a link is broken"""
        try:
            # Skip certain URLs that are commonly valid
            if url.startswith(('mailto:', 'tel:', '#')):
                return False
            
            # Check relative file links
            if not url.startswith(('http://', 'https://')):
                # Relative link - check if file exists
                if url.startswith('/'):
                    # Absolute path from project root
                    target_path = self.config.project_root / url.lstrip('/')
                else:
                    # Relative to current file
                    target_path = source_file.parent / url
                
                # Remove anchor fragments
                if '#' in str(target_path):
                    target_path = Path(str(target_path).split('#')[0])
                
                return not target_path.exists()
            
            # Check HTTP links (with timeout and limited checks)
            try:
                response = requests.head(url, timeout=5, allow_redirects=True)
                return response.status_code >= 400
            except:
                # If we can't check, assume it's working to avoid false positives
                return False
                
        except Exception:
            return False
    
    def _check_documentation_freshness(self, doc_files: List[Path]) -> List[str]:
        """Check for potentially outdated documentation"""
        outdated_docs = []
        
        # Simple heuristic: files not modified in 6 months
        import time
        six_months_ago = time.time() - (6 * 30 * 24 * 60 * 60)
        
        for doc_file in doc_files:
            try:
                if doc_file.stat().st_mtime < six_months_ago:
                    outdated_docs.append(str(doc_file.relative_to(self.config.project_root)))
            except Exception:
                continue
        
        return outdated_docs
    
    def _check_scattered_documentation(self) -> List[str]:
        """Check for documentation files outside the docs directory"""
        scattered_docs = []
        
        try:
            # Look for documentation files in the project root and other directories
            patterns = ["*.md", "*.rst", "*.txt"]
            
            for pattern in patterns:
                for doc_file in self.config.project_root.rglob(pattern):
                    # Skip files in docs directory, hidden directories, and common non-doc files
                    relative_path = doc_file.relative_to(self.config.project_root)
                    
                    if (not str(relative_path).startswith(str(self.config.docs_directory)) and
                        not any(part.startswith('.') for part in relative_path.parts) and
                        not any(part in ['node_modules', '__pycache__', 'venv'] for part in relative_path.parts) and
                        doc_file.name not in ['requirements.txt', 'package.json']):
                        
                        scattered_docs.append(str(relative_path))
        except Exception as e:
            self.logger.warning(f"Failed to check scattered documentation: {e}")
        
        return scattered_docs
    
    def _calculate_documentation_score(self, metrics: Dict[str, Any], issues: List[HealthIssue]) -> float:
        """Calculate documentation health score"""
        base_score = 100.0
        
        # Deduct points for issues
        for issue in issues:
            if issue.severity == Severity.CRITICAL:
                base_score -= 25
            elif issue.severity == Severity.HIGH:
                base_score -= 20
            elif issue.severity == Severity.MEDIUM:
                base_score -= 15
            elif issue.severity == Severity.LOW:
                base_score -= 5
        
        # Bonus points for good metrics
        essential_coverage = metrics.get("essential_coverage", 0)
        if essential_coverage >= 80:
            base_score += 10
        elif essential_coverage >= 60:
            base_score += 5
        
        # Penalty for scattered docs
        scattered_count = metrics.get("scattered_docs", 0)
        if scattered_count > 10:
            base_score -= 15
        elif scattered_count > 5:
            base_score -= 10
        
        return max(0.0, min(100.0, base_score))
    
    def _determine_status(self, score: float) -> str:
        """Determine health status from score"""
        if score >= self.config.warning_threshold:
            return "healthy"
        elif score >= self.config.critical_threshold:
            return "warning"
        else:
            return "critical"