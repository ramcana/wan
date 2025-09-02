"""
Project Structure Analyzer

Implements directory and file scanner that maps project organization
and identifies key components and their purposes.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import fnmatch


@dataclass
class FileInfo:
    """Information about a file in the project."""
    path: str
    name: str
    extension: str
    size: int
    is_executable: bool
    is_config: bool
    is_documentation: bool
    is_test: bool
    is_script: bool
    purpose: Optional[str] = None


@dataclass
class DirectoryInfo:
    """Information about a directory in the project."""
    path: str
    name: str
    file_count: int
    subdirectory_count: int
    total_size: int
    purpose: Optional[str] = None
    is_package: bool = False
    files: List[FileInfo] = None
    subdirectories: List['DirectoryInfo'] = None


@dataclass
class ProjectStructure:
    """Complete project structure analysis."""
    root_path: str
    total_files: int
    total_directories: int
    total_size: int
    main_components: List[DirectoryInfo]
    configuration_files: List[FileInfo]
    documentation_files: List[FileInfo]
    test_files: List[FileInfo]
    script_files: List[FileInfo]
    entry_points: List[FileInfo]
    ignored_patterns: List[str]


class ProjectStructureAnalyzer:
    """Analyzes project directory structure and identifies components."""
    
    # File type classifications
    CONFIG_EXTENSIONS = {'.json', '.yaml', '.yml', '.ini', '.cfg', '.conf', '.toml', '.env'}
    CONFIG_NAMES = {'config', 'configuration', 'settings', 'setup', 'pyproject', 'package', 'requirements'}
    
    DOC_EXTENSIONS = {'.md', '.rst', '.txt', '.html', '.pdf'}
    DOC_NAMES = {'readme', 'changelog', 'license', 'contributing', 'docs', 'documentation'}
    
    TEST_PATTERNS = ['test_*', '*_test', 'tests', 'testing', 'spec_*', '*_spec']
    SCRIPT_PATTERNS = ['*.bat', '*.sh', '*.ps1', 'scripts', 'bin']
    
    # Directories to ignore by default
    DEFAULT_IGNORE_PATTERNS = [
        '__pycache__',
        '.git',
        '.pytest_cache',
        'node_modules',
        '.vscode',
        '.idea',
        '*.egg-info',
        'venv',
        'env',
        '.env',
        'build',
        'dist',
        '.tox'
    ]
    
    def __init__(self, root_path: str, ignore_patterns: Optional[List[str]] = None):
        """Initialize the analyzer with project root path."""
        self.root_path = Path(root_path).resolve()
        self.ignore_patterns = ignore_patterns or self.DEFAULT_IGNORE_PATTERNS
        
    def analyze(self) -> ProjectStructure:
        """Perform complete project structure analysis."""
        print(f"Analyzing project structure at: {self.root_path}")
        
        # Scan the entire project
        all_files = []
        all_directories = []
        
        for root, dirs, files in os.walk(self.root_path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not self._should_ignore(d, root)]
            
            current_dir = Path(root)
            relative_path = current_dir.relative_to(self.root_path)
            
            # Analyze directory
            if not self._should_ignore(current_dir.name, str(current_dir.parent)):
                dir_info = self._analyze_directory(current_dir, files)
                all_directories.append(dir_info)
            
            # Analyze files
            for file in files:
                file_path = current_dir / file
                if not self._should_ignore(file, str(current_dir)):
                    file_info = self._analyze_file(file_path)
                    all_files.append(file_info)
        
        # Categorize files
        config_files = [f for f in all_files if f.is_config]
        doc_files = [f for f in all_files if f.is_documentation]
        test_files = [f for f in all_files if f.is_test]
        script_files = [f for f in all_files if f.is_script]
        entry_points = self._identify_entry_points(all_files)
        
        # Identify main components
        main_components = self._identify_main_components(all_directories)
        
        # Calculate totals
        total_size = sum(f.size for f in all_files)
        
        return ProjectStructure(
            root_path=str(self.root_path),
            total_files=len(all_files),
            total_directories=len(all_directories),
            total_size=total_size,
            main_components=main_components,
            configuration_files=config_files,
            documentation_files=doc_files,
            test_files=test_files,
            script_files=script_files,
            entry_points=entry_points,
            ignored_patterns=self.ignore_patterns
        )
    
    def _should_ignore(self, name: str, parent_path: str) -> bool:
        """Check if a file or directory should be ignored."""
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False
    
    def _analyze_file(self, file_path: Path) -> FileInfo:
        """Analyze a single file and determine its characteristics."""
        try:
            stat = file_path.stat()
            size = stat.st_size
            is_executable = os.access(file_path, os.X_OK)
        except (OSError, PermissionError):
            size = 0
            is_executable = False
        
        name = file_path.name
        extension = file_path.suffix.lower()
        relative_path = str(file_path.relative_to(self.root_path))
        
        # Classify file type
        is_config = self._is_config_file(name, extension)
        is_documentation = self._is_documentation_file(name, extension)
        is_test = self._is_test_file(name, relative_path)
        is_script = self._is_script_file(name, extension, relative_path)
        
        # Determine purpose
        purpose = self._determine_file_purpose(name, extension, relative_path)
        
        return FileInfo(
            path=relative_path,
            name=name,
            extension=extension,
            size=size,
            is_executable=is_executable,
            is_config=is_config,
            is_documentation=is_documentation,
            is_test=is_test,
            is_script=is_script,
            purpose=purpose
        )
    
    def _analyze_directory(self, dir_path: Path, files: List[str]) -> DirectoryInfo:
        """Analyze a directory and determine its characteristics."""
        name = dir_path.name
        relative_path = str(dir_path.relative_to(self.root_path))
        
        # Count subdirectories
        try:
            subdirs = [d for d in dir_path.iterdir() if d.is_dir() and not self._should_ignore(d.name, str(dir_path))]
            subdirectory_count = len(subdirs)
        except (OSError, PermissionError):
            subdirectory_count = 0
        
        # Calculate total size
        total_size = 0
        for file in files:
            try:
                file_path = dir_path / file
                total_size += file_path.stat().st_size
            except (OSError, PermissionError):
                continue
        
        # Check if it's a Python package
        is_package = (dir_path / '__init__.py').exists()
        
        # Determine purpose
        purpose = self._determine_directory_purpose(name, relative_path, files, is_package)
        
        return DirectoryInfo(
            path=relative_path,
            name=name,
            file_count=len(files),
            subdirectory_count=subdirectory_count,
            total_size=total_size,
            purpose=purpose,
            is_package=is_package
        )
    
    def _is_config_file(self, name: str, extension: str) -> bool:
        """Determine if a file is a configuration file."""
        name_lower = name.lower()
        
        # Check extension
        if extension in self.CONFIG_EXTENSIONS:
            return True
        
        # Check name patterns
        for config_name in self.CONFIG_NAMES:
            if config_name in name_lower:
                return True
        
        return False
    
    def _is_documentation_file(self, name: str, extension: str) -> bool:
        """Determine if a file is documentation."""
        name_lower = name.lower()
        
        # Check extension
        if extension in self.DOC_EXTENSIONS:
            return True
        
        # Check name patterns
        for doc_name in self.DOC_NAMES:
            if doc_name in name_lower:
                return True
        
        return False
    
    def _is_test_file(self, name: str, path: str) -> bool:
        """Determine if a file is a test file."""
        name_lower = name.lower()
        path_lower = path.lower()
        
        # Check patterns
        for pattern in self.TEST_PATTERNS:
            if fnmatch.fnmatch(name_lower, pattern.lower()):
                return True
        
        # Check if in test directory
        return 'test' in path_lower.split(os.sep)
    
    def _is_script_file(self, name: str, extension: str, path: str) -> bool:
        """Determine if a file is a script."""
        # Check script extensions
        script_extensions = {'.bat', '.sh', '.ps1', '.py'}
        if extension in script_extensions:
            # Python files are scripts if they're in scripts directory or executable
            if extension == '.py':
                return 'script' in path.lower() or name.lower().startswith('run_') or name.lower().startswith('start_')
            return True
        
        return False
    
    def _determine_file_purpose(self, name: str, extension: str, path: str) -> Optional[str]:
        """Determine the purpose of a file based on its characteristics."""
        name_lower = name.lower()
        path_lower = path.lower()
        
        # Entry points
        if name_lower in ['main.py', 'app.py', 'start.py', '__main__.py']:
            return 'Application Entry Point'
        
        # Configuration
        if 'config' in name_lower:
            return 'Configuration File'
        
        # Documentation
        if name_lower.startswith('readme'):
            return 'Project Documentation'
        elif name_lower in ['changelog.md', 'changelog.txt']:
            return 'Change Log'
        elif name_lower in ['license', 'license.txt', 'license.md']:
            return 'License File'
        
        # Build/Deploy
        if name_lower in ['dockerfile', 'docker-compose.yml', 'docker-compose.yaml']:
            return 'Container Configuration'
        elif name_lower in ['requirements.txt', 'pyproject.toml', 'setup.py', 'package.json']:
            return 'Dependency Management'
        
        # CI/CD
        if '.github' in path_lower or '.gitlab' in path_lower:
            return 'CI/CD Configuration'
        
        return None
    
    def _determine_directory_purpose(self, name: str, path: str, files: List[str], is_package: bool) -> Optional[str]:
        """Determine the purpose of a directory."""
        name_lower = name.lower()
        
        # Common directory purposes
        purposes = {
            'backend': 'Backend Application Code',
            'frontend': 'Frontend Application Code',
            'api': 'API Implementation',
            'core': 'Core Application Logic',
            'models': 'Data Models',
            'services': 'Business Logic Services',
            'utils': 'Utility Functions',
            'helpers': 'Helper Functions',
            'config': 'Configuration Files',
            'configs': 'Configuration Files',
            'docs': 'Documentation',
            'documentation': 'Documentation',
            'tests': 'Test Suite',
            'test': 'Test Files',
            'scripts': 'Automation Scripts',
            'tools': 'Development Tools',
            'migrations': 'Database Migrations',
            'static': 'Static Assets',
            'templates': 'Template Files',
            'logs': 'Log Files',
            'outputs': 'Generated Output Files',
            'cache': 'Cache Files',
            'temp': 'Temporary Files',
            'backup': 'Backup Files',
            'backups': 'Backup Files',
            'examples': 'Example Code',
            'demo': 'Demo Code',
            'local_installation': 'Local Installation Package',
            'local_testing_framework': 'Testing Framework',
            'infrastructure': 'Infrastructure Code'
        }
        
        if name_lower in purposes:
            return purposes[name_lower]
        
        # Check for Python packages
        if is_package:
            return 'Python Package'
        
        # Check file contents for clues
        if any('test' in f.lower() for f in files):
            return 'Test Directory'
        elif any(f.endswith('.md') for f in files):
            return 'Documentation Directory'
        elif any(f.endswith(('.json', '.yaml', '.yml', '.ini')) for f in files):
            return 'Configuration Directory'
        
        return None
    
    def _identify_entry_points(self, files: List[FileInfo]) -> List[FileInfo]:
        """Identify potential application entry points."""
        entry_points = []
        
        entry_point_names = {
            'main.py', 'app.py', 'start.py', '__main__.py',
            'run.py', 'server.py', 'manage.py', 'cli.py'
        }
        
        for file in files:
            if file.name.lower() in entry_point_names:
                entry_points.append(file)
            elif file.is_executable and file.extension == '.py':
                entry_points.append(file)
        
        return entry_points
    
    def _identify_main_components(self, directories: List[DirectoryInfo]) -> List[DirectoryInfo]:
        """Identify the main components of the project."""
        # Sort by importance (size, file count, purpose)
        def component_score(dir_info: DirectoryInfo) -> int:
            score = 0
            
            # Size and file count
            score += dir_info.file_count * 10
            score += dir_info.subdirectory_count * 5
            score += min(dir_info.total_size // 1000, 100)  # Cap size contribution
            
            # Purpose importance
            important_purposes = {
                'Backend Application Code': 100,
                'Frontend Application Code': 100,
                'Core Application Logic': 90,
                'API Implementation': 80,
                'Test Suite': 70,
                'Documentation': 60,
                'Development Tools': 50
            }
            
            if dir_info.purpose in important_purposes:
                score += important_purposes[dir_info.purpose]
            
            # Root level directories are more important
            if '/' not in dir_info.path and '\\' not in dir_info.path:
                score += 50
            
            return score
        
        # Filter and sort directories
        main_dirs = [d for d in directories if d.file_count > 0 or d.subdirectory_count > 0]
        main_dirs.sort(key=component_score, reverse=True)
        
        # Return top components (limit to reasonable number)
        return main_dirs[:20]
    
    def save_analysis(self, analysis: ProjectStructure, output_path: str) -> None:
        """Save analysis results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict for JSON serialization
        analysis_dict = asdict(analysis)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis saved to: {output_file}")
    
    def generate_summary_report(self, analysis: ProjectStructure) -> str:
        """Generate a human-readable summary report."""
        report = []
        report.append("# Project Structure Analysis Report")
        report.append("")
        report.append(f"**Project Root:** {analysis.root_path}")
        report.append(f"**Total Files:** {analysis.total_files:,}")
        report.append(f"**Total Directories:** {analysis.total_directories:,}")
        report.append(f"**Total Size:** {analysis.total_size / (1024*1024):.1f} MB")
        report.append("")
        
        # Main Components
        report.append("## Main Components")
        report.append("")
        for component in analysis.main_components[:10]:  # Top 10
            size_mb = component.total_size / (1024*1024) if component.total_size > 0 else 0
            report.append(f"- **{component.name}** ({component.path})")
            report.append(f"  - Purpose: {component.purpose or 'Unknown'}")
            report.append(f"  - Files: {component.file_count}, Subdirs: {component.subdirectory_count}")
            report.append(f"  - Size: {size_mb:.1f} MB")
            if component.is_package:
                report.append(f"  - Python Package: Yes")
            report.append("")
        
        # File Categories
        report.append("## File Categories")
        report.append("")
        report.append(f"- **Configuration Files:** {len(analysis.configuration_files)}")
        report.append(f"- **Documentation Files:** {len(analysis.documentation_files)}")
        report.append(f"- **Test Files:** {len(analysis.test_files)}")
        report.append(f"- **Script Files:** {len(analysis.script_files)}")
        report.append(f"- **Entry Points:** {len(analysis.entry_points)}")
        report.append("")
        
        # Entry Points
        if analysis.entry_points:
            report.append("## Application Entry Points")
            report.append("")
            for entry in analysis.entry_points:
                report.append(f"- **{entry.name}** ({entry.path})")
                if entry.purpose:
                    report.append(f"  - {entry.purpose}")
            report.append("")
        
        return "\n".join(report)