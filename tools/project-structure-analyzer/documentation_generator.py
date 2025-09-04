import pytest
"""
Comprehensive Documentation Generator

Creates automated documentation for project structure and components,
including component relationships, Local Testing Framework documentation,
and developer onboarding guides.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from structure_analyzer import ProjectStructure, DirectoryInfo, FileInfo
from component_analyzer import ComponentRelationshipMap, ComponentInfo, ComponentDependency
from complexity_analyzer import ProjectComplexityReport, ComponentComplexity


@dataclass
class DocumentationSection:
    """Represents a section of documentation."""
    title: str
    content: str
    subsections: List['DocumentationSection'] = None
    level: int = 1


@dataclass
class DocumentationTemplate:
    """Template for generating documentation."""
    name: str
    sections: List[DocumentationSection]
    metadata: Dict[str, str]


class DocumentationGenerator:
    """Generates comprehensive project documentation."""
    
    def __init__(self, project_root: str):
        """Initialize the documentation generator."""
        self.project_root = Path(project_root).resolve()
        self.generation_time = datetime.now()
        
    def generate_project_overview(self, structure: ProjectStructure,
                                relationships: Optional[ComponentRelationshipMap] = None,
                                complexity: Optional[ProjectComplexityReport] = None) -> str:
        """Generate comprehensive project overview documentation."""
        
        sections = []
        
        # Header
        sections.append(self._create_header_section(structure))
        
        # Executive Summary
        sections.append(self._create_executive_summary(structure, relationships, complexity))
        
        # Project Structure
        sections.append(self._create_structure_section(structure))
        
        # Component Overview
        sections.append(self._create_component_overview(structure, relationships))
        
        # Architecture Overview
        if relationships:
            sections.append(self._create_architecture_section(relationships))
        
        # Getting Started
        sections.append(self._create_getting_started_section(structure))
        
        # Development Guide
        sections.append(self._create_development_guide(structure, complexity))
        
        return self._render_sections(sections)
    
    def generate_component_documentation(self, component: ComponentInfo,
                                       structure: ProjectStructure,
                                       relationships: ComponentRelationshipMap) -> str:
        """Generate detailed documentation for a specific component."""
        
        sections = []
        
        # Component Header
        sections.append(DocumentationSection(
            title=f"# {component.name}",
            content=f"**Type:** {component.component_type}\n**Path:** `{component.path}`\n\n" +
                   (f"**Purpose:** {component.purpose}\n\n" if component.purpose else ""),
            level=1
        ))
        
        # Overview
        overview_content = []
        overview_content.append(f"This component contains {len(component.files)} files.")
        
        if component.dependencies:
            overview_content.append(f"It depends on {len(component.dependencies)} other components.")
        
        if component.dependents:
            overview_content.append(f"It is used by {len(component.dependents)} other components.")
        
        sections.append(DocumentationSection(
            title="## Overview",
            content="\n".join(overview_content),
            level=2
        ))
        
        # Files
        if component.files:
            file_list = []
            for file_path in component.files[:20]:  # Limit to first 20 files
                file_info = self._find_file_info(file_path, structure)
                if file_info:
                    purpose_text = f" - {file_info.purpose}" if file_info.purpose else ""
                    file_list.append(f"- `{file_info.name}`{purpose_text}")
                else:
                    file_list.append(f"- `{Path(file_path).name}`")
            
            if len(component.files) > 20:
                file_list.append(f"- ... and {len(component.files) - 20} more files")
            
            sections.append(DocumentationSection(
                title="## Files",
                content="\n".join(file_list),
                level=2
            ))
        
        # Dependencies
        if component.dependencies:
            dep_content = self._create_dependency_documentation(component.dependencies, "Dependencies")
            sections.append(DocumentationSection(
                title="## Dependencies",
                content=dep_content,
                level=2
            ))
        
        # Dependents
        if component.dependents:
            dep_content = self._create_dependency_documentation(component.dependents, "Used By")
            sections.append(DocumentationSection(
                title="## Used By",
                content=dep_content,
                level=2
            ))
        
        return self._render_sections(sections)
    
    def generate_local_testing_framework_docs(self, structure: ProjectStructure,
                                            relationships: ComponentRelationshipMap) -> str:
        """Generate specific documentation for the Local Testing Framework."""
        
        # Find the local testing framework component
        ltf_component = None
        for component in relationships.components:
            if 'local_testing_framework' in component.name.lower():
                ltf_component = component
                break
        
        sections = []
        
        # Header
        sections.append(DocumentationSection(
            title="# Local Testing Framework",
            content="The Local Testing Framework is a specialized testing system designed to validate " +
                   "the WAN22 project's functionality in local development environments.",
            level=1
        ))
        
        # Purpose and Scope
        purpose_content = [
            "## Purpose",
            "",
            "The Local Testing Framework serves several key purposes:",
            "",
            "1. **Local Validation**: Tests the complete system in a local development environment",
            "2. **Integration Testing**: Validates interactions between different components",
            "3. **Performance Testing**: Measures system performance under various conditions",
            "4. **Edge Case Testing**: Tests unusual or boundary conditions",
            "5. **Regression Testing**: Ensures new changes don't break existing functionality",
            "",
            "## Relationship to Main Application",
            "",
            "The Local Testing Framework is **separate from but complementary to** the main WAN22 application:",
            "",
            "- **Main Application** (`backend/`, `frontend/`, `core/`): The production video generation system",
            "- **Local Testing Framework** (`local_testing_framework/`): Testing and validation tools",
            "",
            "The framework **tests** the main application but is **not part** of the production deployment."
        ]
        
        sections.append(DocumentationSection(
            title="",
            content="\n".join(purpose_content),
            level=2
        ))
        
        # Architecture
        if ltf_component:
            arch_content = [
                "## Architecture",
                "",
                f"The framework is organized as a Python package with {len(ltf_component.files)} files:",
                ""
            ]
            
            # Find framework structure
            ltf_path = self.project_root / "local_testing_framework"
            if ltf_path.exists():
                for item in ltf_path.iterdir():
                    if item.is_file() and item.suffix == '.py':
                        arch_content.append(f"- `{item.name}`: {self._guess_file_purpose(item.name)}")
                    elif item.is_dir() and not item.name.startswith('.'):
                        arch_content.append(f"- `{item.name}/`: {self._guess_directory_purpose(item.name)}")
            
            sections.append(DocumentationSection(
                title="",
                content="\n".join(arch_content),
                level=2
            ))
        
        # Usage
        usage_content = [
            "## Usage",
            "",
            "### Running Tests",
            "",
            "```bash",
            "# Run all tests",
            "python -m local_testing_framework",
            "",
            "# Run specific test type",
            "python -m local_testing_framework --integration",
            "python -m local_testing_framework --performance",
            "```",
            "",
            "### Configuration",
            "",
            "The framework uses configuration files in `local_testing_framework/config_templates/` to define:",
            "",
            "- Test parameters and thresholds",
            "- Model configurations for testing",
            "- Environment-specific settings",
            "",
            "### Test Samples",
            "",
            "Test data is stored in `local_testing_framework/test_samples/` and includes:",
            "",
            "- Sample input prompts",
            "- Expected output formats",
            "- Edge case scenarios"
        ]
        
        sections.append(DocumentationSection(
            title="",
            content="\n".join(usage_content),
            level=2
        ))
        
        # Integration with Main System
        integration_content = [
            "## Integration with Main System",
            "",
            "The Local Testing Framework integrates with the main application through:",
            "",
            "### API Testing",
            "- Tests backend API endpoints",
            "- Validates request/response formats",
            "- Checks error handling",
            "",
            "### Model Testing", 
            "- Loads and tests AI models",
            "- Validates generation quality",
            "- Tests model switching",
            "",
            "### Configuration Testing",
            "- Validates configuration files",
            "- Tests environment setup",
            "- Checks dependency resolution",
            "",
            "### Performance Monitoring",
            "- Measures generation times",
            "- Monitors resource usage",
            "- Tracks system health"
        ]
        
        sections.append(DocumentationSection(
            title="",
            content="\n".join(integration_content),
            level=2
        ))
        
        return self._render_sections(sections) 
   
    def generate_developer_onboarding_guide(self, structure: ProjectStructure,
                                           relationships: Optional[ComponentRelationshipMap] = None,
                                           complexity: Optional[ProjectComplexityReport] = None) -> str:
        """Generate step-by-step developer onboarding guide."""
        
        sections = []
        
        # Header
        sections.append(DocumentationSection(
            title="# Developer Onboarding Guide",
            content="Welcome to the WAN22 project! This guide will help you understand the project " +
                   "structure and get started with development in 30 minutes or less.",
            level=1
        ))
        
        # Quick Start (5 minutes)
        quick_start = [
            "## 🚀 Quick Start (5 minutes)",
            "",
            "### 1. Clone and Setup",
            "```bash",
            "git clone <repository-url>",
            "cd wan22",
            "python -m venv venv",
            "source venv/bin/activate  # On Windows: venv\\Scripts\\activate",
            "pip install -r requirements.txt",
            "```",
            "",
            "### 2. Verify Installation",
            "```bash",
            "python start.py --help",
            "python -m local_testing_framework --quick-test",
            "```",
            "",
            "### 3. Run Your First Test",
            "```bash",
            "python backend/test_real_ai_ready.py",
            "```"
        ]
        
        sections.append(DocumentationSection(
            title="",
            content="\n".join(quick_start),
            level=2
        ))
        
        # Project Structure Overview (10 minutes)
        structure_overview = [
            "## 📁 Project Structure Overview (10 minutes)",
            "",
            "The project is organized into several main areas:",
            ""
        ]
        
        # Add main components with explanations
        for component in structure.main_components[:8]:  # Top 8 components
            icon = self._get_component_icon(component.purpose)
            structure_overview.append(f"### {icon} `{component.name}/`")
            structure_overview.append("")
            if component.purpose:
                structure_overview.append(f"**Purpose:** {component.purpose}")
            structure_overview.append(f"**Files:** {component.file_count}")
            
            # Add specific guidance for key components
            if 'backend' in component.name.lower():
                structure_overview.extend([
                    "",
                    "**Key areas:**",
                    "- `api/` - REST API endpoints",
                    "- `core/` - Business logic",
                    "- `services/` - Service layer",
                    "- `models/` - Data models"
                ])
            elif 'frontend' in component.name.lower():
                structure_overview.extend([
                    "",
                    "**Key areas:**",
                    "- `src/` - React/TypeScript source",
                    "- `public/` - Static assets",
                    "- UI components and styling"
                ])
            elif 'local_testing_framework' in component.name.lower():
                structure_overview.extend([
                    "",
                    "**This is separate from the main app** - it's for testing the system locally.",
                    "Use this to validate your changes before committing."
                ])
            elif 'tests' in component.name.lower():
                structure_overview.extend([
                    "",
                    "**Test types:**",
                    "- `unit/` - Unit tests",
                    "- `integration/` - Integration tests",
                    "- `e2e/` - End-to-end tests"
                ])
            
            structure_overview.append("")
        
        sections.append(DocumentationSection(
            title="",
            content="\n".join(structure_overview),
            level=2
        ))
        
        # Key Concepts (10 minutes)
        key_concepts = [
            "## 🧠 Key Concepts (10 minutes)",
            "",
            "### WAN22 System",
            "WAN22 is an AI-powered video generation system that:",
            "- Converts text prompts to videos (T2V)",
            "- Converts images to videos (I2V)", 
            "- Handles text+image to video (TI2V)",
            "",
            "### Architecture Pattern",
            "The system follows a **layered architecture**:",
            "",
            "```",
            "Frontend (React/TypeScript)",
            "    ↓",
            "Backend API (FastAPI/Python)",
            "    ↓", 
            "Core Services (AI Models)",
            "    ↓",
            "Infrastructure (GPU/Storage)",
            "```",
            "",
            "### Key Technologies",
            "- **Backend**: Python, FastAPI, PyTorch",
            "- **Frontend**: React, TypeScript, Vite",
            "- **AI Models**: Diffusion models, Transformers",
            "- **Infrastructure**: CUDA, Docker (optional)"
        ]
        
        if relationships and relationships.critical_components:
            key_concepts.extend([
                "",
                "### Critical Components",
                "These components are used by many others - be careful when modifying:",
                ""
            ])
            for comp in relationships.critical_components[:5]:
                key_concepts.append(f"- `{comp}`")
        
        sections.append(DocumentationSection(
            title="",
            content="\n".join(key_concepts),
            level=2
        ))
        
        # Development Workflow (5 minutes)
        workflow = [
            "## 🔄 Development Workflow (5 minutes)",
            "",
            "### Making Changes",
            "1. **Create a branch**: `git checkout -b feature/your-feature`",
            "2. **Make your changes** in the appropriate component",
            "3. **Test locally**: `python -m local_testing_framework`",
            "4. **Run unit tests**: `pytest tests/`",
            "5. **Check code quality**: `python tools/health-checker/cli.py`",
            "6. **Commit and push**: Standard git workflow",
            "",
            "### Testing Strategy",
            "- **Unit tests**: Test individual functions/classes",
            "- **Integration tests**: Test component interactions", 
            "- **Local Testing Framework**: Test the complete system",
            "- **Manual testing**: Use the UI for end-to-end validation",
            "",
            "### Common Tasks",
            "",
            "**Adding a new API endpoint:**",
            "1. Add route in `backend/api/`",
            "2. Add business logic in `backend/core/` or `backend/services/`",
            "3. Add tests in `backend/tests/`",
            "4. Update frontend if needed",
            "",
            "**Modifying AI models:**",
            "1. Update model code in `backend/core/`",
            "2. Test with Local Testing Framework",
            "3. Update configuration if needed",
            "4. Validate performance impact"
        ]
        
        sections.append(DocumentationSection(
            title="",
            content="\n".join(workflow),
            level=2
        ))
        
        # Troubleshooting
        troubleshooting = [
            "## 🔧 Troubleshooting",
            "",
            "### Common Issues",
            "",
            "**Import errors:**",
            "- Check virtual environment is activated",
            "- Run `pip install -r requirements.txt`",
            "- Check Python path configuration",
            "",
            "**Model loading errors:**",
            "- Ensure models are downloaded: `python backend/scripts/download_models.py`",
            "- Check GPU availability: `python backend/test_cuda_detection.py`",
            "- Verify disk space for model storage",
            "",
            "**Test failures:**",
            "- Run tests individually to isolate issues",
            "- Check test configuration in `tests/config/`",
            "- Review test logs in `test_logs/`",
            "",
            "### Getting Help",
            "- Check existing documentation in `docs/`",
            "- Review similar issues in git history",
            "- Run diagnostic tools: `python backend/diagnose_system.py`",
            "- Use the health checker: `python tools/health-checker/cli.py`"
        ]
        
        sections.append(DocumentationSection(
            title="",
            content="\n".join(troubleshooting),
            level=2
        ))
        
        # Next Steps
        next_steps = [
            "## 🎯 Next Steps",
            "",
            "Now that you understand the basics:",
            "",
            "1. **Explore the codebase**: Start with the component most relevant to your work",
            "2. **Read detailed docs**: Check `docs/` for specific guides",
            "3. **Join development**: Pick up a task from the issue tracker",
            "4. **Ask questions**: Don't hesitate to ask team members",
            "",
            "### Recommended Reading Order",
            "1. This onboarding guide (you're here!)",
            "2. `docs/SYSTEM_REQUIREMENTS.md` - System setup",
            "3. `docs/USER_GUIDE.md` - How to use the system",
            "4. Component-specific documentation in each directory",
            "",
            "### Development Resources",
            "- **API Documentation**: `docs/api/`",
            "- **Deployment Guide**: `docs/DEPLOYMENT_GUIDE.md`",
            "- **Troubleshooting**: `docs/COMPREHENSIVE_TROUBLESHOOTING.md`",
            "- **Performance Guide**: `docs/WAN22_PERFORMANCE_OPTIMIZATION_GUIDE.md`"
        ]
        
        sections.append(DocumentationSection(
            title="",
            content="\n".join(next_steps),
            level=2
        ))
        
        return self._render_sections(sections)
    
    def generate_component_relationship_docs(self, relationships: ComponentRelationshipMap) -> str:
        """Generate documentation explaining component relationships."""
        
        sections = []
        
        # Header
        sections.append(DocumentationSection(
            title="# Component Relationships",
            content="This document explains how different components in the project interact with each other.",
            level=1
        ))
        
        # Architecture Overview
        arch_content = [
            "## Architecture Overview",
            "",
            f"The project consists of {len(relationships.components)} main components with " +
            f"{len(relationships.dependencies)} dependencies between them.",
            ""
        ]
        
        if relationships.entry_points:
            arch_content.extend([
                "### Entry Points",
                "These components serve as application entry points:",
                ""
            ])
            for entry in relationships.entry_points[:10]:  # Limit to first 10
                component = next((c for c in relationships.components if c.name == entry), None)
                if component and component.purpose:
                    arch_content.append(f"- **{entry}**: {component.purpose}")
                else:
                    arch_content.append(f"- **{entry}**")
            arch_content.append("")
        
        if relationships.critical_components:
            arch_content.extend([
                "### Critical Components",
                "These components are heavily used by others:",
                ""
            ])
            for critical in relationships.critical_components:
                component = next((c for c in relationships.components if c.name == critical), None)
                if component:
                    deps_in = len(component.dependents)
                    arch_content.append(f"- **{critical}**: Used by {deps_in} components")
                    if component.purpose:
                        arch_content.append(f"  - {component.purpose}")
            arch_content.append("")
        
        sections.append(DocumentationSection(
            title="",
            content="\n".join(arch_content),
            level=2
        ))
        
        # Dependency Types
        dep_types = {}
        for dep in relationships.dependencies:
            dep_types[dep.dependency_type] = dep_types.get(dep.dependency_type, 0) + 1
        
        if dep_types:
            dep_content = [
                "## Dependency Types",
                "",
                "The project uses several types of dependencies:",
                ""
            ]
            
            for dep_type, count in sorted(dep_types.items(), key=lambda x: x[1], reverse=True):
                dep_content.append(f"- **{dep_type.replace('_', ' ').title()}**: {count} dependencies")
                
                # Add explanation for each type
                if dep_type == 'import':
                    dep_content.append("  - Python import statements between modules")
                elif dep_type == 'api_call':
                    dep_content.append("  - HTTP API calls between services")
                elif dep_type == 'config':
                    dep_content.append("  - Configuration file references")
                elif dep_type == 'file_reference':
                    dep_content.append("  - Direct file path references in code")
            
            dep_content.append("")
            sections.append(DocumentationSection(
                title="",
                content="\n".join(dep_content),
                level=2
            ))
        
        # Circular Dependencies
        if relationships.circular_dependencies:
            circular_content = [
                "## ⚠️ Circular Dependencies",
                "",
                "The following circular dependencies were detected and should be resolved:",
                ""
            ]
            
            for i, cycle in enumerate(relationships.circular_dependencies, 1):
                circular_content.append(f"### Cycle {i}")
                circular_content.append(f"```")
                circular_content.append(" → ".join(cycle))
                circular_content.append("```")
                circular_content.append("")
                
                # Add suggestions for resolution
                circular_content.extend([
                    "**Suggested resolution:**",
                    "- Extract common functionality into a shared module",
                    "- Use dependency injection or event-driven patterns",
                    "- Refactor to create a clear hierarchy",
                    ""
                ])
            
            sections.append(DocumentationSection(
                title="",
                content="\n".join(circular_content),
                level=2
            ))
        
        # Component Details
        component_details = [
            "## Component Details",
            "",
            "Detailed information about each component and its relationships:",
            ""
        ]
        
        # Sort components by importance (number of connections)
        sorted_components = sorted(
            relationships.components,
            key=lambda c: len(c.dependencies) + len(c.dependents),
            reverse=True
        )
        
        for component in sorted_components[:15]:  # Top 15 most connected
            component_details.append(f"### {component.name}")
            component_details.append("")
            
            if component.purpose:
                component_details.append(f"**Purpose:** {component.purpose}")
            
            component_details.append(f"**Type:** {component.component_type}")
            component_details.append(f"**Files:** {len(component.files)}")
            component_details.append(f"**Dependencies:** {len(component.dependencies)} out, {len(component.dependents)} in")
            
            if component.dependencies:
                component_details.append("")
                component_details.append("**Depends on:**")
                for dep in component.dependencies[:5]:  # Top 5 dependencies
                    component_details.append(f"- {dep.target_component} ({dep.dependency_type})")
                    if dep.details:
                        component_details.append(f"  - {dep.details}")
            
            if component.dependents:
                component_details.append("")
                component_details.append("**Used by:**")
                for dep in component.dependents[:5]:  # Top 5 dependents
                    component_details.append(f"- {dep.source_component} ({dep.dependency_type})")
            
            component_details.append("")
        
        sections.append(DocumentationSection(
            title="",
            content="\n".join(component_details),
            level=2
        ))
        
        return self._render_sections(sections)
    
    def _create_header_section(self, structure: ProjectStructure) -> DocumentationSection:
        """Create the header section for project documentation."""
        content = [
            f"# Project Documentation",
            "",
            f"**Generated:** {self.generation_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Project Root:** `{structure.root_path}`",
            "",
            "This documentation provides a comprehensive overview of the project structure, ",
            "components, and development guidelines.",
            ""
        ]
        
        return DocumentationSection(
            title="",
            content="\n".join(content),
            level=1
        )
    
    def _create_executive_summary(self, structure: ProjectStructure,
                                relationships: Optional[ComponentRelationshipMap],
                                complexity: Optional[ProjectComplexityReport]) -> DocumentationSection:
        """Create executive summary section."""
        content = [
            "## Executive Summary",
            "",
            f"- **Total Files:** {structure.total_files:,}",
            f"- **Total Directories:** {structure.total_directories:,}",
            f"- **Project Size:** {structure.total_size / (1024*1024):.1f} MB",
            f"- **Main Components:** {len(structure.main_components)}",
            f"- **Entry Points:** {len(structure.entry_points)}",
            ""
        ]
        
        if relationships:
            content.extend([
                f"- **Component Dependencies:** {len(relationships.dependencies)}",
                f"- **Critical Components:** {len(relationships.critical_components)}",
            ])
            
            if relationships.circular_dependencies:
                content.append(f"- **⚠️ Circular Dependencies:** {len(relationships.circular_dependencies)}")
        
        if complexity:
            content.extend([
                f"- **Python Files Analyzed:** {complexity.total_files:,}",
                f"- **Lines of Code:** {complexity.total_lines:,}",
                f"- **Average Complexity:** {complexity.average_complexity:.1f}",
            ])
            
            if complexity.high_priority_areas:
                content.append(f"- **⚠️ High Priority Areas:** {len(complexity.high_priority_areas)}")
        
        content.append("")
        
        return DocumentationSection(
            title="",
            content="\n".join(content),
            level=2
        )   
 
    def _create_structure_section(self, structure: ProjectStructure) -> DocumentationSection:
        """Create project structure section."""
        content = [
            "## Project Structure",
            "",
            "### Main Components",
            ""
        ]
        
        for component in structure.main_components[:10]:  # Top 10 components
            icon = self._get_component_icon(component.purpose)
            size_mb = component.total_size / (1024*1024) if component.total_size > 0 else 0
            
            content.append(f"#### {icon} `{component.name}/`")
            content.append("")
            if component.purpose:
                content.append(f"**Purpose:** {component.purpose}")
            content.append(f"**Files:** {component.file_count}")
            content.append(f"**Size:** {size_mb:.1f} MB")
            if component.is_package:
                content.append("**Type:** Python Package")
            content.append("")
        
        # File Categories
        content.extend([
            "### File Categories",
            "",
            f"- **Configuration Files:** {len(structure.configuration_files)}",
            f"- **Documentation Files:** {len(structure.documentation_files)}",
            f"- **Test Files:** {len(structure.test_files)}",
            f"- **Script Files:** {len(structure.script_files)}",
            ""
        ])
        
        # Entry Points
        if structure.entry_points:
            content.extend([
                "### Entry Points",
                "",
                "These files serve as application entry points:",
                ""
            ])
            
            for entry in structure.entry_points[:10]:  # Top 10 entry points
                content.append(f"- `{entry.name}` ({entry.path})")
                if entry.purpose:
                    content.append(f"  - {entry.purpose}")
            
            if len(structure.entry_points) > 10:
                content.append(f"- ... and {len(structure.entry_points) - 10} more")
            
            content.append("")
        
        return DocumentationSection(
            title="",
            content="\n".join(content),
            level=2
        )
    
    def _create_component_overview(self, structure: ProjectStructure,
                                 relationships: Optional[ComponentRelationshipMap]) -> DocumentationSection:
        """Create component overview section."""
        content = [
            "## Component Overview",
            "",
            "This section provides an overview of the main components and their roles:",
            ""
        ]
        
        # Group components by type/purpose
        component_groups = {}
        for component in structure.main_components:
            purpose = component.purpose or "Other"
            if purpose not in component_groups:
                component_groups[purpose] = []
            component_groups[purpose].append(component)
        
        for purpose, components in sorted(component_groups.items()):
            if purpose == "Other":
                continue
                
            content.append(f"### {purpose}")
            content.append("")
            
            for component in components:
                content.append(f"- **{component.name}**: {component.file_count} files")
                
                # Add relationship info if available
                if relationships:
                    rel_component = next((c for c in relationships.components if c.name == component.name), None)
                    if rel_component:
                        deps_out = len(rel_component.dependencies)
                        deps_in = len(rel_component.dependents)
                        if deps_out > 0 or deps_in > 0:
                            content.append(f"  - Dependencies: {deps_out} out, {deps_in} in")
            
            content.append("")
        
        return DocumentationSection(
            title="",
            content="\n".join(content),
            level=2
        )
    
    def _create_architecture_section(self, relationships: ComponentRelationshipMap) -> DocumentationSection:
        """Create architecture overview section."""
        content = [
            "## Architecture Overview",
            "",
            "The system follows a layered architecture with clear separation of concerns:",
            ""
        ]
        
        # Identify layers based on component names and relationships
        layers = {
            "Frontend": [c for c in relationships.components if 'frontend' in c.name.lower() or 'ui' in c.name.lower()],
            "API": [c for c in relationships.components if 'api' in c.name.lower()],
            "Services": [c for c in relationships.components if 'service' in c.name.lower()],
            "Core": [c for c in relationships.components if 'core' in c.name.lower()],
            "Infrastructure": [c for c in relationships.components if 'infrastructure' in c.name.lower()],
            "Tools": [c for c in relationships.components if 'tool' in c.name.lower()],
            "Tests": [c for c in relationships.components if 'test' in c.name.lower()]
        }
        
        for layer_name, components in layers.items():
            if components:
                content.append(f"### {layer_name} Layer")
                content.append("")
                for component in components[:5]:  # Limit to 5 per layer
                    content.append(f"- **{component.name}**: {component.purpose or 'No description'}")
                content.append("")
        
        # Data Flow
        content.extend([
            "### Data Flow",
            "",
            "The typical data flow in the system:",
            "",
            "1. **User Input** → Frontend components",
            "2. **API Requests** → Backend API endpoints", 
            "3. **Business Logic** → Core services",
            "4. **Data Processing** → AI models and utilities",
            "5. **Response** → Back through the layers to user",
            ""
        ])
        
        return DocumentationSection(
            title="",
            content="\n".join(content),
            level=2
        )
    
    def _create_getting_started_section(self, structure: ProjectStructure) -> DocumentationSection:
        """Create getting started section."""
        content = [
            "## Getting Started",
            "",
            "### Prerequisites",
            "",
            "- Python 3.8 or higher",
            "- Git",
            "- CUDA-compatible GPU (recommended)",
            "",
            "### Installation",
            "",
            "1. **Clone the repository:**",
            "   ```bash",
            "   git clone <repository-url>",
            "   cd " + Path(structure.root_path).name,
            "   ```",
            "",
            "2. **Set up virtual environment:**",
            "   ```bash",
            "   python -m venv venv",
            "   source venv/bin/activate  # On Windows: venv\\Scripts\\activate",
            "   ```",
            "",
            "3. **Install dependencies:**",
            "   ```bash",
            "   pip install -r requirements.txt",
            "   ```",
            ""
        ]
        
        # Find main entry points
        main_entries = [e for e in structure.entry_points if e.name.lower() in ['main.py', 'app.py', 'start.py']]
        
        if main_entries:
            content.extend([
                "### Running the Application",
                "",
                "Start the main application:",
                "```bash",
                f"python {main_entries[0].name}",
                "```",
                ""
            ])
        
        # Testing
        content.extend([
            "### Testing",
            "",
            "Run the test suite:",
            "```bash",
            "pytest tests/",
            "```",
            "",
            "Run local testing framework:",
            "```bash",
            "python -m local_testing_framework",
            "```",
            ""
        ])
        
        return DocumentationSection(
            title="",
            content="\n".join(content),
            level=2
        )
    
    def _create_development_guide(self, structure: ProjectStructure,
                                complexity: Optional[ProjectComplexityReport]) -> DocumentationSection:
        """Create development guide section."""
        content = [
            "## Development Guide",
            "",
            "### Code Organization",
            "",
            "Follow these guidelines when adding new code:",
            "",
            "- **Backend logic**: Add to `backend/core/` or `backend/services/`",
            "- **API endpoints**: Add to `backend/api/`",
            "- **Frontend components**: Add to `frontend/src/components/`",
            "- **Utilities**: Add to appropriate utility modules",
            "- **Tests**: Mirror the structure in `tests/`",
            "",
            "### Coding Standards",
            "",
            "- Follow PEP 8 for Python code",
            "- Use type hints where possible",
            "- Write docstrings for public functions and classes",
            "- Keep functions focused and small",
            "- Add tests for new functionality",
            ""
        ]
        
        if complexity and complexity.high_priority_areas:
            content.extend([
                "### Areas Needing Attention",
                "",
                "These components have high complexity and should be refactored:",
                ""
            ])
            
            for area in complexity.high_priority_areas[:5]:
                content.append(f"- `{area}`: High complexity, needs refactoring")
            
            content.append("")
        
        content.extend([
            "### Testing Strategy",
            "",
            "- **Unit Tests**: Test individual functions and classes",
            "- **Integration Tests**: Test component interactions",
            "- **End-to-End Tests**: Test complete workflows",
            "- **Local Testing**: Use the Local Testing Framework",
            "",
            "### Performance Considerations",
            "",
            "- Profile code before optimizing",
            "- Consider memory usage with large models",
            "- Use async/await for I/O operations",
            "- Cache expensive computations",
            ""
        ])
        
        return DocumentationSection(
            title="",
            content="\n".join(content),
            level=2
        )
    
    def _create_dependency_documentation(self, dependencies: List[ComponentDependency], title: str) -> str:
        """Create documentation for a list of dependencies."""
        content = []
        
        # Group by dependency type
        by_type = {}
        for dep in dependencies:
            if dep.dependency_type not in by_type:
                by_type[dep.dependency_type] = []
            by_type[dep.dependency_type].append(dep)
        
        for dep_type, deps in sorted(by_type.items()):
            content.append(f"**{dep_type.replace('_', ' ').title()}:**")
            for dep in deps[:5]:  # Limit to 5 per type
                target = dep.target_component if title == "Dependencies" else dep.source_component
                content.append(f"- {target}")
                if dep.details:
                    content.append(f"  - {dep.details}")
            
            if len(deps) > 5:
                content.append(f"- ... and {len(deps) - 5} more")
            content.append("")
        
        return "\n".join(content)
    
    def _find_file_info(self, file_path: str, structure: ProjectStructure) -> Optional[FileInfo]:
        """Find file info from structure analysis."""
        for file_info in (structure.configuration_files + structure.documentation_files + 
                         structure.test_files + structure.script_files):
            if file_info.path == file_path:
                return file_info
        return None
    
    def _guess_file_purpose(self, filename: str) -> str:
        """Guess the purpose of a file based on its name."""
        name_lower = filename.lower()
        
        purposes = {
            'main': 'Main entry point',
            'app': 'Application entry point',
            'test': 'Test runner',
            'config': 'Configuration management',
            'manager': 'Management utilities',
            'validator': 'Validation logic',
            'monitor': 'Monitoring functionality',
            'report': 'Report generation',
            'sample': 'Sample data management',
            'integration': 'Integration testing',
            'performance': 'Performance testing',
            'diagnostic': 'Diagnostic tools',
            'production': 'Production utilities'
        }
        
        for key, purpose in purposes.items():
            if key in name_lower:
                return purpose
        
        return 'Utility module'
    
    def _guess_directory_purpose(self, dirname: str) -> str:
        """Guess the purpose of a directory based on its name."""
        name_lower = dirname.lower()
        
        purposes = {
            'tests': 'Test files',
            'docs': 'Documentation',
            'config': 'Configuration files',
            'examples': 'Example code',
            'models': 'Data models',
            'cli': 'Command-line interface',
            'edge_case_samples': 'Edge case test data',
            'test_samples': 'Test data samples'
        }
        
        return purposes.get(name_lower, 'Module directory')
    
    def _get_component_icon(self, purpose: Optional[str]) -> str:
        """Get appropriate icon for component purpose."""
        if not purpose:
            return "📁"
        
        purpose_lower = purpose.lower()
        
        icons = {
            'backend': '🔧',
            'frontend': '🎨',
            'api': '🌐',
            'core': '⚙️',
            'models': '📊',
            'services': '🔄',
            'utils': '🛠️',
            'config': '⚙️',
            'test': '🧪',
            'docs': '📚',
            'scripts': '📜',
            'tools': '🔨',
            'local': '🏠',
            'installation': '📦'
        }
        
        for key, icon in icons.items():
            if key in purpose_lower:
                return icon
        
        return "📁"
    
    def _render_sections(self, sections: List[DocumentationSection]) -> str:
        """Render documentation sections to markdown."""
        content = []
        
        for section in sections:
            if section.title:
                content.append(section.title)
                content.append("")
            
            content.append(section.content)
            content.append("")
            
            if section.subsections:
                for subsection in section.subsections:
                    content.append(self._render_sections([subsection]))
        
        return "\n".join(content)
    
    def save_documentation(self, content: str, filename: str, output_dir: str):
        """Save documentation to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Documentation saved: {file_path}")
    
    def generate_all_documentation(self, structure: ProjectStructure,
                                 relationships: Optional[ComponentRelationshipMap] = None,
                                 complexity: Optional[ProjectComplexityReport] = None,
                                 output_dir: str = "./documentation") -> Dict[str, str]:
        """Generate all documentation types."""
        
        docs = {}
        
        # Project overview
        docs['project_overview.md'] = self.generate_project_overview(structure, relationships, complexity)
        
        # Developer onboarding guide
        docs['developer_onboarding.md'] = self.generate_developer_onboarding_guide(structure, relationships, complexity)
        
        # Local Testing Framework documentation
        if relationships:
            docs['local_testing_framework.md'] = self.generate_local_testing_framework_docs(structure, relationships)
            
            # Component relationships
            docs['component_relationships.md'] = self.generate_component_relationship_docs(relationships)
        
        # Individual component documentation
        if relationships:
            for component in relationships.components[:10]:  # Top 10 components
                if component.files:  # Only document components with files
                    filename = f"component_{component.name.replace('.', '_').replace('/', '_')}.md"
                    docs[filename] = self.generate_component_documentation(component, structure, relationships)
        
        # Save all documentation
        for filename, content in docs.items():
            self.save_documentation(content, filename, output_dir)
        
        return docs