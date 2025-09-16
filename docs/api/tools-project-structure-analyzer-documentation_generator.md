---
title: tools.project-structure-analyzer.documentation_generator
category: api
tags: [api, tools]
---

# tools.project-structure-analyzer.documentation_generator



## Classes

### DocumentationSection

Represents a section of documentation.

### DocumentationTemplate

Template for generating documentation.

### DocumentationGenerator

Generates comprehensive project documentation.

#### Methods

##### __init__(self: Any, project_root: str)

Initialize the documentation generator.

##### generate_project_overview(self: Any, structure: ProjectStructure, relationships: <ast.Subscript object at 0x000001942F3BEBC0>, complexity: <ast.Subscript object at 0x000001942F3BEB00>) -> str

Generate comprehensive project overview documentation.

##### generate_component_documentation(self: Any, component: ComponentInfo, structure: ProjectStructure, relationships: ComponentRelationshipMap) -> str

Generate detailed documentation for a specific component.

##### generate_local_testing_framework_docs(self: Any, structure: ProjectStructure, relationships: ComponentRelationshipMap) -> str

Generate specific documentation for the Local Testing Framework.

##### generate_developer_onboarding_guide(self: Any, structure: ProjectStructure, relationships: <ast.Subscript object at 0x00000194318D7010>, complexity: <ast.Subscript object at 0x00000194318D6F50>) -> str

Generate step-by-step developer onboarding guide.

##### generate_component_relationship_docs(self: Any, relationships: ComponentRelationshipMap) -> str

Generate documentation explaining component relationships.

##### _create_header_section(self: Any, structure: ProjectStructure) -> DocumentationSection

Create the header section for project documentation.

##### _create_executive_summary(self: Any, structure: ProjectStructure, relationships: <ast.Subscript object at 0x000001942FBB2260>, complexity: <ast.Subscript object at 0x000001942FBB21A0>) -> DocumentationSection

Create executive summary section.

##### _create_structure_section(self: Any, structure: ProjectStructure) -> DocumentationSection

Create project structure section.

##### _create_component_overview(self: Any, structure: ProjectStructure, relationships: <ast.Subscript object at 0x000001942F3B54B0>) -> DocumentationSection

Create component overview section.

##### _create_architecture_section(self: Any, relationships: ComponentRelationshipMap) -> DocumentationSection

Create architecture overview section.

##### _create_getting_started_section(self: Any, structure: ProjectStructure) -> DocumentationSection

Create getting started section.

##### _create_development_guide(self: Any, structure: ProjectStructure, complexity: <ast.Subscript object at 0x000001942FDFACE0>) -> DocumentationSection

Create development guide section.

##### _create_dependency_documentation(self: Any, dependencies: <ast.Subscript object at 0x000001942FDFBFA0>, title: str) -> str

Create documentation for a list of dependencies.

##### _find_file_info(self: Any, file_path: str, structure: ProjectStructure) -> <ast.Subscript object at 0x000001942FDC9D50>

Find file info from structure analysis.

##### _guess_file_purpose(self: Any, filename: str) -> str

Guess the purpose of a file based on its name.

##### _guess_directory_purpose(self: Any, dirname: str) -> str

Guess the purpose of a directory based on its name.

##### _get_component_icon(self: Any, purpose: <ast.Subscript object at 0x000001942FDCB0A0>) -> str

Get appropriate icon for component purpose.

##### _render_sections(self: Any, sections: <ast.Subscript object at 0x000001942FDCBD30>) -> str

Render documentation sections to markdown.

##### save_documentation(self: Any, content: str, filename: str, output_dir: str)

Save documentation to file.

##### generate_all_documentation(self: Any, structure: ProjectStructure, relationships: <ast.Subscript object at 0x000001942FE39330>, complexity: <ast.Subscript object at 0x000001942FE393F0>, output_dir: str) -> <ast.Subscript object at 0x000001942FE3A7D0>

Generate all documentation types.

