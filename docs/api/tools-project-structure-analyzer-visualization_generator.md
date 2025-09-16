---
title: tools.project-structure-analyzer.visualization_generator
category: api
tags: [api, tools]
---

# tools.project-structure-analyzer.visualization_generator

Mermaid Visualization Generator

Generates Mermaid diagrams for project structure visualization
including component relationships and dependency graphs.

## Classes

### MermaidDiagram

Represents a Mermaid diagram.

### MermaidVisualizationGenerator

Generates Mermaid diagrams for project visualization.

#### Methods

##### __init__(self: Any)

Initialize the visualization generator.

##### generate_structure_diagram(self: Any, structure: ProjectStructure) -> MermaidDiagram

Generate a project structure diagram.

##### generate_dependency_diagram(self: Any, relationships: ComponentRelationshipMap) -> MermaidDiagram

Generate a component dependency diagram.

##### generate_complexity_heatmap(self: Any, complexity_report: Any) -> MermaidDiagram

Generate a complexity heatmap diagram.

##### _get_component_icon(self: Any, purpose: <ast.Subscript object at 0x00000194288F6830>) -> str

Get appropriate icon for component purpose.

##### generate_all_diagrams(self: Any, structure: ProjectStructure, relationships: ComponentRelationshipMap, complexity_report: Any) -> <ast.Subscript object at 0x00000194288F7100>

Generate all available diagrams.

##### save_diagrams(self: Any, diagrams: <ast.Subscript object at 0x00000194288F7DF0>, output_dir: str)

Save diagrams to files.

##### _create_diagram_index(self: Any, diagrams: <ast.Subscript object at 0x00000194288F6980>, output_path: Path)

Create an index file for all diagrams.

