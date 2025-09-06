"""
Mermaid Visualization Generator

Generates Mermaid diagrams for project structure visualization
including component relationships and dependency graphs.
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from pathlib import Path
import json

from structure_analyzer import ProjectStructure, DirectoryInfo
from component_analyzer import ComponentRelationshipMap, ComponentInfo, ComponentDependency


@dataclass
class MermaidDiagram:
    """Represents a Mermaid diagram."""
    title: str
    diagram_type: str
    content: str
    description: str


class MermaidVisualizationGenerator:
    """Generates Mermaid diagrams for project visualization."""
    
    def __init__(self):
        """Initialize the visualization generator."""
        pass
    
    def generate_structure_diagram(self, structure: ProjectStructure) -> MermaidDiagram:
        """Generate a project structure diagram."""
        lines = []
        lines.append("graph TD")
        lines.append("    %% Project Structure Overview")
        lines.append("")
        
        # Add root node
        lines.append("    Root[\"🏠 Project Root\"]")
        lines.append("")
        
        # Add main components
        component_nodes = {}
        for i, component in enumerate(structure.main_components[:10]):  # Limit to top 10
            node_id = f"C{i}"
            component_nodes[component.name] = node_id
            
            # Determine icon based on component purpose
            icon = self._get_component_icon(component.purpose)
            label = f"{icon} {component.name}"
            
            lines.append(f"    {node_id}[\"{label}\"]")
            lines.append(f"    Root --> {node_id}")
        
        lines.append("")
        
        # Add styling
        lines.extend([
            "    %% Styling",
            "    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px",
            "    classDef root fill:#e1f5fe,stroke:#01579b,stroke-width:3px",
            "    classDef component fill:#f3e5f5,stroke:#4a148c,stroke-width:2px",
            "",
            "    class Root root",
            "    class " + ",".join(component_nodes.values()) + " component"
        ])
        
        content = "\n".join(lines)
        
        return MermaidDiagram(
            title="Project Structure Overview",
            diagram_type="graph",
            content=content,
            description="High-level overview of main project components"
        )
    
    def generate_dependency_diagram(self, relationships: ComponentRelationshipMap) -> MermaidDiagram:
        """Generate a component dependency diagram."""
        lines = []
        lines.append("graph LR")
        lines.append("    %% Component Dependencies")
        lines.append("")
        
        # Create nodes for components
        component_nodes = {}
        for i, component in enumerate(relationships.components[:15]):  # Limit for readability
            node_id = f"C{i}"
            component_nodes[component.name] = node_id
            
            # Style based on component type
            if component.name in relationships.critical_components:
                style = "critical"
            elif component.name in relationships.entry_points:
                style = "entry"
            elif component.name in relationships.isolated_components:
                style = "isolated"
            else:
                style = "normal"
            
            lines.append(f"    {node_id}[\"{component.name}\"]")
        
        lines.append("")
        
        # Add dependencies
        added_edges = set()
        for dep in relationships.dependencies:
            if dep.source_component in component_nodes and dep.target_component in component_nodes:
                source_id = component_nodes[dep.source_component]
                target_id = component_nodes[dep.target_component]
                edge_key = (source_id, target_id)
                
                if edge_key not in added_edges:
                    # Style edge based on dependency type
                    if dep.dependency_type == 'import':
                        edge_style = "-->"
                    elif dep.dependency_type == 'api_call':
                        edge_style = "==>"
                    else:
                        edge_style = "-.->"
                    
                    lines.append(f"    {source_id} {edge_style} {target_id}")
                    added_edges.add(edge_key)
        
        lines.append("")
        
        # Add styling
        lines.extend([
            "    %% Styling",
            "    classDef critical fill:#ffcdd2,stroke:#d32f2f,stroke-width:3px",
            "    classDef entry fill:#c8e6c9,stroke:#388e3c,stroke-width:3px", 
            "    classDef isolated fill:#fff3e0,stroke:#f57c00,stroke-width:2px",
            "    classDef normal fill:#f5f5f5,stroke:#616161,stroke-width:2px"
        ])
        
        # Apply styles
        critical_nodes = [component_nodes[name] for name in relationships.critical_components 
                         if name in component_nodes]
        entry_nodes = [component_nodes[name] for name in relationships.entry_points 
                      if name in component_nodes]
        isolated_nodes = [component_nodes[name] for name in relationships.isolated_components 
                         if name in component_nodes]
        
        if critical_nodes:
            lines.append(f"    class {','.join(critical_nodes)} critical")
        if entry_nodes:
            lines.append(f"    class {','.join(entry_nodes)} entry")
        if isolated_nodes:
            lines.append(f"    class {','.join(isolated_nodes)} isolated")
        
        content = "\n".join(lines)
        
        return MermaidDiagram(
            title="Component Dependencies",
            diagram_type="graph",
            content=content,
            description="Dependencies between project components"
        )
    
    def generate_complexity_heatmap(self, complexity_report) -> MermaidDiagram:
        """Generate a complexity heatmap diagram."""
        lines = []
        lines.append("graph TD")
        lines.append("    %% Complexity Heatmap")
        lines.append("")
        
        # Sort components by complexity
        sorted_components = sorted(
            complexity_report.components,
            key=lambda c: c.complexity_score,
            reverse=True
        )
        
        # Create nodes with complexity-based styling
        for i, component in enumerate(sorted_components[:12]):  # Top 12 most complex
            node_id = f"COMP{i}"
            
            # Determine complexity level
            if component.complexity_score > 40:
                complexity_level = "very-high"
                icon = "🔥"
            elif component.complexity_score > 30:
                complexity_level = "high"
                icon = "⚠️"
            elif component.complexity_score > 20:
                complexity_level = "medium"
                icon = "⚡"
            else:
                complexity_level = "low"
                icon = "✅"
            
            label = f"{icon} {component.name}\\n({component.complexity_score})"
            lines.append(f"    {node_id}[\"{label}\"]")
        
        lines.append("")
        
        # Add styling
        lines.extend([
            "    %% Complexity Styling",
            "    classDef very-high fill:#ff1744,color:#fff,stroke:#d50000,stroke-width:3px",
            "    classDef high fill:#ff9800,color:#fff,stroke:#e65100,stroke-width:3px",
            "    classDef medium fill:#ffc107,color:#000,stroke:#ff8f00,stroke-width:2px",
            "    classDef low fill:#4caf50,color:#fff,stroke:#2e7d32,stroke-width:2px"
        ])
        
        # Apply complexity styling
        for i, component in enumerate(sorted_components[:12]):
            node_id = f"COMP{i}"
            if component.complexity_score > 40:
                lines.append(f"    class {node_id} very-high")
            elif component.complexity_score > 30:
                lines.append(f"    class {node_id} high")
            elif component.complexity_score > 20:
                lines.append(f"    class {node_id} medium")
            else:
                lines.append(f"    class {node_id} low")
        
        content = "\n".join(lines)
        
        return MermaidDiagram(
            title="Complexity Heatmap",
            diagram_type="graph", 
            content=content,
            description="Visual representation of component complexity levels"
        )
    
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
            'tools': '🔨'
        }
        
        for key, icon in icons.items():
            if key in purpose_lower:
                return icon
        
        return "📁"
    
    def generate_all_diagrams(self, structure: ProjectStructure, 
                            relationships: ComponentRelationshipMap,
                            complexity_report) -> List[MermaidDiagram]:
        """Generate all available diagrams."""
        diagrams = []
        
        # Structure diagram
        diagrams.append(self.generate_structure_diagram(structure))
        
        # Dependency diagram
        diagrams.append(self.generate_dependency_diagram(relationships))
        
        # Complexity heatmap
        diagrams.append(self.generate_complexity_heatmap(complexity_report))
        
        return diagrams
    
    def save_diagrams(self, diagrams: List[MermaidDiagram], output_dir: str):
        """Save diagrams to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for diagram in diagrams:
            # Save as .mmd file
            filename = diagram.title.lower().replace(' ', '_') + '.mmd'
            file_path = output_path / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"---\ntitle: {diagram.title}\n---\n")
                f.write(diagram.content)
            
            print(f"Saved diagram: {file_path}")
        
        # Create index file
        self._create_diagram_index(diagrams, output_path)
    
    def _create_diagram_index(self, diagrams: List[MermaidDiagram], output_path: Path):
        """Create an index file for all diagrams."""
        index_content = []
        index_content.append("# Project Structure Diagrams")
        index_content.append("")
        index_content.append("This directory contains Mermaid diagrams visualizing the project structure.")
        index_content.append("")
        
        for diagram in diagrams:
            filename = diagram.title.lower().replace(' ', '_') + '.mmd'
            index_content.append(f"## {diagram.title}")
            index_content.append("")
            index_content.append(diagram.description)
            index_content.append("")
            index_content.append(f"**File:** `{filename}`")
            index_content.append("")
            index_content.append("```mermaid")
            index_content.append(diagram.content)
            index_content.append("```")
            index_content.append("")
        
        index_file = output_path / "README.md"
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(index_content))
        
        print(f"Created diagram index: {index_file}")