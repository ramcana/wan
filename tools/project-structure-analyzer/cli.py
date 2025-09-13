"""
CLI interface for Project Structure Analysis Engine
"""

import argparse
import sys
from pathlib import Path

from tools..structure_analyzer import ProjectStructureAnalyzer
from tools..component_analyzer import ComponentRelationshipAnalyzer
from tools..complexity_analyzer import ProjectComplexityAnalyzer
from tools..visualization_generator import MermaidVisualizationGenerator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze project structure and generate documentation"
    )
    
    parser.add_argument(
        "project_path",
        help="Path to the project root directory"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="./project-analysis",
        help="Output directory for analysis results"
    )
    
    parser.add_argument(
        "--skip-complexity",
        action="store_true",
        help="Skip complexity analysis"
    )
    
    parser.add_argument(
        "--skip-relationships", 
        action="store_true",
        help="Skip component relationship analysis"
    )
    
    parser.add_argument(
        "--skip-visualizations",
        action="store_true", 
        help="Skip generating Mermaid diagrams"
    )
    
    args = parser.parse_args()
    
    # Validate project path
    project_path = Path(args.project_path)
    if not project_path.exists():
        print(f"Error: Project path does not exist: {project_path}")
        sys.exit(1)
    
    if not project_path.is_dir():
        print(f"Error: Project path is not a directory: {project_path}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing project: {project_path}")
    print(f"Output directory: {output_path}")
    print()
    
    # Run analyses
    run_analysis(str(project_path), str(output_path), args)


def run_analysis(project_path: str, output_path: str, args):
    """Run the complete project analysis."""
    
    # 1. Structure Analysis
    print("=" * 60)
    print("STEP 1: PROJECT STRUCTURE ANALYSIS")
    print("=" * 60)
    
    structure_analyzer = ProjectStructureAnalyzer(project_path)
    structure = structure_analyzer.analyze()
    
    # Save structure analysis
    structure_analyzer.save_analysis(structure, f"{output_path}/structure_analysis.json")
    
    # Generate structure report
    structure_report = structure_analyzer.generate_summary_report(structure)
    with open(f"{output_path}/structure_report.md", 'w', encoding='utf-8') as f:
        f.write(structure_report)
    
    print(f"✅ Structure analysis complete")
    print(f"   - Found {structure.total_files:,} files in {structure.total_directories:,} directories")
    print(f"   - Identified {len(structure.main_components)} main components")
    print()
    
    # 2. Component Relationship Analysis
    relationships = None
    if not args.skip_relationships:
        print("=" * 60)
        print("STEP 2: COMPONENT RELATIONSHIP ANALYSIS") 
        print("=" * 60)
        
        relationship_analyzer = ComponentRelationshipAnalyzer(project_path)
        relationships = relationship_analyzer.analyze()
        
        # Save relationship analysis
        relationship_analyzer.save_analysis(relationships, f"{output_path}/relationships_analysis.json")
        
        # Generate relationship report
        relationship_report = relationship_analyzer.generate_summary_report(relationships)
        with open(f"{output_path}/relationships_report.md", 'w', encoding='utf-8') as f:
            f.write(relationship_report)
        
        print(f"✅ Relationship analysis complete")
        print(f"   - Analyzed {len(relationships.components)} components")
        print(f"   - Found {len(relationships.dependencies)} dependencies")
        if relationships.circular_dependencies:
            print(f"   - ⚠️  Detected {len(relationships.circular_dependencies)} circular dependencies")
        print()
    
    # 3. Complexity Analysis
    complexity_report = None
    if not args.skip_complexity:
        print("=" * 60)
        print("STEP 3: COMPLEXITY ANALYSIS")
        print("=" * 60)
        
        complexity_analyzer = ProjectComplexityAnalyzer(project_path)
        complexity_report = complexity_analyzer.analyze()
        
        # Save complexity analysis
        complexity_analyzer.save_analysis(complexity_report, f"{output_path}/complexity_analysis.json")
        
        # Generate complexity report
        complexity_summary = complexity_analyzer.generate_summary_report(complexity_report)
        with open(f"{output_path}/complexity_report.md", 'w', encoding='utf-8') as f:
            f.write(complexity_summary)
        
        print(f"✅ Complexity analysis complete")
        print(f"   - Analyzed {complexity_report.total_files:,} Python files")
        print(f"   - Average complexity score: {complexity_report.average_complexity:.1f}")
        if complexity_report.high_priority_areas:
            print(f"   - ⚠️  {len(complexity_report.high_priority_areas)} high-priority areas identified")
        print()


if __name__ == "__main__":
    main() 
   
    # 4. Generate Visualizations
    if not args.skip_visualizations and relationships:
        print("=" * 60)
        print("STEP 4: GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        viz_generator = MermaidVisualizationGenerator()
        
        diagrams = []
        
        # Structure diagram
        structure_diagram = viz_generator.generate_structure_diagram(structure)
        diagrams.append(structure_diagram)
        
        # Dependency diagram
        dependency_diagram = viz_generator.generate_dependency_diagram(relationships)
        diagrams.append(dependency_diagram)
        
        # Complexity heatmap (if complexity analysis was run)
        if complexity_report:
            complexity_diagram = viz_generator.generate_complexity_heatmap(complexity_report)
            diagrams.append(complexity_diagram)
        
        # Save all diagrams
        viz_output = f"{output_path}/diagrams"
        viz_generator.save_diagrams(diagrams, viz_output)
        
        print(f"✅ Visualizations generated")
        print(f"   - Created {len(diagrams)} Mermaid diagrams")
        print(f"   - Saved to: {viz_output}")
        print()
    
    # 5. Generate Master Report
    print("=" * 60)
    print("STEP 5: GENERATING MASTER REPORT")
    print("=" * 60)
    
    master_report = generate_master_report(structure, relationships, complexity_report)
    with open(f"{output_path}/PROJECT_ANALYSIS_REPORT.md", 'w', encoding='utf-8') as f:
        f.write(master_report)
    
    print("✅ Master report generated")
    print()
    
    # Summary
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"📁 All results saved to: {output_path}")
    print()
    print("Generated files:")
    print("- PROJECT_ANALYSIS_REPORT.md (Master report)")
    print("- structure_analysis.json")
    print("- structure_report.md")
    if relationships:
        print("- relationships_analysis.json")
        print("- relationships_report.md")
    if complexity_report:
        print("- complexity_analysis.json") 
        print("- complexity_report.md")
    if not args.skip_visualizations and relationships:
        print("- diagrams/ (Mermaid visualizations)")
    print()


def generate_master_report(structure, relationships, complexity_report):
    """Generate a comprehensive master report."""
    report = []
    
    report.append("# Project Analysis Report")
    report.append("")
    report.append("This report provides a comprehensive analysis of the project structure, ")
    report.append("component relationships, and code complexity.")
    report.append("")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    report.append(f"- **Project Size:** {structure.total_files:,} files, {structure.total_directories:,} directories")
    report.append(f"- **Main Components:** {len(structure.main_components)}")
    
    if relationships:
        report.append(f"- **Component Dependencies:** {len(relationships.dependencies)}")
        if relationships.circular_dependencies:
            report.append(f"- **⚠️ Circular Dependencies:** {len(relationships.circular_dependencies)}")
    
    if complexity_report:
        report.append(f"- **Code Complexity:** {complexity_report.average_complexity:.1f} average score")
        if complexity_report.high_priority_areas:
            report.append(f"- **⚠️ High Priority Areas:** {len(complexity_report.high_priority_areas)}")
    
    report.append("")
    
    # Key Findings
    report.append("## Key Findings")
    report.append("")
    
    # Structure findings
    report.append("### Project Structure")
    report.append("")
    report.append("**Main Components:**")
    for component in structure.main_components[:10]:
        size_mb = component.total_size / (1024*1024) if component.total_size > 0 else 0
        report.append(f"- **{component.name}**: {component.file_count} files, {size_mb:.1f} MB")
        if component.purpose:
            report.append(f"  - {component.purpose}")
    report.append("")
    
    # Entry points
    if structure.entry_points:
        report.append("**Entry Points:**")
        for entry in structure.entry_points:
            report.append(f"- {entry.name} ({entry.path})")
        report.append("")
    
    # Relationship findings
    if relationships:
        report.append("### Component Relationships")
        report.append("")
        
        if relationships.critical_components:
            report.append("**Critical Components** (heavily depended upon):")
            for comp in relationships.critical_components[:5]:
                report.append(f"- {comp}")
            report.append("")
        
        if relationships.circular_dependencies:
            report.append("**⚠️ Circular Dependencies:**")
            for cycle in relationships.circular_dependencies:
                report.append(f"- {' → '.join(cycle)}")
            report.append("")
    
    # Complexity findings
    if complexity_report:
        report.append("### Code Complexity")
        report.append("")
        
        if complexity_report.high_priority_areas:
            report.append("**High Priority Areas** (need immediate attention):")
            for area in complexity_report.high_priority_areas[:5]:
                report.append(f"- {area}")
            report.append("")
        
        if complexity_report.documentation_gaps:
            report.append("**Documentation Gaps:**")
            for gap in complexity_report.documentation_gaps[:5]:
                report.append(f"- {gap}")
            report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    
    recommendations = []
    
    # Structure recommendations
    if len(structure.main_components) > 20:
        recommendations.append("Consider consolidating similar components to reduce complexity")
    
    # Relationship recommendations
    if relationships and relationships.circular_dependencies:
        recommendations.append("Resolve circular dependencies to improve maintainability")
    
    if relationships and len(relationships.isolated_components) > 5:
        recommendations.append("Review isolated components - they may be unused or need better integration")
    
    # Complexity recommendations
    if complexity_report:
        recommendations.extend(complexity_report.recommendations)
    
    for i, rec in enumerate(recommendations, 1):
        report.append(f"{i}. {rec}")
    
    report.append("")
    
    # Detailed Reports
    report.append("## Detailed Reports")
    report.append("")
    report.append("For detailed analysis, see the following files:")
    report.append("")
    report.append("- `structure_report.md` - Detailed project structure analysis")
    if relationships:
        report.append("- `relationships_report.md` - Component dependency analysis")
    if complexity_report:
        report.append("- `complexity_report.md` - Code complexity analysis")
    report.append("- `diagrams/` - Visual diagrams (Mermaid format)")
    report.append("")
    
    return "\n".join(report)
