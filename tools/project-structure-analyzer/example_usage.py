#!/usr/bin/env python3
"""
Example usage of the Project Structure Analysis Engine

This script demonstrates how to use the various components
of the project structure analyzer programmatically.
"""

import os
import sys
import json
from pathlib import Path

# Add the tools directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from structure_analyzer import ProjectStructureAnalyzer
from component_analyzer import ComponentRelationshipAnalyzer
from complexity_analyzer import ProjectComplexityAnalyzer
from visualization_generator import MermaidVisualizationGenerator
from documentation_generator import DocumentationGenerator
from documentation_validator import DocumentationValidator


def main():
    """Run example analysis on the current project."""
    
    # Use the project root (go up from tools/project-structure-analyzer)
    project_root = Path(__file__).parent.parent.parent
    output_dir = Path("./example-analysis-output")
    
    print("🔍 Project Structure Analysis Engine - Example Usage")
    print("=" * 60)
    print(f"Analyzing project: {project_root}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # 1. Basic Structure Analysis
    print("📁 Running Structure Analysis...")
    structure_analyzer = ProjectStructureAnalyzer(str(project_root))
    structure = structure_analyzer.analyze()
    
    print(f"   Found {structure.total_files:,} files in {structure.total_directories:,} directories")
    print(f"   Identified {len(structure.main_components)} main components")
    print(f"   Entry points: {len(structure.entry_points)}")
    print()
    
    # Show top components
    print("🏗️ Top Components:")
    for i, component in enumerate(structure.main_components[:5], 1):
        size_mb = component.total_size / (1024*1024) if component.total_size > 0 else 0
        print(f"   {i}. {component.name} - {component.file_count} files, {size_mb:.1f} MB")
        if component.purpose:
            print(f"      Purpose: {component.purpose}")
    print()
    
    # 2. Component Relationship Analysis
    print("🔗 Running Relationship Analysis...")
    relationship_analyzer = ComponentRelationshipAnalyzer(str(project_root))
    relationships = relationship_analyzer.analyze()
    
    print(f"   Analyzed {len(relationships.components)} components")
    print(f"   Found {len(relationships.dependencies)} dependencies")
    print(f"   Critical components: {len(relationships.critical_components)}")
    print(f"   Entry points: {len(relationships.entry_points)}")
    
    if relationships.circular_dependencies:
        print(f"   ⚠️  Circular dependencies: {len(relationships.circular_dependencies)}")
        for cycle in relationships.circular_dependencies[:3]:  # Show first 3
            print(f"      - {' → '.join(cycle)}")
    print()
    
    # 3. Complexity Analysis
    print("📊 Running Complexity Analysis...")
    complexity_analyzer = ProjectComplexityAnalyzer(str(project_root))
    complexity_report = complexity_analyzer.analyze()
    
    print(f"   Analyzed {complexity_report.total_files:,} Python files")
    print(f"   Total lines of code: {complexity_report.total_lines:,}")
    print(f"   Average complexity: {complexity_report.average_complexity:.1f}")
    
    if complexity_report.high_priority_areas:
        print(f"   ⚠️  High priority areas: {len(complexity_report.high_priority_areas)}")
        for area in complexity_report.high_priority_areas[:3]:  # Show first 3
            print(f"      - {area}")
    print()
    
    # 4. Generate Visualizations
    print("📈 Generating Visualizations...")
    viz_generator = MermaidVisualizationGenerator()
    
    # Generate all diagrams
    diagrams = viz_generator.generate_all_diagrams(structure, relationships, complexity_report)
    
    print(f"   Generated {len(diagrams)} diagrams:")
    for diagram in diagrams:
        print(f"      - {diagram.title}")
    print()
    
    # 5. Save Results
    print("💾 Saving Results...")
    
    # Save structure analysis
    structure_analyzer.save_analysis(structure, str(output_dir / "structure_analysis.json"))
    
    # Save relationship analysis
    relationship_analyzer.save_analysis(relationships, str(output_dir / "relationships_analysis.json"))
    
    # Save complexity analysis
    complexity_analyzer.save_analysis(complexity_report, str(output_dir / "complexity_analysis.json"))
    
    # Save visualizations
    viz_generator.save_diagrams(diagrams, str(output_dir / "diagrams"))
    
    # Generate reports
    structure_report = structure_analyzer.generate_summary_report(structure)
    with open(output_dir / "structure_report.md", 'w', encoding='utf-8') as f:
        f.write(structure_report)
    
    relationship_report = relationship_analyzer.generate_summary_report(relationships)
    with open(output_dir / "relationships_report.md", 'w', encoding='utf-8') as f:
        f.write(relationship_report)
    
    complexity_summary = complexity_analyzer.generate_summary_report(complexity_report)
    with open(output_dir / "complexity_report.md", 'w', encoding='utf-8') as f:
        f.write(complexity_summary)
    
    print("   All results saved!")
    print()
    
    # 6. Generate Documentation
    print("📚 Generating Documentation...")
    doc_generator = DocumentationGenerator(str(project_root))
    
    # Generate all documentation
    docs = doc_generator.generate_all_documentation(
        structure, relationships, complexity_report, str(output_dir / "documentation")
    )
    
    print(f"   Generated {len(docs)} documentation files:")
    for filename in docs.keys():
        print(f"      - {filename}")
    print()
    
    # 7. Validate Documentation
    print("✅ Validating Documentation...")
    doc_validator = DocumentationValidator(str(project_root))
    
    # Run validation
    validation_report = doc_validator.validate_all(structure, relationships)
    
    print(f"   Validated {validation_report.metrics.total_files} documentation files")
    print(f"   Found {len(validation_report.issues)} issues")
    print(f"   Coverage: {validation_report.metrics.coverage_percentage:.1f}%")
    print(f"   Freshness: {validation_report.metrics.freshness_score:.1f}%")
    
    if validation_report.metrics.broken_links > 0:
        print(f"   ⚠️  {validation_report.metrics.broken_links} broken links found")
    
    # Save validation report
    doc_validator.save_report(validation_report, str(output_dir / "documentation_validation.json"))
    
    # Generate validation summary
    validation_summary = doc_validator.generate_summary_report(validation_report)
    with open(output_dir / "documentation_validation_report.md", 'w', encoding='utf-8') as f:
        f.write(validation_summary)
    
    # Generate maintenance plan
    maintenance_plan = doc_validator.generate_maintenance_plan(validation_report)
    with open(output_dir / "documentation_maintenance_plan.json", 'w', encoding='utf-8') as f:
        json.dump(maintenance_plan, f, indent=2)
    
    print()
    
    # 8. Show Key Insights
    print("🎯 Key Insights:")
    print()
    
    # Most complex components
    if complexity_report.components:
        most_complex = max(complexity_report.components, key=lambda c: c.complexity_score)
        print(f"   📊 Most complex component: {most_complex.name} (score: {most_complex.complexity_score})")
    
    # Critical dependencies
    if relationships.critical_components:
        print(f"   🔗 Most critical component: {relationships.critical_components[0]}")
    
    # Documentation gaps
    if complexity_report.documentation_gaps:
        print(f"   📚 Components needing documentation: {len(complexity_report.documentation_gaps)}")
    
    # Entry points
    if relationships.entry_points:
        print(f"   🚪 Application entry points: {', '.join(relationships.entry_points)}")
    
    print()
    print("✅ Analysis Complete!")
    print(f"📁 Check the results in: {output_dir}")


if __name__ == "__main__":
    main()
