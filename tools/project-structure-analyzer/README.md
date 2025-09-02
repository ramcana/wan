# Project Structure Analysis Engine

A comprehensive tool for analyzing project structure, component relationships, and code complexity. This engine provides detailed insights into project organization and generates documentation to help developers understand complex codebases.

## Features

### 🏗️ Structure Analysis

- **Directory Mapping**: Scans and maps the entire project structure
- **Component Identification**: Identifies main components and their purposes
- **File Classification**: Categorizes files by type (config, docs, tests, scripts)
- **Entry Point Detection**: Finds application entry points and main files
- **Size Analysis**: Calculates file and directory sizes

### 🔗 Component Relationship Analysis

- **Dependency Mapping**: Analyzes Python imports and dependencies
- **API Interaction Detection**: Identifies API calls and service interactions
- **Configuration References**: Tracks configuration file usage
- **Circular Dependency Detection**: Finds problematic circular dependencies
- **Critical Component Identification**: Identifies heavily-used components

### 📊 Complexity Analysis

- **Code Complexity Metrics**: Calculates cyclomatic and cognitive complexity
- **Documentation Coverage**: Measures documentation quality
- **Priority Assessment**: Identifies areas needing immediate attention
- **Refactoring Recommendations**: Suggests improvements

### 📈 Visualization Generation

- **Mermaid Diagrams**: Generates visual project structure diagrams
- **Dependency Graphs**: Shows component relationships
- **Complexity Heatmaps**: Visualizes complexity distribution

## Installation

The tool is part of the project's tools directory. No additional installation required.

## Usage

### Command Line Interface

```bash
# Basic analysis
python -m tools.project-structure-analyzer /path/to/project

# Specify output directory
python -m tools.project-structure-analyzer /path/to/project --output ./analysis-results

# Skip specific analyses
python -m tools.project-structure-analyzer /path/to/project --skip-complexity --skip-visualizations
```

### Programmatic Usage

```python
from tools.project_structure_analyzer import (
    ProjectStructureAnalyzer,
    ComponentRelationshipAnalyzer,
    ProjectComplexityAnalyzer,
    MermaidVisualizationGenerator
)

# Analyze project structure
analyzer = ProjectStructureAnalyzer("/path/to/project")
structure = analyzer.analyze()

# Generate report
report = analyzer.generate_summary_report(structure)
print(report)
```

## Output Files

The analysis generates several output files:

### Reports

- `PROJECT_ANALYSIS_REPORT.md` - Master summary report
- `structure_report.md` - Detailed structure analysis
- `relationships_report.md` - Component dependency analysis
- `complexity_report.md` - Code complexity analysis

### Data Files

- `structure_analysis.json` - Raw structure data
- `relationships_analysis.json` - Raw relationship data
- `complexity_analysis.json` - Raw complexity data

### Visualizations

- `diagrams/project_structure_overview.mmd` - Project structure diagram
- `diagrams/component_dependencies.mmd` - Dependency graph
- `diagrams/complexity_heatmap.mmd` - Complexity visualization
- `diagrams/README.md` - Diagram documentation

## Analysis Metrics

### Structure Metrics

- File and directory counts
- Component identification and classification
- Entry point detection
- Configuration file mapping

### Relationship Metrics

- Import dependencies
- API call relationships
- Configuration references
- Circular dependency detection

### Complexity Metrics

- **Cyclomatic Complexity**: Measures decision points in code
- **Cognitive Complexity**: Measures how hard code is to understand
- **Documentation Ratio**: Percentage of documented code
- **Component Complexity Score**: Overall complexity assessment

## Understanding the Results

### Priority Levels

- **High**: Requires immediate attention (complexity > 40, doc coverage < 30%)
- **Medium**: Should be addressed soon (complexity > 25, doc coverage < 50%)
- **Low**: Acceptable current state

### Component Types

- **Package**: Python package with `__init__.py`
- **Module**: Directory with Python files
- **Service**: API or service-related component
- **Directory**: General directory grouping

### Dependency Types

- **Import**: Python import statements
- **API Call**: HTTP API interactions
- **Config**: Configuration file references
- **File Reference**: Direct file path references

## Recommendations

The tool provides actionable recommendations:

1. **Documentation Improvements**: Areas needing better documentation
2. **Complexity Reduction**: Files/components to refactor
3. **Dependency Management**: Circular dependencies to resolve
4. **Structure Organization**: Consolidation opportunities

## Integration

This tool integrates with the broader project quality improvement system:

- **Test Quality Tools**: Identifies untested complex code
- **Configuration Management**: Maps configuration dependencies
- **Documentation Generator**: Provides input for doc generation
- **Health Checker**: Feeds into overall project health metrics

## Requirements

- Python 3.7+
- Standard library only (no external dependencies)
- Read access to project files

## Limitations

- Python-focused analysis (other languages have limited support)
- Large projects may take several minutes to analyze
- Complex circular dependencies may not be fully resolved
- Documentation analysis is heuristic-based

## Contributing

To extend the analyzer:

1. Add new metrics to the appropriate analyzer class
2. Update the CLI to expose new options
3. Add visualization support for new metrics
4. Update documentation and examples

## Examples

See the `examples/` directory for sample analyses and integration patterns.
