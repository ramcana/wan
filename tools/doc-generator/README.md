# WAN22 Documentation Generator

Comprehensive documentation generation, validation, and management tools for the WAN22 project.

## Overview

The Documentation Generator provides a complete suite of tools for managing project documentation:

- **Documentation Generation**: Consolidate scattered docs and generate API documentation
- **Static Site Generation**: MkDocs-based documentation server with search
- **Validation**: Comprehensive link checking and content validation
- **Search & Indexing**: Full-text search with advanced filtering
- **Navigation Generation**: Automatic menu and navigation structure
- **Migration Tools**: Migrate scattered documentation to unified structure

## Quick Start

### Complete Setup (Recommended)

Run the complete documentation pipeline:

```bash
# Install dependencies and set up everything
python tools/doc-generator/cli.py all

# Start development server
python tools/doc-generator/cli.py serve
```

Visit `http://localhost:8000` to view your documentation.

### Individual Commands

```bash
# Generate consolidated documentation
python tools/doc-generator/cli.py generate --generate-api

# Validate documentation
python tools/doc-generator/cli.py validate --output report.html --format html

# Start documentation server
python tools/doc-generator/cli.py serve --setup

# Search documentation
python tools/doc-generator/cli.py search index
python tools/doc-generator/cli.py search search --query "installation"
```

## Tools Overview

### 1. Documentation Generator (`documentation_generator.py`)

Consolidates scattered documentation and generates API docs from code annotations.

**Features:**

- Discovers and consolidates existing documentation
- Generates API documentation from Python code
- Creates unified documentation structure
- Handles metadata and cross-references

**Usage:**

```bash
python tools/doc-generator/generate_docs.py --generate-api
```

### 2. Documentation Server (`server.py`)

MkDocs-based static site generator with development server.

**Features:**

- Static site generation with MkDocs
- Development server with live reload
- Custom theme and styling
- Search functionality
- Navigation generation

**Usage:**

```bash
python tools/doc-generator/server.py setup
python tools/doc-generator/server.py serve
```

### 3. Documentation Validator (`validator.py`)

Comprehensive validation for links, content quality, and style compliance.

**Features:**

- Internal and external link checking
- Content style validation
- Metadata validation
- Freshness checking
- HTML and JSON reports

**Usage:**

```bash
python tools/doc-generator/validator.py --output report.html --format html
```

### 4. Search Indexer (`search_indexer.py`)

Advanced search indexing with full-text search capabilities.

**Features:**

- SQLite-based search index
- Full-text search with FTS5
- Tag and category filtering
- Search suggestions
- Performance metrics

**Usage:**

```bash
python tools/doc-generator/search_indexer.py index
python tools/doc-generator/search_indexer.py search --query "configuration"
```

### 5. Navigation Generator (`navigation_generator.py`)

Automatic navigation structure generation from file organization.

**Features:**

- Hierarchical navigation from file structure
- MkDocs navigation format
- Sidebar JSON for web interfaces
- Breadcrumb generation
- Custom ordering and icons

**Usage:**

```bash
python tools/doc-generator/navigation_generator.py --format all
```

### 6. Metadata Manager (`metadata_manager.py`)

Documentation metadata management and cross-reference tracking.

**Features:**

- Frontmatter metadata extraction
- Cross-reference tracking
- Broken link detection
- Navigation menu generation
- Metadata validation

**Usage:**

```bash
python tools/doc-generator/metadata_manager.py --scan
```

### 7. Migration Tool (`migration_tool.py`)

Migrate scattered documentation to unified structure.

**Features:**

- Automatic file discovery
- Smart categorization
- Metadata generation
- Duplicate handling
- Migration reporting

**Usage:**

```bash
python tools/doc-generator/migration_tool.py --source . --target docs
```

## Configuration

### MkDocs Configuration

The server automatically generates `mkdocs.yml` with:

- Material theme with dark/light mode
- Search functionality
- Navigation structure
- Markdown extensions
- Custom styling

### Validation Configuration

Create `validation_config.yaml`:

```yaml
check_links: true
check_external_links: true
check_style: true
check_freshness: true
external_link_timeout: 10
freshness_threshold_days: 90
max_line_length: 120
required_metadata_fields:
  - title
  - category
  - last_updated
allowed_categories:
  - user-guide
  - developer-guide
  - api
  - deployment
  - reference
```

### Search Configuration

Search indexing is automatic with configurable:

- Stop words filtering
- Content preprocessing
- Index optimization
- Search result ranking

## Documentation Structure

The tools expect and create this structure:

```
docs/
├── index.md                    # Main documentation index
├── user-guide/                 # User documentation
│   ├── index.md
│   ├── installation.md
│   ├── configuration.md
│   └── troubleshooting.md
├── developer-guide/            # Developer documentation
│   ├── index.md
│   ├── architecture.md
│   ├── contributing.md
│   └── testing.md
├── api/                        # API documentation
│   ├── index.md
│   ├── backend-api.md
│   └── frontend-components.md
├── deployment/                 # Deployment guides
│   ├── index.md
│   ├── production-setup.md
│   └── monitoring.md
├── reference/                  # Reference materials
│   ├── index.md
│   ├── configuration/
│   ├── troubleshooting/
│   └── performance/
├── templates/                  # Documentation templates
│   ├── page-template.md
│   ├── api-template.md
│   └── troubleshooting-template.md
├── style-guide.md             # Documentation style guide
└── .metadata/                 # Generated metadata
    ├── index.json
    ├── navigation.json
    └── validation_report.json
```

## Metadata Format

All documentation files should include YAML frontmatter:

```yaml
---
title: Page Title
category: user-guide
tags: [tag1, tag2, tag3]
last_updated: 2024-01-01
author: Author Name
status: published
---
```

## Integration with CI/CD

### GitHub Actions

Add to `.github/workflows/docs.yml`:

```yaml
name: Documentation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          python tools/doc-generator/server.py install

      - name: Validate documentation
        run: |
          python tools/doc-generator/cli.py validate --output validation-report.json

      - name: Build documentation
        run: |
          python tools/doc-generator/cli.py serve --build-only

      - name: Deploy to GitHub Pages
        if: github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site
```

### Pre-commit Hooks

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: doc-validation
        name: Documentation Validation
        entry: python tools/doc-generator/cli.py validate --links-only
        language: python
        files: \.md$
```

## Troubleshooting

### Common Issues

1. **MkDocs not found**

   ```bash
   python tools/doc-generator/server.py install
   ```

2. **Broken links in validation**

   ```bash
   python tools/doc-generator/cli.py validate --links-only --output links-report.html --format html
   ```

3. **Search index issues**

   ```bash
   python tools/doc-generator/cli.py search index
   ```

4. **Navigation not updating**
   ```bash
   python tools/doc-generator/cli.py navigation --format all
   ```

### Performance Tips

- Use `--external-only` for faster link checking during development
- Run full validation in CI/CD only
- Index search incrementally for large documentation sets
- Use caching for external link validation

## Development

### Adding New Validators

Extend `DocumentationValidator` class:

```python
def _validate_custom_rule(self, content: str, file_path: str):
    # Custom validation logic
    if condition:
        self.issues.append(ValidationIssue(
            severity='warning',
            category='custom',
            message='Custom validation message',
            file_path=file_path,
            suggestion='How to fix this issue'
        ))
```

### Custom Search Filters

Extend `SearchIndexer` class:

```python
def search_with_custom_filter(self, query: str, custom_filter: str):
    # Custom search logic
    pass
```

### Adding New Migration Rules

Extend `DocumentationMigrator` class:

```python
self.migration_rules.append(MigrationRule(
    pattern="CUSTOM_*",
    target_category="custom-category",
    target_name="custom-name.md"
))
```

## Dependencies

- Python 3.8+
- MkDocs and extensions
- PyYAML
- Requests (for link checking)
- SQLite3 (built-in)

Install all dependencies:

```bash
pip install mkdocs mkdocs-material mkdocs-awesome-pages-plugin mkdocs-git-revision-date-localized-plugin pymdown-extensions pyyaml requests
```

## License

Part of the WAN22 project. See main project license for details.

## Contributing

1. Follow the documentation style guide
2. Add tests for new functionality
3. Update this README for new features
4. Validate documentation before submitting PRs

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review validation reports
3. Check existing documentation
4. Create an issue with detailed information
