---
title: tools.doc_generator.migration_tool
category: api
tags: [api, tools]
---

# tools.doc_generator.migration_tool

Documentation Migration Tool

Specialized tool for migrating scattered documentation files
to a unified structure with proper categorization and metadata.

## Classes

### MigrationRule

Rule for migrating documentation files

### DocumentationMigrator

Tool for migrating scattered documentation to unified structure

#### Methods

##### __init__(self: Any, source_root: Path, target_root: Path)



##### discover_scattered_docs(self: Any) -> <ast.Subscript object at 0x0000019428879540>

Discover all documentation files that need migration

##### _is_documentation_file(self: Any, file_path: Path) -> bool

Determine if a file is documentation based on name and content

##### categorize_file(self: Any, file_path: Path) -> <ast.Subscript object at 0x000001942B2F5270>

Categorize a file and determine its target location

##### _matches_pattern(self: Any, filename: str, pattern: str) -> bool

Check if filename matches pattern (simple glob-like matching)

##### _generate_target_name(self: Any, filename: str) -> str

Generate a clean target filename

##### _default_categorization(self: Any, file_path: Path) -> <ast.Subscript object at 0x0000019428D73EB0>

Default categorization for files that don't match rules

##### migrate_file(self: Any, source_path: Path, dry_run: bool) -> Dict

Migrate a single file to the target structure

##### _generate_metadata(self: Any, source_path: Path, category: str) -> Dict

Generate metadata for migrated file

##### _extract_title_from_filename(self: Any, filename: str) -> str

Extract a readable title from filename

##### _generate_tags(self: Any, source_path: Path, category: str) -> <ast.Subscript object at 0x0000019427B623E0>

Generate tags based on filename and category

##### migrate_all(self: Any, dry_run: bool) -> Dict

Migrate all discovered documentation files

##### generate_migration_report(self: Any, results: Dict) -> str

Generate a human-readable migration report

