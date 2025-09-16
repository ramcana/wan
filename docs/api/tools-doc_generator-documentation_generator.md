---
title: tools.doc_generator.documentation_generator
category: api
tags: [api, tools]
---

# tools.doc_generator.documentation_generator



## Classes

### DocumentationPage

Represents a documentation page with metadata

### APIDocumentation

Represents API documentation extracted from code

### MigrationReport

Report of documentation migration process

### DocumentationGenerator

Main class for consolidating existing documentation and generating API docs

#### Methods

##### __init__(self: Any, source_dirs: <ast.Subscript object at 0x000001942C653A00>, output_dir: Path)



##### discover_documentation_files(self: Any) -> <ast.Subscript object at 0x00000194285221D0>

Discover all documentation files in source directories

##### _should_exclude_file(self: Any, file_path: Path) -> bool

Check if file should be excluded from documentation

##### parse_documentation_file(self: Any, file_path: Path) -> <ast.Subscript object at 0x000001942C555630>

Parse a documentation file and extract metadata

##### _extract_title(self: Any, content: str, file_path: Path) -> str

Extract title from content or filename

##### _extract_frontmatter(self: Any, content: str) -> <ast.Subscript object at 0x000001942C554610>

Extract YAML frontmatter from content

##### _extract_links(self: Any, content: str) -> <ast.Subscript object at 0x000001942C557D90>

Extract markdown links from content

##### _determine_category(self: Any, file_path: Path, metadata: <ast.Subscript object at 0x000001942C554F70>) -> str

Determine documentation category

##### _extract_tags(self: Any, content: str, metadata: <ast.Subscript object at 0x000001942C554CD0>) -> <ast.Subscript object at 0x00000194284B2650>

Extract tags from content and metadata

##### generate_api_docs(self: Any, code_dirs: <ast.Subscript object at 0x00000194284B1A50>) -> <ast.Subscript object at 0x0000019428DDFD60>

Generate API documentation from code annotations

##### _parse_python_file(self: Any, file_path: Path) -> <ast.Subscript object at 0x0000019428DDD270>

Parse Python file and extract API documentation

##### _get_module_name(self: Any, file_path: Path) -> str

Get module name from file path

##### _extract_class_info(self: Any, node: ast.ClassDef) -> <ast.Subscript object at 0x0000019427FBF4C0>

Extract class information

##### _extract_function_info(self: Any, node: ast.FunctionDef) -> <ast.Subscript object at 0x0000019427FBDA50>

Extract function information

##### _extract_constant_info(self: Any, node: ast.Assign) -> <ast.Subscript object at 0x0000019427FBF5B0>

Extract constant information

##### _get_name(self: Any, node: ast.AST) -> str

Get name from AST node

##### consolidate_existing_docs(self: Any) -> MigrationReport

Consolidate existing documentation into unified structure

##### _get_target_path(self: Any, page: DocumentationPage) -> Path

Determine target path for migrated documentation

##### _clean_filename(self: Any, filename: str) -> str

Clean up filename for better organization

##### _migrate_file(self: Any, page: DocumentationPage, target_path: Path)

Migrate file to target location with updated content

##### _build_link_graph(self: Any)

Build link graph for cross-references

##### _find_broken_links(self: Any) -> <ast.Subscript object at 0x0000019427A6B040>

Find broken links in documentation

##### generate_index(self: Any) -> str

Generate main documentation index

##### generate_api_index(self: Any) -> str

Generate API documentation index

##### save_consolidated_docs(self: Any)

Save all consolidated documentation

##### _generate_detailed_api_doc(self: Any, api_doc: APIDocumentation, api_dir: Path)

Generate detailed API documentation for a module

