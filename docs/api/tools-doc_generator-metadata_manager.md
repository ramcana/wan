---
title: tools.doc_generator.metadata_manager
category: api
tags: [api, tools]
---

# tools.doc_generator.metadata_manager

Documentation Metadata Manager

Manages documentation metadata, cross-references, and relationships
between documentation pages.

## Classes

### DocumentMetadata

Metadata for a documentation page

### CrossReference

Cross-reference between documentation pages

### DocumentationIndex

Complete documentation index with metadata and relationships

### MetadataManager

Manages documentation metadata and cross-references

#### Methods

##### __init__(self: Any, docs_root: Path)



##### scan_documentation(self: Any) -> DocumentationIndex

Scan all documentation files and extract metadata

##### _should_skip_file(self: Any, file_path: Path) -> bool

Check if file should be skipped

##### _extract_metadata(self: Any, file_path: Path) -> <ast.Subscript object at 0x000001942FBC2440>

Extract metadata from a documentation file

##### _parse_frontmatter(self: Any, content: str) -> <ast.Subscript object at 0x000001942FBC1990>

Parse YAML frontmatter from content

##### _generate_basic_metadata(self: Any, file_path: Path) -> <ast.Subscript object at 0x000001942FBC2E60>

Generate basic metadata for files without frontmatter

##### _generate_title(self: Any, file_path: Path) -> str

Generate title from filename

##### _infer_category(self: Any, file_path: Path) -> str

Infer category from file path

##### _infer_tags(self: Any, file_path: Path) -> <ast.Subscript object at 0x000001942F42ED40>

Infer tags from filename and path

##### _extract_cross_references(self: Any, file_path: Path, relative_path: str) -> <ast.Subscript object at 0x0000019432E3B7C0>

Extract cross-references from a documentation file

##### _build_index(self: Any) -> DocumentationIndex

Build complete documentation index

##### save_index(self: Any, index: DocumentationIndex)

Save documentation index to file

##### load_index(self: Any) -> <ast.Subscript object at 0x0000019432E38430>

Load documentation index from file

##### update_page_metadata(self: Any, page_path: str, metadata: DocumentMetadata)

Update metadata for a specific page

##### _update_file_frontmatter(self: Any, file_path: Path, metadata: DocumentMetadata)

Update frontmatter in a documentation file

##### find_broken_references(self: Any) -> <ast.Subscript object at 0x0000019432DE5030>

Find cross-references that point to non-existent pages

##### suggest_related_pages(self: Any, page_path: str) -> <ast.Subscript object at 0x0000019432DE4730>

Suggest related pages based on tags and content

##### generate_navigation_menu(self: Any) -> <ast.Subscript object at 0x0000019432DE7610>

Generate navigation menu structure

