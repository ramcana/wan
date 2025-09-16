---
title: tools.doc_generator.server
category: api
tags: [api, tools]
---

# tools.doc_generator.server

Documentation Server

Static site generator and server for WAN22 documentation with search functionality.
Uses MkDocs for static site generation and provides development server capabilities.

## Classes

### ServerConfig

Configuration for documentation server

### DocumentationServer

Documentation server using MkDocs for static site generation

#### Methods

##### __init__(self: Any, config: ServerConfig, project_root: Path)



##### generate_mkdocs_config(self: Any) -> <ast.Subscript object at 0x000001942EFFB790>

Generate MkDocs configuration

##### _generate_navigation_structure(self: Any) -> <ast.Subscript object at 0x000001942EFFA5C0>

Generate navigation structure from documentation files

##### save_mkdocs_config(self: Any)

Save MkDocs configuration to file

##### install_dependencies(self: Any)

Install required MkDocs dependencies

##### build_site(self: Any) -> bool

Build static documentation site

##### serve_dev(self: Any) -> bool

Start development server with live reload

##### generate_search_index(self: Any) -> bool

Generate search index for documentation

##### _should_skip_file(self: Any, file_path: Path) -> bool

Check if file should be skipped for indexing

##### _extract_content_for_search(self: Any, content: str) -> <ast.Subscript object at 0x000001942FE0FDF0>

Extract title and searchable content from markdown

##### create_custom_theme(self: Any)

Create custom theme files for WAN22 branding

##### setup_complete_server(self: Any) -> bool

Complete server setup including dependencies, config, and build

