---
title: tools.doc_generator.navigation_generator
category: api
tags: [api, tools]
---

# tools.doc_generator.navigation_generator

Navigation Generator

Automatically generates navigation menus and site structure
for WAN22 documentation based on file organization and metadata.

## Classes

### NavigationItem

Navigation menu item

### NavigationConfig

Configuration for navigation generation

### NavigationGenerator

Generates navigation structure for documentation

#### Methods

##### __init__(self: Any, docs_root: Path, config: NavigationConfig)



##### generate_navigation(self: Any) -> NavigationItem

Generate complete navigation structure

##### _discover_pages(self: Any) -> <ast.Subscript object at 0x00000194288D0AC0>

Discover all documentation pages

##### _should_skip_file(self: Any, file_path: Path) -> bool

Check if file should be skipped

##### _extract_page_info(self: Any, file_path: Path) -> <ast.Subscript object at 0x000001942835F640>

Extract page information for navigation

##### _parse_frontmatter(self: Any, content: str) -> <ast.Subscript object at 0x000001942835F0D0>

Parse YAML frontmatter

##### _extract_title_from_content(self: Any, content: str) -> str

Extract title from first heading

##### _infer_category(self: Any, file_path: Path) -> str

Infer category from file path

##### _determine_order(self: Any, file_path: Path, metadata: <ast.Subscript object at 0x0000019427F10C10>) -> int

Determine page order

##### _organize_by_category(self: Any, pages: <ast.Subscript object at 0x0000019427F126B0>) -> <ast.Subscript object at 0x0000019427F12CB0>

Organize pages by category

##### _find_home_page(self: Any, pages: <ast.Subscript object at 0x0000019427F11270>) -> <ast.Subscript object at 0x0000019427F11CC0>

Find the home/index page

##### _build_category_navigation(self: Any, category_name: str, pages: <ast.Subscript object at 0x0000019427F107C0>) -> <ast.Subscript object at 0x000001942C67FA60>

Build navigation for a category

##### _build_hierarchical_structure(self: Any, pages: <ast.Subscript object at 0x000001942C67FAF0>, category: str) -> <ast.Subscript object at 0x000001942C67F160>

Build hierarchical navigation structure

##### _convert_structure_to_navigation(self: Any, structure: <ast.Subscript object at 0x000001942C67D780>) -> <ast.Subscript object at 0x0000019428888940>

Convert nested structure to NavigationItem list

##### _format_directory_title(self: Any, dir_name: str) -> str

Format directory name as title

##### _get_directory_order(self: Any, dir_name: str) -> int

Get order for directory

##### generate_mkdocs_nav(self: Any, root: NavigationItem) -> <ast.Subscript object at 0x000001942888AA70>

Generate MkDocs navigation format

##### _convert_to_mkdocs_format(self: Any, item: NavigationItem) -> Any

Convert NavigationItem to MkDocs format

##### generate_sidebar_json(self: Any, root: NavigationItem) -> <ast.Subscript object at 0x000001942CCB5360>

Generate sidebar JSON for web interface

##### _convert_to_sidebar_format(self: Any, item: NavigationItem) -> <ast.Subscript object at 0x000001942CCB6C20>

Convert NavigationItem to sidebar JSON format

##### generate_breadcrumb_data(self: Any, current_path: str, root: NavigationItem) -> <ast.Subscript object at 0x000001942CCB6B90>

Generate breadcrumb data for a given path

##### _find_path_in_tree(self: Any, target_path: str, root: NavigationItem) -> <ast.Subscript object at 0x000001942CDD1690>

Find path to target in navigation tree

##### save_navigation_files(self: Any, root: NavigationItem, output_dir: Path)

Save navigation files in various formats

