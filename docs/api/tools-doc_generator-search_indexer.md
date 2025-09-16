---
title: tools.doc_generator.search_indexer
category: api
tags: [api, tools]
---

# tools.doc_generator.search_indexer

Documentation Search Indexer

Advanced search indexing for WAN22 documentation with full-text search,
tag-based filtering, and content categorization.

## Classes

### SearchDocument

Document for search indexing

### SearchResult

Search result with relevance scoring

### SearchIndexer

Advanced search indexer for documentation

#### Methods

##### __init__(self: Any, docs_root: Path, index_db_path: <ast.Subscript object at 0x00000194278E7670>)



##### _init_database(self: Any)

Initialize SQLite database for search index

##### index_documentation(self: Any) -> <ast.Subscript object at 0x000001942796FB50>

Index all documentation files

##### _should_skip_file(self: Any, file_path: Path) -> bool

Check if file should be skipped

##### _needs_reindexing(self: Any, file_path: Path) -> bool

Check if file needs to be reindexed

##### _process_document(self: Any, file_path: Path) -> <ast.Subscript object at 0x000001942796C9D0>

Process a documentation file for indexing

##### _extract_metadata(self: Any, content: str) -> <ast.Subscript object at 0x00000194278FD360>

Extract YAML frontmatter metadata

##### _extract_title_from_content(self: Any, content: str) -> str

Extract title from first heading

##### _clean_content_for_indexing(self: Any, content: str) -> str

Clean content for search indexing

##### _extract_headings(self: Any, content: str) -> <ast.Subscript object at 0x000001942888A230>

Extract all headings from content

##### _extract_code_blocks(self: Any, content: str) -> <ast.Subscript object at 0x00000194288D0F70>

Extract code blocks from content

##### _generate_document_id(self: Any, relative_path: str) -> str

Generate unique document ID

##### _store_document(self: Any, document: SearchDocument)

Store document in search index

##### _index_document_terms(self: Any, conn: sqlite3.Connection, document: SearchDocument)

Index individual terms for a document

##### _tokenize_content(self: Any, content: str) -> <ast.Subscript object at 0x000001942C6D25C0>

Tokenize content into searchable terms

##### _document_exists(self: Any, doc_id: str) -> bool

Check if document already exists in index

##### _get_document_count(self: Any) -> int

Get total number of indexed documents

##### _update_fts_index(self: Any)

Update full-text search index

##### search(self: Any, query: str, category: <ast.Subscript object at 0x000001942CD8A170>, tags: <ast.Subscript object at 0x000001942CD8A260>, limit: int) -> <ast.Subscript object at 0x000001942CCF7010>

Search documentation with advanced filtering

##### _fts_search(self: Any, query: str, category: <ast.Subscript object at 0x000001942CCF6C50>, tags: <ast.Subscript object at 0x000001942CCF6B90>, limit: int) -> <ast.Subscript object at 0x00000194279027D0>

Full-text search using SQLite FTS

##### _build_fts_query(self: Any, query: str) -> str

Build FTS query from user input

##### _create_document_from_data(self: Any, data: <ast.Subscript object at 0x0000019427903460>) -> SearchDocument

Create SearchDocument from database row

##### _calculate_relevance_score(self: Any, document: SearchDocument, query: str) -> float

Calculate relevance score for search result

##### _find_matches(self: Any, document: SearchDocument, query: str) -> <ast.Subscript object at 0x0000019428AAE950>

Find specific matches in document

##### _generate_snippet(self: Any, document: SearchDocument, query: str, max_length: int) -> str

Generate search result snippet

##### get_search_suggestions(self: Any, partial_query: str, limit: int) -> <ast.Subscript object at 0x0000019428A78F10>

Get search suggestions based on partial query

##### get_index_stats(self: Any) -> <ast.Subscript object at 0x0000019428A7A170>

Get search index statistics

