---
title: tools.codebase-cleanup.duplicate_detector
category: api
tags: [api, tools]
---

# tools.codebase-cleanup.duplicate_detector



## Classes

### DuplicateFile

Represents a duplicate file with metadata

### DuplicateReport

Report of duplicate detection analysis

### DuplicateDetector

Comprehensive duplicate detection system that identifies:
- Exact file duplicates (by hash)
- Near-duplicate files (by content similarity)
- Code similarity (by AST comparison)

#### Methods

##### __init__(self: Any, root_path: str, backup_dir: str)



##### scan_for_duplicates(self: Any) -> DuplicateReport

Perform comprehensive duplicate detection scan

Returns:
    DuplicateReport with all findings and recommendations

##### _get_files_to_scan(self: Any) -> <ast.Subscript object at 0x00000194319198A0>

Get list of files to scan, excluding patterns

##### _calculate_file_hashes(self: Any, files: <ast.Subscript object at 0x0000019431919750>) -> <ast.Subscript object at 0x0000019431B8FC40>

Calculate both file hash and content hash for each file

Returns:
    Dict mapping file path to (file_hash, content_hash, size)

##### _normalize_content(self: Any, content: bytes, file_extension: str) -> str

Normalize file content for better similarity detection

##### _normalize_code_content(self: Any, content: str, file_extension: str) -> str

Normalize code content by removing comments and normalizing whitespace

##### _normalize_python_content(self: Any, content: str) -> str

Normalize Python code content

##### _normalize_javascript_content(self: Any, content: str) -> str

Normalize JavaScript/TypeScript content

##### _find_exact_duplicates(self: Any, file_hashes: <ast.Subscript object at 0x0000019431B8D780>) -> <ast.Subscript object at 0x000001942F791000>

Find files with identical hashes

##### _find_near_duplicates(self: Any, files: <ast.Subscript object at 0x000001942F790EB0>) -> <ast.Subscript object at 0x000001942F7AF7F0>

Find near-duplicate files using content similarity

##### _calculate_similarity(self: Any, file1: Path, file2: Path) -> float

Calculate similarity between two files

##### _combine_duplicate_results(self: Any, exact: <ast.Subscript object at 0x000001942FC32440>, near: <ast.Subscript object at 0x000001942FC324A0>) -> <ast.Subscript object at 0x000001942FC31A50>

Combine exact and near-duplicate results, removing overlaps

##### _group_duplicates(self: Any, duplicates: <ast.Subscript object at 0x000001942FC31900>) -> <ast.Subscript object at 0x000001942FC310C0>

Group duplicates by their duplicate_group

##### _calculate_potential_savings(self: Any, duplicates: <ast.Subscript object at 0x000001942FC31150>) -> int

Calculate potential disk space savings

##### _generate_recommendations(self: Any, duplicates: <ast.Subscript object at 0x000001942FC30430>) -> <ast.Subscript object at 0x000001942FBADC00>

Generate recommendations for handling duplicates

##### create_backup(self: Any, files_to_remove: <ast.Subscript object at 0x000001942FBADAB0>) -> str

Create backup of files before removal

Returns:
    Path to backup directory

##### safe_remove_duplicates(self: Any, duplicate_groups: <ast.Subscript object at 0x000001942FBAC6D0>, auto_remove_exact: bool) -> <ast.Subscript object at 0x000001942FBC1DB0>

Safely remove duplicate files with backup

Args:
    duplicate_groups: Groups of duplicate files
    auto_remove_exact: Whether to automatically remove exact duplicates
    
Returns:
    Dict mapping operation to result message

##### rollback_removal(self: Any, backup_path: str) -> bool

Rollback duplicate removal using backup

Args:
    backup_path: Path to backup directory
    
Returns:
    True if rollback successful, False otherwise

##### save_report(self: Any, report: DuplicateReport, output_path: str) -> None

Save duplicate detection report to file

