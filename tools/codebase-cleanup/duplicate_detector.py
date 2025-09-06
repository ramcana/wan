import pytest
"""
Duplicate Detection and Removal System

This module provides comprehensive duplicate file and code detection capabilities
with safe removal and backup functionality.
"""

import os
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import difflib
import ast
import re


@dataclass
class DuplicateFile:
    """Represents a duplicate file with metadata"""
    path: str
    size: int
    hash: str
    content_hash: str
    similarity_score: float
    duplicate_group: str


@dataclass
class DuplicateReport:
    """Report of duplicate detection analysis"""
    total_files_scanned: int
    duplicate_files: List[DuplicateFile]
    duplicate_groups: Dict[str, List[str]]
    potential_savings: int
    recommendations: List[str]
    scan_timestamp: str


class DuplicateDetector:
    """
    Comprehensive duplicate detection system that identifies:
    - Exact file duplicates (by hash)
    - Near-duplicate files (by content similarity)
    - Code similarity (by AST comparison)
    """
    
    def __init__(self, root_path: str, backup_dir: str = "backups/duplicates"):
        self.root_path = Path(root_path)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # File extensions to analyze for code similarity
        self.code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h'}
        
        # Files/directories to exclude from analysis
        self.exclude_patterns = {
            '__pycache__', '.git', '.pytest_cache', 'node_modules', 
            '.venv', 'venv', '.env', 'dist', 'build', '.next'
        }
    
    def scan_for_duplicates(self) -> DuplicateReport:
        """
        Perform comprehensive duplicate detection scan
        
        Returns:
            DuplicateReport with all findings and recommendations
        """
        print("Starting duplicate detection scan...")
        
        # Get all files to analyze
        files_to_scan = self._get_files_to_scan()
        print(f"Scanning {len(files_to_scan)} files...")
        
        # Calculate file hashes
        file_hashes = self._calculate_file_hashes(files_to_scan)
        
        # Find exact duplicates
        exact_duplicates = self._find_exact_duplicates(file_hashes)
        
        # Find near-duplicates for code files
        near_duplicates = self._find_near_duplicates(files_to_scan)
        
        # Combine results
        all_duplicates = self._combine_duplicate_results(exact_duplicates, near_duplicates)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_duplicates)
        
        # Calculate potential savings
        potential_savings = self._calculate_potential_savings(all_duplicates)
        
        report = DuplicateReport(
            total_files_scanned=len(files_to_scan),
            duplicate_files=all_duplicates,
            duplicate_groups=self._group_duplicates(all_duplicates),
            potential_savings=potential_savings,
            recommendations=recommendations,
            scan_timestamp=datetime.now().isoformat()
        )
        
        print(f"Scan complete. Found {len(all_duplicates)} duplicate files.")
        return report
    
    def _get_files_to_scan(self) -> List[Path]:
        """Get list of files to scan, excluding patterns"""
        files = []
        
        for root, dirs, filenames in os.walk(self.root_path):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.exclude_patterns)]
            
            for filename in filenames:
                file_path = Path(root) / filename
                
                # Skip if file matches exclude patterns
                if any(pattern in str(file_path) for pattern in self.exclude_patterns):
                    continue
                
                # Skip very small files (likely not meaningful duplicates)
                try:
                    if file_path.stat().st_size < 10:
                        continue
                except (OSError, FileNotFoundError):
                    continue
                
                files.append(file_path)
        
        return files
    
    def _calculate_file_hashes(self, files: List[Path]) -> Dict[Path, Tuple[str, str, int]]:
        """
        Calculate both file hash and content hash for each file
        
        Returns:
            Dict mapping file path to (file_hash, content_hash, size)
        """
        file_hashes = {}
        
        for file_path in files:
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                # File hash (includes metadata)
                file_hash = hashlib.md5(content).hexdigest()
                
                # Content hash (normalized content for better similarity detection)
                normalized_content = self._normalize_content(content, file_path.suffix)
                content_hash = hashlib.md5(normalized_content.encode('utf-8')).hexdigest()
                
                file_hashes[file_path] = (file_hash, content_hash, len(content))
                
            except (OSError, UnicodeDecodeError, PermissionError):
                # Skip files that can't be read
                continue
        
        return file_hashes
    
    def _normalize_content(self, content: bytes, file_extension: str) -> str:
        """
        Normalize file content for better similarity detection
        """
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            # For binary files, use raw content
            return content.hex()
        
        if file_extension in self.code_extensions:
            # For code files, normalize whitespace and comments
            return self._normalize_code_content(text_content, file_extension)
        else:
            # For other text files, normalize whitespace
            return re.sub(r'\s+', ' ', text_content.strip())
    
    def _normalize_code_content(self, content: str, file_extension: str) -> str:
        """Normalize code content by removing comments and normalizing whitespace"""
        if file_extension == '.py':
            return self._normalize_python_content(content)
        elif file_extension in {'.js', '.ts', '.jsx', '.tsx'}:
            return self._normalize_javascript_content(content)
        else:
            # Generic normalization
            return re.sub(r'\s+', ' ', content.strip())
    
    def _normalize_python_content(self, content: str) -> str:
        """Normalize Python code content"""
        try:
            # Parse AST to get structure without comments/whitespace
            tree = ast.parse(content)
            return ast.dump(tree, indent=2)
        except SyntaxError:
            # If parsing fails, use text normalization
            lines = []
            for line in content.split('\n'):
                # Remove comments and normalize whitespace
                line = re.sub(r'#.*$', '', line).strip()
                if line:
                    lines.append(line)
            return '\n'.join(lines)
    
    def _normalize_javascript_content(self, content: str) -> str:
        """Normalize JavaScript/TypeScript content"""
        # Remove single-line comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        # Remove multi-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content.strip())
        return content
    
    def _find_exact_duplicates(self, file_hashes: Dict[Path, Tuple[str, str, int]]) -> List[DuplicateFile]:
        """Find files with identical hashes"""
        hash_groups = {}
        
        for file_path, (file_hash, content_hash, size) in file_hashes.items():
            if file_hash not in hash_groups:
                hash_groups[file_hash] = []
            hash_groups[file_hash].append((file_path, content_hash, size))
        
        duplicates = []
        for file_hash, files in hash_groups.items():
            if len(files) > 1:
                group_id = f"exact_{file_hash[:8]}"
                for file_path, content_hash, size in files:
                    duplicates.append(DuplicateFile(
                        path=str(file_path),
                        size=size,
                        hash=file_hash,
                        content_hash=content_hash,
                        similarity_score=1.0,
                        duplicate_group=group_id
                    ))
        
        return duplicates
    
    def _find_near_duplicates(self, files: List[Path]) -> List[DuplicateFile]:
        """Find near-duplicate files using content similarity"""
        near_duplicates = []
        code_files = [f for f in files if f.suffix in self.code_extensions]
        
        # Compare each pair of code files
        for i, file1 in enumerate(code_files):
            for file2 in code_files[i+1:]:
                similarity = self._calculate_similarity(file1, file2)
                
                if similarity > 0.8:  # 80% similarity threshold
                    group_id = f"similar_{hashlib.md5(f'{file1}{file2}'.encode()).hexdigest()[:8]}"
                    
                    # Add both files to near-duplicates if not already added
                    file1_size = file1.stat().st_size
                    file2_size = file2.stat().st_size
                    
                    near_duplicates.extend([
                        DuplicateFile(
                            path=str(file1),
                            size=file1_size,
                            hash="",
                            content_hash="",
                            similarity_score=similarity,
                            duplicate_group=group_id
                        ),
                        DuplicateFile(
                            path=str(file2),
                            size=file2_size,
                            hash="",
                            content_hash="",
                            similarity_score=similarity,
                            duplicate_group=group_id
                        )
                    ])
        
        return near_duplicates
    
    def _calculate_similarity(self, file1: Path, file2: Path) -> float:
        """Calculate similarity between two files"""
        try:
            with open(file1, 'r', encoding='utf-8') as f1:
                content1 = f1.read()
            with open(file2, 'r', encoding='utf-8') as f2:
                content2 = f2.read()
            
            # Normalize content
            norm1 = self._normalize_content(content1.encode(), file1.suffix)
            norm2 = self._normalize_content(content2.encode(), file2.suffix)
            
            # Calculate similarity using difflib
            similarity = difflib.SequenceMatcher(None, norm1, norm2).ratio()
            return similarity
            
        except (OSError, UnicodeDecodeError):
            return 0.0
    
    def _combine_duplicate_results(self, exact: List[DuplicateFile], near: List[DuplicateFile]) -> List[DuplicateFile]:
        """Combine exact and near-duplicate results, removing overlaps"""
        # Use set to track already included files
        included_files = set()
        combined = []
        
        # Add exact duplicates first (higher priority)
        for dup in exact:
            if dup.path not in included_files:
                combined.append(dup)
                included_files.add(dup.path)
        
        # Add near duplicates that aren't already included
        for dup in near:
            if dup.path not in included_files:
                combined.append(dup)
                included_files.add(dup.path)
        
        return combined
    
    def _group_duplicates(self, duplicates: List[DuplicateFile]) -> Dict[str, List[str]]:
        """Group duplicates by their duplicate_group"""
        groups = {}
        for dup in duplicates:
            if dup.duplicate_group not in groups:
                groups[dup.duplicate_group] = []
            groups[dup.duplicate_group].append(dup.path)
        return groups
    
    def _calculate_potential_savings(self, duplicates: List[DuplicateFile]) -> int:
        """Calculate potential disk space savings"""
        groups = self._group_duplicates(duplicates)
        total_savings = 0
        
        for group_files in groups.values():
            if len(group_files) > 1:
                # Keep the first file, count others as savings
                file_sizes = []
                for file_path in group_files:
                    try:
                        size = Path(file_path).stat().st_size
                        file_sizes.append(size)
                    except OSError:
                        continue
                
                if file_sizes:
                    # Savings = total size - size of file to keep
                    total_savings += sum(file_sizes) - min(file_sizes)
        
        return total_savings
    
    def _generate_recommendations(self, duplicates: List[DuplicateFile]) -> List[str]:
        """Generate recommendations for handling duplicates"""
        recommendations = []
        groups = self._group_duplicates(duplicates)
        
        exact_groups = [g for g in groups.keys() if g.startswith('exact_')]
        similar_groups = [g for g in groups.keys() if g.startswith('similar_')]
        
        if exact_groups:
            recommendations.append(f"Found {len(exact_groups)} groups of exact duplicates - safe to remove")
        
        if similar_groups:
            recommendations.append(f"Found {len(similar_groups)} groups of similar files - review before removal")
        
        # Specific recommendations based on file types
        config_duplicates = [d for d in duplicates if any(ext in d.path.lower() for ext in ['.json', '.yaml', '.yml', '.ini'])]
        if config_duplicates:
            recommendations.append("Configuration file duplicates found - consolidate to avoid conflicts")
        
        test_duplicates = [d for d in duplicates if 'test' in d.path.lower()]
        if test_duplicates:
            recommendations.append("Test file duplicates found - may indicate copy-paste test patterns")
        
        return recommendations
    
    def create_backup(self, files_to_remove: List[str]) -> str:
        """
        Create backup of files before removal
        
        Returns:
            Path to backup directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"duplicate_removal_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        for file_path in files_to_remove:
            src_path = Path(file_path)
            if src_path.exists():
                # Preserve directory structure in backup
                rel_path = src_path.relative_to(self.root_path)
                backup_file_path = backup_path / rel_path
                backup_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                shutil.copy2(src_path, backup_file_path)
        
        # Create backup manifest
        manifest = {
            'timestamp': timestamp,
            'files': files_to_remove,
            'backup_path': str(backup_path)
        }
        
        with open(backup_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return str(backup_path)
    
    def safe_remove_duplicates(self, duplicate_groups: Dict[str, List[str]], 
                             auto_remove_exact: bool = True) -> Dict[str, str]:
        """
        Safely remove duplicate files with backup
        
        Args:
            duplicate_groups: Groups of duplicate files
            auto_remove_exact: Whether to automatically remove exact duplicates
            
        Returns:
            Dict mapping operation to result message
        """
        results = {}
        files_to_remove = []
        
        for group_id, group_files in duplicate_groups.items():
            if len(group_files) <= 1:
                continue
            
            if group_id.startswith('exact_') and auto_remove_exact:
                # For exact duplicates, keep the first file (usually shortest path)
                sorted_files = sorted(group_files, key=len)
                files_to_remove.extend(sorted_files[1:])
                results[group_id] = f"Marked {len(sorted_files)-1} exact duplicates for removal"
            
            elif group_id.startswith('similar_'):
                # For similar files, just report - don't auto-remove
                results[group_id] = f"Found {len(group_files)} similar files - manual review recommended"
        
        if files_to_remove:
            # Create backup
            backup_path = self.create_backup(files_to_remove)
            results['backup'] = f"Created backup at {backup_path}"
            
            # Remove files
            removed_count = 0
            for file_path in files_to_remove:
                try:
                    Path(file_path).unlink()
                    removed_count += 1
                except OSError as e:
                    results['errors'] = results.get('errors', [])
                    results['errors'].append(f"Failed to remove {file_path}: {e}")
            
            results['removal'] = f"Successfully removed {removed_count} duplicate files"
        
        return results
    
    def rollback_removal(self, backup_path: str) -> bool:
        """
        Rollback duplicate removal using backup
        
        Args:
            backup_path: Path to backup directory
            
        Returns:
            True if rollback successful, False otherwise
        """
        backup_dir = Path(backup_path)
        manifest_path = backup_dir / 'manifest.json'
        
        if not manifest_path.exists():
            print(f"Backup manifest not found at {manifest_path}")
            return False
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            restored_count = 0
            for file_path in manifest['files']:
                src_path = Path(file_path)
                rel_path = src_path.relative_to(self.root_path)
                backup_file_path = backup_dir / rel_path
                
                if backup_file_path.exists():
                    # Restore directory structure
                    src_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_file_path, src_path)
                    restored_count += 1
            
            print(f"Successfully restored {restored_count} files from backup")
            return True
            
        except Exception as e:
            print(f"Rollback failed: {e}")
            return False
    
    def save_report(self, report: DuplicateReport, output_path: str) -> None:
        """Save duplicate detection report to file"""
        with open(output_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"Report saved to {output_path}")


def main():
    """Example usage of DuplicateDetector"""
    detector = DuplicateDetector(".")
    
    # Scan for duplicates
    report = detector.scan_for_duplicates()
    
    # Save report
    detector.save_report(report, "duplicate_report.json")
    
    # Print summary
    print(f"\nDuplicate Detection Summary:")
    print(f"Files scanned: {report.total_files_scanned}")
    print(f"Duplicate files found: {len(report.duplicate_files)}")
    print(f"Potential savings: {report.potential_savings / 1024:.1f} KB")
    print(f"Duplicate groups: {len(report.duplicate_groups)}")
    
    for rec in report.recommendations:
        print(f"- {rec}")


if __name__ == "__main__":
    main()