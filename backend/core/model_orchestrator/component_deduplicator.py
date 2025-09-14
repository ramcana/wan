"""
Component Deduplication System - Content-addressed storage for shared model components.

This module implements a deduplication system that identifies common files across
models and creates hardlinks/symlinks to save disk space while maintaining
reference tracking to prevent premature deletion.
"""

import os
import json
import hashlib
import shutil
import platform
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import time
import threading

from .exceptions import ModelOrchestratorError, ErrorCode
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ComponentInfo:
    """Information about a shared component."""
    
    name: str                         # Component name (e.g., "text_encoder", "tokenizer")
    version: str                      # Component version/hash
    content_hash: str                 # SHA256 hash of content
    size: int                         # Size in bytes
    original_path: str                # Original file path within model
    shared_path: str                  # Path in shared component store
    references: Set[str]              # Set of model_ids that reference this component
    created_at: float                 # Timestamp when component was created
    last_accessed: float              # Timestamp of last access


@dataclass
class DeduplicationResult:
    """Result of a deduplication operation."""
    
    total_files_processed: int
    duplicates_found: int
    bytes_saved: int
    links_created: int
    errors: List[str]
    processing_time: float


class ComponentDeduplicator:
    """
    Manages component deduplication with content-addressed storage.
    
    Features:
    - Content-addressed storage for shared components
    - Hardlink/symlink creation based on platform capabilities
    - Reference tracking to prevent premature deletion
    - Cross-platform compatibility (Windows junctions, Unix hardlinks/symlinks)
    """
    
    def __init__(self, models_root: str):
        """
        Initialize the component deduplicator.
        
        Args:
            models_root: Base directory for model storage
        """
        self.models_root = Path(models_root)
        self.components_root = self.models_root / "components"
        self.metadata_file = self.components_root / ".component_registry.json"
        
        # Platform detection for link strategy
        self.is_windows = platform.system() == "Windows"
        self.supports_hardlinks = self._check_hardlink_support()
        self.supports_symlinks = self._check_symlink_support()
        
        # Thread lock for metadata operations
        self._metadata_lock = threading.Lock()
        
        # In-memory cache of component metadata
        self._component_cache: Dict[str, ComponentInfo] = {}
        self._cache_loaded = False
        
        # Common component patterns for WAN2.2 models
        self.common_components = {
            "tokenizer": ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
            "text_encoder": ["config.json", "pytorch_model.bin", "pytorch_model.safetensors"],
            "scheduler": ["scheduler_config.json"],
            "feature_extractor": ["preprocessor_config.json"],
            "safety_checker": ["config.json", "pytorch_model.bin", "pytorch_model.safetensors"]
        }
        
        logger.info(
            "ComponentDeduplicator initialized",
            extra={
                "models_root": str(self.models_root),
                "components_root": str(self.components_root),
                "supports_hardlinks": self.supports_hardlinks,
                "supports_symlinks": self.supports_symlinks,
                "platform": platform.system()
            }
        )
    
    def _check_hardlink_support(self) -> bool:
        """Check if the filesystem supports hardlinks."""
        try:
            # Create a test file and try to hardlink it
            test_dir = self.models_root / ".tmp" / "hardlink_test"
            test_dir.mkdir(parents=True, exist_ok=True)
            
            test_file = test_dir / "test.txt"
            test_link = test_dir / "test_link.txt"
            
            test_file.write_text("test")
            os.link(str(test_file), str(test_link))
            
            # Cleanup
            test_link.unlink()
            test_file.unlink()
            test_dir.rmdir()
            
            return True
        except (OSError, NotImplementedError):
            return False
    
    def _check_symlink_support(self) -> bool:
        """Check if the filesystem supports symbolic links."""
        try:
            # Create a test file and try to symlink it
            test_dir = self.models_root / ".tmp" / "symlink_test"
            test_dir.mkdir(parents=True, exist_ok=True)
            
            test_file = test_dir / "test.txt"
            test_link = test_dir / "test_symlink.txt"
            
            test_file.write_text("test")
            os.symlink(str(test_file), str(test_link))
            
            # Cleanup
            test_link.unlink()
            test_file.unlink()
            test_dir.rmdir()
            
            return True
        except (OSError, NotImplementedError):
            return False
    
    def _load_component_metadata(self) -> None:
        """Load component metadata from disk."""
        if self._cache_loaded:
            return
        
        with self._metadata_lock:
            if self._cache_loaded:  # Double-check after acquiring lock
                return
            
            try:
                if self.metadata_file.exists():
                    with open(self.metadata_file, 'r') as f:
                        data = json.load(f)
                    
                    # Convert loaded data back to ComponentInfo objects
                    for component_key, component_data in data.items():
                        # Convert references back to set
                        component_data['references'] = set(component_data['references'])
                        self._component_cache[component_key] = ComponentInfo(**component_data)
                    
                    logger.debug(f"Loaded {len(self._component_cache)} components from metadata")
                else:
                    logger.debug("No existing component metadata found")
                
                self._cache_loaded = True
                
            except Exception as e:
                logger.warning(f"Failed to load component metadata: {e}")
                self._component_cache = {}
                self._cache_loaded = True
    
    def _save_component_metadata(self) -> None:
        """Save component metadata to disk."""
        with self._metadata_lock:
            try:
                self.components_root.mkdir(parents=True, exist_ok=True)
                
                # Convert ComponentInfo objects to serializable format
                serializable_data = {}
                for component_key, component_info in self._component_cache.items():
                    data = asdict(component_info)
                    # Convert set to list for JSON serialization
                    data['references'] = list(data['references'])
                    serializable_data[component_key] = data
                
                # Write atomically
                temp_file = self.metadata_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(serializable_data, f, indent=2)
                
                # Atomic rename
                temp_file.replace(self.metadata_file)
                
                logger.debug(f"Saved metadata for {len(self._component_cache)} components")
                
            except Exception as e:
                logger.error(f"Failed to save component metadata: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            raise
    
    def _identify_component_type(self, file_path: str) -> Optional[str]:
        """Identify the component type based on file path patterns."""
        file_path_lower = file_path.lower()
        
        # Check for common directory-based components first (most specific)
        path_parts = Path(file_path).parts
        for part in path_parts:
            part_lower = part.lower()
            if part_lower in self.common_components:
                return part_lower
        
        # Check against known component patterns, but only if the directory context matches
        for component_type, patterns in self.common_components.items():
            # Check if the file is in a directory that suggests this component type
            if any(component_type in part.lower() for part in path_parts):
                for pattern in patterns:
                    if pattern.lower() in file_path_lower:
                        return component_type
        
        return None
    
    def _get_component_key(self, component_type: str, content_hash: str) -> str:
        """Generate a unique key for a component."""
        return f"{component_type}@{content_hash[:16]}"
    
    def _create_shared_component_path(self, component_key: str, original_path: str) -> Path:
        """Create the path for a shared component."""
        # Preserve the original filename
        filename = Path(original_path).name
        return self.components_root / component_key / filename
    
    def _create_link(self, source_path: Path, target_path: Path) -> bool:
        """
        Create a link from target to source, choosing the best method available.
        
        Args:
            source_path: Path to the original file (in shared storage)
            target_path: Path where the link should be created (in model directory)
            
        Returns:
            True if link was created successfully
        """
        try:
            # Ensure target directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Remove existing target if it exists
            if target_path.exists():
                target_path.unlink()
            
            # Try hardlink first (most efficient)
            if self.supports_hardlinks:
                try:
                    os.link(str(source_path), str(target_path))
                    logger.debug(f"Created hardlink: {target_path} -> {source_path}")
                    return True
                except OSError as e:
                    logger.debug(f"Hardlink failed, trying symlink: {e}")
            
            # Try symlink as fallback
            if self.supports_symlinks:
                try:
                    os.symlink(str(source_path), str(target_path))
                    logger.debug(f"Created symlink: {target_path} -> {source_path}")
                    return True
                except OSError as e:
                    logger.debug(f"Symlink failed, trying copy: {e}")
            
            # On Windows, try junction for directories or copy for files
            if self.is_windows:
                if source_path.is_dir():
                    try:
                        # Use mklink /J for directory junctions on Windows
                        import subprocess
                        result = subprocess.run([
                            'mklink', '/J', str(target_path), str(source_path)
                        ], shell=True, capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            logger.debug(f"Created junction: {target_path} -> {source_path}")
                            return True
                    except Exception as e:
                        logger.debug(f"Junction creation failed: {e}")
            
            # Fallback to copy (no space savings, but maintains functionality)
            if source_path.is_file():
                shutil.copy2(str(source_path), str(target_path))
                logger.debug(f"Copied file (no link support): {target_path}")
                return True
            elif source_path.is_dir():
                shutil.copytree(str(source_path), str(target_path))
                logger.debug(f"Copied directory (no link support): {target_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to create link {target_path} -> {source_path}: {e}")
            return False
    
    def deduplicate_model(self, model_id: str, model_path: Path) -> DeduplicationResult:
        """
        Deduplicate components within a single model.
        
        Args:
            model_id: Identifier for the model
            model_path: Path to the model directory
            
        Returns:
            DeduplicationResult with statistics
        """
        start_time = time.time()
        result = DeduplicationResult(
            total_files_processed=0,
            duplicates_found=0,
            bytes_saved=0,
            links_created=0,
            errors=[],
            processing_time=0.0
        )
        
        self._load_component_metadata()
        
        logger.info(f"Starting deduplication for model {model_id}")
        
        try:
            # Scan all files in the model directory
            file_hashes: Dict[str, List[Path]] = defaultdict(list)
            
            for file_path in model_path.rglob('*'):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    try:
                        file_hash = self._calculate_file_hash(file_path)
                        file_hashes[file_hash].append(file_path)
                        result.total_files_processed += 1
                    except Exception as e:
                        result.errors.append(f"Failed to hash {file_path}: {e}")
            
            # Process files that can be deduplicated
            for file_hash, file_paths in file_hashes.items():
                if len(file_paths) > 1:
                    # Multiple files with same hash - deduplicate them
                    result.duplicates_found += len(file_paths) - 1
                    
                    # Choose the first file as the canonical one
                    canonical_file = file_paths[0]
                    
                    # Identify component type
                    relative_path = canonical_file.relative_to(model_path)
                    component_type = self._identify_component_type(str(relative_path))
                    
                    logger.debug(f"File {relative_path} identified as component type: {component_type}")
                    
                    if component_type:
                        # Create shared component
                        component_key = self._get_component_key(component_type, file_hash)
                        shared_path = self._create_shared_component_path(component_key, str(relative_path))
                        
                        # Move canonical file to shared storage
                        shared_path.parent.mkdir(parents=True, exist_ok=True)
                        if not shared_path.exists():
                            shutil.move(str(canonical_file), str(shared_path))
                        
                        # Create link back to original location
                        if self._create_link(shared_path, canonical_file):
                            result.links_created += 1
                        
                        # Replace duplicates with links
                        for duplicate_file in file_paths[1:]:
                            file_size = duplicate_file.stat().st_size
                            duplicate_file.unlink()  # Remove duplicate
                            
                            if self._create_link(shared_path, duplicate_file):
                                result.links_created += 1
                                result.bytes_saved += file_size
                        
                        # Update component metadata
                        component_info = ComponentInfo(
                            name=component_type,
                            version=file_hash[:16],
                            content_hash=file_hash,
                            size=shared_path.stat().st_size,
                            original_path=str(relative_path),
                            shared_path=str(shared_path),
                            references={model_id},
                            created_at=time.time(),
                            last_accessed=time.time()
                        )
                        
                        self._component_cache[component_key] = component_info
            
            # Save updated metadata
            self._save_component_metadata()
            
        except Exception as e:
            result.errors.append(f"Deduplication failed: {e}")
            logger.error(f"Deduplication failed for model {model_id}: {e}")
        
        result.processing_time = time.time() - start_time
        
        logger.info(
            f"Deduplication completed for model {model_id}",
            extra={
                "files_processed": result.total_files_processed,
                "duplicates_found": result.duplicates_found,
                "bytes_saved": result.bytes_saved,
                "links_created": result.links_created,
                "processing_time": result.processing_time
            }
        )
        
        return result
    
    def deduplicate_across_models(self, model_paths: Dict[str, Path]) -> DeduplicationResult:
        """
        Deduplicate components across multiple models.
        
        Args:
            model_paths: Dictionary mapping model_id to model directory path
            
        Returns:
            DeduplicationResult with statistics
        """
        start_time = time.time()
        result = DeduplicationResult(
            total_files_processed=0,
            duplicates_found=0,
            bytes_saved=0,
            links_created=0,
            errors=[],
            processing_time=0.0
        )
        
        self._load_component_metadata()
        
        logger.info(f"Starting cross-model deduplication for {len(model_paths)} models")
        
        try:
            # Build a global hash map of all files across models
            global_file_map: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
            
            for model_id, model_path in model_paths.items():
                for file_path in model_path.rglob('*'):
                    if file_path.is_file() and not file_path.name.startswith('.'):
                        try:
                            file_hash = self._calculate_file_hash(file_path)
                            global_file_map[file_hash].append((model_id, file_path))
                            result.total_files_processed += 1
                        except Exception as e:
                            result.errors.append(f"Failed to hash {file_path}: {e}")
            
            # Process files that appear in multiple models
            for file_hash, file_entries in global_file_map.items():
                if len(file_entries) > 1:
                    # Check if these are the same component type
                    component_types = set()
                    for model_id, file_path in file_entries:
                        model_path = model_paths[model_id]
                        relative_path = file_path.relative_to(model_path)
                        component_type = self._identify_component_type(str(relative_path))
                        if component_type:
                            component_types.add(component_type)
                    
                    # Only deduplicate if all files are the same component type
                    if len(component_types) == 1:
                        component_type = component_types.pop()
                        component_key = self._get_component_key(component_type, file_hash)
                        
                        # Choose first file as canonical
                        canonical_model_id, canonical_file = file_entries[0]
                        canonical_model_path = model_paths[canonical_model_id]
                        relative_path = canonical_file.relative_to(canonical_model_path)
                        
                        shared_path = self._create_shared_component_path(component_key, str(relative_path))
                        
                        # Move canonical file to shared storage if not already there
                        shared_path.parent.mkdir(parents=True, exist_ok=True)
                        if not shared_path.exists():
                            shutil.move(str(canonical_file), str(shared_path))
                        
                        # Create link back to canonical location
                        if self._create_link(shared_path, canonical_file):
                            result.links_created += 1
                        
                        # Replace all other instances with links
                        references = {canonical_model_id}
                        for model_id, file_path in file_entries[1:]:
                            file_size = file_path.stat().st_size
                            file_path.unlink()  # Remove duplicate
                            
                            if self._create_link(shared_path, file_path):
                                result.links_created += 1
                                result.bytes_saved += file_size
                                references.add(model_id)
                        
                        result.duplicates_found += len(file_entries) - 1
                        
                        # Update component metadata
                        if component_key in self._component_cache:
                            # Update existing component
                            self._component_cache[component_key].references.update(references)
                            self._component_cache[component_key].last_accessed = time.time()
                        else:
                            # Create new component entry
                            component_info = ComponentInfo(
                                name=component_type,
                                version=file_hash[:16],
                                content_hash=file_hash,
                                size=shared_path.stat().st_size,
                                original_path=str(relative_path),
                                shared_path=str(shared_path),
                                references=references,
                                created_at=time.time(),
                                last_accessed=time.time()
                            )
                            self._component_cache[component_key] = component_info
            
            # Save updated metadata
            self._save_component_metadata()
            
        except Exception as e:
            result.errors.append(f"Cross-model deduplication failed: {e}")
            logger.error(f"Cross-model deduplication failed: {e}")
        
        result.processing_time = time.time() - start_time
        
        logger.info(
            f"Cross-model deduplication completed",
            extra={
                "models_processed": len(model_paths),
                "files_processed": result.total_files_processed,
                "duplicates_found": result.duplicates_found,
                "bytes_saved": result.bytes_saved,
                "links_created": result.links_created,
                "processing_time": result.processing_time
            }
        )
        
        return result
    
    def add_model_reference(self, model_id: str, model_path: Path) -> None:
        """
        Add references for a model that uses shared components.
        
        Args:
            model_id: Identifier for the model
            model_path: Path to the model directory
        """
        self._load_component_metadata()
        
        # Scan for links to shared components
        for file_path in model_path.rglob('*'):
            if file_path.is_file() and file_path.is_symlink():
                # Check if this links to a shared component
                try:
                    target = file_path.resolve()
                    if target.is_relative_to(self.components_root):
                        # Find the component this links to
                        for component_key, component_info in self._component_cache.items():
                            if Path(component_info.shared_path) == target:
                                component_info.references.add(model_id)
                                component_info.last_accessed = time.time()
                                break
                except Exception as e:
                    logger.debug(f"Failed to resolve link {file_path}: {e}")
        
        self._save_component_metadata()
    
    def remove_model_reference(self, model_id: str) -> List[str]:
        """
        Remove references for a model and return list of components that can be cleaned up.
        
        Args:
            model_id: Identifier for the model being removed
            
        Returns:
            List of component keys that have no remaining references
        """
        self._load_component_metadata()
        
        orphaned_components = []
        
        # Remove model_id from all component references
        for component_key, component_info in self._component_cache.items():
            if model_id in component_info.references:
                component_info.references.discard(model_id)
                
                # If no references remain, mark for cleanup
                if not component_info.references:
                    orphaned_components.append(component_key)
        
        self._save_component_metadata()
        
        logger.info(
            f"Removed references for model {model_id}",
            extra={
                "orphaned_components": len(orphaned_components),
                "component_keys": orphaned_components
            }
        )
        
        return orphaned_components
    
    def cleanup_orphaned_components(self, component_keys: List[str]) -> int:
        """
        Clean up components that have no remaining references.
        
        Args:
            component_keys: List of component keys to clean up
            
        Returns:
            Number of bytes reclaimed
        """
        self._load_component_metadata()
        
        bytes_reclaimed = 0
        
        for component_key in component_keys:
            if component_key in self._component_cache:
                component_info = self._component_cache[component_key]
                
                # Double-check that component has no references
                if not component_info.references:
                    try:
                        shared_path = Path(component_info.shared_path)
                        if shared_path.exists():
                            if shared_path.is_file():
                                bytes_reclaimed += shared_path.stat().st_size
                                shared_path.unlink()
                            elif shared_path.is_dir():
                                bytes_reclaimed += sum(
                                    f.stat().st_size for f in shared_path.rglob('*') if f.is_file()
                                )
                                shutil.rmtree(shared_path)
                            
                            # Remove empty parent directories
                            try:
                                shared_path.parent.rmdir()
                            except OSError:
                                pass  # Directory not empty, that's fine
                        
                        # Remove from cache
                        del self._component_cache[component_key]
                        
                        logger.debug(f"Cleaned up orphaned component: {component_key}")
                        
                    except Exception as e:
                        logger.error(f"Failed to cleanup component {component_key}: {e}")
        
        self._save_component_metadata()
        
        logger.info(
            f"Cleaned up {len(component_keys)} orphaned components",
            extra={"bytes_reclaimed": bytes_reclaimed}
        )
        
        return bytes_reclaimed
    
    def get_component_stats(self) -> Dict[str, any]:
        """Get statistics about shared components."""
        self._load_component_metadata()
        
        total_components = len(self._component_cache)
        total_size = sum(info.size for info in self._component_cache.values())
        total_references = sum(len(info.references) for info in self._component_cache.values())
        
        component_types = defaultdict(int)
        for info in self._component_cache.values():
            component_types[info.name] += 1
        
        return {
            "total_components": total_components,
            "total_size_bytes": total_size,
            "total_references": total_references,
            "component_types": dict(component_types),
            "components_root": str(self.components_root),
            "supports_hardlinks": self.supports_hardlinks,
            "supports_symlinks": self.supports_symlinks
        }