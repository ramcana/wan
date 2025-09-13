#!/usr/bin/env python3
"""
Optimized Image Cache and Memory Management for WAN22
Implements efficient caching and memory management for image operations
"""

import weakref
import threading
import time
import psutil
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass, field
from PIL import Image
import hashlib
import io
import gc
from datetime import datetime, timedelta
from collections import OrderedDict
import json

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    image: Image.Image
    size_bytes: int
    access_count: int
    last_accessed: datetime
    created: datetime
    cache_key: str
    
    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = datetime.now()

@dataclass
class CacheStats:
    """Cache performance statistics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_memory_mb: float = 0.0
    entries_count: int = 0
    evictions: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

class OptimizedImageCache:
    """High-performance image cache with intelligent memory management"""
    
    def __init__(self, max_memory_mb: float = 512, max_entries: int = 100, 
                 cleanup_interval: int = 300):
        self.max_memory_mb = max_memory_mb
        self.max_entries = max_entries
        self.cleanup_interval = cleanup_interval
        
        # Thread-safe cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
        
        # Memory monitoring
        self._current_memory_mb = 0.0
        self._last_cleanup = datetime.now()
        
        # Weak references for automatic cleanup
        self._weak_refs: Dict[str, weakref.ref] = {}
        
        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self._cleanup_thread.start()
        
    def get_cache_key(self, image_data: bytes = None, image_path: str = None, 
                     operation: str = "", params: Dict[str, Any] = None) -> str:
        """Generate cache key for image and operation"""
        hasher = hashlib.md5()
        
        if image_data:
            hasher.update(image_data)
        elif image_path:
            hasher.update(image_path.encode())
            
        hasher.update(operation.encode())
        
        if params:
            param_str = json.dumps(params, sort_keys=True)
            hasher.update(param_str.encode())
            
        return hasher.hexdigest()
        
    def get(self, cache_key: str) -> Optional[Image.Image]:
        """Get image from cache"""
        with self._lock:
            self._stats.total_requests += 1
            
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                entry.update_access()
                
                # Move to end (most recently used)
                self._cache.move_to_end(cache_key)
                
                self._stats.cache_hits += 1
                return entry.image.copy() if hasattr(entry.image, 'copy') else entry.image
            else:
                self._stats.cache_misses += 1
                return None
                
    def put(self, cache_key: str, image: Any, operation: str = ""):
        """Store image or data in cache"""
        with self._lock:
            # Calculate image size
            size_bytes = self._calculate_image_size(image)
            
            # Check if we need to make room
            self._ensure_capacity(size_bytes)
            
            # Create cache entry
            stored_data = image.copy() if hasattr(image, 'copy') else image
            entry = CacheEntry(
                image=stored_data,  # Store copy to prevent external modification
                size_bytes=size_bytes,
                access_count=1,
                last_accessed=datetime.now(),
                created=datetime.now(),
                cache_key=cache_key
            )
            
            # Store in cache
            self._cache[cache_key] = entry
            self._current_memory_mb += size_bytes / (1024 * 1024)
            self._stats.entries_count += 1
            
            # Create weak reference for automatic cleanup (only for objects that support it)
            try:
                self._weak_refs[cache_key] = weakref.ref(entry.image, 
                                                       lambda ref: self._on_image_deleted(cache_key))
            except TypeError:
                # Object doesn't support weak references, skip
                pass
            
    def _calculate_image_size(self, image: Any) -> int:
        """Calculate approximate memory size of image or data"""
        if hasattr(image, 'size') and hasattr(image, 'getbands'):
            # PIL Image
            width, height = image.size
            channels = len(image.getbands())
            return width * height * channels
        else:
            # Other data types - estimate based on string representation
            import sys
            return sys.getsizeof(image)
        
    def _ensure_capacity(self, new_size_bytes: float):
        """Ensure cache has capacity for new entry"""
        new_size_mb = new_size_bytes / (1024 * 1024)
        
        # Check memory limit
        while (self._current_memory_mb + new_size_mb > self.max_memory_mb and 
               len(self._cache) > 0):
            self._evict_lru()
            
        # Check entry count limit
        while len(self._cache) >= self.max_entries:
            self._evict_lru()
            
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self._cache:
            return
            
        # Find LRU entry (first in OrderedDict)
        cache_key, entry = self._cache.popitem(last=False)
        
        # Update statistics
        self._current_memory_mb -= entry.size_bytes / (1024 * 1024)
        self._stats.entries_count -= 1
        self._stats.evictions += 1
        
        # Clean up weak reference
        if cache_key in self._weak_refs:
            del self._weak_refs[cache_key]
            
    def _on_image_deleted(self, cache_key: str):
        """Handle automatic cleanup when image is garbage collected"""
        with self._lock:
            if cache_key in self._cache:
                entry = self._cache.pop(cache_key)
                self._current_memory_mb -= entry.size_bytes / (1024 * 1024)
                self._stats.entries_count -= 1
                
    def _background_cleanup(self):
        """Background thread for periodic cleanup"""
        while True:
            time.sleep(self.cleanup_interval)
            self._periodic_cleanup()
            
    def _periodic_cleanup(self):
        """Perform periodic cleanup of expired entries"""
        with self._lock:
            current_time = datetime.now()
            expired_keys = []
            
            # Find expired entries (not accessed in last hour)
            for cache_key, entry in self._cache.items():
                if current_time - entry.last_accessed > timedelta(hours=1):
                    expired_keys.append(cache_key)
                    
            # Remove expired entries
            for cache_key in expired_keys:
                if cache_key in self._cache:
                    entry = self._cache.pop(cache_key)
                    self._current_memory_mb -= entry.size_bytes / (1024 * 1024)
                    self._stats.entries_count -= 1
                    self._stats.evictions += 1
                    
            # Force garbage collection
            gc.collect()
            
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._weak_refs.clear()
            self._current_memory_mb = 0.0
            self._stats.entries_count = 0
            gc.collect()
            
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self._lock:
            stats = CacheStats(
                total_requests=self._stats.total_requests,
                cache_hits=self._stats.cache_hits,
                cache_misses=self._stats.cache_misses,
                total_memory_mb=self._current_memory_mb,
                entries_count=len(self._cache),
                evictions=self._stats.evictions
            )
            return stats
            
    def print_stats(self):
        """Print cache statistics"""
        stats = self.get_stats()
        print(f"\nIMAGE CACHE STATISTICS:")
        print(f"  Total Requests: {stats.total_requests}")
        print(f"  Cache Hits: {stats.cache_hits}")
        print(f"  Cache Misses: {stats.cache_misses}")
        print(f"  Hit Rate: {stats.hit_rate:.2%}")
        print(f"  Memory Usage: {stats.total_memory_mb:.2f}MB / {self.max_memory_mb}MB")
        print(f"  Entries: {stats.entries_count} / {self.max_entries}")
        print(f"  Evictions: {stats.evictions}")

class MemoryManager:
    """Advanced memory management for image operations"""
    
    def __init__(self, memory_threshold_percent: float = 80.0):
        self.memory_threshold_percent = memory_threshold_percent
        self._monitoring = False
        self._monitor_thread = None
        
    def start_monitoring(self):
        """Start memory monitoring"""
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self._monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            
    def _monitor_memory(self):
        """Monitor system memory usage"""
        while self._monitoring:
            memory_percent = psutil.virtual_memory().percent
            
            if memory_percent > self.memory_threshold_percent:
                self._handle_high_memory()
                
            time.sleep(5)  # Check every 5 seconds
            
    def _handle_high_memory(self):
        """Handle high memory usage"""
        # Force garbage collection
        gc.collect()
        
        # Log memory warning
        memory_info = psutil.virtual_memory()
        print(f"WARNING: High memory usage detected: {memory_info.percent:.1f}%")
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory information"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            'system_total_gb': memory.total / (1024**3),
            'system_available_gb': memory.available / (1024**3),
            'system_used_percent': memory.percent,
            'process_memory_mb': process.memory_info().rss / (1024**2),
            'process_memory_percent': process.memory_percent()
        }
        
    def optimize_for_large_images(self) -> Dict[str, Any]:
        """Get optimization settings for large image processing"""
        memory_info = self.get_memory_info()
        available_gb = memory_info['system_available_gb']
        
        # Conservative settings based on available memory
        if available_gb < 2:
            return {
                'max_image_size': (1024, 1024),
                'thumbnail_size': (128, 128),
                'cache_size_mb': 64,
                'batch_size': 1
            }
        elif available_gb < 4:
            return {
                'max_image_size': (1920, 1080),
                'thumbnail_size': (256, 256),
                'cache_size_mb': 128,
                'batch_size': 2
            }
        else:
            return {
                'max_image_size': (4096, 4096),
                'thumbnail_size': (512, 512),
                'cache_size_mb': 256,
                'batch_size': 4
            }

# Global cache instance
_global_cache: Optional[OptimizedImageCache] = None
_cache_lock = threading.Lock()

def get_global_cache() -> OptimizedImageCache:
    """Get global image cache instance"""
    global _global_cache
    
    with _cache_lock:
        if _global_cache is None:
            _global_cache = OptimizedImageCache()
        return _global_cache

def clear_global_cache():
    """Clear global image cache"""
    global _global_cache
    
    with _cache_lock:
        if _global_cache is not None:
            _global_cache.clear()

# Convenience functions for common operations
def cache_image_operation(operation_name: str, image: Image.Image, 
                         operation_func, *args, **kwargs) -> Image.Image:
    """Cache the result of an image operation"""
    cache = get_global_cache()
    
    # Generate cache key
    params = {'args': args, 'kwargs': kwargs}
    cache_key = cache.get_cache_key(
        image_data=image.tobytes(),
        operation=operation_name,
        params=params
    )
    
    # Try to get from cache
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        return cached_result
        
    # Execute operation and cache result
    result = operation_func(image, *args, **kwargs)
    cache.put(cache_key, result, operation_name)
    
    return result

if __name__ == "__main__":
    # Test the cache system
    cache = OptimizedImageCache(max_memory_mb=100, max_entries=10)
    
    # Create test images
    for i in range(15):
        test_image = Image.new('RGB', (512, 512), color=(i*10, i*10, i*10))
        cache_key = f"test_image_{i}"
        cache.put(cache_key, test_image)
        
    cache.print_stats()
