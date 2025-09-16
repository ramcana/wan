---
title: core.model_orchestrator.memory_optimizer
category: api
tags: [api, core]
---

# core.model_orchestrator.memory_optimizer

Memory optimization utilities for large model downloads and processing.

## Classes

### MemoryStats

Memory usage statistics.

### MemoryMonitor

Monitor and track memory usage during downloads.

#### Methods

##### __init__(self: Any, warning_threshold: float, critical_threshold: float)



##### get_memory_stats(self: Any) -> MemoryStats

Get current memory statistics.

##### start_monitoring(self: Any, interval: float)

Start background memory monitoring.

##### stop_monitoring(self: Any)

Stop background memory monitoring.

##### add_callback(self: Any, name: str, callback: callable)

Add callback for memory threshold events.

##### remove_callback(self: Any, name: str)

Remove memory threshold callback.

##### _monitor_loop(self: Any, interval: float)

Background monitoring loop.

##### _trigger_callbacks(self: Any, level: str, stats: MemoryStats)

Trigger registered callbacks for memory events.

### StreamingFileHandler

Memory-efficient file handler for large downloads.

#### Methods

##### __init__(self: Any, chunk_size: int, use_mmap: bool)



##### open_for_streaming_write(self: Any, file_path: Path, expected_size: <ast.Subscript object at 0x000001942809F670>)

Open file for streaming write with memory optimization.

Args:
    file_path: Path to the file to write
    expected_size: Expected file size for pre-allocation

##### open_for_streaming_read(self: Any, file_path: Path)

Open file for streaming read with memory optimization.

##### _preallocate_file(self: Any, file_path: Path, size: int)

Pre-allocate file space to reduce fragmentation.

### StreamingWriter

Memory-efficient streaming writer.

#### Methods

##### __init__(self: Any, file_handle: BinaryIO, chunk_size: int)



##### write(self: Any, data: bytes) -> int

Write data with buffering for efficiency.

##### write_chunk(self: Any, chunk: bytes)

Write a chunk directly (bypassing buffer).

##### flush(self: Any)

Flush any remaining buffered data.

##### _flush_buffer(self: Any)

Flush internal buffer to file.

### StreamingReader

Memory-efficient streaming reader.

#### Methods

##### __init__(self: Any, file_handle: Any, chunk_size: int)



##### read_chunks(self: Any) -> <ast.Subscript object at 0x00000194280594E0>

Read file in chunks to minimize memory usage.

##### read_chunk(self: Any) -> <ast.Subscript object at 0x0000019428059360>

Read a single chunk.

### MemoryOptimizer

Main memory optimization coordinator for model downloads.

#### Methods

##### __init__(self: Any, max_memory_usage: <ast.Subscript object at 0x00000194281A8610>, gc_threshold: float, streaming_threshold: int)



##### start_monitoring(self: Any)

Start memory monitoring.

##### stop_monitoring(self: Any)

Stop memory monitoring.

##### optimized_download_context(self: Any, model_id: str, total_size: int, file_count: int)

Context manager for memory-optimized downloads.

Args:
    model_id: Model identifier
    total_size: Total download size
    file_count: Number of files to download

##### update_download_progress(self: Any, model_id: str, bytes_downloaded: int)

Update download progress for memory tracking.

##### get_memory_stats(self: Any) -> MemoryStats

Get current memory statistics.

##### get_download_stats(self: Any) -> <ast.Subscript object at 0x00000194281AB790>

Get current download statistics.

##### _handle_memory_pressure(self: Any, level: str, stats: MemoryStats)

Handle memory pressure events.

##### _force_garbage_collection(self: Any)

Force garbage collection to free memory.

##### should_use_streaming(self: Any, file_size: int) -> bool

Determine if streaming should be used for a file.

##### get_optimal_chunk_size(self: Any, file_size: int, available_memory: int) -> int

Calculate optimal chunk size based on file size and available memory.

##### cleanup(self: Any)

Clean up resources.

