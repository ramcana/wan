---
title: core.model_orchestrator.download_manager
category: api
tags: [api, core]
---

# core.model_orchestrator.download_manager

Advanced download manager with parallel downloads, bandwidth limiting, and queue management.

## Classes

### DownloadPriority

Priority levels for download queue.

### DownloadTask

Individual download task.

#### Methods

##### __lt__(self: Any, other: Any)

For priority queue ordering.

### DownloadProgress

Progress information for a download.

### BandwidthLimiter

Token bucket bandwidth limiter.

#### Methods

##### acquire_tokens(self: Any, bytes_requested: int) -> float

Acquire tokens for bandwidth limiting.

Returns:
    Time to wait before proceeding (0 if no wait needed)

### ConnectionPool

Enhanced HTTP connection pool with adaptive connection management.

#### Methods

##### __init__(self: Any, max_connections: int, max_connections_per_host: int)



##### _create_session(self: Any) -> Any

Create an optimized HTTP session.

##### get_session(self: Any) -> Any

Get HTTP session for the current thread.

##### update_stats(self: Any, thread_id: int, bytes_downloaded: int, error: bool)

Update session statistics.

##### get_stats(self: Any) -> <ast.Subscript object at 0x000001942FB6DC00>

Get connection pool statistics.

##### cleanup_stale_sessions(self: Any, max_age: float)

Clean up stale sessions that haven't been used recently.

##### close(self: Any)

Close all connections in the pool.

### ModelDownloadQueue

Queue for managing downloads of a specific model.

#### Methods

##### __lt__(self: Any, other: Any)

For priority queue ordering.

### DownloadMetrics

Metrics for download performance analysis.

#### Methods

##### duration(self: Any) -> float

Get download duration.

##### completion_rate(self: Any) -> float

Get completion rate as percentage.

### ParallelDownloadManager

Advanced download manager with parallel downloads, bandwidth limiting,
queue management, and comprehensive performance optimization.

#### Methods

##### __init__(self: Any, max_concurrent_downloads: int, max_concurrent_files_per_model: int, max_bandwidth_bps: <ast.Subscript object at 0x000001942FC20D60>, connection_pool_size: int, chunk_size: int, enable_resume: bool, enable_adaptive_chunking: bool, enable_compression: bool, queue_timeout: float)



##### add_download_task(self: Any, model_id: str, file_path: str, source_url: str, local_path: Path, size: int, priority: DownloadPriority, progress_callback: <ast.Subscript object at 0x000001942F8153F0>) -> str

Add a download task to the queue.

Returns:
    Task ID for tracking the download

##### queue_model_download(self: Any, model_id: str, file_specs: <ast.Subscript object at 0x000001942F8140D0>, source_url: str, local_dir: Path, priority: DownloadPriority, progress_callback: <ast.Subscript object at 0x000001942F767EB0>) -> str

Queue a complete model download with optimized parallel file handling.

Returns:
    Queue ID for tracking the model download

##### download_model_parallel(self: Any, model_id: str, file_specs: <ast.Subscript object at 0x000001942F7A13F0>, source_url: str, local_dir: Path, progress_callback: <ast.Subscript object at 0x000001942F7A1270>) -> <ast.Subscript object at 0x000001942F7A0BB0>

Download all files for a model in parallel with enhanced performance optimization.

Returns:
    Dictionary with download results and comprehensive statistics

##### wait_for_model_completion(self: Any, queue_id: str, timeout: <ast.Subscript object at 0x000001942F8035E0>) -> <ast.Subscript object at 0x000001942F802620>

Wait for a queued model download to complete.

Returns:
    Dictionary with download results and statistics

##### _determine_file_priority(self: Any, file_spec: Any) -> DownloadPriority

Determine download priority for a file based on type and size.

##### _collect_download_results(self: Any, queue_id: str) -> <ast.Subscript object at 0x000001942F827070>

Collect comprehensive download results for a model.

##### _queue_worker(self: Any)

Enhanced background worker that processes model download queues.

##### _process_model_queue(self: Any, model_queue: ModelDownloadQueue)

Process downloads for a specific model with optimized concurrency.

##### _handle_task_completion(self: Any, task: DownloadTask, queue_id: str, success: bool, error: <ast.Subscript object at 0x0000019434319120>)

Handle completion of a download task.

##### _apply_adaptive_chunking(self: Any, task: DownloadTask)

Apply adaptive chunk sizing based on file size and network conditions.

##### _is_io_intensive(self: Any, task: DownloadTask) -> bool

Determine if a task is I/O intensive and should use the I/O executor.

##### _get_source_key(self: Any, source_url: str) -> str

Get a key for tracking performance by source.

##### _metrics_worker(self: Any)

Background worker for collecting and updating performance metrics.

##### _download_file_optimized(self: Any, task: DownloadTask, queue_id: str) -> bool

Optimized download method with advanced performance features.

Returns:
    True if download succeeded, False otherwise

##### _prepare_headers(self: Any, task: DownloadTask, resume_pos: int) -> <ast.Subscript object at 0x000001942F3A1E10>

Prepare optimized HTTP headers for download.

##### _download_file_fresh(self: Any, task: DownloadTask, progress: DownloadProgress) -> bool

Download file from the beginning (used when resume fails).

##### _update_performance_metrics(self: Any, task: DownloadTask, progress: DownloadProgress, queue_id: str)

Update performance metrics during download.

##### _download_file_basic(self: Any, task: DownloadTask, progress: DownloadProgress, resume_pos: int) -> bool

Fallback download method without connection pooling.

##### _notify_progress(self: Any, progress: DownloadProgress)

Notify progress callbacks for a model.

##### get_download_stats(self: Any) -> <ast.Subscript object at 0x00000194340BB910>

Get comprehensive download statistics.

##### get_model_queue_status(self: Any, queue_id: str) -> <ast.Subscript object at 0x00000194340A1390>

Get status of a specific model download queue.

##### cancel_model_download(self: Any, queue_id: str) -> bool

Cancel a model download queue.

##### pause_downloads(self: Any)

Pause all downloads (for maintenance or resource management).

##### resume_downloads(self: Any)

Resume paused downloads.

##### optimize_performance(self: Any)

Trigger performance optimization based on current conditions.

##### shutdown(self: Any)

Shutdown the download manager gracefully.

## Constants

### HIGH

Type: `int`

Value: `1`

### NORMAL

Type: `int`

Value: `2`

### LOW

Type: `int`

Value: `3`

