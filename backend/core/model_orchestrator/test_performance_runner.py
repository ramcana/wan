"""
Simple performance benchmark runner for testing.
"""

import tempfile
import time
from pathlib import Path

from .download_manager import ParallelDownloadManager, DownloadPriority
from .memory_optimizer import MemoryOptimizer


def run_simple_performance_test():
    """Run a simple performance test to verify the advanced features work."""
    print("Starting simple performance test...")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix="perf_test_"))
    
    try:
        # Test 1: Download Manager Performance
        print("Testing download manager performance...")
        dm = ParallelDownloadManager(
            max_concurrent_downloads=4,
            max_concurrent_files_per_model=8,
            enable_adaptive_chunking=True,
            enable_compression=True
        )
        
        # Get initial stats
        stats = dm.get_download_stats()
        print(f"Initial stats: {stats}")
        
        # Test adaptive chunking
        from .download_manager import DownloadTask
        task = DownloadTask(
            task_id="test-task",
            model_id="test-model",
            file_path="test.bin",
            source_url="http://test.com/test.bin",
            local_path=temp_dir / "test.bin",
            size=50 * 1024 * 1024  # 50MB
        )
        
        dm._apply_adaptive_chunking(task)
        chunk_size = dm._adaptive_chunk_sizes.get(task.task_id, dm.chunk_size)
        print(f"Adaptive chunk size: {chunk_size} bytes")
        
        # Test performance optimization
        dm.optimize_performance()
        print("Performance optimization completed")
        
        dm.shutdown()
        
        # Test 2: Memory Optimizer Performance
        print("Testing memory optimizer performance...")
        memory_optimizer = MemoryOptimizer(
            streaming_threshold=10 * 1024 * 1024  # 10MB
        )
        
        memory_optimizer.start_monitoring()
        
        # Test optimized download context
        with memory_optimizer.optimized_download_context(
            model_id="test-model",
            total_size=100 * 1024 * 1024,  # 100MB
            file_count=10
        ) as context:
            print(f"Streaming enabled: {context['use_streaming']}")
            print(f"File handler available: {context['file_handler'] is not None}")
            
            # Simulate some work
            time.sleep(0.1)
        
        # Get memory stats
        stats = memory_optimizer.get_memory_stats()
        print(f"Memory stats: Total={stats.total_memory}, Available={stats.available_memory}")
        
        # Test streaming threshold
        should_stream_small = memory_optimizer.should_use_streaming(1024 * 1024)  # 1MB
        should_stream_large = memory_optimizer.should_use_streaming(50 * 1024 * 1024)  # 50MB
        print(f"Should stream 1MB file: {should_stream_small}")
        print(f"Should stream 50MB file: {should_stream_large}")
        
        # Test optimal chunk size calculation
        chunk_size = memory_optimizer.get_optimal_chunk_size(
            file_size=100 * 1024 * 1024,  # 100MB
            available_memory=1024 * 1024 * 1024  # 1GB
        )
        print(f"Optimal chunk size: {chunk_size} bytes")
        
        memory_optimizer.cleanup()
        
        # Test 3: Connection Pool Performance
        print("Testing connection pool performance...")
        from .download_manager import ConnectionPool
        
        pool = ConnectionPool(max_connections=10, max_connections_per_host=5)
        
        # Get session (may be None if requests not available)
        session = pool.get_session()
        print(f"Session available: {session is not None}")
        
        # Get stats
        pool_stats = pool.get_stats()
        print(f"Pool stats: {pool_stats}")
        
        # Test cleanup
        pool.cleanup_stale_sessions()
        pool.close()
        
        print("All performance tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"Performance test failed: {e}")
        return False
        
    finally:
        # Clean up temp directory
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    success = run_simple_performance_test()
    exit(0 if success else 1)