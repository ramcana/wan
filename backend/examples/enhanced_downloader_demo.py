"""
Enhanced Model Downloader Demo
Demonstrates the enhanced model downloader functionality with retry logic,
progress tracking, and download management features.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from core.enhanced_model_downloader import (
    EnhancedModelDownloader,
    DownloadStatus,
    RetryConfig,
    BandwidthConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def progress_callback(progress):
    """Progress callback to display download progress"""
    status_emoji = {
        DownloadStatus.QUEUED: "⏳",
        DownloadStatus.DOWNLOADING: "⬇️",
        DownloadStatus.PAUSED: "⏸️",
        DownloadStatus.COMPLETED: "✅",
        DownloadStatus.FAILED: "❌",
        DownloadStatus.CANCELLED: "🚫",
        DownloadStatus.VERIFYING: "🔍",
        DownloadStatus.RESUMING: "▶️"
    }
    
    emoji = status_emoji.get(progress.status, "❓")
    
    if progress.total_mb > 0:
        print(f"{emoji} {progress.model_id}: {progress.progress_percent:.1f}% "
              f"({progress.downloaded_mb:.1f}/{progress.total_mb:.1f} MB) "
              f"@ {progress.speed_mbps:.1f} Mbps")
        
        if progress.eta_seconds:
            eta_min = progress.eta_seconds / 60
            print(f"   ETA: {eta_min:.1f} minutes")
    else:
        print(f"{emoji} {progress.model_id}: {progress.status.value}")
    
    if progress.error_message:
        print(f"   Error: {progress.error_message}")


async def demo_basic_functionality():
    """Demonstrate basic enhanced downloader functionality"""
    print("=" * 60)
    print("Enhanced Model Downloader - Basic Functionality Demo")
    print("=" * 60)
    
    # Create temporary directory for demo
    demo_dir = Path("demo_downloads")
    demo_dir.mkdir(exist_ok=True)
    
    # Initialize enhanced downloader
    async with EnhancedModelDownloader(models_dir=str(demo_dir)) as downloader:
        # Add progress callback
        downloader.add_progress_callback(progress_callback)
        
        print(f"✅ Enhanced downloader initialized")
        print(f"📁 Models directory: {downloader.models_dir}")
        print(f"📁 Partial downloads directory: {downloader._partial_downloads_dir}")
        
        # Display configuration
        print(f"\n🔧 Retry Configuration:")
        print(f"   Max retries: {downloader.retry_config.max_retries}")
        print(f"   Initial delay: {downloader.retry_config.initial_delay}s")
        print(f"   Max delay: {downloader.retry_config.max_delay}s")
        print(f"   Backoff factor: {downloader.retry_config.backoff_factor}")
        
        print(f"\n🌐 Bandwidth Configuration:")
        print(f"   Max speed: {downloader.bandwidth_config.max_speed_mbps or 'Unlimited'}")
        print(f"   Chunk size: {downloader.bandwidth_config.chunk_size} bytes")
        print(f"   Adaptive chunking: {downloader.bandwidth_config.adaptive_chunking}")
        
        return downloader


async def demo_retry_logic():
    """Demonstrate retry logic with exponential backoff"""
    print("\n" + "=" * 60)
    print("Retry Logic Demo")
    print("=" * 60)
    
    demo_dir = Path("demo_downloads")
    
    async with EnhancedModelDownloader(models_dir=str(demo_dir)) as downloader:
        print("🔄 Testing retry delay calculation:")
        
        for attempt in range(5):
            delay = await downloader._calculate_retry_delay(attempt)
            print(f"   Attempt {attempt + 1}: {delay:.2f}s delay")
        
        print("\n🚫 Testing rate limit handling:")
        rate_limit_delay = await downloader._calculate_retry_delay(0, "Rate limit exceeded")
        print(f"   Rate limit delay: {rate_limit_delay:.2f}s")
        
        print("\n⚙️ Testing custom retry configuration:")
        downloader.update_retry_config(
            max_retries=5,
            initial_delay=0.5,
            max_delay=30.0,
            backoff_factor=1.5
        )
        
        for attempt in range(3):
            delay = await downloader._calculate_retry_delay(attempt)
            print(f"   Custom attempt {attempt + 1}: {delay:.2f}s delay")


async def demo_bandwidth_management():
    """Demonstrate bandwidth management features"""
    print("\n" + "=" * 60)
    print("Bandwidth Management Demo")
    print("=" * 60)
    
    demo_dir = Path("demo_downloads")
    
    async with EnhancedModelDownloader(models_dir=str(demo_dir)) as downloader:
        print("📊 Testing adaptive chunk sizing:")
        
        speeds = [5.0, 25.0, 75.0, 150.0]  # Mbps
        for speed in speeds:
            chunk_size = downloader._calculate_chunk_size(speed)
            print(f"   {speed:6.1f} Mbps -> {chunk_size:5d} bytes chunk")
        
        print("\n🚦 Testing bandwidth limiting:")
        
        # Set bandwidth limit
        success = downloader.set_bandwidth_limit(10.0)  # 10 Mbps
        print(f"   Set 10 Mbps limit: {'✅' if success else '❌'}")
        
        # Remove bandwidth limit
        success = downloader.set_bandwidth_limit(None)
        print(f"   Remove limit: {'✅' if success else '❌'}")
        
        print("\n⚙️ Testing bandwidth configuration update:")
        downloader.update_bandwidth_config(
            max_speed_mbps=25.0,
            chunk_size=16384,
            adaptive_chunking=False
        )
        
        print(f"   Updated max speed: {downloader.bandwidth_config.max_speed_mbps} Mbps")
        print(f"   Updated chunk size: {downloader.bandwidth_config.chunk_size} bytes")
        print(f"   Adaptive chunking: {downloader.bandwidth_config.adaptive_chunking}")


async def demo_download_management():
    """Demonstrate download management features"""
    print("\n" + "=" * 60)
    print("Download Management Demo")
    print("=" * 60)
    
    demo_dir = Path("demo_downloads")
    
    async with EnhancedModelDownloader(models_dir=str(demo_dir)) as downloader:
        downloader.add_progress_callback(progress_callback)
        
        print("📋 Testing download progress tracking:")
        
        # Simulate active downloads
        from core.enhanced_model_downloader import DownloadProgress
        
        test_downloads = [
            DownloadProgress(
                model_id="model_1",
                status=DownloadStatus.DOWNLOADING,
                progress_percent=25.0,
                downloaded_mb=2.5,
                total_mb=10.0,
                speed_mbps=15.0,
                eta_seconds=120.0
            ),
            DownloadProgress(
                model_id="model_2",
                status=DownloadStatus.PAUSED,
                progress_percent=75.0,
                downloaded_mb=7.5,
                total_mb=10.0,
                speed_mbps=0.0
            ),
            DownloadProgress(
                model_id="model_3",
                status=DownloadStatus.COMPLETED,
                progress_percent=100.0,
                downloaded_mb=5.0,
                total_mb=5.0,
                speed_mbps=0.0
            )
        ]
        
        # Add test downloads
        async with downloader._download_lock:
            for progress in test_downloads:
                downloader._active_downloads[progress.model_id] = progress
        
        # Display all progress
        all_progress = await downloader.get_all_download_progress()
        print(f"\n📊 Active downloads: {len(all_progress)}")
        
        for model_id, progress in all_progress.items():
            await progress_callback(progress)
        
        print("\n🎮 Testing download controls:")
        
        # Test pause
        result = await downloader.pause_download("model_1")
        print(f"   Pause model_1: {'✅' if result else '❌'}")
        
        # Test resume
        result = await downloader.resume_download("model_2")
        print(f"   Resume model_2: {'✅' if result else '❌'}")
        
        # Test cancel
        result = await downloader.cancel_download("model_1")
        print(f"   Cancel model_1: {'✅' if result else '❌'}")


async def demo_integrity_verification():
    """Demonstrate model integrity verification"""
    print("\n" + "=" * 60)
    print("Model Integrity Verification Demo")
    print("=" * 60)
    
    demo_dir = Path("demo_downloads")
    demo_dir.mkdir(exist_ok=True)
    
    async with EnhancedModelDownloader(models_dir=str(demo_dir)) as downloader:
        print("🔍 Testing model integrity verification:")
        
        # Test with non-existent model
        result = await downloader.verify_and_repair_model("non_existent_model")
        print(f"   Non-existent model: {'✅' if result else '❌'} (expected ❌)")
        
        # Create test model file
        test_model_path = demo_dir / "test_model.model"
        with open(test_model_path, 'wb') as f:
            f.write(b"Test model data" * 1000)  # Create non-empty file
        
        result = await downloader.verify_and_repair_model("test_model")
        print(f"   Valid test model: {'✅' if result else '❌'} (expected ✅)")
        
        # Test checksum calculation
        checksum = await downloader._calculate_file_checksum(test_model_path)
        print(f"   Checksum: {checksum[:16]}... (SHA256)")
        
        # Clean up
        test_model_path.unlink()


async def demo_cleanup_operations():
    """Demonstrate cleanup operations"""
    print("\n" + "=" * 60)
    print("Cleanup Operations Demo")
    print("=" * 60)
    
    demo_dir = Path("demo_downloads")
    
    async with EnhancedModelDownloader(models_dir=str(demo_dir)) as downloader:
        print("🧹 Testing cleanup operations:")
        
        # Create test partial files
        partial_files = []
        for i in range(3):
            partial_file = downloader._partial_downloads_dir / f"test_model_{i}.partial"
            with open(partial_file, 'wb') as f:
                f.write(b"partial data")
            partial_files.append(partial_file)
        
        print(f"   Created {len(partial_files)} partial files")
        
        # Test individual cleanup
        await downloader._cleanup_partial_download("test_model_0")
        remaining = len(list(downloader._partial_downloads_dir.glob("*.partial")))
        print(f"   After individual cleanup: {remaining} files remaining")
        
        # Test cleanup all
        await downloader.cleanup_all_partial_downloads()
        remaining = len(list(downloader._partial_downloads_dir.glob("*.partial")))
        print(f"   After cleanup all: {remaining} files remaining")


async def main():
    """Run all demos"""
    try:
        await demo_basic_functionality()
        await demo_retry_logic()
        await demo_bandwidth_management()
        await demo_download_management()
        await demo_integrity_verification()
        await demo_cleanup_operations()
        
        print("\n" + "=" * 60)
        print("✅ All Enhanced Model Downloader demos completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up demo directory
        demo_dir = Path("demo_downloads")
        if demo_dir.exists():
            import shutil
            shutil.rmtree(demo_dir)
            print(f"🧹 Cleaned up demo directory: {demo_dir}")


if __name__ == "__main__":
    asyncio.run(main())