"""
Demo script for Network Failure Recovery System.

This script demonstrates the comprehensive network failure recovery capabilities
including error detection, alternative sources, resumable downloads, and recovery strategies.
"""

import logging
import tempfile
import shutil
from pathlib import Path
import time
import urllib.error
import socket
from typing import Optional

# Import the network failure recovery system
from scripts.network_failure_recovery import (
    NetworkFailureRecovery, NetworkErrorType, NetworkConfiguration,
    ProxyConfiguration, AuthenticationConfiguration, DownloadMirror
)


def setup_logging():
    """Set up logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('network_recovery_demo.log')
        ]
    )
    return logging.getLogger(__name__)


def demo_network_connectivity_testing(recovery_system: NetworkFailureRecovery, logger: logging.Logger):
    """Demonstrate network connectivity testing."""
    logger.info("=" * 60)
    logger.info("DEMO: Network Connectivity Testing")
    logger.info("=" * 60)
    
    # Test basic connectivity
    logger.info("Testing basic network connectivity...")
    result = recovery_system.test_network_connectivity()
    
    if result.success:
        logger.info(f"✓ Network connectivity successful!")
        logger.info(f"  - Latency: {result.latency_ms:.1f}ms")
        logger.info(f"  - Bandwidth: {result.bandwidth_mbps:.1f}Mbps")
        logger.info(f"  - DNS working: {result.dns_working}")
        logger.info(f"  - Proxy working: {result.proxy_working}")
    else:
        logger.warning(f"✗ Network connectivity failed: {result.error_message}")
    
    # Validate network configuration
    logger.info("\nValidating network configuration...")
    validation_result = recovery_system.validate_network_configuration()
    
    if validation_result.success:
        logger.info("✓ Network configuration is valid")
        if validation_result.warnings:
            logger.warning("Warnings:")
            for warning in validation_result.warnings:
                logger.warning(f"  - {warning}")
    else:
        logger.error("✗ Network configuration has issues:")
        for issue in validation_result.details.get('issues', []):
            logger.error(f"  - {issue}")


def demo_error_detection(recovery_system: NetworkFailureRecovery, logger: logging.Logger):
    """Demonstrate network error detection and classification."""
    logger.info("=" * 60)
    logger.info("DEMO: Network Error Detection")
    logger.info("=" * 60)
    
    # Test different error types
    test_errors = [
        (socket.timeout("Connection timed out"), None, "Connection Timeout"),
        (socket.gaierror("Name resolution failed"), None, "DNS Resolution Error"),
        (urllib.error.HTTPError("url", 401, "Unauthorized", {}, None), 401, "Authentication Error"),
        (urllib.error.HTTPError("url", 429, "Too Many Requests", {}, None), 429, "Rate Limiting"),
        (urllib.error.HTTPError("url", 500, "Internal Server Error", {}, None), 500, "Server Error"),
        (ConnectionRefusedError("Connection refused"), None, "Connection Refused"),
    ]
    
    for error, response_code, description in test_errors:
        logger.info(f"\nTesting {description}:")
        error_type = recovery_system.error_detector.detect_error_type(error, response_code)
        is_retryable = recovery_system.error_detector.is_retryable_error(error_type)
        needs_alt_source = recovery_system.error_detector.requires_alternative_source(error_type)
        
        logger.info(f"  - Error type: {error_type.value}")
        logger.info(f"  - Retryable: {is_retryable}")
        logger.info(f"  - Needs alternative source: {needs_alt_source}")


def demo_mirror_management(recovery_system: NetworkFailureRecovery, logger: logging.Logger):
    """Demonstrate download mirror management."""
    logger.info("=" * 60)
    logger.info("DEMO: Download Mirror Management")
    logger.info("=" * 60)
    
    # Show default mirrors
    logger.info("Default mirrors configured:")
    mirror_stats = recovery_system.mirror_manager.get_mirror_statistics()
    
    for service, stats in mirror_stats.items():
        logger.info(f"\n{service.upper()} Service:")
        logger.info(f"  - Total mirrors: {stats['total_mirrors']}")
        logger.info(f"  - Available mirrors: {stats['available_mirrors']}")
        
        for mirror in stats['mirrors'][:3]:  # Show first 3 mirrors
            logger.info(f"    * {mirror['name']}: {mirror['base_url']} (priority: {mirror['priority']})")
    
    # Add a custom mirror
    logger.info("\nAdding custom mirror...")
    custom_mirror = DownloadMirror(
        name="demo_mirror",
        base_url="https://demo.example.com",
        priority=1
    )
    recovery_system.mirror_manager.add_mirror("demo_service", custom_mirror)
    logger.info("✓ Custom mirror added successfully")
    
    # Test mirror failure and success tracking
    logger.info("\nTesting mirror failure tracking...")
    test_mirror = recovery_system.mirror_manager.mirrors["huggingface"][0]
    original_failure_count = test_mirror.failure_count
    
    recovery_system.mirror_manager.mark_mirror_failure(test_mirror, Exception("Test failure"))
    logger.info(f"✓ Mirror failure count increased from {original_failure_count} to {test_mirror.failure_count}")
    
    recovery_system.mirror_manager.mark_mirror_success(test_mirror)
    logger.info(f"✓ Mirror success recorded, still available: {test_mirror.available}")


def demo_authentication_configuration(recovery_system: NetworkFailureRecovery, logger: logging.Logger):
    """Demonstrate authentication configuration."""
    logger.info("=" * 60)
    logger.info("DEMO: Authentication Configuration")
    logger.info("=" * 60)
    
    # Configure different authentication methods
    logger.info("Configuring token-based authentication...")
    recovery_system.configure_authentication(
        token="demo_token_12345",
        headers={"Authorization": "Bearer demo_token_12345"}
    )
    logger.info("✓ Token authentication configured")
    
    logger.info("\nConfiguring API key authentication...")
    recovery_system.configure_authentication(
        api_key="demo_api_key_67890",
        headers={"X-API-Key": "demo_api_key_67890"}
    )
    logger.info("✓ API key authentication configured")
    
    logger.info("\nConfiguring basic authentication...")
    recovery_system.configure_authentication(
        username="demo_user",
        password="demo_password"
    )
    logger.info("✓ Basic authentication configured")


def demo_proxy_configuration(recovery_system: NetworkFailureRecovery, logger: logging.Logger):
    """Demonstrate proxy configuration."""
    logger.info("=" * 60)
    logger.info("DEMO: Proxy Configuration")
    logger.info("=" * 60)
    
    # Configure proxy settings
    logger.info("Configuring proxy settings...")
    recovery_system.configure_proxy(
        http_proxy="http://proxy.example.com:8080",
        https_proxy="https://proxy.example.com:8080",
        no_proxy="localhost,127.0.0.1,*.local"
    )
    logger.info("✓ Proxy configuration updated")
    
    # Show current proxy configuration
    proxy_config = recovery_system.proxy_config
    logger.info(f"Current proxy configuration:")
    logger.info(f"  - HTTP proxy: {proxy_config.http_proxy}")
    logger.info(f"  - HTTPS proxy: {proxy_config.https_proxy}")
    logger.info(f"  - No proxy: {proxy_config.no_proxy}")


def demo_recovery_strategies(recovery_system: NetworkFailureRecovery, logger: logging.Logger):
    """Demonstrate recovery strategies for different error types."""
    logger.info("=" * 60)
    logger.info("DEMO: Recovery Strategies")
    logger.info("=" * 60)
    
    # Test recovery strategies for different error types
    error_types = [
        NetworkErrorType.AUTHENTICATION,
        NetworkErrorType.RATE_LIMITING,
        NetworkErrorType.PROXY_ERROR,
        NetworkErrorType.CONNECTION_TIMEOUT,
        NetworkErrorType.SERVER_ERROR
    ]
    
    for error_type in error_types:
        logger.info(f"\nRecovery strategies for {error_type.value}:")
        strategies = recovery_system._get_recovery_strategies(error_type, "https://example.com/file.txt")
        
        for i, (strategy_name, _) in enumerate(strategies, 1):
            logger.info(f"  {i}. {strategy_name}")


def demo_simulated_recovery(recovery_system: NetworkFailureRecovery, logger: logging.Logger):
    """Demonstrate simulated network failure recovery."""
    logger.info("=" * 60)
    logger.info("DEMO: Simulated Network Failure Recovery")
    logger.info("=" * 60)
    
    # Simulate a failing operation that eventually succeeds
    attempt_count = 0
    
    def simulated_failing_operation():
        nonlocal attempt_count
        attempt_count += 1
        
        if attempt_count == 1:
            raise urllib.error.HTTPError("url", 429, "Too Many Requests", {}, None)
        elif attempt_count == 2:
            raise socket.timeout("Connection timed out")
        else:
            return f"Success after {attempt_count} attempts!"
    
    logger.info("Simulating network operation with failures...")
    
    try:
        result = recovery_system.recover_from_network_failure(
            simulated_failing_operation,
            "simulated_download",
            "https://example.com/test_file.bin",
            Exception("Initial failure"),
            max_attempts=2
        )
        logger.info(f"✓ Recovery successful: {result}")
    except Exception as e:
        logger.error(f"✗ Recovery failed: {e}")
    
    # Show recovery statistics
    stats = recovery_system.get_recovery_statistics()
    logger.info(f"\nRecovery Statistics:")
    logger.info(f"  - Total attempts: {stats['total_attempts']}")
    logger.info(f"  - Successful recoveries: {stats['successful_recoveries']}")
    logger.info(f"  - Failed recoveries: {stats['failed_recoveries']}")
    logger.info(f"  - Success rate: {stats['success_rate']:.1%}")


def demo_download_progress_tracking(recovery_system: NetworkFailureRecovery, logger: logging.Logger):
    """Demonstrate download progress tracking."""
    logger.info("=" * 60)
    logger.info("DEMO: Download Progress Tracking")
    logger.info("=" * 60)
    
    # Create a temporary file for download simulation
    temp_dir = Path(tempfile.mkdtemp())
    try:
        destination = temp_dir / "demo_download.txt"
        
        # Simulate download progress tracking
        logger.info("Simulating download progress tracking...")
        
        # Create mock progress data
        from scripts.network_failure_recovery import DownloadProgress
        from datetime import datetime
        
        progress = DownloadProgress(
            url="https://example.com/demo_file.txt",
            filename="demo_file.txt",
            total_size=1024 * 1024,  # 1MB
            downloaded_size=512 * 1024,  # 512KB downloaded
            start_time=datetime.now(),
            current_speed_mbps=2.5,
            average_speed_mbps=2.0,
            eta_seconds=200,
            resume_count=1
        )
        
        # Add to active downloads
        download_id = f"{progress.url}_{progress.filename}"
        recovery_system.downloader.active_downloads[download_id] = progress
        
        logger.info(f"Download Progress:")
        logger.info(f"  - File: {progress.filename}")
        logger.info(f"  - Progress: {progress.downloaded_size / progress.total_size:.1%}")
        logger.info(f"  - Speed: {progress.current_speed_mbps:.1f} Mbps")
        logger.info(f"  - ETA: {progress.eta_seconds} seconds")
        logger.info(f"  - Resume count: {progress.resume_count}")
        
        # Show active downloads
        active_downloads = recovery_system.downloader.get_active_downloads()
        logger.info(f"\nActive downloads: {len(active_downloads)}")
        
        # Cancel the download
        success = recovery_system.downloader.cancel_download(download_id)
        logger.info(f"✓ Download cancelled: {success}")
        
    finally:
        shutil.rmtree(temp_dir)


def demo_comprehensive_statistics(recovery_system: NetworkFailureRecovery, logger: logging.Logger):
    """Demonstrate comprehensive statistics reporting."""
    logger.info("=" * 60)
    logger.info("DEMO: Comprehensive Statistics")
    logger.info("=" * 60)
    
    # Get comprehensive statistics
    stats = recovery_system.get_recovery_statistics()
    
    logger.info("Network Failure Recovery Statistics:")
    logger.info(f"  - Total recovery attempts: {stats['total_attempts']}")
    logger.info(f"  - Successful recoveries: {stats['successful_recoveries']}")
    logger.info(f"  - Failed recoveries: {stats['failed_recoveries']}")
    logger.info(f"  - Mirror switches: {stats['mirror_switches']}")
    logger.info(f"  - Resumed downloads: {stats['resumed_downloads']}")
    logger.info(f"  - Authentication fixes: {stats['authentication_fixes']}")
    logger.info(f"  - Proxy fixes: {stats['proxy_fixes']}")
    logger.info(f"  - Success rate: {stats['success_rate']:.1%}")
    logger.info(f"  - Active downloads: {stats['active_downloads']}")
    
    # Show mirror statistics
    mirror_stats = stats['mirror_statistics']
    logger.info(f"\nMirror Statistics:")
    for service, service_stats in mirror_stats.items():
        logger.info(f"  {service}: {service_stats['available_mirrors']}/{service_stats['total_mirrors']} available")


def main():
    """Run the comprehensive network failure recovery demo."""
    logger = setup_logging()
    
    logger.info("🚀 Starting Network Failure Recovery System Demo")
    logger.info("=" * 80)
    
    # Create temporary directory for demo
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Initialize the network failure recovery system
        logger.info("Initializing Network Failure Recovery System...")
        recovery_system = NetworkFailureRecovery(str(temp_dir), logger)
        logger.info("✓ Network Failure Recovery System initialized successfully")
        
        # Run all demo sections
        demo_network_connectivity_testing(recovery_system, logger)
        demo_error_detection(recovery_system, logger)
        demo_mirror_management(recovery_system, logger)
        demo_authentication_configuration(recovery_system, logger)
        demo_proxy_configuration(recovery_system, logger)
        demo_recovery_strategies(recovery_system, logger)
        demo_simulated_recovery(recovery_system, logger)
        demo_download_progress_tracking(recovery_system, logger)
        demo_comprehensive_statistics(recovery_system, logger)
        
        logger.info("=" * 80)
        logger.info("🎉 Network Failure Recovery System Demo completed successfully!")
        logger.info("=" * 80)
        
        # Final summary
        final_stats = recovery_system.get_recovery_statistics()
        logger.info("Final Demo Statistics:")
        logger.info(f"  - Total recovery attempts during demo: {final_stats['total_attempts']}")
        logger.info(f"  - Demo success rate: {final_stats['success_rate']:.1%}")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        logger.info("Demo cleanup completed")


if __name__ == "__main__":
    main()