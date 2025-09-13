#!/usr/bin/env python3
"""
WAN Model Weight Management CLI

Command-line interface for managing WAN model weights including downloading,
verification, caching, and updates.

Usage:
    python wan_weight_cli.py download <model_id> [--force] [--verify]
    python wan_weight_cli.py verify <model_id>
    python wan_weight_cli.py list [--cached-only]
    python wan_weight_cli.py cache-info [<model_id>]
    python wan_weight_cli.py cleanup [--max-size-gb <size>] [--retention-days <days>]
    python wan_weight_cli.py check-updates [<model_id>]
    python wan_weight_cli.py update <model_id> [--version <version>] [--strategy <strategy>]
    python wan_weight_cli.py rollback <model_id> <version>
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.models.wan_models.wan_weight_manager import (
    get_wan_weight_manager, download_wan_model, verify_wan_model
)
from backend.core.models.wan_models.wan_model_updater import (
    WANModelUpdater, MigrationStrategy, check_wan_model_updates, update_wan_model
)
from backend.core.models.wan_models.wan_model_config import get_available_wan_models, get_wan_model_info

logger = logging.getLogger(__name__)


class WANWeightCLI:
    """WAN Model Weight Management CLI"""
    
    def __init__(self, models_dir: Optional[str] = None):
        """Initialize CLI with optional models directory"""
        self.models_dir = models_dir
        self.weight_manager = None
        self.updater = None
    
    async def initialize(self):
        """Initialize weight manager and updater"""
        try:
            self.weight_manager = await get_wan_weight_manager(self.models_dir)
            self.updater = WANModelUpdater(self.weight_manager)
            return True
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            return False
    
    async def download_command(self, model_id: str, force: bool = False, verify: bool = True):
        """Download model weights"""
        try:
            print(f"📥 Downloading weights for {model_id}...")
            
            if force:
                print("🔄 Force redownload enabled")
            
            success = await self.weight_manager.download_model_weights(
                model_id, 
                force_redownload=force,
                verify_integrity=verify
            )
            
            if success:
                print(f"✅ Successfully downloaded {model_id}")
                
                if verify:
                    print("🔍 Verifying integrity...")
                    verified = await self.weight_manager.verify_model_integrity(model_id)
                    if verified:
                        print("✅ Integrity verification passed")
                    else:
                        print("❌ Integrity verification failed")
                        return False
            else:
                print(f"❌ Failed to download {model_id}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {model_id}: {e}")
            return False
    
    async def verify_command(self, model_id: str):
        """Verify model integrity"""
        try:
            print(f"🔍 Verifying integrity of {model_id}...")
            
            success = await self.weight_manager.verify_model_integrity(model_id)
            
            if success:
                print(f"✅ {model_id} integrity verification passed")
            else:
                print(f"❌ {model_id} integrity verification failed")
            
            return success
            
        except Exception as e:
            print(f"❌ Error verifying {model_id}: {e}")
            return False
    
    async def list_command(self, cached_only: bool = False):
        """List available or cached models"""
        try:
            if cached_only:
                print("📋 Cached WAN Models:")
                print("-" * 50)
                
                # List cached models
                cache_dir = self.weight_manager.cache_dir
                cache_file = cache_dir / "weight_cache.json"
                
                if cache_file.exists():
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                    
                    if cache_data:
                        for model_id in cache_data.keys():
                            cache_info = await self.weight_manager.get_model_cache_info(model_id)
                            if cache_info:
                                print(f"📦 {model_id}")
                                print(f"   Size: {cache_info['total_size_mb']:.1f} MB")
                                print(f"   Cached: {cache_info['cache_time']}")
                                print(f"   Accessed: {cache_info['access_count']} times")
                                print()
                    else:
                        print("No cached models found")
                else:
                    print("No cache file found")
            else:
                print("📋 Available WAN Models:")
                print("-" * 50)
                
                available_models = get_available_wan_models()
                for model_id in available_models:
                    model_info = get_wan_model_info(model_id)
                    if model_info:
                        print(f"🤖 {model_info['display_name']} ({model_id})")
                        print(f"   Description: {model_info['description']}")
                        print(f"   Parameters: {model_info['parameter_count']:,}")
                        print(f"   VRAM: {model_info['min_vram_gb']:.1f}GB - {model_info['vram_estimate_gb']:.1f}GB")
                        print(f"   Max Resolution: {model_info['max_resolution'][0]}x{model_info['max_resolution'][1]}")
                        print(f"   Max Frames: {model_info['max_frames']}")
                        
                        # Check if cached
                        cache_info = await self.weight_manager.get_model_cache_info(model_id)
                        if cache_info:
                            print(f"   Status: ✅ Cached ({cache_info['total_size_mb']:.1f} MB)")
                        else:
                            print(f"   Status: ⬇️ Not downloaded")
                        print()
            
            return True
            
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return False
    
    async def cache_info_command(self, model_id: Optional[str] = None):
        """Show cache information"""
        try:
            if model_id:
                print(f"📊 Cache Info for {model_id}:")
                print("-" * 50)
                
                cache_info = await self.weight_manager.get_model_cache_info(model_id)
                if cache_info:
                    print(f"Model ID: {cache_info['model_id']}")
                    print(f"Total Size: {cache_info['total_size_mb']:.1f} MB")
                    print(f"Cache Time: {cache_info['cache_time']}")
                    print(f"Last Accessed: {cache_info['last_accessed'] or 'Never'}")
                    print(f"Access Count: {cache_info['access_count']}")
                    print()
                    print("Weight Files:")
                    for weight_type, status in cache_info['weight_status'].items():
                        status_icon = "✅" if status['status'] in ['downloaded', 'verified'] else "❌"
                        print(f"  {status_icon} {weight_type}: {status['size_mb']:.1f} MB ({status['status']})")
                        if status['last_verified']:
                            print(f"     Last verified: {status['last_verified']}")
                else:
                    print(f"No cache info found for {model_id}")
            else:
                print("📊 Overall Cache Statistics:")
                print("-" * 50)
                
                # Calculate overall statistics
                cache_file = self.weight_manager.cache_dir / "weight_cache.json"
                if cache_file.exists():
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                    
                    total_models = len(cache_data)
                    total_size_mb = 0
                    
                    for model_id in cache_data.keys():
                        cache_info = await self.weight_manager.get_model_cache_info(model_id)
                        if cache_info:
                            total_size_mb += cache_info['total_size_mb']
                    
                    print(f"Total Models: {total_models}")
                    print(f"Total Size: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
                    print(f"Cache Directory: {self.weight_manager.cache_dir}")
                    print(f"Models Directory: {self.weight_manager.models_dir}")
                else:
                    print("No cache data found")
            
            return True
            
        except Exception as e:
            print(f"❌ Error getting cache info: {e}")
            return False
    
    async def cleanup_command(self, max_size_gb: Optional[float] = None, 
                            retention_days: Optional[int] = None):
        """Clean up cache"""
        try:
            print("🧹 Cleaning up cache...")
            
            cleanup_stats = await self.weight_manager.cleanup_cache(
                max_size_gb=max_size_gb,
                retention_days=retention_days
            )
            
            if "error" in cleanup_stats:
                print(f"❌ Cleanup failed: {cleanup_stats['error']}")
                return False
            
            print(f"✅ Cleanup completed:")
            print(f"   Cleaned models: {len(cleanup_stats['cleaned_models'])}")
            print(f"   Freed space: {cleanup_stats['freed_size_gb']:.2f} GB")
            print(f"   Remaining models: {cleanup_stats['remaining_models']}")
            print(f"   Remaining size: {cleanup_stats['remaining_size_gb']:.2f} GB")
            
            if cleanup_stats['cleaned_models']:
                print(f"   Removed models: {', '.join(cleanup_stats['cleaned_models'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
            return False
    
    async def check_updates_command(self, model_id: Optional[str] = None):
        """Check for model updates"""
        try:
            print("🔍 Checking for updates...")
            
            update_infos = await self.updater.check_for_updates(model_id)
            
            if not update_infos:
                print("No models to check or no updates available")
                return True
            
            print(f"📋 Update Status:")
            print("-" * 50)
            
            for mid, update_info in update_infos.items():
                if update_info.update_available:
                    print(f"🆕 {mid}: Update available")
                    print(f"   Current: {update_info.current_version}")
                    print(f"   Latest: {update_info.latest_version}")
                    print(f"   Download size: ~{update_info.estimated_download_size_mb:.0f} MB")
                    
                    if update_info.compatibility_issues:
                        print(f"   ⚠️ Compatibility issues: {', '.join(update_info.compatibility_issues)}")
                    
                    if update_info.version_info and update_info.version_info.changelog:
                        print(f"   Changes:")
                        for change in update_info.version_info.changelog:
                            print(f"     - {change}")
                else:
                    print(f"✅ {mid}: Up to date ({update_info.current_version})")
                print()
            
            return True
            
        except Exception as e:
            print(f"❌ Error checking updates: {e}")
            return False
    
    async def update_command(self, model_id: str, version: Optional[str] = None,
                           strategy: str = "conservative"):
        """Update a model"""
        try:
            print(f"🔄 Updating {model_id}...")
            
            # Parse strategy
            strategy_map = {
                "conservative": MigrationStrategy.CONSERVATIVE,
                "aggressive": MigrationStrategy.AGGRESSIVE,
                "parallel": MigrationStrategy.PARALLEL
            }
            
            migration_strategy = strategy_map.get(strategy.lower(), MigrationStrategy.CONSERVATIVE)
            
            success = await self.updater.update_model(
                model_id,
                target_version=version,
                strategy=migration_strategy
            )
            
            if success:
                print(f"✅ Successfully updated {model_id}")
                if version:
                    print(f"   Updated to version: {version}")
            else:
                print(f"❌ Failed to update {model_id}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error updating {model_id}: {e}")
            return False
    
    async def rollback_command(self, model_id: str, version: str):
        """Rollback a model to a previous version"""
        try:
            print(f"⏪ Rolling back {model_id} to version {version}...")
            
            success = await self.updater.rollback_model(model_id, version)
            
            if success:
                print(f"✅ Successfully rolled back {model_id} to version {version}")
            else:
                print(f"❌ Failed to rollback {model_id}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error rolling back {model_id}: {e}")
            return False


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="WAN Model Weight Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s download t2v-A14B --force --verify
  %(prog)s verify i2v-A14B
  %(prog)s list --cached-only
  %(prog)s cache-info t2v-A14B
  %(prog)s cleanup --max-size-gb 20 --retention-days 14
  %(prog)s check-updates
  %(prog)s update t2v-A14B --version 1.1.0 --strategy conservative
  %(prog)s rollback t2v-A14B 1.0.0
        """
    )
    
    parser.add_argument(
        "--models-dir",
        type=str,
        help="Directory for storing model weights"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download model weights")
    download_parser.add_argument("model_id", help="Model identifier")
    download_parser.add_argument("--force", action="store_true", help="Force redownload")
    download_parser.add_argument("--no-verify", action="store_true", help="Skip integrity verification")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify model integrity")
    verify_parser.add_argument("model_id", help="Model identifier")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List models")
    list_parser.add_argument("--cached-only", action="store_true", help="Show only cached models")
    
    # Cache info command
    cache_info_parser = subparsers.add_parser("cache-info", help="Show cache information")
    cache_info_parser.add_argument("model_id", nargs="?", help="Model identifier (optional)")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up cache")
    cleanup_parser.add_argument("--max-size-gb", type=float, help="Maximum cache size in GB")
    cleanup_parser.add_argument("--retention-days", type=int, help="Days to retain unused models")
    
    # Check updates command
    check_updates_parser = subparsers.add_parser("check-updates", help="Check for model updates")
    check_updates_parser.add_argument("model_id", nargs="?", help="Model identifier (optional)")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update model")
    update_parser.add_argument("model_id", help="Model identifier")
    update_parser.add_argument("--version", help="Target version (latest if not specified)")
    update_parser.add_argument("--strategy", choices=["conservative", "aggressive", "parallel"],
                              default="conservative", help="Migration strategy")
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback model to previous version")
    rollback_parser.add_argument("model_id", help="Model identifier")
    rollback_parser.add_argument("version", help="Target version")
    
    return parser


async def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create CLI instance
    cli = WANWeightCLI(models_dir=args.models_dir)
    
    # Initialize
    if not await cli.initialize():
        return 1
    
    # Execute command
    try:
        success = False
        
        if args.command == "download":
            success = await cli.download_command(
                args.model_id, 
                force=args.force, 
                verify=not args.no_verify
            )
        
        elif args.command == "verify":
            success = await cli.verify_command(args.model_id)
        
        elif args.command == "list":
            success = await cli.list_command(cached_only=args.cached_only)
        
        elif args.command == "cache-info":
            success = await cli.cache_info_command(args.model_id)
        
        elif args.command == "cleanup":
            success = await cli.cleanup_command(
                max_size_gb=args.max_size_gb,
                retention_days=args.retention_days
            )
        
        elif args.command == "check-updates":
            success = await cli.check_updates_command(args.model_id)
        
        elif args.command == "update":
            success = await cli.update_command(
                args.model_id,
                version=args.version,
                strategy=args.strategy
            )
        
        elif args.command == "rollback":
            success = await cli.rollback_command(args.model_id, args.version)
        
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user")
        return 1
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
