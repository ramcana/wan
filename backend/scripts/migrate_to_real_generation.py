#!/usr/bin/env python3
"""
Migration script to transition from mock to real AI generation mode.
This script handles configuration updates, database migrations, and validation.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from backend.core.system_integration import SystemIntegration
from backend.core.configuration_bridge import ConfigurationBridge
from backend.core.model_integration_bridge import ModelIntegrationBridge
from database.database import get_database
from models.generation_task import GenerationTaskDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealGenerationMigrator:
    """Handles migration from mock to real AI generation mode."""
    
    def __init__(self):
        self.config_path = Path("config.json")
        self.backup_path = Path("config_backup.json")
        self.system_integration = None
        self.config_bridge = None
        
    async def initialize(self) -> bool:
        """Initialize migration components."""
        try:
            self.system_integration = SystemIntegration()
            await self.system_integration.initialize()
            
            self.config_bridge = ConfigurationBridge()
            await self.config_bridge.initialize()
            
            logger.info("Migration components initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize migration components: {e}")
            return False
    
    def backup_current_config(self) -> bool:
        """Create backup of current configuration."""
        try:
            if self.config_path.exists():
                import shutil
                shutil.copy2(self.config_path, self.backup_path)
                logger.info(f"Configuration backed up to {self.backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to backup configuration: {e}")
            return False
    
    def update_configuration_for_real_generation(self) -> bool:
        """Update configuration to enable real generation mode."""
        try:
            config = {}
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
            
            # Update generation settings
            config.update({
                "generation": {
                    "mode": "real",  # Changed from "mock"
                    "enable_real_models": True,
                    "fallback_to_mock": True,
                    "auto_download_models": True
                },
                "models": {
                    "auto_optimize": True,
                    "enable_offloading": True,
                    "vram_management": True,
                    "quantization_enabled": True
                },
                "hardware": {
                    "auto_detect": True,
                    "optimize_for_hardware": True,
                    "vram_limit_gb": None  # Auto-detect
                },
                "websocket": {
                    "enable_progress_updates": True,
                    "detailed_progress": True,
                    "resource_monitoring": True
                }
            })
            
            # Write updated configuration
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info("Configuration updated for real generation mode")
            return True
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False
    
    async def migrate_database_schema(self) -> bool:
        """Migrate database schema for enhanced task tracking."""
        try:
            db = await get_database()
            
            # Add new columns for real generation tracking
            migration_queries = [
                """
                ALTER TABLE generation_tasks 
                ADD COLUMN IF NOT EXISTS model_used VARCHAR(100);
                """,
                """
                ALTER TABLE generation_tasks 
                ADD COLUMN IF NOT EXISTS generation_time_seconds FLOAT DEFAULT 0.0;
                """,
                """
                ALTER TABLE generation_tasks 
                ADD COLUMN IF NOT EXISTS peak_vram_usage_mb FLOAT DEFAULT 0.0;
                """,
                """
                ALTER TABLE generation_tasks 
                ADD COLUMN IF NOT EXISTS optimizations_applied TEXT;
                """,
                """
                ALTER TABLE generation_tasks 
                ADD COLUMN IF NOT EXISTS error_category VARCHAR(50);
                """,
                """
                ALTER TABLE generation_tasks 
                ADD COLUMN IF NOT EXISTS recovery_suggestions TEXT;
                """
            ]
            
            for query in migration_queries:
                await db.execute(query)
            
            await db.commit()
            logger.info("Database schema migrated successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to migrate database schema: {e}")
            return False
    
    async def validate_system_components(self) -> Dict[str, bool]:
        """Validate that all system components are working."""
        validation_results = {}
        
        try:
            # Test system integration
            validation_results["system_integration"] = await self._test_system_integration()
            
            # Test model bridge
            validation_results["model_bridge"] = await self._test_model_bridge()
            
            # Test configuration bridge
            validation_results["config_bridge"] = await self._test_config_bridge()
            
            # Test database connection
            validation_results["database"] = await self._test_database()
            
            # Test model availability
            validation_results["models"] = await self._test_model_availability()
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_results["error"] = str(e)
        
        return validation_results
    
    async def _test_system_integration(self) -> bool:
        """Test system integration functionality."""
        try:
            status = await self.system_integration.get_system_status()
            return status.get("initialized", False)
        except Exception as e:
            logger.error(f"System integration test failed: {e}")
            return False
    
    async def _test_model_bridge(self) -> bool:
        """Test model integration bridge."""
        try:
            bridge = await self.system_integration.get_model_bridge()
            if bridge:
                # Test basic functionality
                status = bridge.get_system_model_status()
                return isinstance(status, dict)
            return False
        except Exception as e:
            logger.error(f"Model bridge test failed: {e}")
            return False
    
    async def _test_config_bridge(self) -> bool:
        """Test configuration bridge."""
        try:
            config = await self.config_bridge.get_current_config()
            return isinstance(config, dict) and "generation" in config
        except Exception as e:
            logger.error(f"Config bridge test failed: {e}")
            return False
    
    async def _test_database(self) -> bool:
        """Test database connectivity."""
        try:
            db = await get_database()
            # Simple query to test connection
            result = await db.fetch_one("SELECT 1 as test")
            return result is not None
        except Exception as e:
            logger.error(f"Database test failed: {e}")
            return False
    
    async def _test_model_availability(self) -> bool:
        """Test model availability and download capability."""
        try:
            bridge = await self.system_integration.get_model_bridge()
            if bridge:
                # Check if at least one model type is available or can be downloaded
                model_types = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
                for model_type in model_types:
                    try:
                        available = await bridge.ensure_model_available(model_type)
                        if available:
                            return True
                    except Exception:
                        continue
            return False
        except Exception as e:
            logger.error(f"Model availability test failed: {e}")
            return False
    
    def rollback_configuration(self) -> bool:
        """Rollback configuration to backup."""
        try:
            if self.backup_path.exists():
                import shutil
                shutil.copy2(self.backup_path, self.config_path)
                logger.info("Configuration rolled back from backup")
                return True
            else:
                logger.warning("No backup found to rollback")
                return False
        except Exception as e:
            logger.error(f"Failed to rollback configuration: {e}")
            return False
    
    async def run_migration(self) -> bool:
        """Run complete migration process."""
        logger.info("Starting migration to real AI generation mode...")
        
        # Step 1: Initialize components
        if not await self.initialize():
            logger.error("Failed to initialize migration components")
            return False
        
        # Step 2: Backup current configuration
        if not self.backup_current_config():
            logger.error("Failed to backup configuration")
            return False
        
        # Step 3: Update configuration
        if not self.update_configuration_for_real_generation():
            logger.error("Failed to update configuration")
            self.rollback_configuration()
            return False
        
        # Step 4: Migrate database schema
        if not await self.migrate_database_schema():
            logger.error("Failed to migrate database schema")
            self.rollback_configuration()
            return False
        
        # Step 5: Validate system components
        validation_results = await self.validate_system_components()
        
        # Check validation results
        failed_components = [k for k, v in validation_results.items() if not v]
        if failed_components:
            logger.error(f"Validation failed for components: {failed_components}")
            logger.info("Migration completed with warnings. Some components may need manual configuration.")
            return False
        
        logger.info("Migration to real AI generation mode completed successfully!")
        logger.info("You can now use real AI models for video generation.")
        return True

async def main():
    """Main migration function."""
    migrator = RealGenerationMigrator()
    
    try:
        success = await migrator.run_migration()
        if success:
            print("\n✅ Migration completed successfully!")
            print("Your system is now configured for real AI generation.")
            print("You can start the FastAPI server to begin using real models.")
        else:
            print("\n❌ Migration failed or completed with warnings.")
            print("Please check the logs and resolve any issues before proceeding.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Migration interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Migration failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
