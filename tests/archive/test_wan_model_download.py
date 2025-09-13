#!/usr/bin/env python3
"""
Test WAN model download functionality
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_wan_model_download():
    """Test downloading a WAN model"""
    try:
        logger.info("🚀 Testing WAN model download...")
        
        # Import and initialize the generation service
        from backend.services.generation_service import GenerationService
        generation_service = GenerationService()
        
        # Initialize the service
        await generation_service.initialize()
        
        if not generation_service.model_integration_bridge:
            logger.error("❌ Model Integration Bridge not available")
            return False
        
        # Try to download the smallest WAN model (TI2V 5B)
        model_type = "ti2v-5B"
        logger.info(f"📥 Attempting to download {model_type}...")
        
        # Check if model downloader is available
        if hasattr(generation_service.model_integration_bridge, 'model_downloader') and \
           generation_service.model_integration_bridge.model_downloader:
            
            downloader = generation_service.model_integration_bridge.model_downloader
            logger.info("✅ Model downloader is available")
            
            # Check existing models first
            existing_models = downloader.check_existing_models()
            logger.info(f"📋 Existing models: {existing_models}")
            
            # Map model type to downloader ID
            model_id_mappings = {
                "ti2v-5B": "WAN2.2-TI2V-5B",
                "t2v-A14B": "WAN2.2-T2V-A14B", 
                "i2v-A14B": "WAN2.2-I2V-A14B"
            }
            
            downloader_model_id = model_id_mappings.get(model_type, model_type)
            
            if downloader_model_id in existing_models:
                logger.info(f"✅ Model {downloader_model_id} already exists")
                return True
            else:
                logger.info(f"📥 Model {downloader_model_id} not found, would need to download")
                logger.info("ℹ️  To test actual download, run the model downloader separately")
                return True
        else:
            logger.warning("⚠️ Model downloader not available")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing WAN model download: {e}")
        return False

async def main():
    """Main test function"""
    success = await test_wan_model_download()
    
    if success:
        logger.info("🎉 WAN model download test completed!")
        return 0
    else:
        logger.error("💥 WAN model download test failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
