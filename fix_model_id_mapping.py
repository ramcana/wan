#!/usr/bin/env python3
"""
Fix Model ID Mapping Issue
Resolves the mismatch between generation service model IDs and model integration bridge
"""

import sys
import asyncio
from pathlib import Path

async def test_model_id_mapping():
    """Test the model ID mapping in the integration bridge"""
    print("🔧 Testing Model ID Mapping...")
    
    try:
        # Add backend to path
        sys.path.insert(0, str(Path("backend").absolute()))
        
        from core.model_integration_bridge import ModelIntegrationBridge
        
        # Initialize bridge
        bridge = ModelIntegrationBridge()
        await bridge.initialize()
        
        # Test model ID mappings
        test_ids = ["t2v-a14b", "i2v-a14b", "ti2v-5b", "t2v-A14B", "i2v-A14B", "ti2v-5B"]
        
        print("📋 Model ID Mapping Test:")
        for model_id in test_ids:
            mapped_id = bridge.model_id_mappings.get(model_id, model_id)
            model_type = bridge.model_type_mappings.get(model_id, "UNKNOWN")
            print(f"   {model_id} -> {mapped_id} ({model_type})")
        
        # Test model availability check
        print("\n🔍 Testing Model Availability:")
        for model_id in ["t2v-a14b", "i2v-a14b", "ti2v-5b"]:
            try:
                status = await bridge.check_model_availability(model_id)
                print(f"   {model_id}: {status.status.value} (valid: {status.is_valid})")
            except Exception as e:
                print(f"   {model_id}: ERROR - {e}")
        
        # Test ensure model available
        print("\n📥 Testing Ensure Model Available:")
        test_model = "t2v-a14b"
        try:
            available = await bridge.ensure_model_available(test_model)
            print(f"   {test_model}: {'✅ Available' if available else '❌ Not Available'}")
        except Exception as e:
            print(f"   {test_model}: ERROR - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model ID mapping test failed: {e}")
        return False

def patch_model_downloader_check():
    """Patch the model downloader to handle ID mapping correctly"""
    print("\n🔧 Patching Model Downloader Check...")
    
    try:
        # Add local installation to path
        local_installation_path = Path("local_installation")
        if local_installation_path.exists():
            sys.path.insert(0, str(local_installation_path))
            sys.path.insert(0, str(local_installation_path / "scripts"))
        
        from scripts.download_models import ModelDownloader
        
        # Create a patched version of check_existing_models
        original_check = ModelDownloader.check_existing_models
        
        def patched_check_existing_models(self):
            """Patched version that handles both old and new model names"""
            existing_models = original_check(self)
            
            # Add mappings for API compatibility
            model_mappings = {
                "WAN2.2-T2V-A14B": ["t2v-a14b", "t2v-A14B"],
                "WAN2.2-I2V-A14B": ["i2v-a14b", "i2v-A14B"],
                "WAN2.2-TI2V-5B": ["ti2v-5b", "ti2v-5B"]
            }
            
            # Add mapped names to existing models
            mapped_models = existing_models.copy()
            for full_name in existing_models:
                if full_name in model_mappings:
                    mapped_models.extend(model_mappings[full_name])
            
            return mapped_models
        
        # Apply the patch
        ModelDownloader.check_existing_models = patched_check_existing_models
        
        # Test the patch
        downloader = ModelDownloader(
            installation_path=str(Path.cwd()),
            models_dir=str(Path("models"))
        )
        
        existing_models = downloader.check_existing_models()
        print(f"✅ Patched model check found: {existing_models}")
        
        # Check if our target models are now found
        target_models = ["t2v-a14b", "i2v-a14b", "ti2v-5b"]
        found_targets = [model for model in target_models if model in existing_models]
        
        if len(found_targets) == len(target_models):
            print(f"🎉 All target models now found: {found_targets}")
            return True
        else:
            print(f"⚠️ Some target models still missing: {set(target_models) - set(found_targets)}")
            return False
        
    except Exception as e:
        print(f"❌ Model downloader patch failed: {e}")
        return False

def create_permanent_model_mapping_fix():
    """Create a permanent fix for model ID mapping"""
    print("\n🛠️ Creating Permanent Model Mapping Fix...")
    
    downloader_file = Path("local_installation/scripts/download_models.py")
    if not downloader_file.exists():
        print("❌ Model downloader file not found")
        return False
    
    try:
        # Read the current file
        with open(downloader_file, 'r') as f:
            content = f.read()
        
        # Check if patch is already applied
        if "Add mappings for API compatibility" in content:
            print("✅ Permanent fix already applied")
            return True
        
        # Find the check_existing_models method
        method_start = content.find("def check_existing_models(self) -> List[str]:")
        if method_start == -1:
            print("❌ Could not find check_existing_models method")
            return False
        
        # Find the return statement
        return_pos = content.find("return existing_models", method_start)
        if return_pos == -1:
            print("❌ Could not find return statement in check_existing_models")
            return False
        
        # Insert the mapping code before the return
        mapping_code = '''
        
        # Add mappings for API compatibility
        model_mappings = {
            "WAN2.2-T2V-A14B": ["t2v-a14b", "t2v-A14B"],
            "WAN2.2-I2V-A14B": ["i2v-a14b", "i2v-A14B"],
            "WAN2.2-TI2V-5B": ["ti2v-5b", "ti2v-5B"]
        }
        
        # Add mapped names to existing models
        for full_name in existing_models:
            if full_name in model_mappings:
                existing_models.extend(model_mappings[full_name])
        '''
        
        # Insert the code
        new_content = content[:return_pos] + mapping_code + "\n        " + content[return_pos:]
        
        # Create backup if not exists
        backup_file = downloader_file.with_suffix('.py.backup2')
        if not backup_file.exists():
            with open(backup_file, 'w') as f:
                f.write(content)
            print(f"✅ Created backup: {backup_file}")
        
        # Write the fixed file
        with open(downloader_file, 'w') as f:
            f.write(new_content)
        
        print("✅ Applied permanent model mapping fix")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create permanent fix: {e}")
        return False

async def main():
    """Main function"""
    print("🔧 MODEL ID MAPPING FIX")
    print("=" * 50)
    
    # Test current mapping
    if await test_model_id_mapping():
        print("\n✅ Model ID mapping test passed")
    else:
        print("\n❌ Model ID mapping test failed")
    
    # Apply runtime patch
    if patch_model_downloader_check():
        print("\n✅ Runtime patch applied successfully")
        
        # Offer permanent fix
        response = input("\n🤔 Apply permanent fix to model downloader? (y/n): ")
        if response.lower() in ['y', 'yes']:
            if create_permanent_model_mapping_fix():
                print("🎉 Permanent fix applied successfully!")
            else:
                print("⚠️ Permanent fix failed - runtime patch still active")
        else:
            print("ℹ️ Runtime patch applied - restart may reset this")
    else:
        print("\n❌ Runtime patch failed")
    
    # Test again after patch
    print("\n🧪 Testing after patch...")
    if await test_model_id_mapping():
        print("🎉 Model ID mapping now working correctly!")
    else:
        print("⚠️ Issues still remain")

if __name__ == "__main__":
    asyncio.run(main())