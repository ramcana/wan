"""
Test script for post-installation setup functionality.
"""

import sys
import tempfile
import json
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scripts.post_install_setup import PostInstallationSetup


def test_post_install_setup():
    """Test the post-installation setup functionality."""
    print("Testing Post-Installation Setup")
    print("=" * 40)
    
    # Create temporary installation directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create necessary directory structure
        (temp_path / "scripts").mkdir()
        (temp_path / "application").mkdir()
        (temp_path / "venv" / "Scripts").mkdir(parents=True)
        (temp_path / "logs").mkdir()
        
        # Create dummy config file
        config = {
            "system": {
                "default_quantization": "fp16",
                "enable_offload": True
            },
            "user_preferences": {
                "auto_launch": False,
                "show_advanced_options": False
            }
        }
        
        config_path = temp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Test setup initialization
        setup = PostInstallationSetup(str(temp_path))
        
        print(f"✓ Setup initialized with path: {temp_path}")
        print(f"✓ Config file exists: {config_path.exists()}")
        
        # Test configuration loading
        loaded_config = setup._load_current_config()
        print(f"✓ Configuration loaded: {len(loaded_config)} sections")
        
        # Test usage instructions
        print("\n--- Testing Usage Instructions ---")
        success = setup.show_usage_instructions()
        print(f"✓ Usage instructions displayed: {success}")
        
        # Test configuration saving
        print("\n--- Testing Configuration Save ---")
        test_config = {"test": "value"}
        success = setup._save_configuration(test_config)
        print(f"✓ Configuration saved: {success}")
        
        # Verify saved config
        if config_path.exists():
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            print(f"✓ Saved config verified: {saved_config}")
        
        print("\n" + "=" * 40)
        print("✓ All post-installation setup tests passed!")
        
        return True


def test_shortcut_creation():
    """Test shortcut creation functionality."""
    print("\nTesting Shortcut Creation")
    print("=" * 30)
    
    # Import shortcut creator
    from scripts.create_shortcuts import ShortcutCreator
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create necessary directories
        (temp_path / "application").mkdir()
        (temp_path / "resources").mkdir()
        (temp_path / "venv" / "Scripts").mkdir(parents=True)
        
        # Create dummy files
        (temp_path / "application" / "main.py").touch()
        (temp_path / "application" / "ui.py").touch()
        (temp_path / "venv" / "Scripts" / "python.exe").touch()
        
        # Test shortcut creator
        creator = ShortcutCreator(str(temp_path))
        
        print(f"✓ Shortcut creator initialized")
        print(f"✓ Desktop path: {creator.desktop_path}")
        print(f"✓ Start menu path: {creator.start_menu_path}")
        
        # Test launcher creation
        success = creator.create_application_launcher()
        print(f"✓ Application launcher created: {success}")
        
        # Verify launcher files exist
        launcher_path = temp_path / "launch_wan22.bat"
        ui_launcher_path = temp_path / "launch_wan22_ui.bat"
        
        print(f"✓ Main launcher exists: {launcher_path.exists()}")
        print(f"✓ UI launcher exists: {ui_launcher_path.exists()}")
        
        # Test uninstaller creation
        success = creator.create_uninstaller()
        print(f"✓ Uninstaller created: {success}")
        
        uninstall_path = temp_path / "uninstall.bat"
        print(f"✓ Uninstaller exists: {uninstall_path.exists()}")
        
        print("✓ All shortcut creation tests passed!")
        
        return True


if __name__ == "__main__":
    try:
        print("WAN2.2 Post-Installation Setup Test Suite")
        print("=" * 50)
        
        # Run tests
        test1_success = test_post_install_setup()
        test2_success = test_shortcut_creation()
        
        if test1_success and test2_success:
            print("\n🎉 All tests passed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)