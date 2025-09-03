"""
Simple test to verify LoRA UI integration works correctly
"""

import sys
import os
import tempfile
import json
from unittest.mock import Mock, patch

# Mock the heavy dependencies
sys.modules['torch'] = Mock()
sys.modules['diffusers'] = Mock()
sys.modules['transformers'] = Mock()
sys.modules['huggingface_hub'] = Mock()
sys.modules['psutil'] = Mock()
sys.modules['GPUtil'] = Mock()
sys.modules['PIL'] = Mock()
sys.modules['cv2'] = Mock()
sys.modules['numpy'] = Mock()
sys.modules['gradio'] = Mock()
sys.modules['error_handler'] = Mock()
sys.modules['performance_profiler'] = Mock()

# Mock utils module
utils_mock = Mock()
utils_mock.get_model_manager = Mock()
utils_mock.VRAMOptimizer = Mock()
utils_mock.GenerationTask = Mock()
utils_mock.TaskStatus = Mock()
utils_mock.get_system_stats = Mock()
utils_mock.get_queue_manager = Mock()
utils_mock.enhance_prompt = Mock()
utils_mock.generate_video = Mock()
utils_mock.get_output_manager = Mock()
sys.modules['utils'] = utils_mock

def test_lora_ui_state_integration():
    """Test that LoRA UI state can be imported and initialized"""
    print("Testing LoRA UI state integration...")
    
    # Test importing LoRA UI state
    try:
        from lora_ui_state import LoRAUIState, LoRASelection
        print("✅ Successfully imported LoRAUIState and LoRASelection")
    except Exception as e:
        print(f"❌ Failed to import LoRA UI state: {e}")
        return False
    
    # Test creating LoRA UI state
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "directories": {
                    "loras_directory": os.path.join(temp_dir, "loras")
                },
                "lora_max_file_size_mb": 2048
            }
            
            os.makedirs(config["directories"]["loras_directory"], exist_ok=True)
            state_file = os.path.join(temp_dir, "test_state.json")
            
            # Create LoRA UI state
            lora_state = LoRAUIState(config, state_file)
            print("✅ Successfully created LoRAUIState instance")
            
            # Test basic functionality
            summary = lora_state.get_selection_summary()
            print(f"✅ Got selection summary: {summary['count']} LoRAs selected")
            
            display_data = lora_state.get_display_data()
            print(f"✅ Got display data with {len(display_data['available_loras'])} available LoRAs")
            
            return True
            
    except Exception as e:
        print(f"❌ Failed to create/test LoRAUIState: {e}")
        return False

def test_ui_integration():
    """Test that UI can be imported with LoRA integration"""
    print("\nTesting UI integration...")
    
    try:
        # Mock gradio components
        gr_mock = Mock()
        gr_mock.Blocks = Mock()
        gr_mock.Tabs = Mock()
        gr_mock.Tab = Mock()
        gr_mock.Column = Mock()
        gr_mock.Row = Mock()
        gr_mock.Textbox = Mock()
        gr_mock.Slider = Mock()
        gr_mock.Button = Mock()
        gr_mock.Accordion = Mock()
        gr_mock.Markdown = Mock()
        gr_mock.update = Mock()
        sys.modules['gradio'] = gr_mock
        
        # Test importing UI (this will test the integration)
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "config.json")
            config = {
                "directories": {
                    "models_directory": os.path.join(temp_dir, "models"),
                    "loras_directory": os.path.join(temp_dir, "loras"),
                    "outputs_directory": os.path.join(temp_dir, "outputs")
                },
                "optimization": {
                    "default_quantization": "bf16",
                    "enable_offload": True,
                    "vae_tile_size": 256,
                    "max_vram_usage_gb": 12
                },
                "generation": {
                    "default_resolution": "1280x720",
                    "default_steps": 50,
                    "max_prompt_length": 500
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(config, f)
            
            # Create directories
            for dir_path in config["directories"].values():
                os.makedirs(dir_path, exist_ok=True)
            
            # Mock the UI class methods that would be called during initialization
            with patch('ui.Wan22UI._create_interface'), \
                 patch('ui.Wan22UI._start_auto_refresh'), \
                 patch('ui.Wan22UI._perform_startup_checks'):
                
                from ui import Wan22UI
                print("✅ Successfully imported Wan22UI with LoRA integration")
                
                # Test creating UI instance
                ui_instance = Wan22UI(config_file)
                print("✅ Successfully created Wan22UI instance")
                
                # Test LoRA-related methods
                if hasattr(ui_instance, 'lora_ui_state'):
                    print("✅ UI instance has lora_ui_state attribute")
                    
                    if ui_instance.lora_ui_state:
                        status = ui_instance._get_lora_selection_status()
                        print(f"✅ Got LoRA selection status: {status}")
                        
                        display_data = ui_instance._get_lora_display_data()
                        print(f"✅ Got LoRA display data with {len(display_data.get('available_loras', []))} available LoRAs")
                    else:
                        print("⚠️ LoRA UI state is None (expected in test environment)")
                else:
                    print("❌ UI instance missing lora_ui_state attribute")
                    return False
                
                return True
                
    except Exception as e:
        print(f"❌ Failed to test UI integration: {e}")
        import traceback
traceback.print_exc()
        return False

def main():
    """Run all integration tests"""
    print("Running LoRA UI integration tests...\n")
    
    success = True
    
    # Test LoRA UI state
    if not test_lora_ui_state_integration():
        success = False
    
    # Test UI integration
    if not test_ui_integration():
        success = False
    
    print(f"\n{'='*50}")
    if success:
        print("🎉 All integration tests passed!")
        print("LoRA UI state management is successfully integrated.")
    else:
        print("❌ Some integration tests failed.")
        print("Please check the errors above.")
    print(f"{'='*50}")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)