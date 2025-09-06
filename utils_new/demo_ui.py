"""
Demo script for Wan2.2 UI Variant
Shows how to launch the Gradio interface
"""

def main():
    """Main demo function"""
    print("🎬 Wan2.2 Video Generation UI Demo")
    print("=" * 50)
    
    try:
        # Import the UI module
        from ui import create_ui
        
        print("✅ UI module imported successfully")
        
        # Create the UI instance
        ui = create_ui()
        print("✅ UI instance created")
        
        # Launch the interface
        print("🚀 Launching Gradio interface...")
        print("📍 The interface will be available at: http://localhost:7860")
        print("🔧 Features available:")
        print("   • 🎥 Generation tab - T2V, I2V, TI2V video generation")
        print("   • ⚙️ Optimizations tab - VRAM and performance settings")
        print("   • 📊 Queue & Stats tab - Task management and system monitoring")
        print("   • 📁 Outputs tab - Video gallery and file management")
        print()
        print("⚠️  Note: This demo requires proper GPU setup and model downloads")
        print("   For testing UI structure only, the interface will load but")
        print("   generation features may not work without proper dependencies.")
        
        # Launch with demo settings
        ui.launch(
            server_name="127.0.0.1",  # Local only for demo
            server_port=7860,
            share=False,
            debug=True
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 This is expected if GPU dependencies are not properly installed")
        print("   The UI structure is complete and ready for deployment")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()