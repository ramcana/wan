"""
UI application entry point for WAN2.2.
Direct launcher for the WAN2.2 UI application.
"""

import sys
import os
from pathlib import Path

# Add the application directory to Python path
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))

def main():
    """UI entry point for WAN2.2 application."""
    try:
        from wan22_ui import WAN22UI
        app = WAN22UI()
        app.run()
    except ImportError as e:
        print(f"Error importing WAN2.2 UI: {e}")
        print("Please ensure the installation completed successfully and all dependencies are installed.")
        print("You may need to install UI dependencies: pip install tkinter pillow opencv-python")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting WAN2.2 UI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
