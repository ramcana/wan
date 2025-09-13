#!/usr/bin/env python3
"""
Startup script that applies model loading patches
Import this at the beginning of main.py or ui.py
"""

print("Applying emergency model loading fixes...")

try:
    from model_override import patch_model_loading
    success = patch_model_loading()
    if success:
        print("✅ Model loading patches applied successfully")
    else:
        print("⚠️  Model loading patches could not be applied")
except ImportError as e:
    print(f"⚠️  Could not import model override: {e}")
except Exception as e:
    print(f"❌ Failed to apply model patches: {e}")
    import traceback
    traceback.print_exc()
