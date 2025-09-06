#!/usr/bin/env python3
"""
Script to update import statements after functional reorganization
"""

import os
import re
from pathlib import Path

# Define import mappings
IMPORT_MAPPINGS = {
    # Old imports -> New imports
    'from database import': 'from backend.repositories.database import',
    'from models.schemas import': 'from backend.schemas.schemas import',
    'from services.generation_service import': 'from backend.services.generation_service import',
    'from core.system_integration import': 'from backend.core.system_integration import',
    'import utils': 'from core.services import utils',
    'from utils import': 'from core.services.utils import',
    'import error_handler': 'from infrastructure.hardware import error_handler',
    'from error_handler import': 'from infrastructure.hardware.error_handler import',
    'import architecture_detector': 'from infrastructure.hardware import architecture_detector',
    'from architecture_detector import': 'from infrastructure.hardware.architecture_detector import',
    'import wan_pipeline_loader': 'from core.services import wan_pipeline_loader',
    'from wan_pipeline_loader import': 'from core.services.wan_pipeline_loader import',
    'import optimization_manager': 'from core.services import optimization_manager',
    'from optimization_manager import': 'from core.services.optimization_manager import',
    'import performance_profiler': 'from infrastructure.hardware import performance_profiler',
    'from performance_profiler import': 'from infrastructure.hardware.performance_profiler import',
}

def update_file_imports(file_path):
    """Update imports in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply import mappings
        for old_import, new_import in IMPORT_MAPPINGS.items():
            content = content.replace(old_import, new_import)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated imports in: {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def update_imports_in_directory(directory):
    """Update imports in all Python files in a directory"""
    updated_count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if update_file_imports(file_path):
                    updated_count += 1
    
    return updated_count

if __name__ == "__main__":
    # Update imports in key directories
    directories_to_update = [
        "backend",
        "core", 
        "infrastructure",
        "frontend"
    ]
    
    total_updated = 0
    
    for directory in directories_to_update:
        if os.path.exists(directory):
            print(f"\nUpdating imports in {directory}/...")
            count = update_imports_in_directory(directory)
            total_updated += count
            print(f"Updated {count} files in {directory}/")
        else:
            print(f"Directory {directory}/ not found")
    
    print(f"\nTotal files updated: {total_updated}")
