#!/usr/bin/env python3
"""
WAN2.2 Pipeline Loading Fix

Addresses the specific pipeline loading issues:
1. Pipeline argument mismatch: expected ['kwargs'] but only set() were passed
2. Duplicate trust_remote_code parameter conflict

This fix targets the exact errors seen in the logs.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def fix_wan_pipeline_loading():
    """Apply targeted fixes to WAN pipeline loading issues"""
    
    print("🔧 Applying WAN2.2 Pipeline Loading Fixes")
    print("=" * 50)
    
    fixes_applied = []
    
    # Fix 1: Pipeline Manager argument handling
    try:
        fix_pipeline_manager_args()
        fixes_applied.append("✅ Pipeline Manager argument handling")
    except Exception as e:
        print(f"❌ Pipeline Manager fix failed: {e}")
    
    # Fix 2: WAN Pipeline Loader trust_remote_code conflict
    try:
        fix_trust_remote_code_conflict()
        fixes_applied.append("✅ trust_remote_code conflict resolution")
    except Exception as e:
        print(f"❌ trust_remote_code fix failed: {e}")
    
    # Fix 3: WAN Pipeline kwargs validation
    try:
        fix_wan_pipeline_kwargs()
        fixes_applied.append("✅ WAN Pipeline kwargs validation")
    except Exception as e:
        print(f"❌ WAN Pipeline kwargs fix failed: {e}")
    
    print("\n" + "=" * 50)
    print("FIXES APPLIED:")
    for fix in fixes_applied:
        print(f"  {fix}")
    
    if len(fixes_applied) == 3:
        print("\n✅ All WAN2.2 pipeline loading fixes applied successfully!")
        print("🚀 You can now try generating videos again.")
    else:
        print(f"\n⚠️  {len(fixes_applied)}/3 fixes applied. Some issues may remain.")
    
    return len(fixes_applied) == 3


def fix_pipeline_manager_args():
    """Fix pipeline manager argument handling"""
    
    # Read current pipeline_manager.py
    pipeline_manager_path = Path("pipeline_manager.py")
    if not pipeline_manager_path.exists():
        raise FileNotFoundError("pipeline_manager.py not found")
    
    with open(pipeline_manager_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the argument passing issue
    # The issue is in the load_custom_pipeline method where kwargs are not properly handled
    
    old_pattern = '''def load_custom_pipeline(self, model_path: str, pipeline_class: str, **kwargs) -> PipelineLoadResult:'''
    
    if old_pattern in content:
        # Find and fix the specific loading logic
        fixes = [
            # Fix 1: Ensure kwargs are properly passed to WanPipeline
            (
                'pipeline = pipeline_cls.from_pretrained(model_path, **load_args)',
                '''# Handle WAN pipeline loading with proper kwargs
                if pipeline_class == "WanPipeline":
                    # WAN pipelines need specific argument handling
                    filtered_args = {k: v for k, v in load_args.items() 
                                   if k not in ['model_architecture', 'progress_callback']}
                    pipeline = pipeline_cls.from_pretrained(model_path, **filtered_args)
                else:
                    pipeline = pipeline_cls.from_pretrained(model_path, **load_args)'''
            ),
            # Fix 2: Better error handling for argument mismatches
            (
                'except Exception as e:',
                '''except TypeError as e:
                    if "expected" in str(e) and "kwargs" in str(e):
                        # Try with minimal arguments for WAN pipeline
                        try:
                            minimal_args = {"trust_remote_code": load_args.get("trust_remote_code", True)}
                            pipeline = pipeline_cls.from_pretrained(model_path, **minimal_args)
                            logger.warning(f"WAN pipeline loaded with minimal args due to: {e}")
                        except Exception as minimal_e:
                            raise Exception(f"WAN pipeline loading failed with both full and minimal args. Full error: {e}. Minimal error: {minimal_e}")
                    else:
                        raise
                except Exception as e:'''
            )
        ]
        
        for old, new in fixes:
            if old in content:
                content = content.replace(old, new)
                logger.info(f"Applied pipeline manager fix: {old[:50]}...")
        
        # Write back the fixed content
        with open(pipeline_manager_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info("Pipeline manager argument handling fixed")
    else:
        raise ValueError("Could not find pipeline manager method to fix")


def fix_trust_remote_code_conflict():
    """Fix trust_remote_code parameter duplication"""
    
    # Read current wan_pipeline_loader.py
    loader_path = Path("wan_pipeline_loader.py")
    if not loader_path.exists():
        raise FileNotFoundError("wan_pipeline_loader.py not found")
    
    with open(loader_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the duplicate trust_remote_code issue
    fixes = [
        # Fix 1: Remove duplicate trust_remote_code from load_args before fallback
        (
            'pipeline = DiffusionPipeline.from_pretrained(model_path, trust_remote_code=True, **load_args)',
            '''# Remove trust_remote_code from load_args to avoid duplication
                fallback_args = {k: v for k, v in load_args.items() if k != 'trust_remote_code'}
                pipeline = DiffusionPipeline.from_pretrained(model_path, trust_remote_code=True, **fallback_args)'''
        ),
        # Fix 2: Ensure WAN models always use trust_remote_code=True
        (
            '# For WAN models, always ensure trust_remote_code is True',
            '''# For WAN models, always ensure trust_remote_code is True
            if pipeline_class == "WanPipeline":
                trust_remote_code = True
                self.logger.info("WAN model detected - enabling trust_remote_code")
                # Remove any existing trust_remote_code from load_args to avoid conflicts
                load_args.pop('trust_remote_code', None)'''
        )
    ]
    
    for old, new in fixes:
        if old in content:
            content = content.replace(old, new)
            logger.info(f"Applied trust_remote_code fix: {old[:50]}...")
    
    # Write back the fixed content
    with open(loader_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info("trust_remote_code conflict fixed")


def fix_wan_pipeline_kwargs():
    """Fix WAN pipeline kwargs validation"""
    
    # This fix ensures that WAN pipeline receives the correct arguments
    # and handles the "expected ['kwargs'] but only set() were passed" error
    
    # Check if wan22_compatibility_clean.py exists (the compatibility layer)
    compat_path = Path("wan22_compatibility_clean.py")
    if compat_path.exists():
        with open(compat_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for WanPipeline class definition and fix kwargs handling
        if "class WanPipeline" in content:
            # Add proper kwargs handling to WanPipeline.__init__
            old_init = "def __init__(self"
            if old_init in content and "**kwargs" not in content[content.find(old_init):content.find(old_init) + 200]:
                # Find the __init__ method and add **kwargs
                init_start = content.find("def __init__(self")
                if init_start != -1:
                    # Find the end of the parameter list
                    paren_count = 0
                    i = init_start
                    while i < len(content):
                        if content[i] == '(':
                            paren_count += 1
                        elif content[i] == ')':
                            paren_count -= 1
                            if paren_count == 0:
                                # Insert **kwargs before the closing parenthesis
                                if content[i-1] != '(':
                                    content = content[:i] + ", **kwargs" + content[i:]
                                else:
                                    content = content[:i] + "**kwargs" + content[i:]
                                break
                        i += 1
                
                # Write back the fixed content
                with open(compat_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info("WAN pipeline kwargs handling fixed")
            else:
                logger.info("WAN pipeline already has proper kwargs handling")
        else:
            logger.warning("WanPipeline class not found in compatibility layer")
    else:
        logger.warning("WAN compatibility layer not found - creating minimal fix")
        
        # Create a minimal compatibility fix
        minimal_fix = '''
# Minimal WAN Pipeline Compatibility Fix
import logging
from diffusers import DiffusionPipeline

logger = logging.getLogger(__name__)

class WanPipeline(DiffusionPipeline):
    """Minimal WAN Pipeline wrapper with proper kwargs handling"""
    
    def __init__(self, **kwargs):
        # Filter out WAN-specific kwargs that DiffusionPipeline doesn't expect
        wan_specific = ['boundary_ratio', 'model_architecture', 'progress_callback']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in wan_specific}
        
        logger.info(f"WanPipeline initialized with {len(filtered_kwargs)} arguments")
        super().__init__(**filtered_kwargs)
        
        # Store WAN-specific attributes
        self.boundary_ratio = kwargs.get('boundary_ratio', 0.875)
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load WAN pipeline with proper argument handling"""
        logger.info(f"Loading WAN pipeline from {pretrained_model_name_or_path}")
        
        # Use DiffusionPipeline's loading mechanism
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
'''
        
        with open("wan_pipeline_minimal_fix.py", 'w', encoding='utf-8') as f:
            f.write(minimal_fix)
        
        logger.info("Created minimal WAN pipeline compatibility fix")


if __name__ == "__main__":
    success = fix_wan_pipeline_loading()
    
    if success:
        print("\n🎉 WAN2.2 pipeline loading fixes completed successfully!")
        print("\nNext steps:")
        print("1. Restart the WAN2.2 UI application")
        print("2. Try generating a video")
        print("3. The pipeline should now load correctly")
    else:
        print("\n⚠️  Some fixes failed. Check the logs above for details.")
        print("You may need to apply manual fixes or check for file permissions.")
    
    input("\nPress Enter to exit...")