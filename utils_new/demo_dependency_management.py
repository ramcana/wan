"""
Demo script showing DependencyManager functionality for Wan model compatibility
"""

import tempfile
import shutil
from pathlib import Path
from dependency_manager import DependencyManager


def demo_dependency_management():
    """Demonstrate key DependencyManager features"""
    
    print("=== Wan Model Dependency Management Demo ===\n")
    
    # Create temporary cache directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Initialize DependencyManager
        dm = DependencyManager(cache_dir=temp_dir, trust_mode="safe")
        print(f"✓ DependencyManager initialized with cache: {temp_dir}")
        print(f"✓ Trust mode: {dm.trust_mode}")
        print(f"✓ Trusted sources: {list(dm.trusted_sources)}\n")
        
        # Demo 1: Check remote code availability
        print("1. Checking remote code availability...")
        model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        status = dm.check_remote_code_availability(model_id)
        print(f"   Model: {model_id}")
        print(f"   Available: {status.is_available}")
        if status.error_message:
            print(f"   Error: {status.error_message}")
        print()
        
        # Demo 2: Security validation
        print("2. Security validation...")
        trusted_result = dm._validate_source_security("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
        print(f"   Trusted source (Wan-AI): Safe={trusted_result.is_safe}, Risk={trusted_result.risk_level}")
        
        untrusted_result = dm._validate_source_security("untrusted.com/malicious-model")
        print(f"   Untrusted source: Safe={untrusted_result.is_safe}, Risk={untrusted_result.risk_level}")
        if untrusted_result.detected_risks:
            print(f"   Detected risks: {untrusted_result.detected_risks}")
        print()
        
        # Demo 3: Version compatibility
        print("3. Version compatibility checking...")
        # Create a test pipeline file
        test_file = Path(temp_dir) / "test_pipeline.py"
        test_file.write_text('__version__ = "1.0.0"\nclass TestPipeline: pass')
        
        compat_result = dm.validate_code_version(str(test_file), "1.0.0")
        print(f"   Exact version match: Compatible={compat_result.is_compatible}, Score={compat_result.compatibility_score}")
        
        compat_result2 = dm.validate_code_version(str(test_file), "2.0.0")
        print(f"   Version mismatch: Compatible={compat_result2.is_compatible}, Score={compat_result2.compatibility_score:.2f}")
        if compat_result2.warnings:
            print(f"   Warnings: {compat_result2.warnings[0]}")
        print()
        
        # Demo 4: Fallback options
        print("4. Fallback options...")
        fallback_options = dm.get_fallback_options("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
        print("   Available fallback strategies:")
        for i, option in enumerate(fallback_options[:3], 1):  # Show first 3
            print(f"   {i}. {option}")
        print()
        
        # Demo 5: Fetch with security restrictions
        print("5. Fetch with security restrictions...")
        result = dm.fetch_pipeline_code("untrusted.com/model", trust_remote_code=True)
        print(f"   Fetch from untrusted source: Success={result.success}")
        if not result.success:
            print(f"   Error: {result.error_message}")
            print(f"   Fallback options available: {len(result.fallback_options) if result.fallback_options else 0}")
        print()
        
        # Demo 6: Trust mode disabled
        print("6. Remote code fetching disabled...")
        result = dm.fetch_pipeline_code("Wan-AI/Wan2.2-T2V-A14B-Diffusers", trust_remote_code=False)
        print(f"   Fetch with trust_remote_code=False: Success={result.success}")
        if not result.success:
            print(f"   Error: {result.error_message}")
        print()
        
        print("=== Demo completed successfully! ===")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    demo_dependency_management()