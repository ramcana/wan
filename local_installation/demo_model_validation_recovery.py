"""
Demonstration of the Model Validation Recovery System.

This script demonstrates the key features of the ModelValidationRecovery class:
- Model issue identification (missing files, corruption, wrong versions)
- Automatic model re-download with integrity verification
- Model file repair and directory structure fixing
- Detailed model issue reporting when recovery fails

Requirements demonstrated: 4.1, 4.2, 4.3, 4.4, 4.5
"""

import logging
import tempfile
import shutil
from pathlib import Path
from scripts.model_validation_recovery import (
    ModelValidationRecovery, ModelIssueType, ModelIssue
)


def setup_logging():
    """Set up logging for the demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def create_demo_scenarios(recovery_system: ModelValidationRecovery, model_id: str):
    """Create different demo scenarios to showcase the recovery system."""
    model_path = recovery_system._get_model_path(model_id)
    
    scenarios = {
        "valid_model": lambda: create_valid_model(model_path),
        "missing_files": lambda: create_missing_files_scenario(model_path),
        "corrupted_files": lambda: create_corrupted_files_scenario(model_path),
        "incomplete_download": lambda: create_incomplete_download_scenario(model_path),
        "invalid_structure": lambda: create_invalid_structure_scenario(model_path),
    }
    
    return scenarios


def create_valid_model(model_path: Path):
    """Create a valid model for demonstration."""
    print(f"Creating valid model at: {model_path}")
    model_path.mkdir(parents=True, exist_ok=True)
    
    (model_path / "config.json").write_text('{"model_type": "demo", "version": "1.0"}')
    (model_path / "pytorch_model.bin").write_bytes(b"fake model weights" * 10000)
    (model_path / "tokenizer.json").write_text('{"tokenizer": "demo"}')
    
    print("✓ Created valid model with all required files")


def create_missing_files_scenario(model_path: Path):
    """Create a scenario with missing files."""
    print(f"Creating missing files scenario at: {model_path}")
    
    # Remove the entire directory to simulate missing model
    if model_path.exists():
        shutil.rmtree(model_path)
    
    print("✓ Created missing files scenario (no model directory)")


def create_corrupted_files_scenario(model_path: Path):
    """Create a scenario with corrupted files."""
    print(f"Creating corrupted files scenario at: {model_path}")
    model_path.mkdir(parents=True, exist_ok=True)
    
    (model_path / "config.json").write_text('{"model_type": "demo"}')
    (model_path / "pytorch_model.bin").write_text("")  # Empty file (corrupted)
    # Missing tokenizer.json
    
    print("✓ Created corrupted files scenario (empty model weights, missing tokenizer)")


def create_incomplete_download_scenario(model_path: Path):
    """Create a scenario with incomplete download indicators."""
    print(f"Creating incomplete download scenario at: {model_path}")
    model_path.mkdir(parents=True, exist_ok=True)
    
    (model_path / "config.json").write_text('{"model_type": "demo"}')
    (model_path / "pytorch_model.bin.tmp").write_bytes(b"incomplete download")
    (model_path / "tokenizer.json.part").write_text("partial download")
    
    print("✓ Created incomplete download scenario (temporary files present)")


def create_invalid_structure_scenario(model_path: Path):
    """Create a scenario with invalid directory structure."""
    print(f"Creating invalid structure scenario at: {model_path}")
    
    # Remove directory if it exists
    if model_path.exists():
        shutil.rmtree(model_path)
    
    # Create a file where directory should be
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("This should be a directory, not a file")
    
    print("✓ Created invalid structure scenario (file instead of directory)")


def demonstrate_validation(recovery_system: ModelValidationRecovery, model_id: str, scenario_name: str):
    """Demonstrate model validation for a specific scenario."""
    print(f"\n{'='*60}")
    print(f"DEMONSTRATING: {scenario_name.upper().replace('_', ' ')}")
    print(f"{'='*60}")
    
    # Validate the model
    print("\n1. Running model validation...")
    validation_result = recovery_system.validate_model(model_id)
    
    print(f"   Model ID: {validation_result.model_id}")
    print(f"   Valid: {validation_result.is_valid}")
    print(f"   File Count: {validation_result.file_count}")
    print(f"   Total Size: {validation_result.total_size_mb:.2f} MB")
    print(f"   Structure Valid: {validation_result.structure_valid}")
    print(f"   Required Files Present: {validation_result.required_files_present}")
    print(f"   Issues Found: {len(validation_result.issues)}")
    
    if validation_result.issues:
        print("\n   Issues detected:")
        for i, issue in enumerate(validation_result.issues, 1):
            print(f"   {i}. {issue.issue_type.value.upper()}")
            print(f"      Severity: {issue.severity}")
            print(f"      Description: {issue.description}")
            if issue.file_path:
                print(f"      File: {issue.file_path}")
    
    return validation_result


def demonstrate_recovery(recovery_system: ModelValidationRecovery, model_id: str, validation_result):
    """Demonstrate model recovery for issues found."""
    if validation_result.is_valid:
        print("\n2. Model is valid - no recovery needed")
        return None
    
    print("\n2. Attempting model recovery...")
    
    # Mock the download method for demonstration (since we don't have actual models)
    def mock_download_success(model_id, recovery_result):
        print(f"   [MOCK] Successfully downloaded model: {model_id}")
        # Create a valid model after "download"
        model_path = recovery_system._get_model_path(model_id)
        if model_path.exists() and model_path.is_file():
            model_path.unlink()  # Remove file if it exists
        create_valid_model(model_path)
        recovery_result.recovery_method = "mock_download"
        recovery_result.bytes_downloaded = 1000000
        return True
    
    # Temporarily replace the download method
    original_method = recovery_system._download_model_with_retry
    recovery_system._download_model_with_retry = mock_download_success
    
    try:
        recovery_result = recovery_system.recover_model(model_id, validation_result)
        
        print(f"   Recovery Successful: {recovery_result.success}")
        print(f"   Recovery Method: {recovery_result.recovery_method}")
        print(f"   Issues Resolved: {[t.value for t in recovery_result.issues_resolved]}")
        print(f"   Issues Remaining: {len(recovery_result.issues_remaining)}")
        if recovery_result.bytes_downloaded > 0:
            print(f"   Bytes Downloaded: {recovery_result.bytes_downloaded:,}")
        print(f"   Details: {recovery_result.details}")
        
        return recovery_result
        
    finally:
        # Restore original method
        recovery_system._download_model_with_retry = original_method


def demonstrate_reporting(recovery_system: ModelValidationRecovery, model_id: str, 
                         validation_result, recovery_result=None):
    """Demonstrate detailed reporting capabilities."""
    print("\n3. Generating detailed report...")
    
    report = recovery_system.generate_detailed_report(model_id, validation_result, recovery_result)
    print("\n" + "="*50)
    print("DETAILED REPORT")
    print("="*50)
    print(report)
    
    if not validation_result.is_valid and (not recovery_result or not recovery_result.success):
        print("\n4. Getting recovery suggestions...")
        suggestions = recovery_system.get_recovery_suggestions(validation_result)
        print("\n" + "="*50)
        print("RECOVERY SUGGESTIONS")
        print("="*50)
        for suggestion in suggestions:
            print(suggestion)


def main():
    """Main demonstration function."""
    logger = setup_logging()
    
    print("Model Validation Recovery System Demonstration")
    print("=" * 60)
    print("This demo showcases the comprehensive model validation and recovery capabilities")
    print("designed to address the persistent '3 model issues' problem.\n")
    
    # Create temporary directory for demonstration
    with tempfile.TemporaryDirectory() as temp_dir:
        models_dir = Path(temp_dir) / "models"
        
        # Initialize the recovery system
        recovery_system = ModelValidationRecovery(
            installation_path=temp_dir,
            models_directory=str(models_dir),
            logger=logger
        )
        
        model_id = "Wan2.2/T2V-A14B"
        
        # Create demo scenarios
        scenarios = create_demo_scenarios(recovery_system, model_id)
        
        # Demonstrate each scenario
        for scenario_name, create_scenario in scenarios.items():
            try:
                # Set up the scenario
                create_scenario()
                
                # Demonstrate validation
                validation_result = demonstrate_validation(recovery_system, model_id, scenario_name)
                
                # Demonstrate recovery if needed
                recovery_result = demonstrate_recovery(recovery_system, model_id, validation_result)
                
                # Demonstrate reporting
                demonstrate_reporting(recovery_system, model_id, validation_result, recovery_result)
                
                # Wait for user input to continue (optional)
                input(f"\nPress Enter to continue to the next scenario...")
                
            except Exception as e:
                print(f"Error in scenario {scenario_name}: {e}")
                logger.exception(f"Error in scenario {scenario_name}")
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("The Model Validation Recovery System provides:")
    print("✓ Comprehensive model issue detection")
    print("✓ Automatic recovery with multiple strategies")
    print("✓ Integrity verification using checksums")
    print("✓ Detailed reporting and user guidance")
    print("✓ Support for various failure scenarios")
    print("\nThis system addresses requirements 4.1-4.5 for robust model validation and recovery.")


if __name__ == "__main__":
    main()
