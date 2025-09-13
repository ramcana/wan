#!/usr/bin/env python3
"""
WAN Model Validation and Recovery Script
Validates model integrity and provides recovery mechanisms for corrupted models.
"""

import os
import sys
import json
import logging
import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from core.models.wan_models.wan_model_config import get_wan_model_config
    from core.models.wan_models.wan_model_error_handler import WANModelErrorHandler
    from core.models.wan_models.wan_model_downloader import WANModelDownloader
except ImportError as e:
    logging.warning(f"Could not import WAN model components: {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelValidationRecovery:
    """Handles model validation and recovery operations"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.backup_dir = Path("model_backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        # Initialize components if available
        try:
            self.error_handler = WANModelErrorHandler()
            self.downloader = WANModelDownloader()
        except:
            self.error_handler = None
            self.downloader = None
            logger.warning("WAN model components not available - using fallback validation")
        
        # Expected model structure
        self.model_requirements = {
            "T2V-A14B": {
                "required_files": [
                    "model.safetensors",
                    "config.json",
                    "scheduler_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json"
                ],
                "min_size_mb": 25000,  # ~25GB
                "config_keys": ["model_type", "hidden_size", "num_attention_heads"]
            },
            "I2V-A14B": {
                "required_files": [
                    "model.safetensors", 
                    "config.json",
                    "scheduler_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "image_encoder.safetensors"
                ],
                "min_size_mb": 26000,  # ~26GB
                "config_keys": ["model_type", "hidden_size", "num_attention_heads"]
            },
            "TI2V-5B": {
                "required_files": [
                    "model.safetensors",
                    "config.json",
                    "scheduler_config.json", 
                    "tokenizer.json",
                    "tokenizer_config.json"
                ],
                "min_size_mb": 9000,   # ~9GB
                "config_keys": ["model_type", "hidden_size", "num_attention_heads"]
            }
        }
    
    def validate_model_structure(self, model_type: str) -> Dict[str, any]:
        """Validate basic model file structure"""
        if model_type not in self.model_requirements:
            return {"valid": False, "error": f"Unknown model type: {model_type}"}
        
        requirements = self.model_requirements[model_type]
        model_path = self.models_dir / model_type.lower().replace('-', '_')
        
        if not model_path.exists():
            return {"valid": False, "error": f"Model directory not found: {model_path}"}
        
        validation_result = {
            "valid": True,
            "model_path": str(model_path),
            "issues": [],
            "warnings": [],
            "file_status": {}
        }
        
        # Check required files
        missing_files = []
        corrupted_files = []
        total_size = 0
        
        for file_name in requirements["required_files"]:
            file_path = model_path / file_name
            file_status = {"exists": False, "size": 0, "readable": False}
            
            if file_path.exists():
                file_status["exists"] = True
                try:
                    file_size = file_path.stat().st_size
                    file_status["size"] = file_size
                    total_size += file_size
                    
                    # Test readability
                    with open(file_path, 'rb') as f:
                        f.read(1024)  # Read first 1KB to test
                    file_status["readable"] = True
                    
                except Exception as e:
                    corrupted_files.append(file_name)
                    validation_result["issues"].append(f"File {file_name} is corrupted: {e}")
                    file_status["readable"] = False
            else:
                missing_files.append(file_name)
                validation_result["issues"].append(f"Missing required file: {file_name}")
            
            validation_result["file_status"][file_name] = file_status
        
        # Check total size
        total_size_mb = total_size / (1024 * 1024)
        if total_size_mb < requirements["min_size_mb"]:
            validation_result["warnings"].append(
                f"Model size ({total_size_mb:.1f}MB) is smaller than expected ({requirements['min_size_mb']}MB)"
            )
        
        # Validate config.json if it exists
        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Check required config keys
                missing_keys = []
                for key in requirements["config_keys"]:
                    if key not in config:
                        missing_keys.append(key)
                
                if missing_keys:
                    validation_result["warnings"].append(f"Config missing keys: {missing_keys}")
                
                validation_result["config_valid"] = len(missing_keys) == 0
                
            except Exception as e:
                validation_result["issues"].append(f"Invalid config.json: {e}")
                validation_result["config_valid"] = False
        
        # Set overall validity
        if missing_files or corrupted_files:
            validation_result["valid"] = False
        
        validation_result["missing_files"] = missing_files
        validation_result["corrupted_files"] = corrupted_files
        validation_result["total_size_mb"] = total_size_mb
        
        return validation_result
    
    def validate_model_functionality(self, model_type: str) -> Dict[str, any]:
        """Validate model can be loaded and used"""
        result = {"functional": False, "error": None, "tests_passed": []}
        
        try:
            # Use error handler if available
            if self.error_handler:
                test_result = self.error_handler.test_model_loading(model_type)
                result["functional"] = test_result.get("success", False)
                result["error"] = test_result.get("error")
                result["tests_passed"] = test_result.get("tests_passed", [])
            else:
                # Fallback validation - just check if files can be opened
                validation = self.validate_model_structure(model_type)
                if validation["valid"] and not validation["corrupted_files"]:
                    result["functional"] = True
                    result["tests_passed"] = ["file_structure", "file_readability"]
                else:
                    result["error"] = "Model structure validation failed"
        
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Functional validation failed for {model_type}: {e}")
        
        return result
    
    def create_model_backup(self, model_type: str) -> bool:
        """Create backup of model before recovery attempts"""
        model_path = self.models_dir / model_type.lower().replace('-', '_')
        
        if not model_path.exists():
            logger.error(f"Cannot backup non-existent model: {model_type}")
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"{model_type}_{timestamp}"
        
        try:
            logger.info(f"Creating backup: {model_path} -> {backup_path}")
            shutil.copytree(model_path, backup_path)
            
            # Create backup info file
            info_file = backup_path / "backup_info.json"
            with open(info_file, 'w') as f:
                json.dump({
                    "model_type": model_type,
                    "original_path": str(model_path),
                    "backup_created": timestamp,
                    "backup_reason": "pre_recovery"
                }, f, indent=2)
            
            logger.info(f"Backup created successfully: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def recover_from_backup(self, model_type: str, backup_name: Optional[str] = None) -> bool:
        """Recover model from backup"""
        if backup_name:
            backup_path = self.backup_dir / backup_name
        else:
            # Find most recent backup
            backups = list(self.backup_dir.glob(f"{model_type}_*"))
            if not backups:
                logger.error(f"No backups found for {model_type}")
                return False
            
            backup_path = max(backups, key=lambda p: p.stat().st_mtime)
        
        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_path}")
            return False
        
        model_path = self.models_dir / model_type.lower().replace('-', '_')
        
        try:
            # Remove current model if it exists
            if model_path.exists():
                logger.info(f"Removing corrupted model: {model_path}")
                shutil.rmtree(model_path)
            
            # Restore from backup
            logger.info(f"Restoring from backup: {backup_path} -> {model_path}")
            shutil.copytree(backup_path, model_path)
            
            logger.info(f"Model {model_type} restored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False
    
    def repair_model(self, model_type: str) -> Dict[str, any]:
        """Attempt to repair a corrupted model"""
        result = {"success": False, "actions_taken": [], "remaining_issues": []}
        
        # First validate current state
        validation = self.validate_model_structure(model_type)
        
        if validation["valid"]:
            result["success"] = True
            result["actions_taken"].append("No repair needed - model is valid")
            return result
        
        logger.info(f"Attempting to repair model {model_type}")
        
        # Create backup before repair
        if not self.create_model_backup(model_type):
            result["remaining_issues"].append("Failed to create backup before repair")
        else:
            result["actions_taken"].append("Created backup before repair")
        
        # Try to fix missing files by re-downloading
        if validation["missing_files"]:
            logger.info(f"Attempting to re-download missing files: {validation['missing_files']}")
            
            if self.downloader:
                try:
                    download_result = self.downloader.download_missing_files(
                        model_type, validation["missing_files"]
                    )
                    if download_result.get("success"):
                        result["actions_taken"].append("Re-downloaded missing files")
                    else:
                        result["remaining_issues"].append("Failed to re-download missing files")
                except Exception as e:
                    result["remaining_issues"].append(f"Download error: {e}")
            else:
                result["remaining_issues"].append("Downloader not available for file recovery")
        
        # Try to fix corrupted files
        if validation["corrupted_files"]:
            logger.info(f"Attempting to fix corrupted files: {validation['corrupted_files']}")
            
            # For now, just try to re-download corrupted files
            if self.downloader:
                try:
                    download_result = self.downloader.download_missing_files(
                        model_type, validation["corrupted_files"]
                    )
                    if download_result.get("success"):
                        result["actions_taken"].append("Re-downloaded corrupted files")
                    else:
                        result["remaining_issues"].append("Failed to re-download corrupted files")
                except Exception as e:
                    result["remaining_issues"].append(f"Corruption repair error: {e}")
        
        # Re-validate after repair attempts
        final_validation = self.validate_model_structure(model_type)
        result["success"] = final_validation["valid"]
        
        if not result["success"]:
            result["remaining_issues"].extend(final_validation["issues"])
        
        return result
    
    def full_model_recovery(self, model_type: str) -> bool:
        """Complete model recovery process"""
        logger.info(f"Starting full recovery for model {model_type}")
        
        # Step 1: Try repair
        repair_result = self.repair_model(model_type)
        if repair_result["success"]:
            logger.info(f"Model {model_type} repaired successfully")
            return True
        
        # Step 2: Try backup recovery
        logger.info("Repair failed, attempting backup recovery")
        if self.recover_from_backup(model_type):
            logger.info(f"Model {model_type} recovered from backup")
            return True
        
        # Step 3: Complete re-download
        logger.info("Backup recovery failed, attempting complete re-download")
        if self.downloader:
            try:
                download_result = self.downloader.download_complete_model(model_type)
                if download_result.get("success"):
                    logger.info(f"Model {model_type} re-downloaded successfully")
                    return True
            except Exception as e:
                logger.error(f"Complete re-download failed: {e}")
        
        logger.error(f"All recovery attempts failed for model {model_type}")
        return False
    
    def validate_all_models(self) -> Dict[str, Dict]:
        """Validate all available models"""
        results = {}
        
        for model_type in self.model_requirements.keys():
            logger.info(f"Validating {model_type}...")
            
            structure_result = self.validate_model_structure(model_type)
            functional_result = self.validate_model_functionality(model_type)
            
            results[model_type] = {
                "structure": structure_result,
                "functional": functional_result,
                "overall_status": "healthy" if structure_result["valid"] and functional_result["functional"] else "issues"
            }
        
        return results
    
    def generate_validation_report(self, results: Dict[str, Dict]) -> str:
        """Generate human-readable validation report"""
        report = []
        report.append("WAN Model Validation Report")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for model_type, result in results.items():
            status_icon = "✅" if result["overall_status"] == "healthy" else "❌"
            report.append(f"{status_icon} {model_type}")
            report.append("-" * 30)
            
            # Structure validation
            struct = result["structure"]
            if struct["valid"]:
                report.append("  Structure: ✅ Valid")
                report.append(f"    Size: {struct['total_size_mb']:.1f}MB")
            else:
                report.append("  Structure: ❌ Issues found")
                for issue in struct["issues"]:
                    report.append(f"    - {issue}")
            
            # Functional validation
            func = result["functional"]
            if func["functional"]:
                report.append("  Functionality: ✅ Working")
                report.append(f"    Tests passed: {', '.join(func['tests_passed'])}")
            else:
                report.append("  Functionality: ❌ Not working")
                if func["error"]:
                    report.append(f"    Error: {func['error']}")
            
            report.append("")
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Validate and recover WAN models")
    parser.add_argument("--model", choices=["T2V-A14B", "I2V-A14B", "TI2V-5B"],
                       help="Specific model to validate/recover")
    parser.add_argument("--validate", action="store_true", help="Validate models")
    parser.add_argument("--repair", action="store_true", help="Repair corrupted models")
    parser.add_argument("--recover", action="store_true", help="Full recovery process")
    parser.add_argument("--backup", action="store_true", help="Create model backup")
    parser.add_argument("--restore", help="Restore from specific backup")
    parser.add_argument("--report", action="store_true", help="Generate validation report")
    parser.add_argument("--models-dir", default="models", help="Models directory")
    
    args = parser.parse_args()
    
    validator = ModelValidationRecovery(args.models_dir)
    
    if args.validate or args.report:
        if args.model:
            # Validate specific model
            struct_result = validator.validate_model_structure(args.model)
            func_result = validator.validate_model_functionality(args.model)
            results = {args.model: {"structure": struct_result, "functional": func_result}}
        else:
            # Validate all models
            results = validator.validate_all_models()
        
        if args.report:
            report = validator.generate_validation_report(results)
            print(report)
        else:
            print(json.dumps(results, indent=2))
    
    elif args.backup:
        if not args.model:
            logger.error("--backup requires --model")
            sys.exit(1)
        
        success = validator.create_model_backup(args.model)
        sys.exit(0 if success else 1)
    
    elif args.restore:
        if not args.model:
            logger.error("--restore requires --model")
            sys.exit(1)
        
        success = validator.recover_from_backup(args.model, args.restore)
        sys.exit(0 if success else 1)
    
    elif args.repair:
        if not args.model:
            logger.error("--repair requires --model")
            sys.exit(1)
        
        result = validator.repair_model(args.model)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["success"] else 1)
    
    elif args.recover:
        if not args.model:
            logger.error("--recover requires --model")
            sys.exit(1)
        
        success = validator.full_model_recovery(args.model)
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
