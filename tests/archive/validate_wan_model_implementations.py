#!/usr/bin/env python3
"""
WAN Model Implementation Validation Script

Quick validation of WAN model implementations to ensure they're properly integrated
and can be loaded/initialized correctly.
"""

import asyncio
import sys
import logging
from pathlib import Path
import json
import time

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WANModelValidator:
    """Validates WAN model implementations"""
    
    def __init__(self):
        self.validation_results = {}
    
    async def validate_wan_base_model(self):
        """Validate WAN base model implementation"""
        try:
            # Add the project root to the path
            project_root = Path(__file__).parent
            sys.path.insert(0, str(project_root))
            
            from core.models.wan_models.wan_base_model import WANBaseModel
            
            # Test basic instantiation
            base_model = WANBaseModel()
            
            # Check required methods exist
            required_methods = [
                'initialize', 'load_model', 'generate', 'get_model_info',
                'is_ready', 'cleanup', 'estimate_vram_usage'
            ]
            
            missing_methods = []
            for method in required_methods:
                if not hasattr(base_model, method):
                    missing_methods.append(method)
            
            return {
                "success": len(missing_methods) == 0,
                "missing_methods": missing_methods,
                "class_available": True
            }
            
        except ImportError as e:
            return {
                "success": False,
                "error": f"Import error: {e}",
                "class_available": False
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "class_available": True
            }
    
    async def validate_wan_t2v_model(self):
        """Validate WAN T2V-A14B model implementation"""
        try:
            from core.models.wan_models.wan_t2v_a14b import WANT2VA14B
            
            # Test instantiation with config
            from core.models.wan_models.wan_model_config import get_wan_model_config
            config = get_wan_model_config("t2v-A14B")
            model = WANT2VA14B(config)
            
            # Check model-specific methods
            t2v_methods = ['generate_video_from_text', 'encode_text_prompt']
            
            missing_methods = []
            for method in t2v_methods:
                if not hasattr(model, method):
                    missing_methods.append(method)
            
            # Test model info
            try:
                model_info = model.get_model_info()
                has_model_info = isinstance(model_info, dict)
            except:
                has_model_info = False
            
            return {
                "success": len(missing_methods) == 0,
                "missing_methods": missing_methods,
                "has_model_info": has_model_info,
                "class_available": True
            }
            
        except ImportError as e:
            return {
                "success": False,
                "error": f"Import error: {e}",
                "class_available": False
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "class_available": True
            }
    
    async def validate_wan_i2v_model(self):
        """Validate WAN I2V-A14B model implementation"""
        try:
            from core.models.wan_models.wan_i2v_a14b import WANI2VA14B
            
            from core.models.wan_models.wan_model_config import get_wan_model_config
            config = get_wan_model_config("i2v-A14B")
            model = WANI2VA14B(config)
            
            # Check I2V-specific methods
            i2v_methods = ['generate_video_from_image', 'encode_image']
            
            missing_methods = []
            for method in i2v_methods:
                if not hasattr(model, method):
                    missing_methods.append(method)
            
            return {
                "success": len(missing_methods) == 0,
                "missing_methods": missing_methods,
                "class_available": True
            }
            
        except ImportError as e:
            return {
                "success": False,
                "error": f"Import error: {e}",
                "class_available": False
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "class_available": True
            }
    
    async def validate_wan_ti2v_model(self):
        """Validate WAN TI2V-5B model implementation"""
        try:
            from core.models.wan_models.wan_ti2v_5b import WANTI2V5B
            
            from core.models.wan_models.wan_model_config import get_wan_model_config
            config = get_wan_model_config("ti2v-5B")
            model = WANTI2V5B(config)
            
            # Check TI2V-specific methods
            ti2v_methods = ['generate_video_from_text_and_image', 'encode_text_and_image']
            
            missing_methods = []
            for method in ti2v_methods:
                if not hasattr(model, method):
                    missing_methods.append(method)
            
            return {
                "success": len(missing_methods) == 0,
                "missing_methods": missing_methods,
                "class_available": True
            }
            
        except ImportError as e:
            return {
                "success": False,
                "error": f"Import error: {e}",
                "class_available": False
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "class_available": True
            }
    
    async def validate_wan_pipeline_factory(self):
        """Validate WAN pipeline factory"""
        try:
            from core.models.wan_models.wan_pipeline_factory import WANPipelineFactory
            
            factory = WANPipelineFactory()
            
            # Check factory methods
            factory_methods = ['create_wan_pipeline', 'get_available_models', 'get_cache_stats']
            
            missing_methods = []
            for method in factory_methods:
                if not hasattr(factory, method):
                    missing_methods.append(method)
            
            # Test available models
            try:
                available_models = factory.get_available_models()
                has_available_models = isinstance(available_models, (list, dict))
            except:
                has_available_models = False
            
            return {
                "success": len(missing_methods) == 0,
                "missing_methods": missing_methods,
                "has_available_models": has_available_models,
                "class_available": True
            }
            
        except ImportError as e:
            return {
                "success": False,
                "error": f"Import error: {e}",
                "class_available": False
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "class_available": True
            }
    
    async def validate_wan_model_config(self):
        """Validate WAN model configuration"""
        try:
            from core.models.wan_models.wan_model_config import WANModelConfig
            
            from core.models.wan_models.wan_model_config import get_wan_model_config
            # Test getting a model config instead of creating empty one
            config = get_wan_model_config("t2v-A14B")
            
            # Test that config is valid
            if config is None:
                return {
                    "success": False,
                    "error": "Could not get model config for t2v-A14B"
                }
            
            missing_methods = []  # Config is a dataclass, not a class with methods
            
            # Test model configurations
            from core.models.wan_models.wan_model_config import get_wan_model_config
            model_types = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
            config_results = {}
            
            for model_type in model_types:
                try:
                    model_config = get_wan_model_config(model_type)
                    config_results[model_type] = model_config is not None
                except:
                    config_results[model_type] = False
            
            return {
                "success": len(missing_methods) == 0,
                "missing_methods": missing_methods,
                "config_results": config_results,
                "class_available": True
            }
            
        except ImportError as e:
            return {
                "success": False,
                "error": f"Import error: {e}",
                "class_available": False
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "class_available": True
            }
    
    async def validate_wan_error_handler(self):
        """Validate WAN error handler"""
        try:
            from core.models.wan_models.wan_model_error_handler import WANModelErrorHandler
            
            error_handler = WANModelErrorHandler()
            
            # Check error handler methods
            handler_methods = ['handle_wan_error', 'get_wan_error_categories', 'get_wan_recovery_suggestions']
            
            missing_methods = []
            for method in handler_methods:
                if not hasattr(error_handler, method):
                    missing_methods.append(method)
            
            return {
                "success": len(missing_methods) == 0,
                "missing_methods": missing_methods,
                "class_available": True
            }
            
        except ImportError as e:
            return {
                "success": False,
                "error": f"Import error: {e}",
                "class_available": False
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "class_available": True
            }
    
    async def validate_integration_components(self):
        """Validate integration components"""
        try:
            # Test model integration bridge
            from backend.core.model_integration_bridge import ModelIntegrationBridge
            
            bridge = ModelIntegrationBridge()
            
            # Check if bridge has WAN model methods
            wan_methods = ['get_wan_model_status', 'load_wan_model_implementation']
            
            missing_methods = []
            for method in wan_methods:
                if not hasattr(bridge, method):
                    missing_methods.append(method)
            
            return {
                "success": len(missing_methods) == 0,
                "missing_methods": missing_methods,
                "bridge_available": True
            }
            
        except ImportError as e:
            return {
                "success": False,
                "error": f"Import error: {e}",
                "bridge_available": False
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "bridge_available": True
            }
    
    async def run_all_validations(self):
        """Run all WAN model validations"""
        logger.info("🚀 Starting WAN Model Implementation Validation...")
        
        validations = [
            ("WAN Base Model", self.validate_wan_base_model),
            ("WAN T2V-A14B Model", self.validate_wan_t2v_model),
            ("WAN I2V-A14B Model", self.validate_wan_i2v_model),
            ("WAN TI2V-5B Model", self.validate_wan_ti2v_model),
            ("WAN Pipeline Factory", self.validate_wan_pipeline_factory),
            ("WAN Model Config", self.validate_wan_model_config),
            ("WAN Error Handler", self.validate_wan_error_handler),
            ("Integration Components", self.validate_integration_components)
        ]
        
        results = {}
        
        for validation_name, validation_method in validations:
            try:
                logger.info(f"🧪 Validating {validation_name}...")
                
                start_time = time.time()
                result = await validation_method()
                duration = time.time() - start_time
                
                result["duration"] = duration
                results[validation_name] = result
                
                status = "✅" if result["success"] else "❌"
                logger.info(f"{status} {validation_name}: {'PASS' if result['success'] else 'FAIL'} ({duration:.2f}s)")
                
                if not result["success"] and "error" in result:
                    logger.info(f"  Error: {result['error']}")
                
                if "missing_methods" in result and result["missing_methods"]:
                    logger.info(f"  Missing methods: {result['missing_methods']}")
                
            except Exception as e:
                results[validation_name] = {
                    "success": False,
                    "error": str(e),
                    "duration": 0
                }
                logger.error(f"💥 {validation_name}: ERROR - {e}")
        
        # Generate summary
        total_validations = len(results)
        successful_validations = len([r for r in results.values() if r["success"]])
        success_rate = successful_validations / total_validations if total_validations > 0 else 0
        
        summary = {
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "failed_validations": total_validations - successful_validations,
            "success_rate": success_rate,
            "timestamp": time.time()
        }
        
        # Save results
        report = {
            "summary": summary,
            "validation_results": results
        }
        
        report_path = Path("wan_model_validation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        logger.info(f"\n{'='*50}")
        logger.info(f"🎯 WAN MODEL VALIDATION SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"📊 Results:")
        logger.info(f"  Total Validations: {total_validations}")
        logger.info(f"  ✅ Successful: {successful_validations}")
        logger.info(f"  ❌ Failed: {total_validations - successful_validations}")
        logger.info(f"  📈 Success Rate: {success_rate:.1%}")
        
        logger.info(f"\n📋 Validation Details:")
        for validation_name, result in results.items():
            status = "✅" if result["success"] else "❌"
            logger.info(f"  {status} {validation_name}")
            
            if not result["success"]:
                if "error" in result:
                    logger.info(f"    Error: {result['error']}")
                if "missing_methods" in result and result["missing_methods"]:
                    logger.info(f"    Missing: {', '.join(result['missing_methods'])}")
        
        if success_rate >= 0.8:
            logger.info(f"\n🎉 WAN Model implementations are ready for integration!")
        elif success_rate >= 0.6:
            logger.info(f"\n⚠️ WAN Model implementations need some fixes before full integration")
        else:
            logger.info(f"\n🚨 WAN Model implementations require significant work before integration")
        
        logger.info(f"\n📄 Validation report saved to: {report_path}")
        
        return success_rate >= 0.6

async def main():
    """Main validation function"""
    validator = WANModelValidator()
    
    try:
        success = await validator.run_all_validations()
        
        if success:
            logger.info("🎉 WAN model validation completed successfully!")
            return 0
        else:
            logger.error("💥 WAN model validation failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("🛑 Validation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"💥 Validation failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)