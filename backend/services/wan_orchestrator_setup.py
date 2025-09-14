"""
WAN Pipeline Integration Setup.

This module provides setup functions to initialize the WAN pipeline integration
with the Model Orchestrator when both systems are available.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def setup_wan_orchestrator_integration() -> bool:
    """
    Set up WAN pipeline integration with Model Orchestrator.
    
    Returns:
        True if integration was successfully set up, False otherwise
    """
    try:
        # Import Model Orchestrator components
        from backend.core.model_orchestrator.model_ensurer import ModelEnsurer
        from backend.core.model_orchestrator.model_registry import ModelRegistry
        from backend.core.model_orchestrator.model_resolver import ModelResolver
        from backend.core.model_orchestrator.lock_manager import LockManager
        from backend.core.model_orchestrator.storage_backends.hf_store import HFStore
        from backend.services.wan_pipeline_integration import initialize_wan_integration
        
        # Get configuration from environment
        models_root = os.environ.get('MODELS_ROOT', './models')
        manifest_path = os.environ.get('WAN_MODELS_MANIFEST', './config/models.toml')
        
        # Initialize Model Orchestrator components
        logger.info("Initializing Model Orchestrator components...")
        
        registry = ModelRegistry(manifest_path)
        resolver = ModelResolver(models_root)
        lock_manager = LockManager(os.path.join(models_root, '.locks'))
        
        # Set up storage backends
        storage_backends = [HFStore()]
        
        # Create model ensurer
        ensurer = ModelEnsurer(
            registry=registry,
            resolver=resolver,
            lock_manager=lock_manager,
            storage_backends=storage_backends
        )
        
        # Initialize WAN pipeline integration
        initialize_wan_integration(ensurer, registry)
        
        logger.info("WAN pipeline integration with Model Orchestrator initialized successfully")
        return True
        
    except ImportError as e:
        logger.warning(f"Model Orchestrator not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize WAN pipeline integration: {e}")
        return False


def get_wan_model_path(model_id: str, variant: Optional[str] = None) -> Optional[str]:
    """
    Get the path for a WAN model, with fallback if Model Orchestrator is not available.
    
    Args:
        model_id: Model identifier
        variant: Optional variant
        
    Returns:
        Path to the model directory, or None if not available
    """
    try:
        from backend.services.wan_pipeline_integration import get_wan_paths
        return get_wan_paths(model_id, variant)
    except Exception as e:
        logger.warning(f"Could not get WAN model path via Model Orchestrator: {e}")
        
        # Fallback to environment-based path
        models_root = os.environ.get('MODELS_ROOT', './models')
        model_path = os.path.join(models_root, 'wan22', model_id.replace('@', '_'))
        
        if os.path.exists(model_path):
            return model_path
        
        logger.error(f"Model not found at fallback path: {model_path}")
        return None


def validate_wan_model_components(model_id: str, model_path: str) -> bool:
    """
    Validate WAN model components, with fallback if Model Orchestrator is not available.
    
    Args:
        model_id: Model identifier
        model_path: Path to the model directory
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        from backend.services.wan_pipeline_integration import get_wan_integration
        integration = get_wan_integration()
        result = integration.validate_components(model_id, model_path)
        
        if not result.is_valid:
            logger.error(f"Model validation failed for {model_id}: {result.missing_components}")
            return False
        
        # Log warnings
        for warning in result.warnings:
            logger.warning(f"Model validation warning: {warning}")
        
        return True
        
    except Exception as e:
        logger.warning(f"Could not validate model components via Model Orchestrator: {e}")
        
        # Fallback validation - just check if model_index.json exists
        import json
        model_index_path = os.path.join(model_path, 'model_index.json')
        
        if not os.path.exists(model_index_path):
            logger.error(f"model_index.json not found at {model_index_path}")
            return False
        
        try:
            with open(model_index_path, 'r') as f:
                model_index = json.load(f)
            
            # Basic validation - check for required fields
            if '_class_name' not in model_index:
                logger.error("model_index.json missing _class_name field")
                return False
            
            logger.info(f"Basic model validation passed for {model_id}")
            return True
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to read model_index.json: {e}")
            return False


# Auto-initialize if this module is imported and environment is set up
if os.environ.get('AUTO_SETUP_WAN_ORCHESTRATOR', '').lower() in ('true', '1', 'yes'):
    setup_wan_orchestrator_integration()