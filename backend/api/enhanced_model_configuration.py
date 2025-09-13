"""
Enhanced Model Configuration API Endpoints

Provides REST API endpoints for managing enhanced model availability configuration,
including user preferences, admin policies, and feature flags.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer
from typing import Dict, Any, Optional, List
import logging

from backend.core.enhanced_model_config import (
    get_config_manager, ConfigurationManager,
    UserPreferences, AdminPolicies, FeatureFlagConfig,
    AutomationLevel, FeatureFlag
)
from backend.core.config_validation import ValidationResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/config", tags=["Enhanced Model Configuration"])
security = HTTPBearer()


def get_current_user_id(token: str = Depends(security)) -> str:
    """Extract user ID from authentication token (placeholder implementation)"""
    # TODO: Implement proper JWT token validation
    return "default_user"


def require_admin_role(token: str = Depends(security)) -> bool:
    """Verify admin role for administrative operations (placeholder implementation)"""
    # TODO: Implement proper role-based access control
    return True


@router.get("/user-preferences")
async def get_user_preferences(
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """Get current user preferences for enhanced model features"""
    try:
        config_manager = get_config_manager()
        preferences = config_manager.get_user_preferences()
        
        # Convert to dictionary for JSON serialization
        from dataclasses import asdict
        return {
            "user_id": user_id,
            "preferences": asdict(preferences),
            "last_updated": config_manager.config.last_updated.isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting user preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user preferences"
        )


@router.put("/user-preferences")
async def update_user_preferences(
    preferences_data: Dict[str, Any],
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """Update user preferences for enhanced model features"""
    try:
        config_manager = get_config_manager()
        
        # Convert dictionary to UserPreferences object
        preferences = _dict_to_user_preferences(preferences_data)
        
        # Update preferences
        success = await config_manager.update_user_preferences(preferences)
        
        if success:
            return {
                "success": True,
                "message": "User preferences updated successfully",
                "user_id": user_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update user preferences"
            )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid preferences data: {e}"
        )
    except Exception as e:
        logger.error(f"Error updating user preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user preferences"
        )


@router.get("/admin-policies")
async def get_admin_policies(
    _: bool = Depends(require_admin_role)
) -> Dict[str, Any]:
    """Get current administrative policies (admin only)"""
    try:
        config_manager = get_config_manager()
        policies = config_manager.get_admin_policies()
        
        from dataclasses import asdict
        return {
            "policies": asdict(policies),
            "last_updated": config_manager.config.last_updated.isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting admin policies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve admin policies"
        )


@router.put("/admin-policies")
async def update_admin_policies(
    policies_data: Dict[str, Any],
    _: bool = Depends(require_admin_role)
) -> Dict[str, Any]:
    """Update administrative policies (admin only)"""
    try:
        config_manager = get_config_manager()
        
        # Convert dictionary to AdminPolicies object
        policies = _dict_to_admin_policies(policies_data)
        
        # Update policies
        success = await config_manager.update_admin_policies(policies)
        
        if success:
            return {
                "success": True,
                "message": "Admin policies updated successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update admin policies"
            )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid policies data: {e}"
        )
    except Exception as e:
        logger.error(f"Error updating admin policies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update admin policies"
        )


@router.get("/feature-flags")
async def get_feature_flags(
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """Get feature flags status for current user"""
    try:
        config_manager = get_config_manager()
        
        # Get all feature flags with user-specific overrides
        flags_status = {}
        for flag in FeatureFlag:
            flags_status[flag.value] = config_manager.is_feature_enabled(flag, user_id)
        
        return {
            "user_id": user_id,
            "feature_flags": flags_status,
            "global_flags": config_manager.config.feature_flags.flags,
            "user_overrides": config_manager.config.feature_flags.user_overrides.get(user_id, {})
        }
    except Exception as e:
        logger.error(f"Error getting feature flags: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve feature flags"
        )


@router.put("/feature-flags/{flag_name}")
async def update_feature_flag(
    flag_name: str,
    enabled: bool,
    user_specific: bool = False,
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """Update a feature flag (admin for global, user for personal overrides)"""
    try:
        # Check admin role for global changes
        if not user_specific:
            require_admin_role()
        
        config_manager = get_config_manager()
        
        # Validate flag name
        try:
            flag = FeatureFlag(flag_name)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid feature flag: {flag_name}"
            )
        
        # Update flag
        target_user = user_id if user_specific else None
        success = await config_manager.update_feature_flag(flag, enabled, target_user)
        
        if success:
            return {
                "success": True,
                "message": f"Feature flag {flag_name} updated successfully",
                "flag": flag_name,
                "enabled": enabled,
                "user_specific": user_specific,
                "user_id": user_id if user_specific else None
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update feature flag"
            )
    
    except Exception as e:
        logger.error(f"Error updating feature flag: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update feature flag"
        )


@router.post("/validate-preferences")
async def validate_user_preferences(
    preferences_data: Dict[str, Any],
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """Validate user preferences without saving them"""
    try:
        config_manager = get_config_manager()
        
        # Convert dictionary to UserPreferences object
        preferences = _dict_to_user_preferences(preferences_data)
        
        # Validate preferences
        validation_result = config_manager.validate_user_preferences(preferences)
        
        return {
            "is_valid": validation_result.is_valid,
            "errors": [
                {
                    "field": error.field,
                    "message": error.message,
                    "severity": error.severity,
                    "suggested_value": error.suggested_value
                }
                for error in validation_result.errors
            ],
            "warnings": [
                {
                    "field": warning.field,
                    "message": warning.message,
                    "severity": warning.severity,
                    "suggested_value": warning.suggested_value
                }
                for warning in validation_result.warnings
            ]
        }
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid preferences data: {e}"
        )
    except Exception as e:
        logger.error(f"Error validating preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate preferences"
        )


@router.post("/validate-policies")
async def validate_admin_policies(
    policies_data: Dict[str, Any],
    _: bool = Depends(require_admin_role)
) -> Dict[str, Any]:
    """Validate admin policies without saving them (admin only)"""
    try:
        config_manager = get_config_manager()
        
        # Convert dictionary to AdminPolicies object
        policies = _dict_to_admin_policies(policies_data)
        
        # Validate policies
        validation_result = config_manager.validate_admin_policies(policies)
        
        return {
            "is_valid": validation_result.is_valid,
            "errors": [
                {
                    "field": error.field,
                    "message": error.message,
                    "severity": error.severity,
                    "suggested_value": error.suggested_value
                }
                for error in validation_result.errors
            ],
            "warnings": [
                {
                    "field": warning.field,
                    "message": warning.message,
                    "severity": warning.severity,
                    "suggested_value": warning.suggested_value
                }
                for warning in validation_result.warnings
            ]
        }
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid policies data: {e}"
        )
    except Exception as e:
        logger.error(f"Error validating policies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate policies"
        )

@router.get("/configuration-status")
async def get_configuration_status(
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """Get overall configuration status and health"""
    try:
        config_manager = get_config_manager()
        
        # Validate current configuration
        preferences_validation = config_manager.validate_user_preferences(
            config_manager.get_user_preferences()
        )
        policies_validation = config_manager.validate_admin_policies(
            config_manager.get_admin_policies()
        )
        
        return {
            "user_id": user_id,
            "configuration_version": config_manager.config.version,
            "schema_version": config_manager.config.config_schema_version,
            "last_updated": config_manager.config.last_updated.isoformat(),
            "preferences_valid": preferences_validation.is_valid,
            "policies_valid": policies_validation.is_valid,
            "total_errors": len(preferences_validation.errors) + len(policies_validation.errors),
            "total_warnings": len(preferences_validation.warnings) + len(policies_validation.warnings),
            "enabled_features": [
                flag.value for flag in FeatureFlag 
                if config_manager.is_feature_enabled(flag, user_id)
            ]
        }
    except Exception as e:
        logger.error(f"Error getting configuration status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve configuration status"
        )


@router.post("/reset-to-defaults")
async def reset_configuration_to_defaults(
    section: Optional[str] = None,
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """Reset configuration section to defaults"""
    try:
        # Check admin role for admin section changes
        if section == "admin" or section is None:
            require_admin_role()
        
        config_manager = get_config_manager()
        
        if section == "preferences" or section is None:
            # Reset user preferences to defaults
            default_preferences = UserPreferences()
            success = await config_manager.update_user_preferences(default_preferences)
            
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to reset user preferences"
                )
        
        if section == "admin" or section is None:
            # Reset admin policies to defaults (admin only)
            default_policies = AdminPolicies()
            success = await config_manager.update_admin_policies(default_policies)
            
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to reset admin policies"
                )
        
        return {
            "success": True,
            "message": f"Configuration {'section ' + section if section else 'sections'} reset to defaults",
            "reset_section": section or "all"
        }
    
    except Exception as e:
        logger.error(f"Error resetting configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset configuration"
        )


# Helper functions for data conversion

def _dict_to_user_preferences(data: Dict[str, Any]) -> UserPreferences:
    """Convert dictionary to UserPreferences object"""
    from dataclasses import fields
    
    # Handle automation level enum conversion
    if 'automation_level' in data and isinstance(data['automation_level'], str):
        data['automation_level'] = AutomationLevel(data['automation_level'])
    
    # Convert nested dictionaries to dataclass objects
    nested_configs = {
        'download_config': 'DownloadConfig',
        'health_monitoring': 'HealthMonitoringConfig',
        'fallback_config': 'FallbackConfig',
        'analytics_config': 'AnalyticsConfig',
        'update_config': 'UpdateConfig',
        'notification_config': 'NotificationConfig',
        'storage_config': 'StorageConfig'
    }
    
    for field_name, class_name in nested_configs.items():
        if field_name in data and isinstance(data[field_name], dict):
            # Get the class from the config module
            from backend.core.enhanced_model_config import (
                DownloadConfig, HealthMonitoringConfig, FallbackConfig,
                AnalyticsConfig, UpdateConfig, NotificationConfig, StorageConfig
            )
            
            config_class = globals()[class_name]
            data[field_name] = config_class(**data[field_name])
    
    return UserPreferences(**data)


def _dict_to_admin_policies(data: Dict[str, Any]) -> AdminPolicies:
    """Convert dictionary to AdminPolicies object"""
    return AdminPolicies(**data)
