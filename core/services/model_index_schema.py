"""
Model Index Schema Validation for Wan Model Compatibility

This module provides Pydantic schemas and validation for model_index.json files,
with specific support for Wan model architectures and their custom attributes.
"""

from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
from dataclasses import dataclass
from enum import Enum


class ModelType(str, Enum):
    """Supported model types"""
    STABLE_DIFFUSION = "stable_diffusion"
    WAN_T2V = "wan_t2v"
    WAN_T2I = "wan_t2i"
    WAN_I2V = "wan_i2v"
    UNKNOWN = "unknown"


class ComponentType(BaseModel):
    """Schema for individual component entries in model_index.json"""
    component_class: str = Field(alias="0", description="Component class name")
    component_path: str = Field(alias="1", description="Path to component")
    
    model_config = ConfigDict(populate_by_name=True)


class ModelIndexSchema(BaseModel):
    """Pydantic schema for model_index.json validation"""
    
    # Required fields for all models
    class_name: str = Field(alias="_class_name", description="Pipeline class name")
    diffusers_version: str = Field(alias="_diffusers_version", description="Diffusers version")
    
    # Standard Diffusers components (optional)
    scheduler: Optional[List[str]] = None
    text_encoder: Optional[List[str]] = None
    text_encoder_2: Optional[List[str]] = None
    tokenizer: Optional[List[str]] = None
    tokenizer_2: Optional[List[str]] = None
    vae: Optional[List[str]] = None
    unet: Optional[List[str]] = None
    
    # Wan-specific components
    transformer: Optional[List[str]] = None
    transformer_2: Optional[List[str]] = None
    
    # Wan-specific attributes
    boundary_ratio: Optional[float] = None
    
    # Additional custom attributes
    custom_attributes: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    
    @field_validator('class_name')
    @classmethod
    def validate_class_name(cls, v):
        """Validate pipeline class name format"""
        if not v or not isinstance(v, str):
            raise ValueError("Pipeline class name must be a non-empty string")
        return v
    
    @field_validator('diffusers_version')
    @classmethod
    def validate_diffusers_version(cls, v):
        """Validate diffusers version format"""
        if not v or not isinstance(v, str):
            raise ValueError("Diffusers version must be a non-empty string")
        return v
    
    @field_validator('boundary_ratio')
    @classmethod
    def validate_boundary_ratio(cls, v):
        """Validate boundary_ratio is within reasonable bounds"""
        if v is not None and (v < 0 or v > 1):
            raise ValueError("boundary_ratio must be between 0 and 1")
        return v
    
    def detect_model_type(self) -> ModelType:
        """Detect model type based on architecture signatures"""
        # Check for Wan-specific indicators
        if self.transformer or self.transformer_2 or self.boundary_ratio is not None:
            if "T2V" in self.class_name.upper():
                return ModelType.WAN_T2V
            elif "T2I" in self.class_name.upper():
                return ModelType.WAN_T2I
            elif "I2V" in self.class_name.upper():
                return ModelType.WAN_I2V
            else:
                return ModelType.WAN_T2V  # Default Wan type
        
        # Check for standard SD components
        if self.unet and self.vae and self.text_encoder:
            return ModelType.STABLE_DIFFUSION
        
        return ModelType.UNKNOWN
    
    def is_wan_architecture(self) -> bool:
        """Check if this represents a Wan architecture"""
        return self.detect_model_type() in [ModelType.WAN_T2V, ModelType.WAN_T2I, ModelType.WAN_I2V]
    
    def get_required_components(self) -> List[str]:
        """Get list of required components based on model type"""
        model_type = self.detect_model_type()
        
        if model_type == ModelType.STABLE_DIFFUSION:
            return ["unet", "vae", "text_encoder", "tokenizer", "scheduler"]
        elif model_type in [ModelType.WAN_T2V, ModelType.WAN_T2I, ModelType.WAN_I2V]:
            return ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
        elif model_type == ModelType.UNKNOWN:
            # For unknown types, check what components are present and suggest based on that
            if self.unet:
                return ["unet", "vae", "text_encoder", "tokenizer", "scheduler"]
            elif self.transformer:
                return ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
        
        return []
    
    def get_missing_components(self) -> List[str]:
        """Get list of missing required components"""
        required = self.get_required_components()
        missing = []
        
        for component in required:
            if not getattr(self, component, None):
                missing.append(component)
        
        return missing
    
    def validate_wan_specific_attributes(self) -> List[str]:
        """Validate Wan-specific attributes and return any issues"""
        issues = []
        
        if self.is_wan_architecture():
            # Check for required Wan components
            if not self.transformer and not self.transformer_2:
                issues.append("Wan models require either 'transformer' or 'transformer_2' component")
            
            # Validate pipeline class name matches architecture
            if not self.class_name.startswith("Wan"):
                issues.append(f"Pipeline class '{self.class_name}' should start with 'Wan' for Wan architecture")
            
            # Check for boundary_ratio in T2V models
            if self.detect_model_type() == ModelType.WAN_T2V and self.boundary_ratio is None:
                issues.append("Wan T2V models typically require 'boundary_ratio' attribute")
        
        return issues


@dataclass
class SchemaValidationResult:
    """Result of schema validation"""
    is_valid: bool
    schema: Optional[ModelIndexSchema]
    errors: List[str]
    warnings: List[str]
    suggested_fixes: List[str]
    model_type: ModelType
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization"""
        return {
            "is_valid": self.is_valid,
            "model_type": self.model_type.value,
            "errors": self.errors,
            "warnings": self.warnings,
            "suggested_fixes": self.suggested_fixes,
            "schema_data": self.schema.model_dump() if self.schema else None
        }


class SchemaValidator:
    """Validator for model_index.json files with comprehensive error reporting"""
    
    def __init__(self):
        self.validation_history: List[SchemaValidationResult] = []
    
    def validate_model_index(self, model_index_path: str) -> SchemaValidationResult:
        """
        Validate model_index.json against schema
        
        Args:
            model_index_path: Path to model_index.json file
            
        Returns:
            SchemaValidationResult with validation details
        """
        errors = []
        warnings = []
        suggested_fixes = []
        schema = None
        model_type = ModelType.UNKNOWN
        
        try:
            # Load and parse JSON
            path = Path(model_index_path)
            if not path.exists():
                errors.append(f"model_index.json not found at {model_index_path}")
                return SchemaValidationResult(
                    is_valid=False,
                    schema=None,
                    errors=errors,
                    warnings=warnings,
                    suggested_fixes=["Ensure model_index.json exists in the model directory"],
                    model_type=model_type
                )
            
            with open(path, 'r', encoding='utf-8') as f:
                model_index_data = json.load(f)
            
            # Validate against schema
            try:
                schema = ModelIndexSchema(**model_index_data)
                model_type = schema.detect_model_type()
                
                # Check for missing components
                missing_components = schema.get_missing_components()
                if missing_components:
                    warnings.append(f"Missing recommended components: {', '.join(missing_components)}")
                    suggested_fixes.append(f"Consider adding missing components: {', '.join(missing_components)}")
                
                # Validate Wan-specific attributes
                wan_issues = schema.validate_wan_specific_attributes()
                if wan_issues:
                    warnings.extend(wan_issues)
                    suggested_fixes.extend([f"Fix Wan attribute issue: {issue}" for issue in wan_issues])
                
                # Additional validation checks
                self._perform_additional_validation(schema, warnings, suggested_fixes)
                
            except ValidationError as e:
                errors.extend(self._format_pydantic_errors(e.errors()))
                suggested_fixes.extend(self._generate_fix_suggestions(e.errors()))
            
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON format: {str(e)}")
            suggested_fixes.append("Fix JSON syntax errors in model_index.json")
        except Exception as e:
            errors.append(f"Unexpected error during validation: {str(e)}")
            suggested_fixes.append("Check file permissions and format")
        
        # Create result
        result = SchemaValidationResult(
            is_valid=len(errors) == 0,
            schema=schema,
            errors=errors,
            warnings=warnings,
            suggested_fixes=suggested_fixes,
            model_type=model_type
        )
        
        # Store in history
        self.validation_history.append(result)
        
        return result
    
    def validate_model_index_dict(self, model_index_data: Dict[str, Any]) -> SchemaValidationResult:
        """
        Validate model_index data from dictionary
        
        Args:
            model_index_data: Dictionary containing model index data
            
        Returns:
            SchemaValidationResult with validation details
        """
        errors = []
        warnings = []
        suggested_fixes = []
        schema = None
        model_type = ModelType.UNKNOWN
        
        try:
            # Validate against schema
            schema = ModelIndexSchema(**model_index_data)
            model_type = schema.detect_model_type()
            
            # Check for missing components
            missing_components = schema.get_missing_components()
            if missing_components:
                warnings.append(f"Missing recommended components: {', '.join(missing_components)}")
                suggested_fixes.append(f"Consider adding missing components: {', '.join(missing_components)}")
            
            # Validate Wan-specific attributes
            wan_issues = schema.validate_wan_specific_attributes()
            if wan_issues:
                warnings.extend(wan_issues)
                suggested_fixes.extend([f"Fix Wan attribute issue: {issue}" for issue in wan_issues])
            
            # Additional validation checks
            self._perform_additional_validation(schema, warnings, suggested_fixes)
            
        except ValidationError as e:
            errors.extend(self._format_pydantic_errors(e.errors()))
            suggested_fixes.extend(self._generate_fix_suggestions(e.errors()))
        except Exception as e:
            errors.append(f"Unexpected error during validation: {str(e)}")
            suggested_fixes.append("Check data format and structure")
        
        # Create result
        result = SchemaValidationResult(
            is_valid=len(errors) == 0,
            schema=schema,
            errors=errors,
            warnings=warnings,
            suggested_fixes=suggested_fixes,
            model_type=model_type
        )
        
        # Store in history
        self.validation_history.append(result)
        
        return result
    
    def _perform_additional_validation(self, schema: ModelIndexSchema, 
                                     warnings: List[str], suggested_fixes: List[str]):
        """Perform additional validation checks beyond basic schema validation"""
        
        # Check for version compatibility
        if schema.diffusers_version:
            try:
                version_parts = schema.diffusers_version.split('.')
                major, minor = int(version_parts[0]), int(version_parts[1])
                
                if major == 0 and minor < 20:
                    warnings.append(f"Old diffusers version {schema.diffusers_version} may have compatibility issues")
                    suggested_fixes.append("Consider upgrading to diffusers >= 0.20.0")
                    
            except (ValueError, IndexError):
                warnings.append(f"Invalid diffusers version format: {schema.diffusers_version}")
                suggested_fixes.append("Use semantic versioning format (e.g., '0.21.0')")
        
        # Check for component consistency
        if schema.text_encoder and not schema.tokenizer:
            warnings.append("text_encoder found but tokenizer missing")
            suggested_fixes.append("Add tokenizer component to match text_encoder")
        
        if schema.text_encoder_2 and not schema.tokenizer_2:
            warnings.append("text_encoder_2 found but tokenizer_2 missing")
            suggested_fixes.append("Add tokenizer_2 component to match text_encoder_2")
    
    def _format_pydantic_errors(self, pydantic_errors: List[Dict[str, Any]]) -> List[str]:
        """Format Pydantic validation errors into user-friendly messages"""
        formatted_errors = []
        
        for error in pydantic_errors:
            field = '.'.join(str(loc) for loc in error['loc'])
            msg = error['msg']
            error_type = error['type']
            
            if error_type == 'value_error.missing':
                formatted_errors.append(f"Required field '{field}' is missing")
            elif error_type == 'type_error':
                formatted_errors.append(f"Field '{field}' has incorrect type: {msg}")
            elif error_type == 'value_error':
                formatted_errors.append(f"Field '{field}' has invalid value: {msg}")
            else:
                formatted_errors.append(f"Field '{field}': {msg}")
        
        return formatted_errors
    
    def _generate_fix_suggestions(self, pydantic_errors: List[Dict[str, Any]]) -> List[str]:
        """Generate fix suggestions based on validation errors"""
        suggestions = []
        
        for error in pydantic_errors:
            field = '.'.join(str(loc) for loc in error['loc'])
            error_type = error['type']
            
            if error_type == 'missing':
                if field == 'class_name':
                    suggestions.append("Add '_class_name' field with the pipeline class name (e.g., 'WanPipeline')")
                elif field == 'diffusers_version':
                    suggestions.append("Add '_diffusers_version' field with version string (e.g., '0.21.0')")
                else:
                    suggestions.append(f"Add required field '{field}' to model_index.json")
            
            elif error_type == 'value_error.missing':
                if field == '_class_name' or field == 'class_name':
                    suggestions.append("Add '_class_name' field with the pipeline class name (e.g., 'WanPipeline')")
                elif field == '_diffusers_version' or field == 'diffusers_version':
                    suggestions.append("Add '_diffusers_version' field with version string (e.g., '0.21.0')")
                else:
                    suggestions.append(f"Add required field '{field}' to model_index.json")
            
            elif 'type_error' in error_type:
                suggestions.append(f"Ensure field '{field}' has the correct data type")
            
            elif 'boundary_ratio' in field:
                suggestions.append("Set boundary_ratio to a value between 0 and 1")
        
        return suggestions
    
    def generate_schema_report(self, result: SchemaValidationResult) -> str:
        """Generate a comprehensive schema validation report"""
        report_lines = []
        
        report_lines.append("=" * 60)
        report_lines.append("MODEL INDEX SCHEMA VALIDATION REPORT")
        report_lines.append("=" * 60)
        
        # Validation status
        status = "✓ VALID" if result.is_valid else "✗ INVALID"
        report_lines.append(f"Status: {status}")
        report_lines.append(f"Model Type: {result.model_type.value}")
        
        if result.schema:
            report_lines.append(f"Pipeline Class: {result.schema.class_name}")
            report_lines.append(f"Diffusers Version: {result.schema.diffusers_version}")
            
            if result.schema.is_wan_architecture():
                report_lines.append("Architecture: Wan Model (Custom Pipeline Required)")
            else:
                report_lines.append("Architecture: Standard Diffusers Model")
        
        # Errors
        if result.errors:
            report_lines.append("\nERRORS:")
            for i, error in enumerate(result.errors, 1):
                report_lines.append(f"  {i}. {error}")
        
        # Warnings
        if result.warnings:
            report_lines.append("\nWARNINGS:")
            for i, warning in enumerate(result.warnings, 1):
                report_lines.append(f"  {i}. {warning}")
        
        # Suggested fixes
        if result.suggested_fixes:
            report_lines.append("\nSUGGESTED FIXES:")
            for i, fix in enumerate(result.suggested_fixes, 1):
                report_lines.append(f"  {i}. {fix}")
        
        # Component analysis
        if result.schema:
            report_lines.append("\nCOMPONENT ANALYSIS:")
            components = []
            for attr in ['transformer', 'transformer_2', 'unet', 'vae', 'text_encoder', 
                        'text_encoder_2', 'tokenizer', 'tokenizer_2', 'scheduler']:
                if getattr(result.schema, attr, None):
                    components.append(attr)
            
            if components:
                report_lines.append(f"  Found components: {', '.join(components)}")
            
            missing = result.schema.get_missing_components()
            if missing:
                report_lines.append(f"  Missing components: {', '.join(missing)}")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def get_validation_history(self) -> List[SchemaValidationResult]:
        """Get history of all validation results"""
        return self.validation_history.copy()
    
    def clear_validation_history(self):
        """Clear validation history"""
        self.validation_history.clear()