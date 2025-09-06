"""
LoRA Upload Handler Component
Handles file validation, processing, and metadata extraction for LoRA uploads
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
import shutil

import torch
from PIL import Image
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class LoRAUploadHandler:
    """Handles LoRA file uploads with validation and processing"""
    
    # Supported LoRA file formats
    SUPPORTED_FORMATS = {'.safetensors', '.ckpt', '.pt', '.pth'}
    
    # File size limits (in MB)
    MAX_FILE_SIZE_MB = 2048  # 2GB max
    MIN_FILE_SIZE_MB = 0.1   # 100KB min (more reasonable for testing)
    
    def __init__(self, loras_directory: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the upload handler
        
        Args:
            loras_directory: Directory where LoRA files are stored
            config: Optional configuration dictionary
        """
        self.loras_directory = Path(loras_directory)
        self.loras_directory.mkdir(exist_ok=True)
        
        # Create thumbnails directory
        self.thumbnails_dir = self.loras_directory / ".thumbnails"
        self.thumbnails_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.config = config or {}
        self.max_file_size_mb = self.config.get("max_lora_file_size_mb", self.MAX_FILE_SIZE_MB)
        self.min_file_size_mb = self.config.get("min_lora_file_size_mb", self.MIN_FILE_SIZE_MB)
        
        logger.info(f"LoRAUploadHandler initialized with directory: {self.loras_directory}")
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate a LoRA file for upload
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                return False, f"File does not exist: {file_path}"
            
            # Check file extension
            if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                supported_list = ", ".join(self.SUPPORTED_FORMATS)
                return False, f"Unsupported file format. Supported formats: {supported_list}"
            
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            if file_size_mb < self.min_file_size_mb:
                return False, f"File too small ({file_size_mb:.1f}MB). Minimum size: {self.min_file_size_mb}MB"
            
            if file_size_mb > self.max_file_size_mb:
                return False, f"File too large ({file_size_mb:.1f}MB). Maximum size: {self.max_file_size_mb}MB"
            
            # Validate file content
            content_valid, content_error = self._validate_file_content(file_path)
            if not content_valid:
                return False, content_error
            
            logger.info(f"File validation passed: {file_path.name} ({file_size_mb:.1f}MB)")
            return True, "File validation successful"
            
        except Exception as e:
            error_msg = f"Error validating file: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _validate_file_content(self, file_path: Path) -> Tuple[bool, str]:
        """Validate the content of a LoRA file
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Load the file to check if it's a valid LoRA
            if file_path.suffix.lower() == '.safetensors':
                try:
                    from safetensors.torch import load_file
                    weights = load_file(str(file_path))
                except ImportError:
                    return False, "safetensors library not available for .safetensors files"
                except Exception as e:
                    return False, f"Invalid .safetensors file: {str(e)}"
            else:
                # For .pt, .pth, .ckpt files
                try:
                    weights = torch.load(str(file_path), map_location='cpu', weights_only=False)
                except Exception as e:
                    return False, f"Invalid PyTorch file: {str(e)}"
            
            # Check if it looks like a LoRA file
            if not self._is_lora_weights(weights):
                return False, "File does not appear to contain LoRA weights"
            
            return True, "File content validation successful"
            
        except Exception as e:
            return False, f"Error validating file content: {str(e)}"
    
    def _is_lora_weights(self, weights: Dict[str, Any]) -> bool:
        """Check if the loaded weights appear to be LoRA weights
        
        Args:
            weights: Dictionary of loaded weights
            
        Returns:
            True if weights appear to be LoRA weights
        """
        if not isinstance(weights, dict):
            return False
        
        # Look for LoRA-specific keys
        lora_indicators = ['lora_up', 'lora_down', 'lora_A', 'lora_B', 'alpha']
        
        weight_keys = list(weights.keys())
        has_lora_keys = any(indicator in str(weight_keys) for indicator in lora_indicators)
        
        if has_lora_keys:
            return True
        
        # Check for common LoRA patterns - be more specific about up/down pairing
        up_keys = [k for k in weight_keys if 'up' in k.lower() and ('lora' in k.lower() or 'adapter' in k.lower())]
        down_keys = [k for k in weight_keys if 'down' in k.lower() and ('lora' in k.lower() or 'adapter' in k.lower())]
        
        # If we have paired up/down weights, it's likely a LoRA
        if len(up_keys) > 0 and len(down_keys) > 0:
            return True
        
        # Check for tensor shapes that suggest LoRA structure - be more restrictive
        low_rank_count = 0
        for key, tensor in weights.items():
            if hasattr(tensor, 'shape') and len(tensor.shape) == 2:
                # LoRA weights are typically 2D matrices with significant rank difference
                min_dim = min(tensor.shape)
                max_dim = max(tensor.shape)
                if min_dim < max_dim / 4:  # One dimension is much smaller (low rank)
                    low_rank_count += 1
        
        # Need multiple low-rank matrices to be confident it's a LoRA
        return low_rank_count >= 2
    
    def check_duplicate_filename(self, filename: str) -> Tuple[bool, Optional[str]]:
        """Check if a filename already exists and suggest alternatives
        
        Args:
            filename: Name of the file to check
            
        Returns:
            Tuple of (exists, suggested_name)
        """
        file_path = self.loras_directory / filename
        
        if not file_path.exists():
            return False, None
        
        # Generate alternative filename
        name_stem = Path(filename).stem
        extension = Path(filename).suffix
        
        counter = 1
        while True:
            suggested_name = f"{name_stem}_{counter}{extension}"
            suggested_path = self.loras_directory / suggested_name
            
            if not suggested_path.exists():
                return True, suggested_name
            
            counter += 1
            
            # Prevent infinite loop
            if counter > 1000:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                suggested_name = f"{name_stem}_{timestamp}{extension}"
                return True, suggested_name
    
    def process_upload(self, file_data: bytes, filename: str, overwrite: bool = False) -> Dict[str, Any]:
        """Process a LoRA file upload
        
        Args:
            file_data: Raw file data as bytes
            filename: Name for the uploaded file
            overwrite: Whether to overwrite existing files
            
        Returns:
            Dictionary with upload results
        """
        try:
            # Validate filename
            if not filename or filename.strip() == "":
                return {
                    "success": False,
                    "error": "Filename cannot be empty",
                    "filename": filename
                }
            
            # Sanitize filename
            safe_filename = self._sanitize_filename(filename)
            
            # Check for duplicates if not overwriting
            if not overwrite:
                exists, suggested_name = self.check_duplicate_filename(safe_filename)
                if exists:
                    return {
                        "success": False,
                        "error": f"File already exists: {safe_filename}",
                        "suggested_name": suggested_name,
                        "filename": safe_filename
                    }
            
            # Create temporary file for validation
            temp_path = self.loras_directory / f".temp_{safe_filename}"
            
            try:
                # Write file data to temporary location
                with open(temp_path, 'wb') as f:
                    f.write(file_data)
                
                # Validate the temporary file
                is_valid, error_message = self.validate_file(str(temp_path))
                
                if not is_valid:
                    return {
                        "success": False,
                        "error": error_message,
                        "filename": safe_filename
                    }
                
                # Move to final location
                final_path = self.loras_directory / safe_filename
                shutil.move(str(temp_path), str(final_path))
                
                # Extract metadata
                metadata = self.extract_metadata(str(final_path))
                
                # Generate thumbnail (optional)
                thumbnail_path = None
                try:
                    thumbnail_path = self.generate_thumbnail(safe_filename)
                except Exception as e:
                    logger.warning(f"Failed to generate thumbnail for {safe_filename}: {e}")
                
                # Calculate file hash for integrity checking
                file_hash = self._calculate_file_hash(str(final_path))
                
                result = {
                    "success": True,
                    "filename": safe_filename,
                    "path": str(final_path),
                    "size_mb": len(file_data) / (1024 * 1024),
                    "upload_time": datetime.now().isoformat(),
                    "metadata": metadata,
                    "thumbnail_path": thumbnail_path,
                    "file_hash": file_hash
                }
                
                logger.info(f"Successfully processed upload: {safe_filename}")
                return result
                
            finally:
                # Clean up temporary file if it exists
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to clean up temporary file: {e}")
            
        except Exception as e:
            error_msg = f"Error processing upload: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "filename": filename
            }
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename to prevent path traversal and other issues
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove path components
        filename = os.path.basename(filename)
        
        # Replace problematic characters
        problematic_chars = '<>:"/\\|?*'
        for char in problematic_chars:
            filename = filename.replace(char, '_')
        
        # Remove leading/trailing whitespace and dots, but preserve underscores
        filename = filename.strip(' .')
        
        # Ensure filename is not empty after sanitization
        # Check if only underscores and dots remain (no actual content)
        content_check = filename.replace('_', '').replace('.', '').strip()
        if not filename or not content_check:
            filename = f"lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Ensure it has a valid extension
        if not any(filename.lower().endswith(ext) for ext in self.SUPPORTED_FORMATS):
            filename += '.safetensors'  # Default extension
        
        return filename
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hexadecimal hash string
        """
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def extract_metadata(self, lora_path: str) -> Dict[str, Any]:
        """Extract metadata from a LoRA file
        
        Args:
            lora_path: Path to the LoRA file
            
        Returns:
            Dictionary containing extracted metadata
        """
        try:
            file_path = Path(lora_path)
            
            # Basic file metadata
            stat = file_path.stat()
            metadata = {
                "filename": file_path.name,
                "size_bytes": stat.st_size,
                "size_mb": stat.st_size / (1024 * 1024),
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "file_extension": file_path.suffix.lower()
            }
            
            # Load weights to extract LoRA-specific metadata
            try:
                if file_path.suffix.lower() == '.safetensors':
                    from safetensors.torch import load_file
                    weights = load_file(str(file_path))
                else:
                    weights = torch.load(str(file_path), map_location='cpu', weights_only=False)
                
                # Extract LoRA structure information
                lora_info = self._analyze_lora_structure(weights)
                metadata.update(lora_info)
                
            except Exception as e:
                logger.warning(f"Failed to extract LoRA metadata from {file_path.name}: {e}")
                metadata["extraction_error"] = str(e)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {"error": str(e)}
    
    def _analyze_lora_structure(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the structure of LoRA weights
        
        Args:
            weights: Dictionary of loaded weights
            
        Returns:
            Dictionary with LoRA structure information
        """
        analysis = {
            "total_parameters": 0,
            "lora_layers": 0,
            "parameter_types": set(),
            "layer_names": [],
            "rank_info": {}
        }
        
        try:
            # Count parameters and analyze structure
            for key, tensor in weights.items():
                if hasattr(tensor, 'numel'):
                    analysis["total_parameters"] += tensor.numel()
                
                # Identify parameter types
                if 'lora_up' in key.lower() or 'lora_down' in key.lower():
                    analysis["lora_layers"] += 1
                    
                    # Extract layer name
                    layer_name = key.split('.lora_')[0] if '.lora_' in key else key
                    if layer_name not in analysis["layer_names"]:
                        analysis["layer_names"].append(layer_name)
                    
                    # Analyze rank information
                    if hasattr(tensor, 'shape') and len(tensor.shape) == 2:
                        rank = min(tensor.shape)
                        analysis["rank_info"][key] = {
                            "shape": list(tensor.shape),
                            "estimated_rank": rank
                        }
                
                # Track parameter types
                if 'weight' in key.lower():
                    analysis["parameter_types"].add('weight')
                elif 'bias' in key.lower():
                    analysis["parameter_types"].add('bias')
                elif 'alpha' in key.lower():
                    analysis["parameter_types"].add('alpha')
            
            # Convert set to list for JSON serialization
            analysis["parameter_types"] = list(analysis["parameter_types"])
            
            # Calculate average rank
            if analysis["rank_info"]:
                ranks = [info["estimated_rank"] for info in analysis["rank_info"].values()]
                analysis["average_rank"] = sum(ranks) / len(ranks)
                analysis["max_rank"] = max(ranks)
                analysis["min_rank"] = min(ranks)
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Error analyzing LoRA structure: {e}")
            return {"analysis_error": str(e)}
    
    def generate_thumbnail(self, lora_name: str) -> Optional[str]:
        """Generate a thumbnail for LoRA preview (placeholder implementation)
        
        Args:
            lora_name: Name of the LoRA file
            
        Returns:
            Path to generated thumbnail or None if failed
        """
        try:
            # For now, create a simple placeholder thumbnail
            # In a full implementation, this could generate preview images
            # based on LoRA characteristics or sample generations
            
            thumbnail_filename = f"{Path(lora_name).stem}.jpg"
            thumbnail_path = self.thumbnails_dir / thumbnail_filename
            
            # Create a simple placeholder image
            placeholder_image = self._create_placeholder_thumbnail(lora_name)
            placeholder_image.save(str(thumbnail_path), "JPEG", quality=85)
            
            logger.info(f"Generated thumbnail: {thumbnail_path}")
            return str(thumbnail_path)
            
        except Exception as e:
            logger.warning(f"Failed to generate thumbnail for {lora_name}: {e}")
            return None
    
    def _create_placeholder_thumbnail(self, lora_name: str) -> Image.Image:
        """Create a placeholder thumbnail image
        
        Args:
            lora_name: Name of the LoRA
            
        Returns:
            PIL Image object
        """
        # Create a 256x256 placeholder image
        width, height = 256, 256
        
        # Generate a color based on the LoRA name hash
        name_hash = hashlib.md5(lora_name.encode()).hexdigest()
        r = int(name_hash[0:2], 16)
        g = int(name_hash[2:4], 16)
        b = int(name_hash[4:6], 16)
        
        # Create gradient background
        image = Image.new('RGB', (width, height), (r, g, b))
        
        # Add some visual elements (simple pattern)
        pixels = np.array(image)
        
        # Add diagonal pattern
        for i in range(height):
            for j in range(width):
                if (i + j) % 20 < 10:
                    pixels[i, j] = [min(255, c + 30) for c in pixels[i, j]]
        
        return Image.fromarray(pixels)
    
    def get_upload_stats(self) -> Dict[str, Any]:
        """Get statistics about uploaded LoRA files
        
        Returns:
            Dictionary with upload statistics
        """
        try:
            stats = {
                "total_files": 0,
                "total_size_mb": 0.0,
                "file_types": {},
                "largest_file": {"name": "", "size_mb": 0.0},
                "smallest_file": {"name": "", "size_mb": float('inf')},
                "upload_directory": str(self.loras_directory)
            }
            
            for lora_file in self.loras_directory.iterdir():
                if lora_file.is_file() and lora_file.suffix.lower() in self.SUPPORTED_FORMATS:
                    stats["total_files"] += 1
                    
                    size_mb = lora_file.stat().st_size / (1024 * 1024)
                    stats["total_size_mb"] += size_mb
                    
                    # Track file types
                    ext = lora_file.suffix.lower()
                    stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
                    
                    # Track largest file
                    if size_mb > stats["largest_file"]["size_mb"]:
                        stats["largest_file"] = {"name": lora_file.name, "size_mb": size_mb}
                    
                    # Track smallest file
                    if size_mb < stats["smallest_file"]["size_mb"]:
                        stats["smallest_file"] = {"name": lora_file.name, "size_mb": size_mb}
            
            # Handle case where no files exist
            if stats["total_files"] == 0:
                stats["smallest_file"] = {"name": "", "size_mb": 0.0}
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting upload stats: {e}")
            return {"error": str(e)}