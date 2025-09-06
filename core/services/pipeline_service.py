"""
Core Pipeline Service
Handles video generation pipeline management for T2V, I2V, and TI2V modes
Extracted from utils.py as part of functional organization
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from PIL import Image
import torch

# Import model and optimization services
from core.services.model_manager import get_model_manager
from core.services.optimization_service import get_vram_optimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoGenerationEngine:
    """Core video generation engine for T2V, I2V, and TI2V modes"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.model_manager = get_model_manager()
        self.vram_optimizer = get_vram_optimizer(self.config)
        self.loaded_pipelines: Dict[str, Any] = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load config: {e}")
            # Return default config
            return {
                "generation": {
                    "default_steps": 50,
                    "default_guidance_scale": 7.5,
                    "max_resolution": "1920x1080",
                    "supported_resolutions": ["854x480", "480x854", "1280x720", "1280x704", "1920x1080"]
                },
                "optimization": {
                    "default_quantization": "bf16",
                    "enable_offload": True,
                    "vae_tile_size": 256
                }
            }
    
    def _get_pipeline(self, model_type: str, **optimization_kwargs) -> Any:
        """Get or load a pipeline for the specified model type"""
        if model_type in self.loaded_pipelines:
            return self.loaded_pipelines[model_type]
        
        # Load the model
        pipeline, model_info = self.model_manager.load_model(model_type)
        
        # Apply optimizations using the optimization service
        quantization = optimization_kwargs.get('quantization_level', 
                                              self.config['optimization']['default_quantization'])
        enable_offload = optimization_kwargs.get('enable_offload', 
                                                self.config['optimization']['enable_offload'])
        vae_tile_size = optimization_kwargs.get('vae_tile_size', 
                                               self.config['optimization']['vae_tile_size'])
        
        # Check if we should skip large components (for WAN models that might hang)
        # Default to True for WAN models to prevent hanging during quantization
        is_wan_model = 'wan' in model_type.lower() or 't2v' in model_type.lower()
        default_skip_large = is_wan_model  # Skip large components by default for WAN models
        skip_large = optimization_kwargs.get('skip_large_components', default_skip_large)
        
        # Apply optimizations
        optimized_pipeline = self.vram_optimizer.apply_quantization(pipeline, quantization, skip_large_components=skip_large)
        
        if enable_offload:
            optimized_pipeline = self.vram_optimizer.enable_cpu_offload(optimized_pipeline)
        
        optimized_pipeline = self.vram_optimizer.enable_vae_tiling(optimized_pipeline, vae_tile_size)
        
        # Move to GPU if available and not using CPU offloading
        if torch.cuda.is_available() and not enable_offload:
            optimized_pipeline = optimized_pipeline.to("cuda")
        elif enable_offload:
            logger.info("Skipping GPU move due to CPU offloading being enabled")
        
        self.loaded_pipelines[model_type] = optimized_pipeline
        return optimized_pipeline
    
    def generate_t2v(self, prompt: str, resolution: str = "1280x720", 
                     num_inference_steps: int = 50, guidance_scale: float = 7.5,
                     **kwargs) -> Dict[str, Any]:
        """
        Generate video from text prompt (Text-to-Video)
        
        Args:
            prompt: Text description for video generation
            resolution: Output resolution (e.g., "1280x720")
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated video frames and metadata
        """
        logger.info(f"Starting T2V generation: '{prompt[:50]}...' at {resolution}")
        
        try:
            # Get the T2V pipeline
            pipeline = self._get_pipeline("t2v-A14B", **kwargs)
            
            # Parse resolution
            width, height = map(int, resolution.split('x'))
            
            # Set generation parameters
            generation_kwargs = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_frames": kwargs.get("num_frames", 16),  # Default 16 frames
                # Filter out optimization and unsupported parameters
                **{k: v for k, v in kwargs.items() if k not in [
                    'quantization_level', 'enable_offload', 'vae_tile_size', 'skip_large_components',
                    'duration', 'fps'  # These are handled separately or not supported by WanPipeline
                ]}
            }
            
            # Generate video
            logger.info(f"Generating T2V with {num_inference_steps} steps...")
            result = pipeline(**generation_kwargs)
            
            # Extract frames
            if hasattr(result, 'frames'):
                frames = result.frames[0]  # Get first (and typically only) video
            elif hasattr(result, 'videos'):
                frames = result.videos[0]
            else:
                # Fallback - assume result is the frames directly
                frames = result
            
            logger.info(f"T2V generation completed: {len(frames)} frames at {width}x{height}")
            
            return {
                "frames": frames,
                "metadata": {
                    "model_type": "t2v-A14B",
                    "prompt": prompt,
                    "resolution": resolution,
                    "width": width,
                    "height": height,
                    "num_frames": len(frames),
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "generated_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"T2V generation failed: {e}")
            raise
    
    def generate_i2v(self, prompt: str, image: Image.Image, resolution: str = "1280x720",
                     num_inference_steps: int = 50, guidance_scale: float = 7.5,
                     **kwargs) -> Dict[str, Any]:
        """
        Generate video from image and text prompt (Image-to-Video)
        
        Args:
            prompt: Text description for video generation
            image: Input PIL Image to animate
            resolution: Output resolution (e.g., "1280x720")
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated video frames and metadata
        """
        logger.info(f"Starting I2V generation: '{prompt[:50]}...' with image at {resolution}")
        
        try:
            # Get the I2V pipeline
            pipeline = self._get_pipeline("i2v-A14B", **kwargs)
            
            # Parse resolution
            width, height = map(int, resolution.split('x'))
            
            # Resize input image to match target resolution
            resized_image = image.resize((width, height), Image.Resampling.LANCZOS)
            
            # Set generation parameters
            generation_kwargs = {
                "prompt": prompt,
                "image": resized_image,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_frames": kwargs.get("num_frames", 16),  # Default 16 frames
                **{k: v for k, v in kwargs.items() if k not in ['quantization_level', 'enable_offload', 'vae_tile_size']}
            }
            
            # Generate video
            logger.info(f"Generating I2V with {num_inference_steps} steps...")
            result = pipeline(**generation_kwargs)
            
            # Extract frames
            if hasattr(result, 'frames'):
                frames = result.frames[0]  # Get first (and typically only) video
            elif hasattr(result, 'videos'):
                frames = result.videos[0]
            else:
                # Fallback - assume result is the frames directly
                frames = result
            
            logger.info(f"I2V generation completed: {len(frames)} frames at {width}x{height}")
            
            return {
                "frames": frames,
                "metadata": {
                    "model_type": "i2v-A14B",
                    "prompt": prompt,
                    "resolution": resolution,
                    "width": width,
                    "height": height,
                    "num_frames": len(frames),
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "input_image_size": f"{image.width}x{image.height}",
                    "generated_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"I2V generation failed: {e}")
            raise
    
    def generate_ti2v(self, prompt: str, start_image: Image.Image, end_image: Image.Image,
                      resolution: str = "1280x720", num_inference_steps: int = 50,
                      guidance_scale: float = 7.5, **kwargs) -> Dict[str, Any]:
        """
        Generate video from text prompt with start and end images (Text-Image-to-Video)
        
        Args:
            prompt: Text description for video generation
            start_image: Starting frame PIL Image
            end_image: Ending frame PIL Image
            resolution: Output resolution (e.g., "1280x720")
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated video frames and metadata
        """
        logger.info(f"Starting TI2V generation: '{prompt[:50]}...' with start/end images at {resolution}")
        
        try:
            # Get the TI2V pipeline
            pipeline = self._get_pipeline("ti2v-5B", **kwargs)
            
            # Parse resolution
            width, height = map(int, resolution.split('x'))
            
            # Resize input images to match target resolution
            resized_start = start_image.resize((width, height), Image.Resampling.LANCZOS)
            resized_end = end_image.resize((width, height), Image.Resampling.LANCZOS)
            
            # Set generation parameters
            generation_kwargs = {
                "prompt": prompt,
                "start_image": resized_start,
                "end_image": resized_end,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_frames": kwargs.get("num_frames", 16),  # Default 16 frames
                **{k: v for k, v in kwargs.items() if k not in ['quantization_level', 'enable_offload', 'vae_tile_size']}
            }
            
            # Generate video
            logger.info(f"Generating TI2V with {num_inference_steps} steps...")
            result = pipeline(**generation_kwargs)
            
            # Extract frames
            if hasattr(result, 'frames'):
                frames = result.frames[0]  # Get first (and typically only) video
            elif hasattr(result, 'videos'):
                frames = result.videos[0]
            else:
                # Fallback - assume result is the frames directly
                frames = result
            
            logger.info(f"TI2V generation completed: {len(frames)} frames at {width}x{height}")
            
            return {
                "frames": frames,
                "metadata": {
                    "model_type": "ti2v-5B",
                    "prompt": prompt,
                    "resolution": resolution,
                    "width": width,
                    "height": height,
                    "num_frames": len(frames),
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "start_image_size": f"{start_image.width}x{start_image.height}",
                    "end_image_size": f"{end_image.width}x{end_image.height}",
                    "generated_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"TI2V generation failed: {e}")
            raise
    
    def unload_pipeline(self, model_type: str):
        """Unload a specific pipeline from memory"""
        if model_type in self.loaded_pipelines:
            del self.loaded_pipelines[model_type]
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Unloaded pipeline: {model_type}")
    
    def unload_all_pipelines(self):
        """Unload all pipelines from memory"""
        for model_type in list(self.loaded_pipelines.keys()):
            self.unload_pipeline(model_type)
        
        logger.info("Unloaded all pipelines")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get status of all loaded pipelines"""
        status = {}
        
        for model_type, pipeline in self.loaded_pipelines.items():
            try:
                # Get VRAM usage for this pipeline
                vram_info = self.vram_optimizer.get_vram_usage()
                
                status[model_type] = {
                    "loaded": True,
                    "pipeline_type": type(pipeline).__name__,
                    "device": str(getattr(pipeline, 'device', 'unknown')),
                    "memory_usage_mb": vram_info.get("used_mb", 0)
                }
            except Exception as e:
                status[model_type] = {
                    "loaded": True,
                    "error": str(e)
                }
        
        return status
    
    def validate_generation_request(self, model_type: str, prompt: str, 
                                  resolution: str = "1280x720", **kwargs) -> Dict[str, Any]:
        """Validate a generation request before processing"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate model type
        supported_models = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
        if model_type not in supported_models:
            validation_result["errors"].append(f"Unsupported model type: {model_type}")
            validation_result["valid"] = False
        
        # Validate prompt
        if not prompt or len(prompt.strip()) == 0:
            validation_result["errors"].append("Prompt cannot be empty")
            validation_result["valid"] = False
        elif len(prompt) > 1000:
            validation_result["warnings"].append("Very long prompt may affect generation quality")
        
        # Validate resolution
        try:
            width, height = map(int, resolution.split('x'))
            if width <= 0 or height <= 0:
                validation_result["errors"].append("Invalid resolution dimensions")
                validation_result["valid"] = False
            elif width > 1920 or height > 1080:
                validation_result["warnings"].append("High resolution may require significant VRAM")
        except ValueError:
            validation_result["errors"].append(f"Invalid resolution format: {resolution}")
            validation_result["valid"] = False
        
        # Validate steps
        steps = kwargs.get("num_inference_steps", 50)
        if steps < 1 or steps > 200:
            validation_result["warnings"].append("Unusual number of inference steps")
        
        # Check VRAM availability
        try:
            vram_info = self.vram_optimizer.get_vram_usage()
            if vram_info["percent"] > 80:
                validation_result["warnings"].append("High VRAM usage detected - generation may fail")
        except Exception:
            pass
        
        return validation_result

# Global pipeline engine instance
_pipeline_engine = None

def get_pipeline_engine(config_path: str = "config.json") -> VideoGenerationEngine:
    """Get the global pipeline engine instance"""
    global _pipeline_engine
    if _pipeline_engine is None:
        _pipeline_engine = VideoGenerationEngine(config_path)
    return _pipeline_engine

# Convenience functions for generation
def generate_text_to_video(prompt: str, **kwargs) -> Dict[str, Any]:
    """Generate video from text prompt"""
    engine = get_pipeline_engine()
    return engine.generate_t2v(prompt, **kwargs)

def generate_image_to_video(prompt: str, image: Image.Image, **kwargs) -> Dict[str, Any]:
    """Generate video from image and text prompt"""
    engine = get_pipeline_engine()
    return engine.generate_i2v(prompt, image, **kwargs)

def generate_text_image_to_video(prompt: str, start_image: Image.Image, 
                                end_image: Image.Image, **kwargs) -> Dict[str, Any]:
    """Generate video from text prompt with start and end images"""
    engine = get_pipeline_engine()
    return engine.generate_ti2v(prompt, start_image, end_image, **kwargs)

def unload_pipeline(model_type: str):
    """Unload a specific pipeline from memory"""
    engine = get_pipeline_engine()
    engine.unload_pipeline(model_type)

def get_pipeline_status() -> Dict[str, Any]:
    """Get status of all loaded pipelines"""
    engine = get_pipeline_engine()
    return engine.get_pipeline_status()
