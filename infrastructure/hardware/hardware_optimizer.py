"""
Hardware Optimizer for WAN22 System Optimization - RTX 4080 Implementation
"""

import os
import platform
import psutil
import logging
import subprocess
import multiprocessing
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numa
    NUMA_AVAILABLE = True
except ImportError:
    NUMA_AVAILABLE = False

@dataclass
class HardwareProfile:
    """Hardware profile containing system specifications"""
    cpu_model: str
    cpu_cores: int
    total_memory_gb: int
    gpu_model: str
    vram_gb: int
    cuda_version: str = "Unknown"
    driver_version: str = "Unknown"
    is_rtx_4080: bool = False
    is_threadripper_pro: bool = False

@dataclass
class OptimalSettings:
    """Optimal settings for hardware configuration"""
    tile_size: Tuple[int, int]
    batch_size: int
    enable_cpu_offload: bool
    enable_tensor_cores: bool
    memory_fraction: float
    num_threads: int
    enable_xformers: bool
    vae_tile_size: Tuple[int, int]
    text_encoder_offload: bool
    vae_offload: bool
    use_fp16: bool
    use_bf16: bool
    gradient_checkpointing: bool
    # Threadripper PRO specific settings
    numa_nodes: Optional[List[int]] = None
    cpu_affinity: Optional[List[int]] = None
    parallel_workers: int = 1
    enable_numa_optimization: bool = False
    preprocessing_threads: int = 1
    io_threads: int = 1

@dataclass
class OptimizationResult:
    """Result of hardware optimization"""
    success: bool
    optimizations_applied: List[str]
    performance_improvement: float
    memory_savings: int
    warnings: List[str]
    errors: List[str]
    settings: Optional[OptimalSettings] = None

class HardwareOptimizer:
    """Hardware-specific optimizer for WAN22 system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize hardware optimizer"""
        self.config_path = config_path or "config.json"
        self.logger = logging.getLogger(__name__)
        self.hardware_profile: Optional[HardwareProfile] = None
        self.optimal_settings: Optional[OptimalSettings] = None
    
    def generate_rtx_4080_settings(self, profile: HardwareProfile) -> OptimalSettings:
        """Generate optimal settings for RTX 4080"""
        self.logger.info("Generating RTX 4080 specific optimizations")
        
        # RTX 4080 specific optimizations
        settings = OptimalSettings(
            # Optimal tile size for RTX 4080 (256x256 for VAE as specified)
            tile_size=(512, 512),  # General tile size
            vae_tile_size=(256, 256),  # VAE specific tile size
            
            # Batch size optimization for 16GB VRAM
            batch_size=2 if profile.vram_gb >= 16 else 1,
            
            # Enable CPU offloading for text encoder and VAE
            enable_cpu_offload=True,
            text_encoder_offload=True,
            vae_offload=True,
            
            # Enable tensor cores for mixed precision
            enable_tensor_cores=True,
            use_fp16=True,
            use_bf16=True,  # RTX 4080 supports BF16
            
            # Memory optimization
            memory_fraction=0.9,  # Use 90% of VRAM
            gradient_checkpointing=True,
            
            # Threading optimization
            num_threads=min(profile.cpu_cores, 8),  # Optimal for most workloads
            
            # Enable xFormers for memory efficiency
            enable_xformers=True
        )
        
        return settings
    
    def generate_threadripper_pro_settings(self, profile: HardwareProfile) -> OptimalSettings:
        """Generate optimal settings for Threadripper PRO 5995WX"""
        self.logger.info("Generating Threadripper PRO 5995WX specific optimizations")
        
        # Threadripper PRO 5995WX has 64 cores/128 threads
        total_cores = profile.cpu_cores
        logical_cores = psutil.cpu_count(logical=True)
        
        # NUMA topology detection
        numa_nodes = self._detect_numa_nodes()
        
        # Threadripper PRO specific optimizations
        settings = OptimalSettings(
            # Standard settings
            tile_size=(512, 512),
            vae_tile_size=(384, 384),  # Larger tiles for powerful CPU preprocessing
            batch_size=4 if profile.vram_gb >= 16 else 2,  # Higher batch size with CPU support
            
            # CPU offloading - less aggressive due to powerful CPU
            enable_cpu_offload=True,
            text_encoder_offload=False,  # Keep on GPU with powerful CPU support
            vae_offload=False,  # Keep on GPU with powerful CPU support
            
            # GPU settings
            enable_tensor_cores=True,
            use_fp16=True,
            use_bf16=True,
            memory_fraction=0.95,  # Higher memory fraction with CPU support
            gradient_checkpointing=False,  # Disable with abundant CPU resources
            enable_xformers=True,
            
            # Threadripper PRO specific settings
            num_threads=min(total_cores, 32),  # Optimal threading for AI workloads
            numa_nodes=numa_nodes,
            cpu_affinity=self._generate_cpu_affinity(total_cores),
            parallel_workers=min(8, total_cores // 8),  # Parallel preprocessing workers
            enable_numa_optimization=len(numa_nodes) > 1 if numa_nodes else False,
            preprocessing_threads=min(16, total_cores // 4),  # Dedicated preprocessing threads
            io_threads=min(4, total_cores // 16)  # Dedicated I/O threads
        )
        
        return settings
    
    def _detect_numa_nodes(self) -> List[int]:
        """Detect available NUMA nodes"""
        numa_nodes = []
        
        if NUMA_AVAILABLE:
            try:
                numa_nodes = list(range(numa.get_max_node() + 1))
                self.logger.info(f"Detected NUMA nodes: {numa_nodes}")
            except Exception as e:
                self.logger.warning(f"NUMA detection via library failed: {e}")
        
        # Fallback: try to detect via /proc/meminfo or system commands
        if not numa_nodes:
            try:
                if os.path.exists('/proc/meminfo'):
                    # Linux NUMA detection
                    result = subprocess.run(['numactl', '--hardware'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        lines = result.stdout.split('\n')
                        for line in lines:
                            if 'available:' in line and 'nodes' in line:
                                # Parse "available: 2 nodes (0-1)"
                                parts = line.split()
                                if len(parts) >= 2:
                                    num_nodes = int(parts[1])
                                    numa_nodes = list(range(num_nodes))
                                break
                elif platform.system() == 'Windows':
                    # Windows NUMA detection
                    result = subprocess.run(['wmic', 'computersystem', 'get', 'NumberOfProcessors'], 
                                          capture_output=True, text=True, timeout=5)
                    # Simplified: assume 2 NUMA nodes for high-end Threadripper PRO
                    numa_nodes = [0, 1]
            except Exception as e:
                self.logger.warning(f"System NUMA detection failed: {e}")
        
        # Default assumption for Threadripper PRO 5995WX
        if not numa_nodes:
            numa_nodes = [0, 1]  # Assume 2 NUMA nodes
            self.logger.info("Using default NUMA configuration for Threadripper PRO")
        
        return numa_nodes
    
    def _generate_cpu_affinity(self, total_cores: int) -> List[int]:
        """Generate optimal CPU affinity for AI workloads"""
        # For Threadripper PRO, use cores from first NUMA node for main processing
        # and distribute workload across nodes
        if total_cores >= 32:
            # Use first 16 cores from each NUMA node
            affinity = list(range(0, min(16, total_cores // 2)))
            if total_cores > 32:
                affinity.extend(range(total_cores // 2, total_cores // 2 + 16))
        else:
            # Use all available cores for smaller configurations
            affinity = list(range(total_cores))
        
        return affinity
    
    def apply_threadripper_pro_optimizations(self, profile: HardwareProfile) -> OptimizationResult:
        """Apply Threadripper PRO 5995WX specific optimizations"""
        optimizations_applied = []
        warnings = []
        errors = []
        
        try:
            # Generate optimal settings
            settings = self.generate_threadripper_pro_settings(profile)
            self.optimal_settings = settings
            
            # Apply PyTorch optimizations
            if TORCH_AVAILABLE:
                # Set optimal thread count for PyTorch
                torch.set_num_threads(settings.num_threads)
                optimizations_applied.append(f"Set PyTorch threads to {settings.num_threads}")
                
                # Enable inter-op parallelism
                torch.set_num_interop_threads(settings.preprocessing_threads)
                optimizations_applied.append(f"Set PyTorch interop threads to {settings.preprocessing_threads}")
                
                # Configure CUDA settings if available
                if torch.cuda.is_available():
                    torch.cuda.set_per_process_memory_fraction(settings.memory_fraction)
                    optimizations_applied.append(f"Set CUDA memory fraction to {settings.memory_fraction}")
                    
                    # Enable tensor cores
                    if settings.enable_tensor_cores:
                        torch.backends.cudnn.allow_tf32 = True
                        torch.backends.cuda.matmul.allow_tf32 = True
                        optimizations_applied.append("Enabled Tensor Cores (TF32)")
            
            # Apply NUMA optimizations
            if settings.enable_numa_optimization and settings.numa_nodes:
                self._apply_numa_optimizations(settings)
                optimizations_applied.append(f"Applied NUMA optimizations for nodes: {settings.numa_nodes}")
            
            # Set CPU affinity for optimal performance
            if settings.cpu_affinity:
                try:
                    current_process = psutil.Process()
                    current_process.cpu_affinity(settings.cpu_affinity)
                    optimizations_applied.append(f"Set CPU affinity to cores: {settings.cpu_affinity[:8]}...")
                except Exception as e:
                    warnings.append(f"Could not set CPU affinity: {e}")
            
            # Configure environment variables for multi-core optimization
            os.environ['OMP_NUM_THREADS'] = str(settings.num_threads)
            os.environ['MKL_NUM_THREADS'] = str(settings.num_threads)
            os.environ['NUMEXPR_NUM_THREADS'] = str(settings.num_threads)
            optimizations_applied.append("Set multi-threading environment variables")
            
            # Configure parallel processing
            os.environ['TORCH_NUM_THREADS'] = str(settings.num_threads)
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Larger for high-memory system
            optimizations_applied.append("Configured parallel processing environment")
            
            # Enable optimized memory allocation
            if platform.system() == 'Linux':
                os.environ['MALLOC_ARENA_MAX'] = '4'  # Optimize memory allocation
                optimizations_applied.append("Optimized memory allocation settings")
            
            self.logger.info(f"Applied {len(optimizations_applied)} Threadripper PRO optimizations")
            
            return OptimizationResult(
                success=True,
                optimizations_applied=optimizations_applied,
                performance_improvement=0.0,
                memory_savings=0,
                warnings=warnings,
                errors=errors,
                settings=settings
            )
            
        except Exception as e:
            error_msg = f"Failed to apply Threadripper PRO optimizations: {e}"
            self.logger.error(error_msg)
            errors.append(error_msg)
            
            return OptimizationResult(
                success=False,
                optimizations_applied=optimizations_applied,
                performance_improvement=0.0,
                memory_savings=0,
                warnings=warnings,
                errors=errors
            )
    
    def _apply_numa_optimizations(self, settings: OptimalSettings) -> None:
        """Apply NUMA-aware memory allocation optimizations"""
        try:
            if NUMA_AVAILABLE and settings.numa_nodes:
                # Set memory policy for optimal NUMA allocation
                numa.set_preferred_node(settings.numa_nodes[0])
                self.logger.info(f"Set preferred NUMA node to {settings.numa_nodes[0]}")
                
                # Configure memory interleaving across NUMA nodes
                if len(settings.numa_nodes) > 1:
                    numa.set_interleave_mask(settings.numa_nodes)
                    self.logger.info(f"Enabled memory interleaving across nodes: {settings.numa_nodes}")
            
            # Set NUMA-related environment variables
            if settings.numa_nodes:
                os.environ['NUMA_PREFERRED_NODE'] = str(settings.numa_nodes[0])
                if len(settings.numa_nodes) > 1:
                    os.environ['NUMA_INTERLEAVE_NODES'] = ','.join(map(str, settings.numa_nodes))
                
        except Exception as e:
            self.logger.warning(f"NUMA optimization failed: {e}")
    
    def configure_parallel_preprocessing(self, num_workers: int = None) -> Dict[str, int]:
        """Configure parallel preprocessing for multi-core CPU utilization"""
        if num_workers is None:
            # Auto-detect optimal worker count
            cpu_cores = psutil.cpu_count(logical=False)
            num_workers = min(8, max(2, cpu_cores // 8))
        
        config = {
            'preprocessing_workers': num_workers,
            'io_workers': min(4, max(1, num_workers // 2)),
            'batch_processing_workers': min(2, max(1, num_workers // 4))
        }
        
        try:
            # Set multiprocessing start method for optimal performance
            if hasattr(multiprocessing, 'set_start_method'):
                try:
                    multiprocessing.set_start_method('spawn', force=True)
                except RuntimeError:
                    pass  # Already set
            
            self.parallel_config = config
            self.logger.info(f"Configured parallel preprocessing: {config}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to configure parallel preprocessing: {e}")
            return {'preprocessing_workers': 1, 'io_workers': 1, 'batch_processing_workers': 1}
    
    def apply_rtx_4080_optimizations(self, profile: HardwareProfile) -> OptimizationResult:
        """Apply RTX 4080 specific optimizations"""
        optimizations_applied = []
        warnings = []
        errors = []
        
        try:
            # Generate optimal settings
            settings = self.generate_rtx_4080_settings(profile)
            self.optimal_settings = settings
            
            # Apply PyTorch optimizations
            if TORCH_AVAILABLE:
                # Enable tensor cores
                if settings.enable_tensor_cores:
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                    optimizations_applied.append("Enabled Tensor Cores (TF32)")
                
                # Set memory fraction
                if torch.cuda.is_available():
                    torch.cuda.set_per_process_memory_fraction(settings.memory_fraction)
                    optimizations_applied.append(f"Set CUDA memory fraction to {settings.memory_fraction}")
                
                # Apply threading optimizations
                if settings.num_threads:
                    torch.set_num_threads(settings.num_threads)
                    optimizations_applied.append(f"Set PyTorch threads to {settings.num_threads}")
            
            # Environment variables for optimization
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            optimizations_applied.append("Set CUDA memory allocator optimization")
            
            # Enable mixed precision training
            if settings.use_fp16 or settings.use_bf16:
                os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
                optimizations_applied.append("Enabled cuDNN v8 API for mixed precision")
            
            self.logger.info(f"Applied {len(optimizations_applied)} RTX 4080 optimizations")
            
            return OptimizationResult(
                success=True,
                optimizations_applied=optimizations_applied,
                performance_improvement=0.0,
                memory_savings=0,
                warnings=warnings,
                errors=errors,
                settings=settings
            )
            
        except Exception as e:
            error_msg = f"Failed to apply RTX 4080 optimizations: {e}"
            self.logger.error(error_msg)
            errors.append(error_msg)
            
            return OptimizationResult(
                success=False,
                optimizations_applied=optimizations_applied,
                performance_improvement=0.0,
                memory_savings=0,
                warnings=warnings,
                errors=errors
            )
    
    def configure_vae_tiling(self, tile_size: Tuple[int, int] = (256, 256)) -> bool:
        """Configure VAE tiling for memory efficiency"""
        try:
            self.vae_tile_size = tile_size
            self.logger.info(f"Configured VAE tiling: {tile_size}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to configure VAE tiling: {e}")
            return False
    
    def configure_cpu_offloading(self, text_encoder: bool = True, vae: bool = True) -> Dict[str, bool]:
        """Configure CPU offloading for components"""
        config = {
            'text_encoder_offload': text_encoder,
            'vae_offload': vae
        }
        
        try:
            self.cpu_offload_config = config
            self.logger.info(f"Configured CPU offloading: {config}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to configure CPU offloading: {e}")
            return {'text_encoder_offload': False, 'vae_offload': False}
    
    def get_memory_optimization_settings(self, available_vram_gb: int) -> Dict[str, Any]:
        """Get memory optimization settings based on available VRAM"""
        if available_vram_gb >= 16:  # RTX 4080
            return {
                'enable_attention_slicing': False,
                'enable_vae_slicing': False,
                'enable_cpu_offload': True,
                'batch_size': 2,
                'tile_size': (512, 512),
                'vae_tile_size': (256, 256)
            }
        elif available_vram_gb >= 12:
            return {
                'enable_attention_slicing': True,
                'enable_vae_slicing': True,
                'enable_cpu_offload': True,
                'batch_size': 1,
                'tile_size': (384, 384),
                'vae_tile_size': (192, 192)
            }
        else:
            return {
                'enable_attention_slicing': True,
                'enable_vae_slicing': True,
                'enable_cpu_offload': True,
                'batch_size': 1,
                'tile_size': (256, 256),
                'vae_tile_size': (128, 128)
            }    

    def detect_hardware_profile(self) -> HardwareProfile:
        """Detect current hardware configuration"""
        try:
            # CPU Detection
            cpu_model = platform.processor() or "Unknown"
            cpu_cores = psutil.cpu_count(logical=False)
            total_memory_gb = round(psutil.virtual_memory().total / (1024**3))
            
            # GPU Detection
            gpu_model = "Unknown"
            vram_gb = 0
            cuda_version = "Unknown"
            driver_version = "Unknown"
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_model = torch.cuda.get_device_name(0)
                vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3))
                cuda_version = torch.version.cuda or "Unknown"
            
            # Hardware-specific detection
            is_rtx_4080 = "RTX 4080" in gpu_model.upper()
            is_threadripper_pro = "THREADRIPPER PRO" in cpu_model.upper() or "5995WX" in cpu_model
            
            profile = HardwareProfile(
                cpu_model=cpu_model,
                cpu_cores=cpu_cores,
                total_memory_gb=total_memory_gb,
                gpu_model=gpu_model,
                vram_gb=vram_gb,
                cuda_version=cuda_version,
                driver_version=driver_version,
                is_rtx_4080=is_rtx_4080,
                is_threadripper_pro=is_threadripper_pro
            )
            
            self.hardware_profile = profile
            self.logger.info(f"Detected hardware profile: {profile}")
            return profile
            
        except Exception as e:
            self.logger.error(f"Hardware detection failed: {e}")
            # Return minimal profile
            return HardwareProfile(
                cpu_model="Unknown",
                cpu_cores=psutil.cpu_count(logical=False) or 4,
                total_memory_gb=8,
                gpu_model="Unknown",
                vram_gb=4,
                cuda_version="Unknown",
                driver_version="Unknown"
            )
    
    def save_optimization_profile(self, filepath: str) -> bool:
        """Save current optimization profile to file"""
        try:
            import json
            from dataclasses import asdict
            from datetime import datetime
            
            profile_data = {
                'hardware_profile': asdict(self.hardware_profile) if self.hardware_profile else None,
                'optimal_settings': asdict(self.optimal_settings) if self.optimal_settings else None,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(profile_data, f, indent=2)
            
            self.logger.info(f"Saved optimization profile to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save optimization profile: {e}")
            return False
    
    def load_optimization_profile(self, filepath: str) -> bool:
        """Load optimization profile from file"""
        try:
            import json
            
            with open(filepath, 'r') as f:
                profile_data = json.load(f)
            
            if profile_data.get('hardware_profile'):
                self.hardware_profile = HardwareProfile(**profile_data['hardware_profile'])
            
            if profile_data.get('optimal_settings'):
                settings_data = profile_data['optimal_settings']
                # Convert lists back to tuples for tile sizes
                if 'tile_size' in settings_data and isinstance(settings_data['tile_size'], list):
                    settings_data['tile_size'] = tuple(settings_data['tile_size'])
                if 'vae_tile_size' in settings_data and isinstance(settings_data['vae_tile_size'], list):
                    settings_data['vae_tile_size'] = tuple(settings_data['vae_tile_size'])
                
                self.optimal_settings = OptimalSettings(**settings_data)
            
            self.logger.info(f"Loaded optimization profile from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load optimization profile: {e}")
            return False
    
    def generate_optimal_settings(self, profile: HardwareProfile) -> OptimalSettings:
        """Generate optimal settings based on hardware profile"""
        if profile.is_threadripper_pro:
            return self.generate_threadripper_pro_settings(profile)
        elif profile.is_rtx_4080:
            return self.generate_rtx_4080_settings(profile)
        else:
            # Default settings for other hardware
            return OptimalSettings(
                tile_size=(384, 384),
                batch_size=1,
                enable_cpu_offload=True,
                enable_tensor_cores=False,
                memory_fraction=0.8,
                num_threads=min(profile.cpu_cores, 4),
                enable_xformers=True,
                vae_tile_size=(192, 192),
                text_encoder_offload=True,
                vae_offload=True,
                use_fp16=True,
                use_bf16=False,
                gradient_checkpointing=True
            )
    
    def apply_hardware_optimizations(self, profile: HardwareProfile = None) -> OptimizationResult:
        """Apply hardware-specific optimizations"""
        if profile is None:
            profile = self.detect_hardware_profile()
        
        if profile.is_threadripper_pro:
            self.logger.info("Applying Threadripper PRO 5995WX optimizations")
            return self.apply_threadripper_pro_optimizations(profile)
        elif profile.is_rtx_4080:
            self.logger.info("Applying RTX 4080 optimizations")
            return self.apply_rtx_4080_optimizations(profile)
        else:
            self.logger.info("Applying default hardware optimizations")
            return self._apply_default_optimizations(profile)
    
    def _apply_default_optimizations(self, profile: HardwareProfile) -> OptimizationResult:
        """Apply default optimizations for unrecognized hardware"""
        optimizations_applied = []
        warnings = []
        errors = []
        
        try:
            settings = self.generate_optimal_settings(profile)
            self.optimal_settings = settings
            
            if TORCH_AVAILABLE:
                torch.set_num_threads(settings.num_threads)
                optimizations_applied.append(f"Set PyTorch threads to {settings.num_threads}")
                
                if torch.cuda.is_available():
                    torch.cuda.set_per_process_memory_fraction(settings.memory_fraction)
                    optimizations_applied.append(f"Set CUDA memory fraction to {settings.memory_fraction}")
            
            return OptimizationResult(
                success=True,
                optimizations_applied=optimizations_applied,
                performance_improvement=0.0,
                memory_savings=0,
                warnings=warnings,
                errors=errors,
                settings=settings
            )
            
        except Exception as e:
            error_msg = f"Failed to apply default optimizations: {e}"
            self.logger.error(error_msg)
            errors.append(error_msg)
            
            return OptimizationResult(
                success=False,
                optimizations_applied=optimizations_applied,
                performance_improvement=0.0,
                memory_savings=0,
                warnings=warnings,
                errors=errors
            )
