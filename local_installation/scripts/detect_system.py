"""
System hardware detection module for WAN2.2 installation.
Detects CPU, memory, GPU, and storage specifications for optimal configuration.
"""

import os
import platform
import subprocess
import json
import re
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

try:
    import GPUtil
except ImportError:
    GPUtil = None

from interfaces import (
    ISystemDetector, HardwareProfile, CPUInfo, MemoryInfo, 
    GPUInfo, StorageInfo, OSInfo, ValidationResult, InstallationError, ErrorCategory
)
from base_classes import BaseInstallationComponent


class SystemDetector(BaseInstallationComponent, ISystemDetector):
    """Comprehensive system hardware detection."""
    
    def __init__(self, installation_path: str, logger: Optional[logging.Logger] = None):
        super().__init__(installation_path, logger)
        self.minimum_requirements = {
            "cpu_cores": 4,
            "memory_gb": 8,
            "storage_gb": 50,
            "gpu_vram_gb": 4
        }
    
    def detect_hardware(self) -> HardwareProfile:
        """Detect and return complete hardware profile."""
        self.logger.info("Starting hardware detection...")
        
        try:
            cpu_info = self._detect_cpu()
            memory_info = self._detect_memory()
            gpu_info = self._detect_gpu()
            storage_info = self._detect_storage()
            os_info = self._detect_os()
            
            profile = HardwareProfile(
                cpu=cpu_info,
                memory=memory_info,
                gpu=gpu_info,
                storage=storage_info,
                os=os_info
            )
            
            self.logger.info("Hardware detection completed successfully")
            self._log_hardware_summary(profile)
            return profile
            
        except Exception as e:
            self.logger.error(f"Hardware detection failed: {str(e)}")
            raise InstallationError(
                f"Failed to detect system hardware: {str(e)}",
                ErrorCategory.SYSTEM,
                ["Check system permissions", "Ensure WMI service is running", "Try running as administrator"]
            )
    
    def _detect_cpu(self) -> CPUInfo:
        """Detect CPU specifications."""
        self.logger.debug("Detecting CPU information...")
        
        try:
            # Get basic CPU info from platform
            cpu_model = platform.processor() or "Unknown CPU"
            
            # Use psutil if available for more detailed info
            if psutil:
                cpu_count = psutil.cpu_count(logical=False)  # Physical cores
                cpu_threads = psutil.cpu_count(logical=True)  # Logical cores
                cpu_freq = psutil.cpu_freq()
                base_clock = cpu_freq.current / 1000 if cpu_freq else 0.0
                boost_clock = cpu_freq.max / 1000 if cpu_freq and cpu_freq.max else base_clock
            else:
                # Fallback to basic detection
                cpu_count = os.cpu_count() or 4
                cpu_threads = cpu_count
                base_clock = 0.0
                boost_clock = 0.0
            
            # Try to get more detailed CPU info from Windows WMI
            try:
                wmi_info = self._get_cpu_wmi_info()
                if wmi_info:
                    cpu_model = wmi_info.get('name', cpu_model)
                    if wmi_info.get('max_clock_speed'):
                        boost_clock = float(wmi_info['max_clock_speed']) / 1000
                    if wmi_info.get('current_clock_speed'):
                        base_clock = float(wmi_info['current_clock_speed']) / 1000
            except Exception as e:
                self.logger.debug(f"WMI CPU detection failed: {e}")
            
            # Determine architecture
            architecture = platform.machine().lower()
            if architecture in ['amd64', 'x86_64']:
                architecture = 'x64'
            elif architecture in ['i386', 'i686']:
                architecture = 'x86'
            
            cpu_info = CPUInfo(
                model=cpu_model.strip(),
                cores=cpu_count,
                threads=cpu_threads,
                base_clock=base_clock,
                boost_clock=boost_clock,
                architecture=architecture
            )
            
            self.logger.debug(f"CPU detected: {cpu_info.model} ({cpu_info.cores}C/{cpu_info.threads}T)")
            return cpu_info
            
        except Exception as e:
            self.logger.warning(f"CPU detection failed, using defaults: {e}")
            return CPUInfo(
                model="Unknown CPU",
                cores=4,
                threads=4,
                base_clock=0.0,
                boost_clock=0.0,
                architecture="x64"
            )
    
    def _get_cpu_wmi_info(self) -> Optional[Dict[str, Any]]:
        """Get CPU information from Windows WMI."""
        try:
            cmd = [
                'wmic', 'cpu', 'get', 
                'Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,CurrentClockSpeed',
                '/format:csv'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    # Parse CSV output
                    headers = lines[0].split(',')
                    data = lines[1].split(',')
                    
                    if len(data) >= len(headers):
                        wmi_data = {}
                        for i, header in enumerate(headers):
                            if i < len(data) and data[i].strip():
                                key = header.strip().lower()
                                value = data[i].strip()
                                if key == 'name':
                                    wmi_data['name'] = value
                                elif key == 'maxclockspeed':
                                    wmi_data['max_clock_speed'] = value
                                elif key == 'currentclockspeed':
                                    wmi_data['current_clock_speed'] = value
                        return wmi_data
        except Exception as e:
            self.logger.debug(f"WMI CPU query failed: {e}")
        
        return None
    
    def _detect_memory(self) -> MemoryInfo:
        """Detect memory specifications."""
        self.logger.debug("Detecting memory information...")
        
        try:
            if psutil:
                memory = psutil.virtual_memory()
                total_gb = int(memory.total / (1024**3))
                available_gb = int(memory.available / (1024**3))
            else:
                # Fallback using Windows commands
                total_gb = self._get_memory_from_wmic()
                available_gb = max(1, int(total_gb * 0.8))  # Estimate 80% available
            
            # Try to get memory type and speed from WMI
            memory_type = "Unknown"
            memory_speed = 0
            
            try:
                memory_details = self._get_memory_wmi_info()
                if memory_details:
                    memory_type = memory_details.get('type', 'Unknown')
                    memory_speed = memory_details.get('speed', 0)
            except Exception as e:
                self.logger.debug(f"Memory WMI detection failed: {e}")
            
            memory_info = MemoryInfo(
                total_gb=total_gb,
                available_gb=available_gb,
                type=memory_type,
                speed=memory_speed
            )
            
            self.logger.debug(f"Memory detected: {memory_info.total_gb}GB total, {memory_info.available_gb}GB available")
            return memory_info
            
        except Exception as e:
            self.logger.warning(f"Memory detection failed, using defaults: {e}")
            return MemoryInfo(
                total_gb=8,
                available_gb=6,
                type="Unknown",
                speed=0
            )
    
    def _get_memory_from_wmic(self) -> int:
        """Get total memory from Windows WMI command."""
        try:
            cmd = ['wmic', 'computersystem', 'get', 'TotalPhysicalMemory', '/value']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'TotalPhysicalMemory=' in line:
                        memory_bytes = int(line.split('=')[1].strip())
                        return int(memory_bytes / (1024**3))
        except Exception as e:
            self.logger.debug(f"WMIC memory query failed: {e}")
        
        return 8  # Default fallback
    
    def _get_memory_wmi_info(self) -> Optional[Dict[str, Any]]:
        """Get detailed memory information from WMI."""
        try:
            cmd = ['wmic', 'memorychip', 'get', 'MemoryType,Speed', '/format:csv']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    # Parse first memory module
                    data = lines[1].split(',')
                    if len(data) >= 3:
                        memory_type_code = data[1].strip() if len(data) > 1 else ""
                        speed = data[2].strip() if len(data) > 2 else "0"
                        
                        # Convert memory type code to readable format
                        memory_types = {
                            "20": "DDR",
                            "21": "DDR2",
                            "24": "DDR3",
                            "26": "DDR4",
                            "34": "DDR5"
                        }
                        
                        memory_type = memory_types.get(memory_type_code, "Unknown")
                        memory_speed = int(speed) if speed.isdigit() else 0
                        
                        return {
                            'type': memory_type,
                            'speed': memory_speed
                        }
        except Exception as e:
            self.logger.debug(f"Memory WMI detail query failed: {e}")
        
        return None
    
    def _detect_gpu(self) -> Optional[GPUInfo]:
        """Detect GPU specifications."""
        self.logger.debug("Detecting GPU information...")
        
        try:
            # Try multiple methods to detect GPU
            gpu_info = None
            
            # Method 1: Try GPUtil if available
            if GPUtil:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        gpu_info = GPUInfo(
                            model=gpu.name,
                            vram_gb=int(gpu.memoryTotal / 1024),  # Convert MB to GB
                            cuda_version=self._get_cuda_version(),
                            driver_version=gpu.driver,
                            compute_capability=self._get_compute_capability(gpu.name)
                        )
                except Exception as e:
                    self.logger.debug(f"GPUtil detection failed: {e}")
            
            # Method 2: Try Windows WMI
            if not gpu_info:
                gpu_info = self._get_gpu_wmi_info()
            
            # Method 3: Try nvidia-smi for NVIDIA GPUs
            if not gpu_info:
                gpu_info = self._get_nvidia_smi_info()
            
            # Method 4: Try DirectX diagnostic
            if not gpu_info:
                gpu_info = self._get_dxdiag_gpu_info()
            
            if gpu_info:
                self.logger.debug(f"GPU detected: {gpu_info.model} ({gpu_info.vram_gb}GB VRAM)")
            else:
                self.logger.debug("No GPU detected or GPU detection failed")
            
            return gpu_info
            
        except Exception as e:
            self.logger.warning(f"GPU detection failed: {e}")
            return None
    
    def _get_cuda_version(self) -> str:
        """Get CUDA version if available."""
        try:
            # Try nvidia-smi first
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Parse CUDA version from nvidia-smi output
                cuda_match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
                if cuda_match:
                    return cuda_match.group(1)
            
            # Try nvcc if available
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version_match = re.search(r'release (\d+\.\d+)', result.stdout)
                if version_match:
                    return version_match.group(1)
        except Exception as e:
            self.logger.debug(f"CUDA version detection failed: {e}")
        
        return "Unknown"
    
    def _get_compute_capability(self, gpu_name: str) -> str:
        """Get compute capability based on GPU name."""
        # Common GPU compute capabilities
        gpu_capabilities = {
            # RTX 40 series
            "RTX 4090": "8.9",
            "RTX 4080": "8.9",
            "RTX 4070": "8.9",
            "RTX 4060": "8.9",
            # RTX 30 series
            "RTX 3090": "8.6",
            "RTX 3080": "8.6",
            "RTX 3070": "8.6",
            "RTX 3060": "8.6",
            # RTX 20 series
            "RTX 2080": "7.5",
            "RTX 2070": "7.5",
            "RTX 2060": "7.5",
            # GTX 16 series
            "GTX 1660": "7.5",
            "GTX 1650": "7.5",
            # GTX 10 series
            "GTX 1080": "6.1",
            "GTX 1070": "6.1",
            "GTX 1060": "6.1",
        }
        
        for gpu_model, capability in gpu_capabilities.items():
            if gpu_model.lower() in gpu_name.lower():
                return capability
        
        return "Unknown"
    
    def _get_gpu_wmi_info(self) -> Optional[GPUInfo]:
        """Get GPU information from Windows WMI."""
        try:
            cmd = ['wmic', 'path', 'win32_VideoController', 'get', 'Name,AdapterRAM,DriverVersion', '/format:csv']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        parts = line.split(',')
                        if len(parts) >= 4:
                            gpu_name = parts[2].strip()
                            adapter_ram = parts[1].strip()
                            driver_version = parts[3].strip()
                            
                            # Skip basic display adapters
                            if any(skip in gpu_name.lower() for skip in ['basic', 'standard', 'microsoft']):
                                continue
                            
                            # Calculate VRAM in GB
                            vram_gb = 0
                            if adapter_ram and adapter_ram.isdigit():
                                vram_gb = max(1, int(adapter_ram) // (1024**3))
                            
                            return GPUInfo(
                                model=gpu_name,
                                vram_gb=vram_gb,
                                cuda_version=self._get_cuda_version(),
                                driver_version=driver_version,
                                compute_capability=self._get_compute_capability(gpu_name)
                            )
        except Exception as e:
            self.logger.debug(f"WMI GPU detection failed: {e}")
        
        return None
    
    def _get_nvidia_smi_info(self) -> Optional[GPUInfo]:
        """Get GPU information from nvidia-smi."""
        try:
            cmd = ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0].strip():
                    parts = lines[0].split(',')
                    if len(parts) >= 3:
                        gpu_name = parts[0].strip()
                        vram_mb = int(parts[1].strip())
                        driver_version = parts[2].strip()
                        
                        return GPUInfo(
                            model=gpu_name,
                            vram_gb=int(vram_mb / 1024),
                            cuda_version=self._get_cuda_version(),
                            driver_version=driver_version,
                            compute_capability=self._get_compute_capability(gpu_name)
                        )
        except Exception as e:
            self.logger.debug(f"nvidia-smi detection failed: {e}")
        
        return None
    
    def _get_dxdiag_gpu_info(self) -> Optional[GPUInfo]:
        """Get GPU information from DirectX diagnostic."""
        try:
            # Run dxdiag and save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                temp_path = temp_file.name
            
            cmd = ['dxdiag', '/t', temp_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                try:
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Parse GPU information from dxdiag output
                    gpu_match = re.search(r'Card name: (.+)', content)
                    memory_match = re.search(r'Dedicated Memory: (\d+) MB', content)
                    driver_match = re.search(r'Driver Version: (.+)', content)
                    
                    if gpu_match:
                        gpu_name = gpu_match.group(1).strip()
                        vram_gb = 0
                        
                        if memory_match:
                            vram_mb = int(memory_match.group(1))
                            vram_gb = max(1, vram_mb // 1024)
                        
                        driver_version = driver_match.group(1).strip() if driver_match else "Unknown"
                        
                        return GPUInfo(
                            model=gpu_name,
                            vram_gb=vram_gb,
                            cuda_version=self._get_cuda_version(),
                            driver_version=driver_version,
                            compute_capability=self._get_compute_capability(gpu_name)
                        )
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
        except Exception as e:
            self.logger.debug(f"dxdiag GPU detection failed: {e}")
        
        return None
    
    def _detect_storage(self) -> StorageInfo:
        """Detect storage specifications."""
        self.logger.debug("Detecting storage information...")
        
        try:
            # Get available space on installation drive
            if psutil:
                disk_usage = psutil.disk_usage(str(self.installation_path))
                available_gb = int(disk_usage.free / (1024**3))
            else:
                # Fallback using Windows dir command
                available_gb = self._get_storage_from_dir()
            
            # Try to determine drive type (SSD vs HDD)
            drive_type = self._get_drive_type()
            
            storage_info = StorageInfo(
                available_gb=available_gb,
                type=drive_type
            )
            
            self.logger.debug(f"Storage detected: {storage_info.available_gb}GB available ({storage_info.type})")
            return storage_info
            
        except Exception as e:
            self.logger.warning(f"Storage detection failed, using defaults: {e}")
            return StorageInfo(
                available_gb=100,
                type="Unknown"
            )
    
    def _get_storage_from_dir(self) -> int:
        """Get available storage using Windows dir command."""
        try:
            drive = str(self.installation_path)[:2]  # Get drive letter (e.g., "C:")
            cmd = ['dir', drive, '/-c']
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True, timeout=10)
            
            if result.returncode == 0:
                # Parse free space from dir output
                lines = result.stdout.split('\n')
                for line in reversed(lines):
                    if 'bytes free' in line.lower():
                        # Extract number before "bytes free"
                        match = re.search(r'([\d,]+)\s+bytes free', line, re.IGNORECASE)
                        if match:
                            bytes_free = int(match.group(1).replace(',', ''))
                            return int(bytes_free / (1024**3))
        except Exception as e:
            self.logger.debug(f"Dir storage query failed: {e}")
        
        return 100  # Default fallback
    
    def _get_drive_type(self) -> str:
        """Determine drive type (SSD, HDD, NVMe)."""
        try:
            # Try to get drive type from WMI
            drive_letter = str(self.installation_path)[0]
            cmd = [
                'wmic', 'diskdrive', 'where', f'DeviceID like "%{drive_letter}%"',
                'get', 'MediaType,Model', '/format:csv'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    data = lines[1].split(',')
                    if len(data) >= 3:
                        media_type = data[1].strip().lower()
                        model = data[2].strip().lower()
                        
                        # Determine drive type based on media type and model
                        if 'ssd' in model or 'solid state' in media_type:
                            if 'nvme' in model or 'pcie' in model:
                                return "NVMe SSD"
                            return "SSD"
                        elif 'fixed hard disk' in media_type or 'hdd' in model:
                            return "HDD"
            
            # Alternative method using PowerShell
            ps_cmd = [
                'powershell', '-Command',
                f'Get-PhysicalDisk | Where-Object {{$_.DeviceId -eq (Get-Partition -DriveLetter {drive_letter}).DiskNumber}} | Select-Object MediaType'
            ]
            result = subprocess.run(ps_cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                media_type = result.stdout.strip().lower()
                if 'ssd' in media_type:
                    return "SSD"
                elif 'hdd' in media_type:
                    return "HDD"
        
        except Exception as e:
            self.logger.debug(f"Drive type detection failed: {e}")
        
        return "Unknown"
    
    def _detect_os(self) -> OSInfo:
        """Detect operating system information."""
        self.logger.debug("Detecting OS information...")
        
        try:
            os_name = platform.system()
            os_version = platform.version()
            os_release = platform.release()
            architecture = platform.machine()
            
            # Get more detailed Windows version info
            if os_name == "Windows":
                try:
                    # Try to get Windows version from registry or WMI
                    detailed_version = self._get_windows_version()
                    if detailed_version:
                        os_version = detailed_version
                except Exception as e:
                    self.logger.debug(f"Detailed Windows version detection failed: {e}")
            
            # Normalize architecture
            if architecture.lower() in ['amd64', 'x86_64']:
                architecture = 'x64'
            elif architecture.lower() in ['i386', 'i686']:
                architecture = 'x86'
            
            os_info = OSInfo(
                name=f"{os_name} {os_release}",
                version=os_version,
                architecture=architecture
            )
            
            self.logger.debug(f"OS detected: {os_info.name} {os_info.version} ({os_info.architecture})")
            return os_info
            
        except Exception as e:
            self.logger.warning(f"OS detection failed, using defaults: {e}")
            return OSInfo(
                name="Windows 10",
                version="Unknown",
                architecture="x64"
            )
    
    def _get_windows_version(self) -> Optional[str]:
        """Get detailed Windows version information."""
        try:
            cmd = ['wmic', 'os', 'get', 'Caption,Version', '/format:csv']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    data = lines[1].split(',')
                    if len(data) >= 3:
                        caption = data[1].strip()
                        version = data[2].strip()
                        return f"{caption} (Build {version})"
        except Exception as e:
            self.logger.debug(f"Windows version WMI query failed: {e}")
        
        return None  
  
    def get_optimal_settings(self, profile: HardwareProfile) -> Dict[str, Any]:
        """Generate optimal settings for the given hardware profile."""
        self.logger.info("Generating optimal settings based on hardware profile...")
        
        # Classify hardware performance tier
        performance_tier = self._classify_performance_tier(profile)
        self.logger.info(f"Hardware classified as: {performance_tier}")
        
        # Base settings
        settings = {
            "performance_tier": performance_tier,
            "cpu_threads": min(profile.cpu.threads, 32),  # Cap at 32 for stability
            "memory_allocation_gb": max(4, min(profile.memory.available_gb - 2, 32)),
            "enable_gpu_acceleration": profile.gpu is not None,
            "storage_optimization": profile.storage.type
        }
        
        # GPU-specific settings
        if profile.gpu:
            settings.update({
                "gpu_model": profile.gpu.model,
                "max_vram_usage_gb": max(2, profile.gpu.vram_gb - 2),  # Leave 2GB for system
                "cuda_enabled": "cuda" in profile.gpu.cuda_version.lower() if profile.gpu.cuda_version != "Unknown" else False,
                "gpu_precision": self._get_optimal_precision(profile.gpu)
            })
        
        # Performance tier specific optimizations
        if performance_tier == "high_performance":
            settings.update({
                "quantization": "bf16",
                "enable_model_offload": False,
                "vae_tile_size": 512,
                "max_queue_size": 20,
                "worker_threads": min(profile.cpu.threads // 2, 16),
                "preload_models": True,
                "cache_models": True
            })
        elif performance_tier == "mid_range":
            settings.update({
                "quantization": "fp16",
                "enable_model_offload": True,
                "vae_tile_size": 256,
                "max_queue_size": 10,
                "worker_threads": min(profile.cpu.threads // 4, 8),
                "preload_models": False,
                "cache_models": False
            })
        else:  # budget
            settings.update({
                "quantization": "int8",
                "enable_model_offload": True,
                "vae_tile_size": 128,
                "max_queue_size": 5,
                "worker_threads": min(profile.cpu.threads // 4, 4),
                "preload_models": False,
                "cache_models": False
            })
        
        self.logger.debug(f"Generated optimal settings: {settings}")
        return settings
    
    def _classify_performance_tier(self, profile: HardwareProfile) -> str:
        """Classify hardware into performance tiers."""
        # High performance criteria
        high_perf_criteria = [
            profile.cpu.cores >= 16,
            profile.cpu.threads >= 32,
            profile.memory.total_gb >= 32,
            profile.gpu and profile.gpu.vram_gb >= 12,
            profile.storage.type in ["SSD", "NVMe SSD"]
        ]
        
        # Mid range criteria
        mid_range_criteria = [
            profile.cpu.cores >= 6,
            profile.cpu.threads >= 8,
            profile.memory.total_gb >= 16,
            profile.gpu and profile.gpu.vram_gb >= 6,
        ]
        
        if sum(high_perf_criteria) >= 4:
            return "high_performance"
        elif sum(mid_range_criteria) >= 3:
            return "mid_range"
        else:
            return "budget"
    
    def _get_optimal_precision(self, gpu: GPUInfo) -> str:
        """Get optimal precision based on GPU capabilities."""
        if gpu.vram_gb >= 12:
            return "fp16"
        elif gpu.vram_gb >= 8:
            return "fp16"
        else:
            return "int8"
    
    def validate_requirements(self, profile: HardwareProfile) -> ValidationResult:
        """Validate that hardware meets minimum requirements."""
        self.logger.info("Validating hardware against minimum requirements...")
        
        issues = []
        warnings = []
        
        # Check CPU requirements
        if profile.cpu.cores < self.minimum_requirements["cpu_cores"]:
            issues.append(f"CPU cores: {profile.cpu.cores} < {self.minimum_requirements['cpu_cores']} required")
        
        # Check memory requirements
        if profile.memory.total_gb < self.minimum_requirements["memory_gb"]:
            issues.append(f"Memory: {profile.memory.total_gb}GB < {self.minimum_requirements['memory_gb']}GB required")
        
        # Check storage requirements
        if profile.storage.available_gb < self.minimum_requirements["storage_gb"]:
            issues.append(f"Storage: {profile.storage.available_gb}GB < {self.minimum_requirements['storage_gb']}GB required")
        
        # Check GPU requirements (warning if not met)
        if not profile.gpu:
            warnings.append("No GPU detected - CPU-only mode will be significantly slower")
        elif profile.gpu.vram_gb < self.minimum_requirements["gpu_vram_gb"]:
            warnings.append(f"GPU VRAM: {profile.gpu.vram_gb}GB < {self.minimum_requirements['gpu_vram_gb']}GB recommended")
        
        # Check OS compatibility
        if "windows" not in profile.os.name.lower():
            issues.append(f"Unsupported OS: {profile.os.name} (Windows required)")
        
        # Check architecture
        if profile.os.architecture not in ["x64", "amd64"]:
            issues.append(f"Unsupported architecture: {profile.os.architecture} (x64 required)")
        
        success = len(issues) == 0
        message = "Hardware validation passed" if success else f"Hardware validation failed: {len(issues)} issues found"
        
        result = ValidationResult(
            success=success,
            message=message,
            details={
                "issues": issues,
                "performance_tier": self._classify_performance_tier(profile),
                "recommended_settings": self.get_optimal_settings(profile) if success else None
            },
            warnings=warnings if warnings else None
        )
        
        if success:
            self.logger.info("Hardware validation passed")
        else:
            self.logger.error(f"Hardware validation failed: {issues}")
        
        if warnings:
            self.logger.warning(f"Hardware warnings: {warnings}")
        
        return result
    
    def _log_hardware_summary(self, profile: HardwareProfile) -> None:
        """Log a summary of detected hardware."""
        self.logger.info("=== Hardware Detection Summary ===")
        self.logger.info(f"CPU: {profile.cpu.model}")
        self.logger.info(f"  Cores: {profile.cpu.cores}, Threads: {profile.cpu.threads}")
        self.logger.info(f"  Clock: {profile.cpu.base_clock:.1f}GHz base, {profile.cpu.boost_clock:.1f}GHz boost")
        
        self.logger.info(f"Memory: {profile.memory.total_gb}GB {profile.memory.type}")
        self.logger.info(f"  Available: {profile.memory.available_gb}GB")
        if profile.memory.speed > 0:
            self.logger.info(f"  Speed: {profile.memory.speed}MHz")
        
        if profile.gpu:
            self.logger.info(f"GPU: {profile.gpu.model}")
            self.logger.info(f"  VRAM: {profile.gpu.vram_gb}GB")
            self.logger.info(f"  Driver: {profile.gpu.driver_version}")
            if profile.gpu.cuda_version != "Unknown":
                self.logger.info(f"  CUDA: {profile.gpu.cuda_version}")
        else:
            self.logger.info("GPU: None detected")
        
        self.logger.info(f"Storage: {profile.storage.available_gb}GB available ({profile.storage.type})")
        self.logger.info(f"OS: {profile.os.name} {profile.os.version} ({profile.os.architecture})")
        self.logger.info("=== End Hardware Summary ===")


def main():
    """Test hardware detection functionality."""
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create detector
    detector = SystemDetector(".")
    
    try:
        # Detect hardware
        profile = detector.detect_hardware()
        
        # Get optimal settings
        settings = detector.get_optimal_settings(profile)
        print("\nOptimal Settings:")
        print(json.dumps(settings, indent=2))
        
        # Validate requirements
        validation = detector.validate_requirements(profile)
        print(f"\nValidation Result: {validation.message}")
        if validation.warnings:
            print("Warnings:")
            for warning in validation.warnings:
                print(f"  - {warning}")
        
        if not validation.success and validation.details:
            print("Issues:")
            for issue in validation.details.get("issues", []):
                print(f"  - {issue}")
        
    except Exception as e:
        print(f"Hardware detection failed: {e}")


if __name__ == "__main__":
    main()
