"""
Sample Manager for Local Testing Framework

Generates test data, configuration templates, and edge cases
for comprehensive testing of the Wan2.2 UI Variant system.
"""

import json
import os
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import uuid

from .models.test_results import ValidationResult, ValidationStatus
from .models.configuration import LocalTestConfiguration


class SampleManager:
    """
    Main orchestrator for test data generation including realistic video prompts,
    configuration templates, and edge case scenarios.
    """
    
    def __init__(self, config: Optional[LocalTestConfiguration] = None):
        self.config = config or LocalTestConfiguration()
        self.output_dir = Path("test_samples")
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_realistic_video_prompts(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Generate realistic video prompts for testing
        """
        # Base prompt categories with variations
        prompt_templates = {
            "nature": [
                "A serene mountain landscape with flowing rivers and lush forests",
                "Ocean waves crashing against rocky cliffs during sunset",
                "A peaceful meadow with wildflowers swaying in the breeze",
                "Dense forest with sunlight filtering through tall trees",
                "Desert landscape with sand dunes and cacti under starry sky"
            ],
            "urban": [
                "Bustling city street with people walking and cars passing",
                "Modern skyscrapers reflecting sunlight in downtown area",
                "Quiet suburban neighborhood with tree-lined streets",
                "Industrial district with warehouses and factory buildings",
                "Historic town square with cobblestone streets and old buildings"
            ],
            "abstract": [
                "Colorful geometric shapes morphing and transforming",
                "Flowing liquid patterns with metallic reflections",
                "Particle systems creating dynamic light displays",
                "Abstract art coming to life with vibrant colors",
                "Minimalist design elements in motion"
            ],
            "fantasy": [
                "Magical forest with glowing mushrooms and fairy lights",
                "Ancient castle on a hilltop surrounded by mist",
                "Dragon flying over medieval village at twilight",
                "Enchanted garden with floating islands and waterfalls",
                "Mystical portal opening in a dark cave"
            ],
            "sci-fi": [
                "Futuristic city with flying cars and neon lights",
                "Space station orbiting a distant planet",
                "Robot walking through abandoned industrial complex",
                "Alien landscape with strange rock formations",
                "Cyberpunk street scene with holographic advertisements"
            ]
        }
        
        # Style modifiers
        style_modifiers = [
            "cinematic lighting",
            "high detail",
            "photorealistic",
            "artistic style",
            "dramatic atmosphere",
            "soft lighting",
            "vibrant colors",
            "moody atmosphere",
            "epic scale",
            "intimate perspective"
        ]
        
        # Technical specifications
        camera_angles = [
            "wide shot",
            "close-up",
            "aerial view",
            "low angle",
            "high angle",
            "tracking shot",
            "static shot",
            "panning movement"
        ]
        
        prompts = []
        categories = list(prompt_templates.keys())
        
        for i in range(count):
            # Select random category and base prompt
            category = random.choice(categories)
            base_prompt = random.choice(prompt_templates[category])
            
            # Add style modifiers (1-3 modifiers)
            num_modifiers = random.randint(1, 3)
            selected_modifiers = random.sample(style_modifiers, num_modifiers)
            
            # Add camera angle
            camera_angle = random.choice(camera_angles)
            
            # Construct full prompt
            full_prompt = f"{base_prompt}, {camera_angle}, {', '.join(selected_modifiers)}"
            
            prompt_data = {
                "id": f"prompt_{i+1:03d}",
                "category": category,
                "base_prompt": base_prompt,
                "full_prompt": full_prompt,
                "style_modifiers": selected_modifiers,
                "camera_angle": camera_angle,
                "estimated_complexity": self._estimate_prompt_complexity(full_prompt),
                "recommended_resolution": self._recommend_resolution(category),
                "expected_duration_range": self._estimate_generation_time(category, full_prompt)
            }
            
            prompts.append(prompt_data)
        
        return prompts
    
    def create_sample_input_json(self, prompt_data: Dict[str, Any], 
                                resolution: str = "720p",
                                output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create sample_input.json with required fields: input, resolution, output_path
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_video_{prompt_data.get('id', 'unknown')}_{resolution}_{timestamp}.mp4"
            output_path = f"outputs/{filename}"
        
        sample_input = {
            "input": prompt_data["full_prompt"],
            "resolution": resolution,
            "output_path": output_path,
            "metadata": {
                "prompt_id": prompt_data.get("id"),
                "category": prompt_data.get("category"),
                "complexity": prompt_data.get("estimated_complexity"),
                "generated_at": datetime.now().isoformat(),
                "test_session": str(uuid.uuid4())
            }
        }
        
        return sample_input
    
    def generate_sample_input_files(self, count: int = 5, 
                                   resolutions: List[str] = None) -> List[Path]:
        """
        Generate multiple sample_input.json files for testing
        """
        if resolutions is None:
            resolutions = ["720p", "1080p"]
        
        # Generate prompts
        prompts = self.generate_realistic_video_prompts(count)
        generated_files = []
        
        for prompt in prompts:
            for resolution in resolutions:
                # Create sample input
                sample_input = self.create_sample_input_json(prompt, resolution)
                
                # Save to file
                filename = f"sample_input_{prompt['id']}_{resolution}.json"
                file_path = self.output_dir / filename
                
                with open(file_path, 'w') as f:
                    json.dump(sample_input, f, indent=2)
                
                generated_files.append(file_path)
        
        return generated_files
    
    def _estimate_prompt_complexity(self, prompt: str) -> str:
        """
        Estimate prompt complexity based on content analysis
        """
        complexity_indicators = {
            "high": ["detailed", "complex", "intricate", "elaborate", "multiple", "dynamic"],
            "medium": ["moderate", "balanced", "standard", "typical", "normal"],
            "low": ["simple", "basic", "minimal", "clean", "straightforward"]
        }
        
        prompt_lower = prompt.lower()
        word_count = len(prompt.split())
        
        # Check for complexity indicators
        high_score = sum(1 for word in complexity_indicators["high"] if word in prompt_lower)
        low_score = sum(1 for word in complexity_indicators["low"] if word in prompt_lower)
        
        # Determine complexity
        if high_score > 0 or word_count > 20:
            return "high"
        elif low_score > 0 or word_count < 10:
            return "low"
        else:
            return "medium"
    
    def _recommend_resolution(self, category: str) -> str:
        """
        Recommend resolution based on content category
        """
        resolution_mapping = {
            "nature": "1080p",  # Nature benefits from high resolution
            "urban": "1080p",   # Urban scenes have lots of detail
            "abstract": "720p", # Abstract can work well at lower res
            "fantasy": "1080p", # Fantasy benefits from detail
            "sci-fi": "1080p"   # Sci-fi often has complex details
        }
        
        return resolution_mapping.get(category, "720p")
    
    def _estimate_generation_time(self, category: str, prompt: str) -> Dict[str, float]:
        """
        Estimate generation time range based on category and complexity
        """
        base_times = {
            "nature": {"720p": 6.0, "1080p": 12.0},
            "urban": {"720p": 8.0, "1080p": 15.0},
            "abstract": {"720p": 5.0, "1080p": 10.0},
            "fantasy": {"720p": 7.0, "1080p": 14.0},
            "sci-fi": {"720p": 8.0, "1080p": 16.0}
        }
        
        complexity = self._estimate_prompt_complexity(prompt)
        complexity_multiplier = {
            "low": 0.8,
            "medium": 1.0,
            "high": 1.3
        }
        
        base = base_times.get(category, {"720p": 7.0, "1080p": 14.0})
        multiplier = complexity_multiplier[complexity]
        
        return {
            "720p_minutes": base["720p"] * multiplier,
            "1080p_minutes": base["1080p"] * multiplier,
            "complexity_factor": multiplier
        }
    
    # Delegation methods to other generators
    def generate_config_json_template(self, optimization_level: str = "balanced") -> Dict[str, Any]:
        """Generate config.json template"""
        generator = ConfigurationTemplateGenerator()
        return generator.generate_config_json_template(optimization_level)
    
    def generate_env_template(self, include_examples: bool = True) -> Dict[str, str]:
        """Generate .env template"""
        generator = ConfigurationTemplateGenerator()
        return generator.generate_env_template(include_examples)
    
    def save_config_template(self, config: Dict[str, Any], filename: str = "config_template.json") -> Path:
        """Save config template"""
        generator = ConfigurationTemplateGenerator()
        return generator.save_config_template(config, filename)
    
    def save_env_template(self, env_vars: Dict[str, str], filename: str = "env_template.txt") -> Path:
        """Save env template"""
        generator = ConfigurationTemplateGenerator()
        return generator.save_env_template(env_vars, filename)
    
    def generate_edge_case_prompts(self) -> List[Dict[str, Any]]:
        """Generate edge case prompts"""
        generator = EdgeCaseGenerator()
        return generator.generate_edge_case_prompts()
    
    def create_invalid_input_samples(self) -> List[Dict[str, Any]]:
        """Create invalid input samples"""
        generator = EdgeCaseGenerator()
        return generator.create_invalid_input_samples()


class ConfigurationTemplateGenerator:
    """
    Generates configuration templates for config.json and .env files
    """
    
    def __init__(self):
        self.template_dir = Path("config_templates")
        self.template_dir.mkdir(exist_ok=True)
    
    def generate_config_json_template(self, optimization_level: str = "balanced") -> Dict[str, Any]:
        """
        Generate config.json template with all required sections
        """
        optimization_presets = {
            "performance": {
                "enable_attention_slicing": True,
                "enable_vae_tiling": True,
                "use_fp16": True,
                "torch_compile": True,
                "enable_xformers": True,
                "memory_efficient_attention": True,
                "use_int8": False
            },
            "balanced": {
                "enable_attention_slicing": True,
                "enable_vae_tiling": True,
                "use_fp16": True,
                "torch_compile": False,
                "enable_xformers": True,
                "memory_efficient_attention": True,
                "use_int8": False
            },
            "quality": {
                "enable_attention_slicing": False,
                "enable_vae_tiling": False,
                "use_fp16": False,
                "torch_compile": False,
                "enable_xformers": False,
                "memory_efficient_attention": False,
                "use_int8": False
            }
        }
        
        config_template = {
            "system": {
                "device": "cuda",
                "gpu_memory_fraction": 0.9,
                "allow_tf32": True,
                "cudnn_benchmark": True,
                "empty_cache_frequency": 5,
                "max_workers": 1,
                "log_level": "INFO"
            },
            "directories": {
                "models": "models/",
                "outputs": "outputs/",
                "cache": "cache/",
                "logs": "logs/",
                "temp": "temp/"
            },
            "optimization": optimization_presets.get(optimization_level, optimization_presets["balanced"]),
            "performance": {
                "batch_size": 1,
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "scheduler": "DPMSolverMultistepScheduler",
                "safety_checker": True,
                "nsfw_filter": True,
                "stats_refresh_interval": 5,
                "vram_warning_threshold": 0.9,
                "cpu_warning_percent": 80
            },
            "model": {
                "base_model": "runwayml/stable-diffusion-v1-5",
                "vae_model": "stabilityai/sd-vae-ft-mse",
                "safety_model": "CompVis/stable-diffusion-safety-checker",
                "feature_extractor": "openai/clip-vit-base-patch32",
                "cache_models": True,
                "offload_to_cpu": False
            },
            "ui": {
                "port": 7860,
                "host": "127.0.0.1",
                "share": False,
                "auth": False,
                "theme": "default",
                "queue_max_size": 10,
                "show_api": True
            },
            "logging": {
                "level": "INFO",
                "file": "wan22_ui.log",
                "error_file": "wan22_errors.log",
                "max_size_mb": 100,
                "backup_count": 5,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        return config_template
    
    def generate_env_template(self, include_examples: bool = True) -> Dict[str, str]:
        """
        Generate .env template with HF_TOKEN, CUDA_VISIBLE_DEVICES, PYTORCH_CUDA_ALLOC_CONF
        """
        env_template = {}
        
        if include_examples:
            env_template.update({
                "# Hugging Face Token for model downloads": "",
                "HF_TOKEN": "hf_your_token_here",
                "_blank1": "",
                "# CUDA Configuration": "",
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
                "_blank2": "",
                "# Optional: Additional CUDA settings": "",
                "CUDA_LAUNCH_BLOCKING": "0",
                "TORCH_CUDNN_V8_API_ENABLED": "1",
                "_blank3": "",
                "# Optional: Performance tuning": "",
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "_blank4": "",
                "# Optional: Logging": "",
                "PYTHONUNBUFFERED": "1",
                "LOG_LEVEL": "INFO"
            })
        else:
            env_template.update({
                "HF_TOKEN": "",
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"
            })
        
        return env_template
    
    def save_config_template(self, config: Dict[str, Any], 
                           filename: str = "config_template.json") -> Path:
        """
        Save configuration template to file
        """
        file_path = self.template_dir / filename
        
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return file_path
    
    def save_env_template(self, env_vars: Dict[str, str], 
                         filename: str = "env_template.txt") -> Path:
        """
        Save environment template to file
        """
        file_path = self.template_dir / filename
        
        with open(file_path, 'w') as f:
            for key, value in env_vars.items():
                if key.startswith("#"):
                    f.write(f"{key}\n")
                elif key.startswith("_blank"):
                    f.write("\n")  # Write blank line
                else:
                    f.write(f"{key}={value}\n")
        
        return file_path
    
    def validate_config_template(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate configuration template using basic schema validation
        """
        required_sections = ["system", "directories", "optimization", "performance"]
        missing_sections = []
        
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
        
        if missing_sections:
            return ValidationResult(
                component="config_template",
                status=ValidationStatus.FAILED,
                message=f"Missing required sections: {', '.join(missing_sections)}",
                details={
                    "missing_sections": missing_sections,
                    "present_sections": list(config.keys())
                },
                remediation_steps=[
                    f"Add missing sections: {', '.join(missing_sections)}",
                    "Use generate_config_json_template() to create complete template",
                    "Refer to documentation for section specifications"
                ]
            )
        
        # Validate specific fields
        validation_issues = []
        
        # Check system section
        if "system" in config:
            system = config["system"]
            if "device" not in system:
                validation_issues.append("system.device is required")
            if "gpu_memory_fraction" in system:
                fraction = system["gpu_memory_fraction"]
                if not (0.1 <= fraction <= 1.0):
                    validation_issues.append("system.gpu_memory_fraction must be between 0.1 and 1.0")
        
        # Check directories section
        if "directories" in config:
            directories = config["directories"]
            required_dirs = ["models", "outputs", "cache", "logs"]
            for dir_key in required_dirs:
                if dir_key not in directories:
                    validation_issues.append(f"directories.{dir_key} is required")
        
        if validation_issues:
            return ValidationResult(
                component="config_template",
                status=ValidationStatus.WARNING,
                message=f"Configuration validation issues: {len(validation_issues)}",
                details={
                    "issues": validation_issues,
                    "sections": list(config.keys())
                },
                remediation_steps=[
                    "Fix configuration validation issues",
                    "Review field specifications",
                    "Use recommended default values"
                ]
            )
        
        return ValidationResult(
            component="config_template",
            status=ValidationStatus.PASSED,
            message=f"Configuration template valid with {len(required_sections)} required sections",
            details={
                "sections": list(config.keys()),
                "validation_passed": True
            }
        )


class EdgeCaseGenerator:
    """
    Generates edge cases and stress test data for robustness testing
    """
    
    def __init__(self):
        self.edge_case_dir = Path("edge_case_samples")
        self.edge_case_dir.mkdir(exist_ok=True)
    
    def generate_edge_case_prompts(self) -> List[Dict[str, Any]]:
        """
        Generate edge case prompts for error testing
        """
        edge_cases = []
        
        # 1. Empty and minimal prompts
        edge_cases.extend([
            {
                "id": "edge_empty",
                "category": "empty",
                "prompt": "",
                "expected_behavior": "should handle gracefully or provide default",
                "test_type": "empty_input"
            },
            {
                "id": "edge_single_word",
                "category": "minimal",
                "prompt": "cat",
                "expected_behavior": "should generate basic content",
                "test_type": "minimal_input"
            },
            {
                "id": "edge_whitespace",
                "category": "whitespace",
                "prompt": "   \n\t   ",
                "expected_behavior": "should handle whitespace-only input",
                "test_type": "whitespace_input"
            }
        ])
        
        # 2. Extremely long prompts
        long_prompt = "A very detailed and extremely long prompt that goes on and on describing every possible aspect of the scene including " + \
                     "the lighting conditions, weather patterns, atmospheric effects, character descriptions, background elements, " * 20
        
        edge_cases.append({
            "id": "edge_very_long",
            "category": "length",
            "prompt": long_prompt,
            "expected_behavior": "should truncate or handle long input gracefully",
            "test_type": "long_input",
            "length": len(long_prompt)
        })
        
        # 3. Special characters and Unicode
        edge_cases.extend([
            {
                "id": "edge_special_chars",
                "category": "special_characters",
                "prompt": "A scene with symbols: !@#$%^&*()_+-=[]{}|;':\",./<>?",
                "expected_behavior": "should handle special characters",
                "test_type": "special_characters"
            },
            {
                "id": "edge_unicode",
                "category": "unicode",
                "prompt": "A beautiful scene with émojis 🌟🎨🎭 and ünïcödé characters: café, naïve, résumé",
                "expected_behavior": "should handle Unicode characters",
                "test_type": "unicode_input"
            },
            {
                "id": "edge_mixed_languages",
                "category": "multilingual",
                "prompt": "A scene mixing languages: Hello, 你好, Bonjour, Hola, こんにちは, Привет",
                "expected_behavior": "should handle multiple languages",
                "test_type": "multilingual_input"
            }
        ])
        
        # 4. Potentially problematic content
        edge_cases.extend([
            {
                "id": "edge_numbers_only",
                "category": "numeric",
                "prompt": "123456789 0.5 -42 3.14159",
                "expected_behavior": "should handle numeric-only input",
                "test_type": "numeric_input"
            },
            {
                "id": "edge_repeated_words",
                "category": "repetitive",
                "prompt": "red red red red red red red red red red",
                "expected_behavior": "should handle repetitive input",
                "test_type": "repetitive_input"
            },
            {
                "id": "edge_contradictory",
                "category": "contradictory",
                "prompt": "A bright dark scene with loud silence and moving stillness",
                "expected_behavior": "should handle contradictory descriptions",
                "test_type": "contradictory_input"
            }
        ])
        
        # 5. Technical/System prompts
        edge_cases.extend([
            {
                "id": "edge_code_injection",
                "category": "security",
                "prompt": "A scene with <script>alert('test')</script> and ${system.exit()}",
                "expected_behavior": "should sanitize potential code injection",
                "test_type": "security_test"
            },
            {
                "id": "edge_path_traversal",
                "category": "security",
                "prompt": "A scene in ../../../etc/passwd directory",
                "expected_behavior": "should handle path-like strings safely",
                "test_type": "path_traversal_test"
            },
            {
                "id": "edge_sql_like",
                "category": "security",
                "prompt": "A scene with 1'; DROP TABLE users; --",
                "expected_behavior": "should handle SQL-like strings safely",
                "test_type": "sql_injection_test"
            }
        ])
        
        return edge_cases
    
    def create_invalid_input_samples(self) -> List[Dict[str, Any]]:
        """
        Create invalid input samples for robustness testing
        """
        invalid_samples = []
        
        # 1. Malformed JSON structures
        invalid_samples.extend([
            {
                "id": "invalid_json_missing_quotes",
                "content": '{input: A test scene, resolution: 720p}',
                "error_type": "json_syntax_error",
                "expected_behavior": "should return JSON parsing error"
            },
            {
                "id": "invalid_json_trailing_comma",
                "content": '{"input": "A test scene", "resolution": "720p",}',
                "error_type": "json_syntax_error",
                "expected_behavior": "should handle trailing comma gracefully"
            },
            {
                "id": "invalid_json_missing_bracket",
                "content": '{"input": "A test scene", "resolution": "720p"',
                "error_type": "json_syntax_error",
                "expected_behavior": "should return JSON parsing error"
            }
        ])
        
        # 2. Missing required fields
        invalid_samples.extend([
            {
                "id": "invalid_missing_input",
                "content": '{"resolution": "720p", "output_path": "test.mp4"}',
                "error_type": "missing_required_field",
                "expected_behavior": "should return validation error for missing input"
            },
            {
                "id": "invalid_missing_resolution",
                "content": '{"input": "A test scene", "output_path": "test.mp4"}',
                "error_type": "missing_required_field",
                "expected_behavior": "should return validation error for missing resolution"
            },
            {
                "id": "invalid_missing_output_path",
                "content": '{"input": "A test scene", "resolution": "720p"}',
                "error_type": "missing_required_field",
                "expected_behavior": "should return validation error for missing output_path"
            }
        ])
        
        # 3. Invalid field values
        invalid_samples.extend([
            {
                "id": "invalid_resolution",
                "content": '{"input": "A test scene", "resolution": "999p", "output_path": "test.mp4"}',
                "error_type": "invalid_field_value",
                "expected_behavior": "should return validation error for invalid resolution"
            },
            {
                "id": "invalid_output_path",
                "content": '{"input": "A test scene", "resolution": "720p", "output_path": "/invalid/path/test.mp4"}',
                "error_type": "invalid_field_value",
                "expected_behavior": "should handle invalid output path gracefully"
            },
            {
                "id": "invalid_data_types",
                "content": '{"input": 123, "resolution": true, "output_path": ["test.mp4"]}',
                "error_type": "invalid_data_type",
                "expected_behavior": "should return type validation errors"
            }
        ])
        
        # 4. Oversized data
        large_input = "A" * 10000  # 10KB string
        invalid_samples.append({
            "id": "invalid_oversized_input",
            "content": f'{{"input": "{large_input}", "resolution": "720p", "output_path": "test.mp4"}}',
            "error_type": "oversized_input",
            "expected_behavior": "should handle or reject oversized input",
            "size_bytes": len(large_input)
        })
        
        return invalid_samples
    
    def create_multi_resolution_test_suite(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create comprehensive test suite for 720p and 1080p resolutions
        """
        test_suite = {
            "720p": [],
            "1080p": [],
            "unsupported": []
        }
        
        # Base test scenarios
        base_scenarios = [
            {
                "name": "simple_scene",
                "prompt": "A simple landscape with mountains and trees",
                "complexity": "low",
                "expected_duration_720p": 6.0,
                "expected_duration_1080p": 12.0
            },
            {
                "name": "complex_scene",
                "prompt": "A bustling medieval marketplace with detailed architecture, multiple characters, dynamic lighting, and atmospheric effects",
                "complexity": "high",
                "expected_duration_720p": 10.0,
                "expected_duration_1080p": 20.0
            },
            {
                "name": "motion_heavy",
                "prompt": "Fast-moving cars racing through city streets with motion blur and dynamic camera angles",
                "complexity": "high",
                "expected_duration_720p": 12.0,
                "expected_duration_1080p": 24.0
            },
            {
                "name": "particle_effects",
                "prompt": "Magical spell casting with glowing particles, energy beams, and light effects",
                "complexity": "high",
                "expected_duration_720p": 11.0,
                "expected_duration_1080p": 22.0
            }
        ]
        
        # Generate test cases for each resolution
        for scenario in base_scenarios:
            # 720p test case
            test_suite["720p"].append({
                "id": f"720p_{scenario['name']}",
                "input": scenario["prompt"],
                "resolution": "720p",
                "output_path": f"outputs/test_720p_{scenario['name']}.mp4",
                "expected_duration_minutes": scenario["expected_duration_720p"],
                "complexity": scenario["complexity"],
                "test_type": "resolution_performance"
            })
            
            # 1080p test case
            test_suite["1080p"].append({
                "id": f"1080p_{scenario['name']}",
                "input": scenario["prompt"],
                "resolution": "1080p",
                "output_path": f"outputs/test_1080p_{scenario['name']}.mp4",
                "expected_duration_minutes": scenario["expected_duration_1080p"],
                "complexity": scenario["complexity"],
                "test_type": "resolution_performance"
            })
        
        # Unsupported resolutions for error testing
        unsupported_resolutions = ["480p", "4K", "8K", "invalid", "720x480", "1920x1080"]
        
        for res in unsupported_resolutions:
            test_suite["unsupported"].append({
                "id": f"unsupported_{res}",
                "input": "A test scene for unsupported resolution",
                "resolution": res,
                "output_path": f"outputs/test_{res}.mp4",
                "expected_behavior": "should return resolution not supported error",
                "test_type": "unsupported_resolution"
            })
        
        return test_suite
    
    def create_stress_test_scenarios(self) -> List[Dict[str, Any]]:
        """
        Create stress test scenarios for system limits testing
        """
        stress_tests = []
        
        # 1. Concurrent generation simulation
        stress_tests.append({
            "id": "stress_concurrent_requests",
            "type": "concurrent_load",
            "description": "Simulate 100+ concurrent generation requests",
            "parameters": {
                "concurrent_requests": 100,
                "request_interval_seconds": 0.1,
                "test_duration_minutes": 10
            },
            "expected_behavior": "should handle queue management gracefully",
            "success_criteria": [
                "No system crashes",
                "Proper queue management",
                "Resource usage within limits",
                "All requests processed or properly rejected"
            ]
        })
        
        # 2. Memory exhaustion test
        stress_tests.append({
            "id": "stress_memory_exhaustion",
            "type": "memory_stress",
            "description": "Test system behavior under memory pressure",
            "parameters": {
                "large_batch_size": 10,
                "high_resolution": "1080p",
                "complex_prompts": True,
                "disable_optimizations": True
            },
            "expected_behavior": "should handle memory pressure gracefully",
            "success_criteria": [
                "Proper error handling for OOM conditions",
                "Graceful degradation",
                "System recovery after stress",
                "Memory cleanup after operations"
            ]
        })
        
        # 3. Long-running session test
        stress_tests.append({
            "id": "stress_long_session",
            "type": "endurance",
            "description": "Test system stability over extended periods",
            "parameters": {
                "session_duration_hours": 4,
                "requests_per_hour": 20,
                "mixed_resolutions": True,
                "varied_complexity": True
            },
            "expected_behavior": "should maintain stability over long periods",
            "success_criteria": [
                "No memory leaks",
                "Consistent performance",
                "Proper resource cleanup",
                "No degradation over time"
            ]
        })
        
        # 4. Rapid request cycling
        stress_tests.append({
            "id": "stress_rapid_cycling",
            "type": "rapid_requests",
            "description": "Test rapid request submission and cancellation",
            "parameters": {
                "cycle_count": 1000,
                "request_cancel_ratio": 0.3,
                "rapid_fire_interval": 0.05
            },
            "expected_behavior": "should handle rapid request changes",
            "success_criteria": [
                "Proper request cancellation",
                "No resource leaks from cancelled requests",
                "Queue state consistency",
                "System responsiveness maintained"
            ]
        })
        
        return stress_tests
    
    def save_edge_case_samples(self) -> List[Path]:
        """
        Save all edge case samples to files
        """
        saved_files = []
        
        # Save edge case prompts
        edge_prompts = self.generate_edge_case_prompts()
        for prompt in edge_prompts:
            filename = f"edge_case_{prompt['id']}.json"
            file_path = self.edge_case_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(prompt, f, indent=2, ensure_ascii=False)
            
            saved_files.append(file_path)
        
        # Save invalid input samples
        invalid_samples = self.create_invalid_input_samples()
        for sample in invalid_samples:
            filename = f"invalid_{sample['id']}.json"
            file_path = self.edge_case_dir / filename
            
            with open(file_path, 'w') as f:
                json.dump(sample, f, indent=2)
            
            saved_files.append(file_path)
        
        # Save multi-resolution test suite
        test_suite = self.create_multi_resolution_test_suite()
        for resolution, tests in test_suite.items():
            filename = f"resolution_tests_{resolution}.json"
            file_path = self.edge_case_dir / filename
            
            with open(file_path, 'w') as f:
                json.dump(tests, f, indent=2)
            
            saved_files.append(file_path)
        
        # Save stress test scenarios
        stress_tests = self.create_stress_test_scenarios()
        filename = "stress_test_scenarios.json"
        file_path = self.edge_case_dir / filename
        
        with open(file_path, 'w') as f:
            json.dump(stress_tests, f, indent=2)
        
        saved_files.append(file_path)
        
        return saved_files
