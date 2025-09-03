"""
Test configuration and utilities for the WAN2.2 local installation test suite.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

# Test configuration
TEST_CONFIG = {
    "timeout": {
        "unit_test": 30,      # 30 seconds per unit test
        "integration_test": 120,  # 2 minutes per integration test
        "hardware_test": 60,   # 1 minute per hardware simulation test
        "total_suite": 1800    # 30 minutes total timeout
    },
    "hardware_profiles": {
        "test_high_end": True,
        "test_mid_range": True,
        "test_budget": True,
        "test_minimum": True,
        "test_no_gpu": True,
        "test_legacy": False  # Disabled by default (may fail minimum requirements)
    },
    "mock_downloads": True,    # Mock model downloads in tests
    "mock_subprocess": True,   # Mock subprocess calls
    "create_temp_files": True, # Create temporary files for testing
    "cleanup_after_tests": True,
    "verbose_output": False,
    "save_test_artifacts": True
}


@dataclass
class TestEnvironment:
    """Test environment configuration."""
    temp_dir: str
    mock_downloads: bool
    mock_subprocess: bool
    verbose: bool
    
    def __post_init__(self):
        """Initialize test environment."""
        self.temp_path = Path(self.temp_dir)
        self.logs_dir = self.temp_path / "logs"
        self.models_dir = self.temp_path / "models"
        self.scripts_dir = self.temp_path / "scripts"
        
        # Create directories
        for directory in [self.logs_dir, self.models_dir, self.scripts_dir]:
            directory.mkdir(parents=True, exist_ok=True)


class TestDataGenerator:
    """Generates test data and mock structures."""
    
    @staticmethod
    def create_mock_model_structure(models_dir: Path) -> None:
        """Create mock model directory structure."""
        models = [
            {
                "name": "WAN2.2-T2V-A14B",
                "size_mb": 28000,  # 28GB
                "files": ["pytorch_model.bin", "config.json", "tokenizer.json"]
            },
            {
                "name": "WAN2.2-I2V-A14B", 
                "size_mb": 28000,  # 28GB
                "files": ["pytorch_model.bin", "config.json", "tokenizer.json"]
            },
            {
                "name": "WAN2.2-TI2V-5B",
                "size_mb": 10000,  # 10GB
                "files": ["pytorch_model.bin", "config.json", "tokenizer.json"]
            }
        ]
        
        for model in models:
            model_path = models_dir / model["name"]
            model_path.mkdir(parents=True, exist_ok=True)
            
            for file_name in model["files"]:
                file_path = model_path / file_name
                
                if file_name == "pytorch_model.bin":
                    # Create large file (but not actually the full size for testing)
                    file_path.write_bytes(b"0" * (1024 * 1024 * 10))  # 10MB instead of full size
                elif file_name == "config.json":
                    config_data = {
                        "model_type": "wan22",
                        "num_parameters": "14B" if "A14B" in model["name"] else "5B",
                        "architecture": "transformer"
                    }
                    file_path.write_text(json.dumps(config_data, indent=2))
                elif file_name == "tokenizer.json":
                    tokenizer_data = {
                        "version": "1.0",
                        "tokenizer": {
                            "vocab_size": 50000,
                            "model_type": "BPE"
                        }
                    }
                    file_path.write_text(json.dumps(tokenizer_data, indent=2))
    
    @staticmethod
    def create_mock_venv_structure(venv_dir: Path) -> None:
        """Create mock virtual environment structure."""
        scripts_dir = venv_dir / "Scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock executables
        (scripts_dir / "python.exe").touch()
        (scripts_dir / "pip.exe").touch()
        (scripts_dir / "activate.bat").write_text("@echo Virtual environment activated")
        
        # Create mock site-packages
        site_packages = venv_dir / "Lib" / "site-packages"
        site_packages.mkdir(parents=True, exist_ok=True)
        
        # Mock installed packages
        packages = [
            "torch", "torchvision", "torchaudio", "transformers", 
            "numpy", "pillow", "requests", "tqdm", "psutil"
        ]
        
        for package in packages:
            package_dir = site_packages / package
            package_dir.mkdir(exist_ok=True)
            (package_dir / "__init__.py").touch()
            
            # Create mock package info
            dist_info = site_packages / f"{package}-2.0.0.dist-info"
            dist_info.mkdir(exist_ok=True)
            (dist_info / "METADATA").write_text(f"Name: {package}\nVersion: 2.0.0")
    
    @staticmethod
    def create_mock_requirements_file(resources_dir: Path) -> None:
        """Create mock requirements.txt file."""
        requirements = [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "torchaudio>=2.0.0",
            "transformers>=4.30.0",
            "numpy>=1.24.0",
            "pillow>=9.5.0",
            "requests>=2.31.0",
            "tqdm>=4.65.0",
            "psutil>=5.9.0",
            "huggingface-hub>=0.16.0"
        ]
        
        req_file = resources_dir / "requirements.txt"
        req_file.write_text("\n".join(requirements))
    
    @staticmethod
    def create_mock_config_files(temp_dir: Path) -> None:
        """Create mock configuration files."""
        # Default configuration
        default_config = {
            "system": {
                "enable_gpu_acceleration": True,
                "default_quantization": "fp16",
                "enable_model_offload": True,
                "vae_tile_size": 256,
                "max_queue_size": 10,
                "worker_threads": 8
            },
            "optimization": {
                "cpu_threads": 8,
                "memory_pool_gb": 8,
                "max_vram_usage_gb": 6
            },
            "paths": {
                "models_dir": "models",
                "output_dir": "outputs",
                "logs_dir": "logs"
            }
        }
        
        config_file = temp_dir / "config.json"
        config_file.write_text(json.dumps(default_config, indent=2))


class TestReportGenerator:
    """Generates test reports and summaries."""
    
    @staticmethod
    def generate_test_summary(results: Dict[str, Any], output_file: Path) -> None:
        """Generate a comprehensive test summary report."""
        summary = {
            "test_execution": {
                "timestamp": results.get("timestamp"),
                "total_duration": results.get("total_duration", 0),
                "environment": {
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "test_config": TEST_CONFIG
                }
            },
            "results_summary": results.get("overall_stats", {}),
            "suite_breakdown": results.get("suite_results", {}),
            "recommendations": results.get("summary", {}).get("recommendations", []),
            "critical_failures": results.get("summary", {}).get("critical_failures", [])
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def generate_html_report(results: Dict[str, Any], output_file: Path) -> None:
        """Generate an HTML test report."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>WAN2.2 Installation Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .summary { margin: 20px 0; }
        .suite { margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }
        .suite-header { background-color: #e9e9e9; padding: 10px; font-weight: bold; }
        .test-result { padding: 10px; border-bottom: 1px solid #eee; }
        .passed { color: green; }
        .failed { color: red; }
        .stats { display: flex; gap: 20px; }
        .stat-box { padding: 10px; border: 1px solid #ddd; border-radius: 3px; text-align: center; }
    </style>
</head>
<body>
    <div class="header">
        <h1>WAN2.2 Local Installation Test Report</h1>
        <p>Generated: {timestamp}</p>
        <p>Duration: {duration:.2f} seconds</p>
    </div>
    
    <div class="summary">
        <h2>Test Summary</h2>
        <div class="stats">
            <div class="stat-box">
                <h3>{total_tests}</h3>
                <p>Total Tests</p>
            </div>
            <div class="stat-box">
                <h3 class="passed">{total_passed}</h3>
                <p>Passed</p>
            </div>
            <div class="stat-box">
                <h3 class="failed">{total_failed}</h3>
                <p>Failed</p>
            </div>
            <div class="stat-box">
                <h3>{success_rate:.1%}</h3>
                <p>Success Rate</p>
            </div>
        </div>
    </div>
    
    <div class="suites">
        <h2>Test Suite Results</h2>
        {suite_results}
    </div>
</body>
</html>
        """
        
        # Generate suite results HTML
        suite_html = ""
        for suite_name, suite_data in results.get("suite_results", {}).items():
            suite_html += f"""
            <div class="suite">
                <div class="suite-header">
                    {suite_name.replace('_', ' ').title()} 
                    ({suite_data['passed']}/{suite_data['total']} passed)
                </div>
            """
            
            for test_result in suite_data.get("results", []):
                status_class = "passed" if test_result.success else "failed"
                status_text = "PASSED" if test_result.success else "FAILED"
                error_text = f" - {test_result.error_message}" if test_result.error_message else ""
                
                suite_html += f"""
                <div class="test-result">
                    <span class="{status_class}">{status_text}</span>
                    {test_result.test_name} ({test_result.duration:.2f}s){error_text}
                </div>
                """
            
            suite_html += "</div>"
        
        # Fill template
        overall_stats = results.get("overall_stats", {})
        html_content = html_template.format(
            timestamp=results.get("timestamp", "Unknown"),
            duration=results.get("total_duration", 0),
            total_tests=overall_stats.get("total_tests", 0),
            total_passed=overall_stats.get("total_passed", 0),
            total_failed=overall_stats.get("total_failed", 0),
            success_rate=overall_stats.get("success_rate", 0),
            suite_results=suite_html
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)


class MockHelpers:
    """Helper functions for mocking in tests."""
    
    @staticmethod
    def mock_subprocess_run(command: List[str], **kwargs):
        """Mock subprocess.run calls."""
        if "python" in command[0] and "-c" in command:
            # Mock Python version check
            if "import sys; print(sys.version)" in command[2]:
                return type('MockResult', (), {
                    'returncode': 0,
                    'stdout': '3.9.7 (default, Sep 16 2021, 16:59:28)',
                    'stderr': ''
                })()
            
            # Mock package version check
            if "import" in command[2]:
                return type('MockResult', (), {
                    'returncode': 0,
                    'stdout': '2.0.0',
                    'stderr': ''
                })()
        
        elif "nvidia-smi" in command[0]:
            # Mock nvidia-smi output
            return type('MockResult', (), {
                'returncode': 0,
                'stdout': 'GeForce RTX 3070, 8192 MiB, 537.13',
                'stderr': ''
            })()
        
        # Default mock response
        return type('MockResult', (), {
            'returncode': 0,
            'stdout': '',
            'stderr': ''
        })()
    
    @staticmethod
    def mock_requests_get(url: str, **kwargs):
        """Mock requests.get calls."""
        # Mock model download
        mock_response = type('MockResponse', (), {
            'status_code': 200,
            'headers': {'content-length': '1048576'},  # 1MB
            'iter_content': lambda chunk_size: [b'0' * chunk_size] * 10
        })()
        
        return mock_response


def setup_test_environment(config: Dict[str, Any] = None) -> TestEnvironment:
    """Set up test environment with configuration."""
    if config is None:
        config = TEST_CONFIG
    
    temp_dir = tempfile.mkdtemp(prefix="wan22_test_")
    
    env = TestEnvironment(
        temp_dir=temp_dir,
        mock_downloads=config.get("mock_downloads", True),
        mock_subprocess=config.get("mock_subprocess", True),
        verbose=config.get("verbose_output", False)
    )
    
    # Create test data
    if config.get("create_temp_files", True):
        TestDataGenerator.create_mock_model_structure(env.models_dir)
        TestDataGenerator.create_mock_venv_structure(env.temp_path / "venv")
        
        resources_dir = env.temp_path / "resources"
        resources_dir.mkdir(exist_ok=True)
        TestDataGenerator.create_mock_requirements_file(resources_dir)
        TestDataGenerator.create_mock_config_files(env.temp_path)
    
    return env


def cleanup_test_environment(env: TestEnvironment) -> None:
    """Clean up test environment."""
    if TEST_CONFIG.get("cleanup_after_tests", True):
        import shutil
shutil.rmtree(env.temp_dir, ignore_errors=True)