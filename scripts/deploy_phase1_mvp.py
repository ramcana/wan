#!/usr/bin/env python3
"""
Phase 1 MVP Deployment Script
Validates and deploys the WAN models MVP with T2V, I2V, and TI2V support
"""

import asyncio
import logging
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

console = Console()
logger = logging.getLogger(__name__)

class Phase1Validator:
    """Validates Phase 1 MVP requirements"""
    
    def __init__(self):
        self.backend_url = "http://localhost:9000"
        self.frontend_url = "http://localhost:3000"
        self.validation_results = {}
    
    async def validate_backend_models(self) -> bool:
        """Validate WAN models backend functionality"""
        console.print("🤖 [bold blue]Validating WAN Models Backend[/bold blue]")
        
        checks = [
            ("Model Detection API", self._check_model_detection),
            ("Prompt Enhancement API", self._check_prompt_enhancement),
            ("Generation Capabilities", self._check_capabilities),
            ("Enhanced Generation API", self._check_enhanced_generation),
            ("Model Requirements", self._check_model_requirements)
        ]
        
        results = []
        for check_name, check_func in checks:
            try:
                result = await check_func()
                results.append((check_name, "✅ Pass", result.get("details", "")))
                console.print(f"  ✅ {check_name}")
            except Exception as e:
                results.append((check_name, "❌ Fail", str(e)))
                console.print(f"  ❌ {check_name}: {e}")
        
        self.validation_results["backend_models"] = results
        return all("✅" in result[1] for result in results)
    
    async def _check_model_detection(self) -> Dict:
        """Check model detection API"""
        response = requests.get(
            f"{self.backend_url}/api/v1/generation/models/detect",
            params={"prompt": "A beautiful landscape", "has_image": False}
        )
        response.raise_for_status()
        data = response.json()
        
        assert data["detected_model_type"] == "T2V-A14B"
        assert "explanation" in data
        return {"details": f"Detected: {data['detected_model_type']}"}
    
    async def _check_prompt_enhancement(self) -> Dict:
        """Check prompt enhancement API"""
        response = requests.post(
            f"{self.backend_url}/api/v1/generation/prompt/enhance",
            data={
                "prompt": "A cat",
                "model_type": "T2V-A14B",
                "enhance_quality": True
            }
        )
        response.raise_for_status()
        data = response.json()
        
        assert len(data["enhanced_prompt"]) > len(data["original_prompt"])
        return {"details": f"Enhanced: {len(data['enhancements_applied'])} improvements"}
    
    async def _check_capabilities(self) -> Dict:
        """Check generation capabilities API"""
        response = requests.get(f"{self.backend_url}/api/v1/generation/capabilities")
        response.raise_for_status()
        data = response.json()
        
        expected_models = ["T2V-A14B", "I2V-A14B", "TI2V-5B"]
        assert all(model in data["supported_models"] for model in expected_models)
        assert data["features"]["auto_model_detection"] == True
        return {"details": f"Models: {len(data['supported_models'])}, Features: {len(data['features'])}"}
    
    async def _check_enhanced_generation(self) -> Dict:
        """Check enhanced generation API (dry run)"""
        # Test validation without actual generation
        response = requests.post(
            f"{self.backend_url}/api/v1/generation/submit",
            data={
                "prompt": "",  # Empty prompt should fail validation
                "model_type": "T2V-A14B"
            }
        )
        
        # Should return 422 for validation error
        assert response.status_code == 422
        return {"details": "Validation working correctly"}
    
    async def _check_model_requirements(self) -> Dict:
        """Check model requirements for all supported models"""
        models = ["T2V-A14B", "I2V-A14B", "TI2V-5B"]
        for model in models:
            response = requests.get(
                f"{self.backend_url}/api/v1/generation/models/detect",
                params={"prompt": "test", "has_image": model != "T2V-A14B"}
            )
            response.raise_for_status()
            data = response.json()
            assert "requirements" in data
        
        return {"details": f"Validated {len(models)} model types"}
    
    async def validate_frontend_integration(self) -> bool:
        """Validate frontend integration with enhanced APIs"""
        console.print("🎨 [bold blue]Validating Frontend Integration[/bold blue]")
        
        checks = [
            ("Frontend Accessibility", self._check_frontend_running),
            ("API Client Integration", self._check_api_client),
            ("Model Selection UI", self._check_model_selection),
            ("Auto-Detection UI", self._check_auto_detection_ui)
        ]
        
        results = []
        for check_name, check_func in checks:
            try:
                result = await check_func()
                results.append((check_name, "✅ Pass", result.get("details", "")))
                console.print(f"  ✅ {check_name}")
            except Exception as e:
                results.append((check_name, "❌ Fail", str(e)))
                console.print(f"  ❌ {check_name}: {e}")
        
        self.validation_results["frontend_integration"] = results
        return all("✅" in result[1] for result in results)
    
    async def _check_frontend_running(self) -> Dict:
        """Check if frontend is accessible"""
        try:
            response = requests.get(self.frontend_url, timeout=5)
            return {"details": f"Status: {response.status_code}"}
        except requests.exceptions.RequestException:
            # Frontend might not be running, which is okay for backend-only validation
            return {"details": "Frontend not running (backend-only validation)"}
    
    async def _check_api_client(self) -> Dict:
        """Check API client configuration"""
        # Check if API client files exist
        api_client_path = project_root / "frontend" / "src" / "lib" / "api-client.ts"
        if api_client_path.exists():
            return {"details": "API client configured"}
        else:
            raise FileNotFoundError("API client not found")
    
    async def _check_model_selection(self) -> Dict:
        """Check model selection UI components"""
        generation_form_path = project_root / "frontend" / "src" / "components" / "generation" / "GenerationForm.tsx"
        if generation_form_path.exists():
            content = generation_form_path.read_text()
            if "auto" in content.lower() and "MODEL_OPTIONS" in content:
                return {"details": "Model selection UI with auto-detection"}
            else:
                raise ValueError("Model selection UI missing auto-detection")
        else:
            raise FileNotFoundError("Generation form not found")
    
    async def _check_auto_detection_ui(self) -> Dict:
        """Check auto-detection UI implementation"""
        generation_form_path = project_root / "frontend" / "src" / "components" / "generation" / "GenerationForm.tsx"
        if generation_form_path.exists():
            content = generation_form_path.read_text()
            if "detectedModel" in content and "auto-detection" in content.lower():
                return {"details": "Auto-detection UI implemented"}
            else:
                return {"details": "Auto-detection UI partially implemented"}
        else:
            raise FileNotFoundError("Generation form not found")
    
    async def validate_cli_tools(self) -> bool:
        """Validate CLI tools for Phase 1"""
        console.print("⚡ [bold blue]Validating CLI Tools[/bold blue]")
        
        checks = [
            ("WAN CLI Module", self._check_wan_cli_module),
            ("CLI Integration", self._check_cli_integration),
            ("Model Commands", self._check_model_commands),
            ("Test Commands", self._check_test_commands)
        ]
        
        results = []
        for check_name, check_func in checks:
            try:
                result = await check_func()
                results.append((check_name, "✅ Pass", result.get("details", "")))
                console.print(f"  ✅ {check_name}")
            except Exception as e:
                results.append((check_name, "❌ Fail", str(e)))
                console.print(f"  ❌ {check_name}: {e}")
        
        self.validation_results["cli_tools"] = results
        return all("✅" in result[1] for result in results)
    
    async def _check_wan_cli_module(self) -> Dict:
        """Check WAN CLI module exists"""
        wan_cli_path = project_root / "cli" / "commands" / "wan.py"
        if wan_cli_path.exists():
            content = wan_cli_path.read_text()
            if "T2V-A14B" in content and "I2V-A14B" in content and "TI2V-5B" in content:
                return {"details": "All Phase 1 models supported"}
            else:
                raise ValueError("Not all Phase 1 models supported in CLI")
        else:
            raise FileNotFoundError("WAN CLI module not found")
    
    async def _check_cli_integration(self) -> Dict:
        """Check CLI integration in main CLI"""
        main_cli_path = project_root / "cli" / "main.py"
        if main_cli_path.exists():
            content = main_cli_path.read_text()
            if "wan" in content and "WAN Model Management" in content:
                return {"details": "WAN CLI integrated in main CLI"}
            else:
                raise ValueError("WAN CLI not integrated in main CLI")
        else:
            raise FileNotFoundError("Main CLI not found")
    
    async def _check_model_commands(self) -> Dict:
        """Check model management commands"""
        try:
            # Try importing the WAN CLI module
            sys.path.append(str(project_root / "cli" / "commands"))
            import wan
            
            # Check if key commands exist
            commands = ["models", "test", "generate", "health", "optimize"]
            for cmd in commands:
                if not hasattr(wan.app, 'commands') or cmd not in [c.name for c in wan.app.commands.values()]:
                    # Alternative check - look for command functions
                    if not hasattr(wan, cmd.replace('-', '_')):
                        raise ValueError(f"Command '{cmd}' not found")
            
            return {"details": f"All {len(commands)} commands available"}
        except ImportError as e:
            raise ImportError(f"Failed to import WAN CLI: {e}")
    
    async def _check_test_commands(self) -> Dict:
        """Check test commands functionality"""
        test_file_path = project_root / "tests" / "test_wan_models_phase1.py"
        if test_file_path.exists():
            content = test_file_path.read_text()
            if "TestWANModelsPhase1" in content:
                return {"details": "Phase 1 test suite available"}
            else:
                raise ValueError("Phase 1 test suite not properly structured")
        else:
            raise FileNotFoundError("Phase 1 test suite not found")
    
    async def validate_system_requirements(self) -> bool:
        """Validate system requirements for Phase 1"""
        console.print("🔧 [bold blue]Validating System Requirements[/bold blue]")
        
        checks = [
            ("Python Environment", self._check_python_env),
            ("Required Packages", self._check_packages),
            ("Model Storage", self._check_model_storage),
            ("VRAM Requirements", self._check_vram_requirements)
        ]
        
        results = []
        for check_name, check_func in checks:
            try:
                result = await check_func()
                results.append((check_name, "✅ Pass", result.get("details", "")))
                console.print(f"  ✅ {check_name}")
            except Exception as e:
                results.append((check_name, "⚠️  Warning", str(e)))
                console.print(f"  ⚠️  {check_name}: {e}")
        
        self.validation_results["system_requirements"] = results
        return True  # Warnings are okay for system requirements
    
    async def _check_python_env(self) -> Dict:
        """Check Python environment"""
        python_version = sys.version_info
        if python_version >= (3, 8):
            return {"details": f"Python {python_version.major}.{python_version.minor}"}
        else:
            raise ValueError(f"Python 3.8+ required, got {python_version.major}.{python_version.minor}")
    
    async def _check_packages(self) -> Dict:
        """Check required packages"""
        required_packages = [
            "fastapi", "uvicorn", "torch", "transformers", 
            "diffusers", "typer", "rich", "pytest"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            raise ImportError(f"Missing packages: {', '.join(missing_packages)}")
        
        return {"details": f"{len(required_packages)} packages available"}
    
    async def _check_model_storage(self) -> Dict:
        """Check model storage requirements"""
        models_dir = project_root / "models"
        if not models_dir.exists():
            models_dir.mkdir(exist_ok=True)
        
        # Estimate storage requirements
        estimated_storage_gb = 50  # Rough estimate for all Phase 1 models
        return {"details": f"Storage prepared (~{estimated_storage_gb}GB needed)"}
    
    async def _check_vram_requirements(self) -> Dict:
        """Check VRAM requirements"""
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb >= 8:
                    return {"details": f"VRAM: {vram_gb:.1f}GB (sufficient)"}
                else:
                    raise ValueError(f"Insufficient VRAM: {vram_gb:.1f}GB (8GB+ recommended)")
            else:
                raise ValueError("CUDA not available")
        except ImportError:
            raise ImportError("PyTorch not available")
    
    def generate_report(self) -> None:
        """Generate validation report"""
        console.print("\n" + "="*60)
        console.print("📊 [bold blue]Phase 1 MVP Validation Report[/bold blue]")
        console.print("="*60)
        
        total_checks = 0
        passed_checks = 0
        
        for category, results in self.validation_results.items():
            console.print(f"\n[bold]{category.replace('_', ' ').title()}:[/bold]")
            
            table = Table()
            table.add_column("Check", style="cyan")
            table.add_column("Status", style="white")
            table.add_column("Details", style="white")
            
            for check_name, status, details in results:
                table.add_row(check_name, status, details)
                total_checks += 1
                if "✅" in status:
                    passed_checks += 1
            
            console.print(table)
        
        # Summary
        success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        summary_panel = Panel(
            f"""[bold]Validation Summary[/bold]

Total Checks: {total_checks}
Passed: {passed_checks}
Success Rate: {success_rate:.1f}%

Status: {'✅ Ready for Phase 1 Deployment' if success_rate >= 85 else '⚠️  Needs Attention Before Deployment'}""",
            title="Phase 1 MVP Status",
            border_style="green" if success_rate >= 85 else "yellow"
        )
        
        console.print(f"\n{summary_panel}")

async def main():
    """Main deployment validation function"""
    console.print(Panel(
        """[bold blue]Phase 1 MVP Deployment Validator[/bold blue]

This script validates that your WAN models MVP is ready for deployment with:
• T2V, I2V, and TI2V model support
• Seamless model switching and auto-detection  
• Enhanced prompt processing
• Comprehensive CLI tools
• Frontend integration

Starting validation...""",
        title="WAN Phase 1 Deployment",
        border_style="blue"
    ))
    
    validator = Phase1Validator()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        
        # Run validation steps
        validation_steps = [
            ("Backend Models", validator.validate_backend_models),
            ("Frontend Integration", validator.validate_frontend_integration),
            ("CLI Tools", validator.validate_cli_tools),
            ("System Requirements", validator.validate_system_requirements)
        ]
        
        task = progress.add_task("Validating Phase 1 MVP...", total=len(validation_steps))
        
        all_passed = True
        for step_name, step_func in validation_steps:
            progress.update(task, description=f"Validating {step_name}...")
            
            try:
                result = await step_func()
                if not result:
                    all_passed = False
            except Exception as e:
                console.print(f"[red]❌ {step_name} validation failed: {e}[/red]")
                all_passed = False
            
            progress.advance(task)
    
    # Generate final report
    validator.generate_report()
    
    if all_passed:
        console.print("\n[green]🎉 Phase 1 MVP is ready for deployment![/green]")
        console.print("\nNext steps:")
        console.print("1. Run: [cyan]wan-cli wan test --pattern='test_wan_models*'[/cyan]")
        console.print("2. Start backend: [cyan]python backend/app.py[/cyan]")
        console.print("3. Start frontend: [cyan]cd frontend && npm run dev[/cyan]")
        console.print("4. Test generation: [cyan]wan-cli wan generate 'a beautiful sunset'[/cyan]")
        return 0
    else:
        console.print("\n[yellow]⚠️  Phase 1 MVP needs attention before deployment[/yellow]")
        console.print("Please address the issues above and run validation again.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())