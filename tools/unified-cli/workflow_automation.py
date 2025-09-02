"""
Workflow Automation System

Automatically runs appropriate tools based on development context,
file changes, and project state.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from tools.unified-cli.cli import UnifiedCLI, WorkflowContext, ToolResult


@dataclass
class FileChangeEvent:
    """Represents a file change event"""
    path: str
    event_type: str  # 'created', 'modified', 'deleted', 'moved'
    timestamp: datetime
    file_type: str  # 'python', 'config', 'test', 'doc', etc.


@dataclass
class AutomationRule:
    """Rule for automated workflow execution"""
    name: str
    trigger_patterns: List[str]  # File patterns that trigger this rule
    workflow_context: WorkflowContext
    delay_seconds: int = 0  # Delay before execution
    debounce_seconds: int = 5  # Minimum time between executions
    conditions: List[str] = None  # Additional conditions


class WorkflowAutomation:
    """Automated workflow execution based on development context"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.cli = UnifiedCLI()
        self.observer = None
        self.last_execution = {}  # Track last execution time per rule
        self.pending_executions = {}  # Track pending executions
        
        # Default automation rules
        self.rules = [
            AutomationRule(
                name="python_code_changes",
                trigger_patterns=["*.py"],
                workflow_context=WorkflowContext.PRE_COMMIT,
                delay_seconds=2,
                debounce_seconds=10,
                conditions=["git_staged_changes"]
            ),
            AutomationRule(
                name="test_file_changes",
                trigger_patterns=["test_*.py", "*_test.py", "tests/**/*.py"],
                workflow_context=WorkflowContext.POST_COMMIT,
                delay_seconds=1,
                debounce_seconds=5
            ),
            AutomationRule(
                name="config_changes",
                trigger_patterns=["*.json", "*.yaml", "*.yml", "*.ini", "*.toml"],
                workflow_context=WorkflowContext.DAILY_MAINTENANCE,
                delay_seconds=5,
                debounce_seconds=15
            ),
            AutomationRule(
                name="documentation_changes",
                trigger_patterns=["*.md", "*.rst", "docs/**/*"],
                workflow_context=WorkflowContext.POST_COMMIT,
                delay_seconds=3,
                debounce_seconds=10
            ),
            AutomationRule(
                name="frontend_changes",
                trigger_patterns=["*.js", "*.ts", "*.jsx", "*.tsx", "*.vue", "*.svelte"],
                workflow_context=WorkflowContext.PRE_COMMIT,
                delay_seconds=2,
                debounce_seconds=8
            ),
            AutomationRule(
                name="critical_file_changes",
                trigger_patterns=["requirements.txt", "package.json", "Dockerfile", "*.lock"],
                workflow_context=WorkflowContext.RELEASE_PREP,
                delay_seconds=10,
                debounce_seconds=30
            )
        ]
        
        # Load custom rules if they exist
        self.load_custom_rules()
    
    def load_custom_rules(self):
        """Load custom automation rules from config file"""
        config_file = self.project_root / ".kiro" / "automation-rules.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    custom_rules_data = json.load(f)
                
                for rule_data in custom_rules_data.get('rules', []):
                    rule = AutomationRule(
                        name=rule_data['name'],
                        trigger_patterns=rule_data['trigger_patterns'],
                        workflow_context=WorkflowContext(rule_data['workflow_context']),
                        delay_seconds=rule_data.get('delay_seconds', 0),
                        debounce_seconds=rule_data.get('debounce_seconds', 5),
                        conditions=rule_data.get('conditions', [])
                    )
                    self.rules.append(rule)
                    
            except Exception as e:
                print(f"Warning: Could not load custom automation rules: {e}")
    
    def save_custom_rules(self):
        """Save current automation rules to config file"""
        config_dir = self.project_root / ".kiro"
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / "automation-rules.json"
        
        # Only save non-default rules
        custom_rules = [rule for rule in self.rules if not self.is_default_rule(rule)]
        
        config_data = {
            'rules': [asdict(rule) for rule in custom_rules]
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
    
    def is_default_rule(self, rule: AutomationRule) -> bool:
        """Check if a rule is one of the default rules"""
        default_names = {
            "python_code_changes", "test_file_changes", 
            "config_changes", "documentation_changes"
        }
        return rule.name in default_names
    
    def get_file_type(self, file_path: str) -> str:
        """Determine file type based on path and extension"""
        path = Path(file_path)
        
        if path.suffix == '.py':
            if 'test' in path.name or path.parts and 'tests' in path.parts:
                return 'test'
            return 'python'
        elif path.suffix in ['.json', '.yaml', '.yml', '.ini', '.toml']:
            return 'config'
        elif path.suffix in ['.md', '.rst'] or 'docs' in path.parts:
            return 'doc'
        elif path.suffix in ['.js', '.ts', '.jsx', '.tsx']:
            return 'frontend'
        else:
            return 'other'
    
    def matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if file path matches a pattern"""
        import fnmatch
        
        # Handle directory patterns
        if '**' in pattern:
            # Convert glob pattern to more specific matching
            parts = pattern.split('**')
            if len(parts) == 2:
                prefix, suffix = parts
                return (file_path.startswith(prefix.rstrip('/')) and 
                       file_path.endswith(suffix.lstrip('/')))
        
        return fnmatch.fnmatch(file_path, pattern)
    
    def find_matching_rules(self, file_path: str) -> List[AutomationRule]:
        """Find automation rules that match the given file path"""
        matching_rules = []
        
        for rule in self.rules:
            for pattern in rule.trigger_patterns:
                if self.matches_pattern(file_path, pattern):
                    matching_rules.append(rule)
                    break
        
        return matching_rules
    
    def should_execute_rule(self, rule: AutomationRule) -> bool:
        """Check if a rule should be executed based on debounce logic"""
        now = datetime.now()
        last_exec = self.last_execution.get(rule.name)
        
        if last_exec is None:
            return True
        
        time_since_last = (now - last_exec).total_seconds()
        return time_since_last >= rule.debounce_seconds
    
    def check_rule_conditions(self, rule: AutomationRule) -> bool:
        """Check if rule conditions are met"""
        if not rule.conditions:
            return True
        
        for condition in rule.conditions:
            if condition == "git_staged_changes":
                if not self.has_git_staged_changes():
                    return False
            elif condition == "tests_passing":
                if not self.are_tests_passing():
                    return False
            # Add more conditions as needed
        
        return True
    
    def has_git_staged_changes(self) -> bool:
        """Check if there are staged changes in git"""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only'],
                capture_output=True, text=True, timeout=10,
                cwd=self.project_root
            )
            return bool(result.stdout.strip())
        except Exception:
            return False
    
    def are_tests_passing(self) -> bool:
        """Quick check if tests are passing"""
        # This is a simplified check - in practice, you might want to
        # run a subset of fast tests or check recent test results
        try:
            import subprocess
            result = subprocess.run(
                ['python', '-m', 'pytest', '--collect-only', '-q'],
                capture_output=True, text=True, timeout=30,
                cwd=self.project_root
            )
            return result.returncode == 0
        except Exception:
            return True  # Assume tests are okay if we can't check
    
    async def execute_rule(self, rule: AutomationRule, trigger_file: str = None):
        """Execute a workflow rule"""
        if not self.should_execute_rule(rule):
            return
        
        if not self.check_rule_conditions(rule):
            print(f"Skipping rule {rule.name}: conditions not met")
            return
        
        print(f"Executing automation rule: {rule.name}")
        if trigger_file:
            print(f"Triggered by: {trigger_file}")
        
        # Apply delay if specified
        if rule.delay_seconds > 0:
            await asyncio.sleep(rule.delay_seconds)
        
        # Execute the workflow
        try:
            results = await self.cli.run_workflow(rule.workflow_context)
            
            # Update last execution time
            self.last_execution[rule.name] = datetime.now()
            
            # Log results
            success_count = sum(1 for r in results if r.success)
            print(f"Rule {rule.name} completed: {success_count}/{len(results)} tools succeeded")
            
            # Save execution log
            self.log_execution(rule, results, trigger_file)
            
        except Exception as e:
            print(f"Error executing rule {rule.name}: {e}")
    
    def log_execution(self, rule: AutomationRule, results: List[ToolResult], 
                     trigger_file: str = None):
        """Log automation execution for audit purposes"""
        log_dir = self.project_root / ".kiro" / "automation-logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'rule_name': rule.name,
            'workflow_context': rule.workflow_context.value,
            'trigger_file': trigger_file,
            'results': [asdict(result) for result in results],
            'success_count': sum(1 for r in results if r.success),
            'total_tools': len(results)
        }
        
        log_file = log_dir / f"automation-{datetime.now().strftime('%Y-%m')}.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    async def handle_file_change(self, file_path: str, event_type: str):
        """Handle a file change event"""
        # Convert to relative path
        try:
            rel_path = str(Path(file_path).relative_to(self.project_root))
        except ValueError:
            # File is outside project root
            return
        
        # Skip certain files/directories
        skip_patterns = [
            '.git/**', '__pycache__/**', '*.pyc', '.pytest_cache/**',
            'node_modules/**', '.venv/**', 'venv/**'
        ]
        
        for pattern in skip_patterns:
            if self.matches_pattern(rel_path, pattern):
                return
        
        # Find matching rules
        matching_rules = self.find_matching_rules(rel_path)
        
        if not matching_rules:
            return
        
        print(f"File change detected: {rel_path} ({event_type})")
        
        # Execute matching rules
        for rule in matching_rules:
            # Schedule execution (with debouncing)
            rule_key = f"{rule.name}:{rel_path}"
            
            if rule_key in self.pending_executions:
                # Cancel previous execution
                self.pending_executions[rule_key].cancel()
            
            # Schedule new execution
            task = asyncio.create_task(
                self.execute_rule(rule, rel_path)
            )
            self.pending_executions[rule_key] = task
            
            # Clean up completed tasks
            def cleanup_task(t):
                if rule_key in self.pending_executions:
                    del self.pending_executions[rule_key]
            
            task.add_done_callback(cleanup_task)


class FileWatcher(FileSystemEventHandler):
    """File system event handler for workflow automation"""
    
    def __init__(self, automation: WorkflowAutomation):
        self.automation = automation
        self.loop = None
    
    def set_event_loop(self, loop):
        """Set the event loop for async operations"""
        self.loop = loop
    
    def on_modified(self, event):
        if not event.is_directory and self.loop:
            asyncio.run_coroutine_threadsafe(
                self.automation.handle_file_change(event.src_path, 'modified'),
                self.loop
            )
    
    def on_created(self, event):
        if not event.is_directory and self.loop:
            asyncio.run_coroutine_threadsafe(
                self.automation.handle_file_change(event.src_path, 'created'),
                self.loop
            )
    
    def on_deleted(self, event):
        if not event.is_directory and self.loop:
            asyncio.run_coroutine_threadsafe(
                self.automation.handle_file_change(event.src_path, 'deleted'),
                self.loop
            )


async def start_automation_daemon(project_root: Path = None):
    """Start the workflow automation daemon"""
    automation = WorkflowAutomation(project_root)
    
    # Set up file watcher
    event_handler = FileWatcher(automation)
    event_handler.set_event_loop(asyncio.get_event_loop())
    
    observer = Observer()
    observer.schedule(event_handler, str(automation.project_root), recursive=True)
    observer.start()
    
    print(f"Workflow automation started for {automation.project_root}")
    print("Watching for file changes...")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Stopping workflow automation...")
        observer.stop()
    
    observer.join()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'daemon':
        asyncio.run(start_automation_daemon())
    else:
        print("Usage: python workflow_automation.py daemon")