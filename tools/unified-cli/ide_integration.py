"""
IDE Integration System

Provides real-time quality feedback and integrates with popular IDEs
through Language Server Protocol (LSP) and other mechanisms.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import queue
import time

from tools.unified-cli.cli import UnifiedCLI, ToolResult


@dataclass
class QualityIssue:
    """Represents a code quality issue"""
    file_path: str
    line: int
    column: int
    severity: str  # 'error', 'warning', 'info', 'hint'
    message: str
    rule: str
    tool: str
    fix_suggestion: Optional[str] = None


@dataclass
class RealTimeFeedback:
    """Real-time feedback data"""
    file_path: str
    issues: List[QualityIssue]
    metrics: Dict[str, Any]
    timestamp: datetime


class IDEIntegration:
    """IDE integration for real-time quality feedback"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.cli = UnifiedCLI()
        self.feedback_queue = queue.Queue()
        self.active_files = set()
        self.file_cache = {}  # Cache for file analysis results
        self.watchers = {}  # File watchers for active files
        
        # Quality thresholds
        self.thresholds = {
            'complexity': 10,
            'line_length': 88,
            'function_length': 50,
            'test_coverage': 80
        }
        
        # Load IDE-specific configuration
        self.load_ide_config()
    
    def load_ide_config(self):
        """Load IDE-specific configuration"""
        config_file = self.project_root / ".kiro" / "ide-config.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                self.thresholds.update(config.get('thresholds', {}))
                
            except Exception as e:
                print(f"Warning: Could not load IDE config: {e}")
    
    def save_ide_config(self):
        """Save IDE configuration"""
        config_dir = self.project_root / ".kiro"
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / "ide-config.json"
        
        config_data = {
            'thresholds': self.thresholds,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    async def analyze_file_realtime(self, file_path: str) -> RealTimeFeedback:
        """Analyze a file in real-time and return feedback"""
        issues = []
        metrics = {}
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Run lightweight analysis tools
            issues.extend(await self.check_syntax(file_path, content))
            issues.extend(await self.check_style(file_path, content))
            issues.extend(await self.check_complexity(file_path, content))
            issues.extend(await self.check_imports(file_path, content))
            
            # Calculate metrics
            metrics = await self.calculate_metrics(file_path, content)
            
        except Exception as e:
            issues.append(QualityIssue(
                file_path=file_path,
                line=1,
                column=1,
                severity='error',
                message=f"Analysis failed: {str(e)}",
                rule='analysis_error',
                tool='ide_integration'
            ))
        
        return RealTimeFeedback(
            file_path=file_path,
            issues=issues,
            metrics=metrics,
            timestamp=datetime.now()
        )
    
    async def check_syntax(self, file_path: str, content: str) -> List[QualityIssue]:
        """Check syntax errors"""
        issues = []
        
        if file_path.endswith('.py'):
            try:
                import ast
                ast.parse(content)
            except SyntaxError as e:
                issues.append(QualityIssue(
                    file_path=file_path,
                    line=e.lineno or 1,
                    column=e.offset or 1,
                    severity='error',
                    message=f"Syntax error: {e.msg}",
                    rule='syntax_error',
                    tool='python_ast'
                ))
        
        return issues
    
    async def check_style(self, file_path: str, content: str) -> List[QualityIssue]:
        """Check style issues"""
        issues = []
        
        if not file_path.endswith('.py'):
            return issues
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > self.thresholds['line_length']:
                issues.append(QualityIssue(
                    file_path=file_path,
                    line=i,
                    column=self.thresholds['line_length'] + 1,
                    severity='warning',
                    message=f"Line too long ({len(line)} > {self.thresholds['line_length']})",
                    rule='line_length',
                    tool='style_checker',
                    fix_suggestion="Consider breaking this line into multiple lines"
                ))
            
            # Check trailing whitespace
            if line.endswith(' ') or line.endswith('\t'):
                issues.append(QualityIssue(
                    file_path=file_path,
                    line=i,
                    column=len(line.rstrip()) + 1,
                    severity='info',
                    message="Trailing whitespace",
                    rule='trailing_whitespace',
                    tool='style_checker',
                    fix_suggestion="Remove trailing whitespace"
                ))
        
        return issues
    
    async def check_complexity(self, file_path: str, content: str) -> List[QualityIssue]:
        """Check code complexity"""
        issues = []
        
        if not file_path.endswith('.py'):
            return issues
        
        try:
            import ast
            
            class ComplexityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.complexity_issues = []
                
                def visit_FunctionDef(self, node):
                    # Calculate cyclomatic complexity (simplified)
                    complexity = self.calculate_complexity(node)
                    
                    if complexity > self.parent.thresholds['complexity']:
                        self.complexity_issues.append(QualityIssue(
                            file_path=file_path,
                            line=node.lineno,
                            column=node.col_offset,
                            severity='warning',
                            message=f"Function '{node.name}' is too complex (complexity: {complexity})",
                            rule='cyclomatic_complexity',
                            tool='complexity_analyzer',
                            fix_suggestion="Consider breaking this function into smaller functions"
                        ))
                    
                    # Check function length
                    func_lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
                    if func_lines > self.parent.thresholds['function_length']:
                        self.complexity_issues.append(QualityIssue(
                            file_path=file_path,
                            line=node.lineno,
                            column=node.col_offset,
                            severity='info',
                            message=f"Function '{node.name}' is too long ({func_lines} lines)",
                            rule='function_length',
                            tool='complexity_analyzer',
                            fix_suggestion="Consider breaking this function into smaller functions"
                        ))
                    
                    self.generic_visit(node)
                
                def calculate_complexity(self, node):
                    """Calculate cyclomatic complexity (simplified)"""
                    complexity = 1  # Base complexity
                    
                    for child in ast.walk(node):
                        if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                            complexity += 1
                        elif isinstance(child, ast.ExceptHandler):
                            complexity += 1
                        elif isinstance(child, ast.BoolOp):
                            complexity += len(child.values) - 1
                    
                    return complexity
            
            tree = ast.parse(content)
            visitor = ComplexityVisitor()
            visitor.parent = self  # Give visitor access to thresholds
            visitor.visit(tree)
            issues.extend(visitor.complexity_issues)
            
        except Exception as e:
            # Don't fail the entire analysis for complexity issues
            pass
        
        return issues
    
    async def check_imports(self, file_path: str, content: str) -> List[QualityIssue]:
        """Check import issues"""
        issues = []
        
        if not file_path.endswith('.py'):
            return issues
        
        try:
            import ast
            
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append((alias.name, node.lineno))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append((f"{module}.{alias.name}", node.lineno))
            
            # Check for unused imports (simplified check)
            used_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
            
            for import_name, line_no in imports:
                base_name = import_name.split('.')[0]
                if base_name not in used_names and base_name not in ['os', 'sys', 'json']:
                    issues.append(QualityIssue(
                        file_path=file_path,
                        line=line_no,
                        column=1,
                        severity='info',
                        message=f"Potentially unused import: {import_name}",
                        rule='unused_import',
                        tool='import_checker',
                        fix_suggestion=f"Remove unused import: {import_name}"
                    ))
            
        except Exception as e:
            # Don't fail the entire analysis for import issues
            pass
        
        return issues
    
    async def calculate_metrics(self, file_path: str, content: str) -> Dict[str, Any]:
        """Calculate file metrics"""
        metrics = {
            'lines_of_code': len(content.split('\n')),
            'file_size': len(content),
            'last_analyzed': datetime.now().isoformat()
        }
        
        if file_path.endswith('.py'):
            try:
                import ast
                tree = ast.parse(content)
                
                # Count different node types
                node_counts = {}
                for node in ast.walk(tree):
                    node_type = type(node).__name__
                    node_counts[node_type] = node_counts.get(node_type, 0) + 1
                
                metrics.update({
                    'functions': node_counts.get('FunctionDef', 0) + node_counts.get('AsyncFunctionDef', 0),
                    'classes': node_counts.get('ClassDef', 0),
                    'imports': node_counts.get('Import', 0) + node_counts.get('ImportFrom', 0),
                    'complexity_score': self.estimate_complexity(tree)
                })
                
            except Exception:
                pass
        
        return metrics
    
    def estimate_complexity(self, tree) -> int:
        """Estimate overall file complexity"""
        complexity = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                complexity += 2
            elif isinstance(node, ast.ClassDef):
                complexity += 3
        
        return complexity
    
    def start_file_watcher(self, file_path: str):
        """Start watching a file for changes"""
        if file_path in self.watchers:
            return
        
        def watch_file():
            last_modified = 0
            
            while file_path in self.active_files:
                try:
                    current_modified = os.path.getmtime(file_path)
                    
                    if current_modified > last_modified:
                        last_modified = current_modified
                        
                        # Queue analysis
                        self.feedback_queue.put(('analyze', file_path))
                    
                    time.sleep(0.5)  # Check every 500ms
                    
                except (OSError, FileNotFoundError):
                    # File was deleted or moved
                    break
        
        watcher_thread = threading.Thread(target=watch_file, daemon=True)
        watcher_thread.start()
        self.watchers[file_path] = watcher_thread
    
    def stop_file_watcher(self, file_path: str):
        """Stop watching a file"""
        if file_path in self.active_files:
            self.active_files.remove(file_path)
        
        if file_path in self.watchers:
            del self.watchers[file_path]
    
    async def provide_realtime_feedback(self, file_path: str) -> RealTimeFeedback:
        """Provide real-time feedback for a file"""
        # Add to active files
        self.active_files.add(file_path)
        
        # Start watching the file
        self.start_file_watcher(file_path)
        
        # Perform initial analysis
        feedback = await self.analyze_file_realtime(file_path)
        
        # Cache the result
        self.file_cache[file_path] = feedback
        
        return feedback
    
    def get_cached_feedback(self, file_path: str) -> Optional[RealTimeFeedback]:
        """Get cached feedback for a file"""
        return self.file_cache.get(file_path)
    
    def format_feedback_for_ide(self, feedback: RealTimeFeedback, format_type: str = 'lsp') -> Dict[str, Any]:
        """Format feedback for specific IDE integration"""
        
        if format_type == 'lsp':
            # Language Server Protocol format
            diagnostics = []
            
            for issue in feedback.issues:
                severity_map = {
                    'error': 1,
                    'warning': 2,
                    'info': 3,
                    'hint': 4
                }
                
                diagnostic = {
                    'range': {
                        'start': {'line': issue.line - 1, 'character': issue.column - 1},
                        'end': {'line': issue.line - 1, 'character': issue.column + 10}
                    },
                    'severity': severity_map.get(issue.severity, 3),
                    'message': issue.message,
                    'source': issue.tool,
                    'code': issue.rule
                }
                
                if issue.fix_suggestion:
                    diagnostic['codeActions'] = [{
                        'title': 'Fix: ' + issue.fix_suggestion,
                        'kind': 'quickfix'
                    }]
                
                diagnostics.append(diagnostic)
            
            return {
                'uri': f"file://{feedback.file_path}",
                'diagnostics': diagnostics
            }
        
        elif format_type == 'vscode':
            # VS Code specific format
            return {
                'file': feedback.file_path,
                'issues': [asdict(issue) for issue in feedback.issues],
                'metrics': feedback.metrics,
                'timestamp': feedback.timestamp.isoformat()
            }
        
        else:
            # Generic format
            return asdict(feedback)
    
    async def start_feedback_server(self, port: int = 8765):
        """Start a WebSocket server for IDE integration"""
        import websockets
        
        async def handle_client(websocket, path):
            print(f"IDE client connected: {websocket.remote_address}")
            
            try:
                async for message in websocket:
                    data = json.loads(message)
                    command = data.get('command')
                    
                    if command == 'analyze_file':
                        file_path = data.get('file_path')
                        feedback = await self.provide_realtime_feedback(file_path)
                        
                        response = {
                            'command': 'feedback',
                            'data': self.format_feedback_for_ide(feedback, data.get('format', 'lsp'))
                        }
                        
                        await websocket.send(json.dumps(response))
                    
                    elif command == 'stop_watching':
                        file_path = data.get('file_path')
                        self.stop_file_watcher(file_path)
                        
                        await websocket.send(json.dumps({
                            'command': 'stopped',
                            'file_path': file_path
                        }))
            
            except websockets.exceptions.ConnectionClosed:
                print(f"IDE client disconnected: {websocket.remote_address}")
        
        print(f"Starting IDE integration server on port {port}")
        await websockets.serve(handle_client, "localhost", port)
    
    async def run_feedback_processor(self):
        """Process feedback queue in background"""
        while True:
            try:
                # Check for queued analysis requests
                try:
                    command, file_path = self.feedback_queue.get_nowait()
                    
                    if command == 'analyze':
                        feedback = await self.analyze_file_realtime(file_path)
                        self.file_cache[file_path] = feedback
                        
                        # Notify any connected IDEs
                        # This would send updates to connected WebSocket clients
                        
                except queue.Empty:
                    pass
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"Error in feedback processor: {e}")
                await asyncio.sleep(1)


async def start_ide_integration_server(project_root: Path = None, port: int = 8765):
    """Start the IDE integration server"""
    integration = IDEIntegration(project_root)
    
    # Start the feedback processor
    processor_task = asyncio.create_task(integration.run_feedback_processor())
    
    # Start the WebSocket server
    server_task = asyncio.create_task(integration.start_feedback_server(port))
    
    print(f"IDE integration server started on port {port}")
    print("Connect your IDE to ws://localhost:{port} for real-time feedback")
    
    try:
        await asyncio.gather(processor_task, server_task)
    except KeyboardInterrupt:
        print("Stopping IDE integration server...")
        processor_task.cancel()
        server_task.cancel()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'server':
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
        asyncio.run(start_ide_integration_server(port=port))
    else:
        print("Usage: python ide_integration.py server [port]")