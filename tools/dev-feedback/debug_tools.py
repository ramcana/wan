#!/usr/bin/env python3
"""
Debug Tools with Comprehensive Logging

This module provides debugging tools with comprehensive logging and error reporting.
"""

import os
import sys
import json
import traceback
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import re
import subprocess

@dataclass
class LogEntry:
    """Log entry structure"""
    timestamp: datetime
    level: str
    logger: str
    message: str
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    thread_id: Optional[int] = None
    exception: Optional[str] = None

@dataclass
class ErrorPattern:
    """Error pattern for analysis"""
    pattern: str
    description: str
    category: str
    severity: str
    fix_suggestion: Optional[str] = None

@dataclass
class DebugSession:
    """Debug session information"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    log_entries: List[LogEntry] = None
    error_count: int = 0
    warning_count: int = 0
    performance_metrics: Dict[str, Any] = None

class DebugLogHandler(logging.Handler):
    """Custom log handler for debug tools"""
    
    def __init__(self, debug_tools: 'DebugTools'):
        super().__init__()
        self.debug_tools = debug_tools
    
    def emit(self, record):
        try:
            # Extract information from log record
            log_entry = LogEntry(
                timestamp=datetime.fromtimestamp(record.created),
                level=record.levelname,
                logger=record.name,
                message=record.getMessage(),
                module=record.module if hasattr(record, 'module') else None,
                function=record.funcName if hasattr(record, 'funcName') else None,
                line_number=record.lineno if hasattr(record, 'lineno') else None,
                thread_id=record.thread if hasattr(record, 'thread') else None,
                exception=self.format_exception(record) if record.exc_info else None
            )
            
            self.debug_tools.add_log_entry(log_entry)
            
        except Exception:
            # Don't let logging errors break the application
            pass
    
    def format_exception(self, record) -> str:
        """Format exception information"""
        if record.exc_info:
            return ''.join(traceback.format_exception(*record.exc_info))
        return None

class DebugTools:
    """Comprehensive debugging tools"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.logger = logging.getLogger(__name__)
        
        # Debug session management
        self.current_session = None
        self.sessions = {}
        
        # Log management
        self.log_entries = []
        self.max_log_entries = 10000
        self.log_handler = None
        
        # Error patterns for analysis
        self.error_patterns = self._load_error_patterns()
        
        # Performance monitoring
        self.performance_data = {}
        self.function_timings = {}
        
        # Debug configuration
        self.debug_enabled = False
        self.log_file = self.project_root / "logs" / "debug.log"
        self.log_file.parent.mkdir(exist_ok=True)
    
    def _load_error_patterns(self) -> List[ErrorPattern]:
        """Load common error patterns for analysis"""
        return [
            ErrorPattern(
                pattern=r"ModuleNotFoundError: No module named '(.+)'",
                description="Missing Python module",
                category="dependency",
                severity="error",
                fix_suggestion="Install missing module: pip install {}"
            ),
            ErrorPattern(
                pattern=r"FileNotFoundError: \[Errno 2\] No such file or directory: '(.+)'",
                description="Missing file",
                category="filesystem",
                severity="error",
                fix_suggestion="Check if file exists: {}"
            ),
            ErrorPattern(
                pattern=r"ConnectionError|ConnectionRefusedError",
                description="Network connection issue",
                category="network",
                severity="error",
                fix_suggestion="Check network connectivity and service availability"
            ),
            ErrorPattern(
                pattern=r"PermissionError: \[Errno 13\]",
                description="Permission denied",
                category="permissions",
                severity="error",
                fix_suggestion="Check file/directory permissions"
            ),
            ErrorPattern(
                pattern=r"CUDA out of memory",
                description="GPU memory exhausted",
                category="gpu",
                severity="error",
                fix_suggestion="Reduce batch size or model size"
            ),
            ErrorPattern(
                pattern=r"Port \d+ is already in use",
                description="Port conflict",
                category="network",
                severity="warning",
                fix_suggestion="Use a different port or stop the conflicting service"
            ),
            ErrorPattern(
                pattern=r"DeprecationWarning",
                description="Deprecated API usage",
                category="compatibility",
                severity="warning",
                fix_suggestion="Update code to use newer API"
            )
        ]
    
    def enable_debug_logging(self, level: str = "DEBUG"):
        """Enable comprehensive debug logging"""
        if self.debug_enabled:
            return
        
        self.debug_enabled = True
        
        # Set up file logging
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(getattr(logging, level))
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Set up custom handler for analysis
        self.log_handler = DebugLogHandler(self)
        self.log_handler.setLevel(getattr(logging, level))
        
        # Add handlers to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(self.log_handler)
        root_logger.setLevel(getattr(logging, level))
        
        self.logger.info(f"Debug logging enabled at level {level}")
        self.logger.info(f"Log file: {self.log_file}")
    
    def disable_debug_logging(self):
        """Disable debug logging"""
        if not self.debug_enabled:
            return
        
        self.debug_enabled = False
        
        # Remove custom handler
        if self.log_handler:
            root_logger = logging.getLogger()
            root_logger.removeHandler(self.log_handler)
            self.log_handler = None
        
        self.logger.info("Debug logging disabled")
    
    def start_debug_session(self, session_id: Optional[str] = None) -> str:
        """Start a new debug session"""
        if not session_id:
            session_id = f"debug_{int(time.time())}"
        
        session = DebugSession(
            session_id=session_id,
            start_time=datetime.now(),
            log_entries=[],
            performance_metrics={}
        )
        
        self.current_session = session
        self.sessions[session_id] = session
        
        self.logger.info(f"Started debug session: {session_id}")
        return session_id
    
    def end_debug_session(self) -> Optional[DebugSession]:
        """End the current debug session"""
        if not self.current_session:
            return None
        
        self.current_session.end_time = datetime.now()
        self.current_session.log_entries = self.log_entries.copy()
        
        # Calculate session metrics
        self.current_session.error_count = len([
            entry for entry in self.log_entries 
            if entry.level == 'ERROR'
        ])
        self.current_session.warning_count = len([
            entry for entry in self.log_entries 
            if entry.level == 'WARNING'
        ])
        
        session = self.current_session
        self.current_session = None
        
        self.logger.info(f"Ended debug session: {session.session_id}")
        return session
    
    def add_log_entry(self, log_entry: LogEntry):
        """Add log entry to current session"""
        self.log_entries.append(log_entry)
        
        # Limit log entries to prevent memory issues
        if len(self.log_entries) > self.max_log_entries:
            self.log_entries = self.log_entries[-self.max_log_entries//2:]
    
    def analyze_logs(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze logs for patterns and issues"""
        if session_id and session_id in self.sessions:
            log_entries = self.sessions[session_id].log_entries or []
        else:
            log_entries = self.log_entries
        
        analysis = {
            'total_entries': len(log_entries),
            'error_count': 0,
            'warning_count': 0,
            'info_count': 0,
            'debug_count': 0,
            'patterns_found': [],
            'frequent_errors': {},
            'timeline': [],
            'recommendations': []
        }
        
        # Count log levels
        for entry in log_entries:
            if entry.level == 'ERROR':
                analysis['error_count'] += 1
            elif entry.level == 'WARNING':
                analysis['warning_count'] += 1
            elif entry.level == 'INFO':
                analysis['info_count'] += 1
            elif entry.level == 'DEBUG':
                analysis['debug_count'] += 1
        
        # Analyze error patterns
        for entry in log_entries:
            if entry.level in ['ERROR', 'WARNING']:
                message = entry.message
                if entry.exception:
                    message += "\n" + entry.exception
                
                # Check against known patterns
                for pattern in self.error_patterns:
                    if re.search(pattern.pattern, message, re.IGNORECASE):
                        match_info = {
                            'pattern': pattern.description,
                            'category': pattern.category,
                            'severity': pattern.severity,
                            'fix_suggestion': pattern.fix_suggestion,
                            'timestamp': entry.timestamp.isoformat(),
                            'message': entry.message
                        }
                        analysis['patterns_found'].append(match_info)
                
                # Count frequent errors
                error_key = entry.message[:100]  # First 100 chars
                analysis['frequent_errors'][error_key] = analysis['frequent_errors'].get(error_key, 0) + 1
        
        # Create timeline of significant events
        significant_entries = [
            entry for entry in log_entries 
            if entry.level in ['ERROR', 'WARNING'] or 'started' in entry.message.lower()
        ]
        
        for entry in significant_entries[-20:]:  # Last 20 significant events
            analysis['timeline'].append({
                'timestamp': entry.timestamp.isoformat(),
                'level': entry.level,
                'message': entry.message,
                'logger': entry.logger
            })
        
        # Generate recommendations
        if analysis['error_count'] > 0:
            analysis['recommendations'].append("Review error logs and fix critical issues")
        
        if analysis['warning_count'] > 10:
            analysis['recommendations'].append("Address warnings to improve code quality")
        
        # Pattern-based recommendations
        pattern_categories = set(p['category'] for p in analysis['patterns_found'])
        if 'dependency' in pattern_categories:
            analysis['recommendations'].append("Check and install missing dependencies")
        if 'network' in pattern_categories:
            analysis['recommendations'].append("Verify network connectivity and service availability")
        if 'gpu' in pattern_categories:
            analysis['recommendations'].append("Optimize GPU memory usage")
        
        return analysis
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function execution time"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                duration = end_time - start_time
                
                func_name = f"{func.__module__}.{func.__name__}"
                if func_name not in self.function_timings:
                    self.function_timings[func_name] = []
                
                self.function_timings[func_name].append(duration)
                
                # Log slow functions
                if duration > 1.0:  # More than 1 second
                    self.logger.warning(f"Slow function: {func_name} took {duration:.2f}s")
        
        return wrapper
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance analysis report"""
        report = {
            'function_timings': {},
            'slow_functions': [],
            'total_functions_profiled': len(self.function_timings)
        }
        
        for func_name, timings in self.function_timings.items():
            if not timings:
                continue
            
            avg_time = sum(timings) / len(timings)
            max_time = max(timings)
            min_time = min(timings)
            call_count = len(timings)
            
            func_stats = {
                'call_count': call_count,
                'avg_time': avg_time,
                'max_time': max_time,
                'min_time': min_time,
                'total_time': sum(timings)
            }
            
            report['function_timings'][func_name] = func_stats
            
            # Identify slow functions
            if avg_time > 0.5 or max_time > 2.0:
                report['slow_functions'].append({
                    'function': func_name,
                    'avg_time': avg_time,
                    'max_time': max_time,
                    'call_count': call_count
                })
        
        # Sort slow functions by average time
        report['slow_functions'].sort(key=lambda x: x['avg_time'], reverse=True)
        
        return report
    
    def export_debug_report(self, output_file: Path, session_id: Optional[str] = None):
        """Export comprehensive debug report"""
        log_analysis = self.analyze_logs(session_id)
        performance_report = self.get_performance_report()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'project_root': str(self.project_root),
            'log_analysis': log_analysis,
            'performance_report': performance_report,
            'system_info': self._get_system_info()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Debug report exported to {output_file}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for debug report"""
        import platform
        
        return {
            'platform': platform.system(),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'working_directory': str(Path.cwd()),
            'environment_variables': dict(os.environ),
            'installed_packages': self._get_installed_packages()
        }
    
    def _get_installed_packages(self) -> List[str]:
        """Get list of installed Python packages"""
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.splitlines()
        except Exception:
            pass
        return []
    
    def clear_logs(self):
        """Clear accumulated log entries"""
        self.log_entries.clear()
        self.logger.info("Debug logs cleared")
    
    def get_recent_errors(self, count: int = 10) -> List[LogEntry]:
        """Get recent error log entries"""
        error_entries = [
            entry for entry in self.log_entries 
            if entry.level == 'ERROR'
        ]
        return error_entries[-count:]
    
    def search_logs(self, query: str, level: Optional[str] = None) -> List[LogEntry]:
        """Search log entries by message content"""
        results = []
        
        for entry in self.log_entries:
            if level and entry.level != level:
                continue
            
            if query.lower() in entry.message.lower():
                results.append(entry)
        
        return results

def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug tools with comprehensive logging")
    parser.add_argument('--enable', action='store_true', help='Enable debug logging')
    parser.add_argument('--disable', action='store_true', help='Disable debug logging')
    parser.add_argument('--analyze', action='store_true', help='Analyze current logs')
    parser.add_argument('--report', type=str, help='Generate debug report to file')
    parser.add_argument('--clear', action='store_true', help='Clear accumulated logs')
    parser.add_argument('--errors', type=int, default=10, help='Show recent errors')
    parser.add_argument('--search', type=str, help='Search logs for text')
    parser.add_argument('--level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='DEBUG', help='Log level')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup basic logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    # Create debug tools
    debug_tools = DebugTools()
    
    if args.enable:
        debug_tools.enable_debug_logging(args.level)
        print(f"✅ Debug logging enabled at level {args.level}")
        print(f"Log file: {debug_tools.log_file}")
    
    if args.disable:
        debug_tools.disable_debug_logging()
        print("✅ Debug logging disabled")
    
    if args.analyze:
        analysis = debug_tools.analyze_logs()
        
        print("\n📊 LOG ANALYSIS")
        print("=" * 50)
        print(f"Total entries: {analysis['total_entries']}")
        print(f"Errors: {analysis['error_count']}")
        print(f"Warnings: {analysis['warning_count']}")
        print(f"Info: {analysis['info_count']}")
        print(f"Debug: {analysis['debug_count']}")
        
        if analysis['patterns_found']:
            print(f"\n🔍 PATTERNS FOUND ({len(analysis['patterns_found'])}):")
            for pattern in analysis['patterns_found'][:5]:  # Show first 5
                print(f"  - {pattern['pattern']} ({pattern['category']})")
                if pattern['fix_suggestion']:
                    print(f"    Fix: {pattern['fix_suggestion']}")
        
        if analysis['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in analysis['recommendations']:
                print(f"  - {rec}")
    
    if args.errors:
        recent_errors = debug_tools.get_recent_errors(args.errors)
        
        if recent_errors:
            print(f"\n❌ RECENT ERRORS ({len(recent_errors)}):")
            for error in recent_errors:
                print(f"  [{error.timestamp}] {error.logger}: {error.message}")
        else:
            print("✅ No recent errors found")
    
    if args.search:
        results = debug_tools.search_logs(args.search)
        
        print(f"\n🔍 SEARCH RESULTS for '{args.search}' ({len(results)}):")
        for result in results[-10:]:  # Show last 10 results
            print(f"  [{result.timestamp}] {result.level}: {result.message}")
    
    if args.clear:
        debug_tools.clear_logs()
        print("✅ Debug logs cleared")
    
    if args.report:
        output_file = Path(args.report)
        debug_tools.export_debug_report(output_file)
        print(f"✅ Debug report exported to {output_file}")

if __name__ == "__main__":
    main()
