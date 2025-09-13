"""
Log analysis and management utilities for video generation system.

This module provides tools for analyzing log files, extracting insights,
and managing log data for troubleshooting and optimization.
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Iterator
from dataclasses import dataclass
from collections import defaultdict, Counter
import gzip
import shutil

from generation_logger import get_logger


@dataclass
class LogEntry:
    """Structured representation of a log entry."""
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    session_id: Optional[str] = None
    raw_line: str = ""
    
    @classmethod
    def parse(cls, line: str) -> Optional['LogEntry']:
        """Parse a log line into a LogEntry object."""
        # Standard log format: timestamp - logger_name - level - message
        pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ([^-]+) - ([^-]+) - (.+)'
        match = re.match(pattern, line.strip())
        
        if not match:
            return None
        
        timestamp_str, logger_name, level, message = match.groups()
        
        try:
            # Parse timestamp
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
        except ValueError:
            return None
        
        # Extract session ID if present
        session_id = None
        session_match = re.search(r'\[([a-f0-9-]{36})\]', message)
        if session_match:
            session_id = session_match.group(1)
        
        return cls(
            timestamp=timestamp,
            level=level.strip(),
            logger_name=logger_name.strip(),
            message=message.strip(),
            session_id=session_id,
            raw_line=line.strip()
        )


@dataclass
class SessionAnalysis:
    """Analysis results for a generation session."""
    session_id: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration: Optional[float]
    status: str  # 'success', 'error', 'incomplete'
    model_type: Optional[str]
    generation_mode: Optional[str]
    error_count: int
    warning_count: int
    stages_completed: List[str]
    errors: List[str]
    performance_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'status': self.status,
            'model_type': self.model_type,
            'generation_mode': self.generation_mode,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'stages_completed': self.stages_completed,
            'errors': self.errors,
            'performance_metrics': self.performance_metrics
        }


@dataclass
class LogAnalysisReport:
    """Comprehensive log analysis report."""
    analysis_period: Tuple[datetime, datetime]
    total_sessions: int
    successful_sessions: int
    failed_sessions: int
    incomplete_sessions: int
    average_duration: float
    error_patterns: Dict[str, int]
    performance_trends: Dict[str, List[float]]
    model_usage: Dict[str, int]
    peak_usage_times: List[datetime]
    session_analyses: List[SessionAnalysis]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'analysis_period': [
                self.analysis_period[0].isoformat(),
                self.analysis_period[1].isoformat()
            ],
            'total_sessions': self.total_sessions,
            'successful_sessions': self.successful_sessions,
            'failed_sessions': self.failed_sessions,
            'incomplete_sessions': self.incomplete_sessions,
            'average_duration': self.average_duration,
            'error_patterns': self.error_patterns,
            'performance_trends': self.performance_trends,
            'model_usage': self.model_usage,
            'peak_usage_times': [t.isoformat() for t in self.peak_usage_times],
            'session_analyses': [s.to_dict() for s in self.session_analyses]
        }


class LogAnalyzer:
    """Comprehensive log analysis and management system."""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the log analyzer.
        
        Args:
            log_dir: Directory containing log files
        """
        self.log_dir = Path(log_dir)
        self.logger = get_logger()
    
    def analyze_logs(self, 
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    log_types: Optional[List[str]] = None) -> LogAnalysisReport:
        """
        Perform comprehensive log analysis.
        
        Args:
            start_time: Start of analysis period
            end_time: End of analysis period
            log_types: Types of logs to analyze ('generation', 'errors', 'performance')
            
        Returns:
            Comprehensive analysis report
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=7)  # Last week by default
        if end_time is None:
            end_time = datetime.now()
        if log_types is None:
            log_types = ['generation', 'errors', 'performance']
        
        # Collect all log entries in the time period
        all_entries = []
        for log_type in log_types:
            entries = self._read_log_entries(log_type, start_time, end_time)
            all_entries.extend(entries)
        
        # Sort by timestamp
        all_entries.sort(key=lambda x: x.timestamp)
        
        # Group entries by session
        sessions = self._group_by_session(all_entries)
        
        # Analyze each session
        session_analyses = []
        for session_id, entries in sessions.items():
            analysis = self._analyze_session(session_id, entries)
            session_analyses.append(analysis)
        
        # Generate overall statistics
        total_sessions = len(session_analyses)
        successful_sessions = sum(1 for s in session_analyses if s.status == 'success')
        failed_sessions = sum(1 for s in session_analyses if s.status == 'error')
        incomplete_sessions = sum(1 for s in session_analyses if s.status == 'incomplete')
        
        # Calculate average duration
        durations = [s.duration for s in session_analyses if s.duration is not None]
        average_duration = sum(durations) / len(durations) if durations else 0.0
        
        # Analyze error patterns
        error_patterns = self._analyze_error_patterns(session_analyses)
        
        # Analyze performance trends
        performance_trends = self._analyze_performance_trends(session_analyses)
        
        # Analyze model usage
        model_usage = Counter(s.model_type for s in session_analyses if s.model_type)
        
        # Find peak usage times
        peak_usage_times = self._find_peak_usage_times(all_entries)
        
        return LogAnalysisReport(
            analysis_period=(start_time, end_time),
            total_sessions=total_sessions,
            successful_sessions=successful_sessions,
            failed_sessions=failed_sessions,
            incomplete_sessions=incomplete_sessions,
            average_duration=average_duration,
            error_patterns=dict(error_patterns),
            performance_trends=performance_trends,
            model_usage=dict(model_usage),
            peak_usage_times=peak_usage_times,
            session_analyses=session_analyses
        )
    
    def _read_log_entries(self, 
                         log_type: str, 
                         start_time: datetime, 
                         end_time: datetime) -> List[LogEntry]:
        """Read log entries from a specific log file within time range."""
        log_file = self.log_dir / f"{log_type}.log"
        entries = []
        
        if not log_file.exists():
            return entries
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = LogEntry.parse(line)
                    if entry and start_time <= entry.timestamp <= end_time:
                        entries.append(entry)
        except Exception as e:
            self.logger.error_logger.error(f"Error reading log file {log_file}: {e}")
        
        return entries
    
    def _group_by_session(self, entries: List[LogEntry]) -> Dict[str, List[LogEntry]]:
        """Group log entries by session ID."""
        sessions = defaultdict(list)
        
        for entry in entries:
            if entry.session_id:
                sessions[entry.session_id].append(entry)
            else:
                # Try to extract session ID from message
                session_match = re.search(r'session ([a-f0-9-]{36})', entry.message.lower())
                if session_match:
                    session_id = session_match.group(1)
                    entry.session_id = session_id
                    sessions[session_id].append(entry)
                else:
                    # Group under 'unknown' session
                    sessions['unknown'].append(entry)
        
        return dict(sessions)
    
    def _analyze_session(self, session_id: str, entries: List[LogEntry]) -> SessionAnalysis:
        """Analyze a single generation session."""
        # Sort entries by timestamp
        entries.sort(key=lambda x: x.timestamp)
        
        start_time = entries[0].timestamp if entries else None
        end_time = entries[-1].timestamp if entries else None
        duration = (end_time - start_time).total_seconds() if start_time and end_time else None
        
        # Count errors and warnings
        error_count = sum(1 for e in entries if e.level == 'ERROR')
        warning_count = sum(1 for e in entries if e.level == 'WARNING')
        
        # Determine status
        status = 'incomplete'
        if any('completed successfully' in e.message.lower() for e in entries):
            status = 'success'
        elif error_count > 0 or any('failed' in e.message.lower() for e in entries):
            status = 'error'
        
        # Extract model type and generation mode
        model_type = None
        generation_mode = None
        for entry in entries:
            if 'model:' in entry.message.lower():
                model_match = re.search(r'model:\s*([^,\s]+)', entry.message.lower())
                if model_match:
                    model_type = model_match.group(1)
            
            if 'mode:' in entry.message.lower():
                mode_match = re.search(r'mode:\s*([^,\s]+)', entry.message.lower())
                if mode_match:
                    generation_mode = mode_match.group(1)
        
        # Extract completed stages
        stages_completed = []
        stage_patterns = [
            'validation', 'model loading', 'preprocessing', 
            'generation', 'postprocessing', 'saving'
        ]
        for stage in stage_patterns:
            if any(stage in e.message.lower() for e in entries):
                stages_completed.append(stage)
        
        # Collect error messages
        errors = [e.message for e in entries if e.level == 'ERROR']
        
        # Extract performance metrics
        performance_metrics = {}
        for entry in entries:
            if 'duration:' in entry.message.lower():
                duration_match = re.search(r'duration:\s*([\d.]+)', entry.message.lower())
                if duration_match:
                    performance_metrics['generation_duration'] = float(duration_match.group(1))
            
            if 'vram' in entry.message.lower():
                vram_match = re.search(r'vram.*?(\d+\.?\d*)\s*gb', entry.message.lower())
                if vram_match:
                    performance_metrics['vram_usage'] = float(vram_match.group(1))
        
        return SessionAnalysis(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            status=status,
            model_type=model_type,
            generation_mode=generation_mode,
            error_count=error_count,
            warning_count=warning_count,
            stages_completed=stages_completed,
            errors=errors,
            performance_metrics=performance_metrics
        )
    
    def _analyze_error_patterns(self, sessions: List[SessionAnalysis]) -> Counter:
        """Analyze common error patterns across sessions."""
        error_patterns = Counter()
        
        for session in sessions:
            for error in session.errors:
                # Categorize errors by common patterns
                error_lower = error.lower()
                
                if 'cuda' in error_lower or 'gpu' in error_lower:
                    error_patterns['GPU/CUDA Error'] += 1
                elif 'memory' in error_lower or 'vram' in error_lower:
                    error_patterns['Memory Error'] += 1
                elif 'model' in error_lower and ('load' in error_lower or 'not found' in error_lower):
                    error_patterns['Model Loading Error'] += 1
                elif 'validation' in error_lower or 'invalid' in error_lower:
                    error_patterns['Input Validation Error'] += 1
                elif 'timeout' in error_lower:
                    error_patterns['Timeout Error'] += 1
                elif 'permission' in error_lower or 'access' in error_lower:
                    error_patterns['File Access Error'] += 1
                else:
                    error_patterns['Other Error'] += 1
        
        return error_patterns
    
    def _analyze_performance_trends(self, sessions: List[SessionAnalysis]) -> Dict[str, List[float]]:
        """Analyze performance trends over time."""
        trends = {
            'generation_duration': [],
            'vram_usage': [],
            'success_rate_by_hour': []
        }
        
        # Collect performance metrics
        for session in sessions:
            if 'generation_duration' in session.performance_metrics:
                trends['generation_duration'].append(session.performance_metrics['generation_duration'])
            
            if 'vram_usage' in session.performance_metrics:
                trends['vram_usage'].append(session.performance_metrics['vram_usage'])
        
        # Calculate success rate by hour
        hourly_stats = defaultdict(lambda: {'total': 0, 'success': 0})
        for session in sessions:
            if session.start_time:
                hour = session.start_time.hour
                hourly_stats[hour]['total'] += 1
                if session.status == 'success':
                    hourly_stats[hour]['success'] += 1
        
        for hour in range(24):
            if hourly_stats[hour]['total'] > 0:
                success_rate = hourly_stats[hour]['success'] / hourly_stats[hour]['total']
                trends['success_rate_by_hour'].append(success_rate)
            else:
                trends['success_rate_by_hour'].append(0.0)
        
        return trends
    
    def _find_peak_usage_times(self, entries: List[LogEntry]) -> List[datetime]:
        """Find peak usage times based on log entry frequency."""
        # Group entries by hour
        hourly_counts = defaultdict(int)
        for entry in entries:
            hour_key = entry.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour_key] += 1
        
        # Find hours with above-average activity
        if not hourly_counts:
            return []
        
        average_count = sum(hourly_counts.values()) / len(hourly_counts)
        peak_times = [
            hour for hour, count in hourly_counts.items()
            if count > average_count * 1.5  # 50% above average
        ]
        
        return sorted(peak_times)
    
    def find_error_correlations(self, 
                               sessions: List[SessionAnalysis]) -> Dict[str, Dict[str, float]]:
        """Find correlations between different types of errors."""
        correlations = {}
        
        # Group sessions by error types
        error_sessions = defaultdict(list)
        for session in sessions:
            for error in session.errors:
                error_type = self._categorize_error(error)
                error_sessions[error_type].append(session.session_id)
        
        # Calculate correlations
        error_types = list(error_sessions.keys())
        for i, error_type1 in enumerate(error_types):
            correlations[error_type1] = {}
            for j, error_type2 in enumerate(error_types):
                if i != j:
                    # Calculate Jaccard similarity
                    set1 = set(error_sessions[error_type1])
                    set2 = set(error_sessions[error_type2])
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    correlation = intersection / union if union > 0 else 0.0
                    correlations[error_type1][error_type2] = correlation
        
        return correlations
    
    def _categorize_error(self, error: str) -> str:
        """Categorize an error message into a type."""
        error_lower = error.lower()
        
        if 'cuda' in error_lower or 'gpu' in error_lower:
            return 'GPU/CUDA Error'
        elif 'memory' in error_lower or 'vram' in error_lower:
            return 'Memory Error'
        elif 'model' in error_lower and ('load' in error_lower or 'not found' in error_lower):
            return 'Model Loading Error'
        elif 'validation' in error_lower or 'invalid' in error_lower:
            return 'Input Validation Error'
        elif 'timeout' in error_lower:
            return 'Timeout Error'
        elif 'permission' in error_lower or 'access' in error_lower:
            return 'File Access Error'
        else:
            return 'Other Error'
    
    def export_analysis_report(self, 
                              report: LogAnalysisReport, 
                              output_path: str,
                              format: str = 'json') -> str:
        """
        Export analysis report to file.
        
        Args:
            report: Analysis report to export
            output_path: Path to save the report
            format: Output format ('json' or 'html')
            
        Returns:
            Path to the exported file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
        elif format.lower() == 'html':
            self._export_html_report(report, output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(output_file)
    
    def _export_html_report(self, report: LogAnalysisReport, output_file: Path):
        """Export analysis report as HTML."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>WAN2.2 Log Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 5px; }}
                .error-list {{ background-color: #ffe6e6; padding: 10px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>WAN2.2 Video Generation Log Analysis Report</h1>
                <p>Analysis Period: {report.analysis_period[0].strftime('%Y-%m-%d %H:%M')} to {report.analysis_period[1].strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            
            <div class="section">
                <h2>Summary Statistics</h2>
                <div class="metric">Total Sessions: {report.total_sessions}</div>
                <div class="metric">Successful: {report.successful_sessions}</div>
                <div class="metric">Failed: {report.failed_sessions}</div>
                <div class="metric">Incomplete: {report.incomplete_sessions}</div>
                <div class="metric">Average Duration: {report.average_duration:.2f}s</div>
            </div>
            
            <div class="section">
                <h2>Error Patterns</h2>
                <table>
                    <tr><th>Error Type</th><th>Count</th></tr>
        """
        
        for error_type, count in report.error_patterns.items():
            html_content += f"<tr><td>{error_type}</td><td>{count}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Model Usage</h2>
                <table>
                    <tr><th>Model Type</th><th>Usage Count</th></tr>
        """
        
        for model_type, count in report.model_usage.items():
            html_content += f"<tr><td>{model_type}</td><td>{count}</td></tr>"
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def compress_old_logs(self, days_to_keep: int = 7):
        """Compress log files older than specified days."""
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)
        
        for log_file in self.log_dir.glob('*.log'):
            try:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_time:
                    # Compress the file
                    compressed_file = log_file.with_suffix('.log.gz')
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(compressed_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Remove original file
                    log_file.unlink()
                    
                    self.logger.generation_logger.info(f"Compressed old log file: {log_file}")
            except Exception as e:
                self.logger.error_logger.error(f"Error compressing log file {log_file}: {e}")


# Global log analyzer instance
_analyzer_instance = None


def get_log_analyzer() -> LogAnalyzer:
    """Get the global log analyzer instance."""
    global _analyzer_instance
    
    if _analyzer_instance is None:
        _analyzer_instance = LogAnalyzer()
    
    return _analyzer_instance
