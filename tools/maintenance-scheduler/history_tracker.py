"""
History tracking system for maintenance operations.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from threading import Lock

from models import MaintenanceTask, MaintenanceResult, MaintenanceHistory, MaintenanceMetrics


class MaintenanceHistoryTracker:
    """
    Tracks and manages maintenance operation history and metrics.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or "data/maintenance/history.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.history: List[MaintenanceHistory] = []
        self.history_lock = Lock()
        
        # Load existing history
        self._load_history()
        
        self.logger.info(f"HistoryTracker initialized with {len(self.history)} records")
    
    def record_execution(self, task: MaintenanceTask, result: MaintenanceResult) -> None:
        """Record a maintenance task execution."""
        history_record = MaintenanceHistory(
            task_id=task.id,
            result=result,
            triggered_by="scheduler",  # This could be parameterized
            environment="development",  # This could be detected
            git_commit=self._get_current_git_commit()
        )
        
        with self.history_lock:
            self.history.append(history_record)
            self._save_history()
        
        self.logger.info(
            f"Recorded execution: {task.name} - "
            f"Status: {result.status.value} - "
            f"Duration: {result.duration_seconds:.2f}s"
        )
    
    def get_task_history(self, task_id: str, limit: int = 50) -> List[MaintenanceHistory]:
        """Get execution history for a specific task."""
        task_history = [
            record for record in self.history
            if record.task_id == task_id
        ]
        
        # Sort by execution time (most recent first)
        task_history.sort(key=lambda x: x.result.started_at, reverse=True)
        
        return task_history[:limit]
    
    def get_recent_execution(self, task_id: str) -> Optional[MaintenanceHistory]:
        """Get the most recent execution for a task."""
        task_history = self.get_task_history(task_id, limit=1)
        return task_history[0] if task_history else None
    
    def get_history_by_date_range(self, start_date: datetime, 
                                 end_date: datetime) -> List[MaintenanceHistory]:
        """Get history records within a date range."""
        return [
            record for record in self.history
            if start_date <= record.result.started_at <= end_date
        ]
    
    def get_failed_executions(self, days: int = 7) -> List[MaintenanceHistory]:
        """Get failed executions from the last N days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        return [
            record for record in self.history
            if (record.result.started_at >= cutoff_date and 
                not record.result.success)
        ]
    
    def get_task_success_rate(self, task_id: str, days: int = 30) -> float:
        """Calculate success rate for a task over the last N days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_executions = [
            record for record in self.history
            if (record.task_id == task_id and 
                record.result.started_at >= cutoff_date)
        ]
        
        if not recent_executions:
            return 1.0  # Assume success if no recent history
        
        successful = sum(1 for record in recent_executions if record.result.success)
        return successful / len(recent_executions)
    
    def get_average_duration(self, task_id: str, days: int = 30) -> Optional[float]:
        """Get average execution duration for a task."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_executions = [
            record for record in self.history
            if (record.task_id == task_id and 
                record.result.started_at >= cutoff_date and
                record.result.duration_seconds > 0)
        ]
        
        if not recent_executions:
            return None
        
        total_duration = sum(record.result.duration_seconds for record in recent_executions)
        return total_duration / len(recent_executions)
    
    def get_maintenance_metrics(self, days: int = 30) -> MaintenanceMetrics:
        """Calculate comprehensive maintenance metrics."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_history = [
            record for record in self.history
            if record.result.started_at >= cutoff_date
        ]
        
        if not recent_history:
            return MaintenanceMetrics(
                period_start=cutoff_date,
                period_end=datetime.now()
            )
        
        # Calculate basic metrics
        total_tasks = len(recent_history)
        successful_tasks = sum(1 for record in recent_history if record.result.success)
        failed_tasks = total_tasks - successful_tasks
        
        # Calculate duration metrics
        durations = [
            record.result.duration_seconds for record in recent_history
            if record.result.duration_seconds > 0
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        
        # Calculate quality metrics
        total_issues_fixed = sum(record.result.issues_fixed for record in recent_history)
        total_files_modified = sum(record.result.files_modified for record in recent_history)
        total_lines_changed = sum(record.result.lines_changed for record in recent_history)
        
        quality_improvements = [
            record.result.quality_improvement for record in recent_history
            if record.result.quality_improvement > 0
        ]
        avg_quality_improvement = (
            sum(quality_improvements) / len(quality_improvements)
            if quality_improvements else 0.0
        )
        
        # Find last successful run
        successful_runs = [
            record for record in recent_history
            if record.result.success
        ]
        last_successful_run = None
        if successful_runs:
            last_successful_run = max(
                successful_runs,
                key=lambda x: x.result.started_at
            ).result.started_at
        
        # Calculate consecutive failures
        consecutive_failures = 0
        for record in reversed(recent_history):
            if record.result.success:
                break
            consecutive_failures += 1
        
        return MaintenanceMetrics(
            total_tasks_run=total_tasks,
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            average_duration_seconds=avg_duration,
            total_issues_fixed=total_issues_fixed,
            total_files_modified=total_files_modified,
            total_lines_changed=total_lines_changed,
            average_quality_improvement=avg_quality_improvement,
            last_successful_run=last_successful_run,
            consecutive_failures=consecutive_failures,
            period_start=cutoff_date,
            period_end=datetime.now()
        )
    
    def get_task_performance_trends(self, task_id: str, days: int = 90) -> Dict[str, List[float]]:
        """Get performance trends for a task over time."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        task_history = [
            record for record in self.history
            if (record.task_id == task_id and 
                record.result.started_at >= cutoff_date)
        ]
        
        # Sort by date
        task_history.sort(key=lambda x: x.result.started_at)
        
        trends = {
            'dates': [],
            'durations': [],
            'success_rates': [],
            'quality_improvements': [],
            'issues_fixed': []
        }
        
        # Group by week for trend analysis
        weekly_data = {}
        for record in task_history:
            week_start = record.result.started_at.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=record.result.started_at.weekday())
            
            if week_start not in weekly_data:
                weekly_data[week_start] = []
            weekly_data[week_start].append(record)
        
        # Calculate weekly metrics
        for week_start in sorted(weekly_data.keys()):
            week_records = weekly_data[week_start]
            
            trends['dates'].append(week_start.isoformat())
            
            # Average duration
            durations = [r.result.duration_seconds for r in week_records if r.result.duration_seconds > 0]
            trends['durations'].append(sum(durations) / len(durations) if durations else 0)
            
            # Success rate
            successes = sum(1 for r in week_records if r.result.success)
            trends['success_rates'].append(successes / len(week_records))
            
            # Average quality improvement
            improvements = [r.result.quality_improvement for r in week_records if r.result.quality_improvement > 0]
            trends['quality_improvements'].append(sum(improvements) / len(improvements) if improvements else 0)
            
            # Total issues fixed
            trends['issues_fixed'].append(sum(r.result.issues_fixed for r in week_records))
        
        return trends
    
    def cleanup_old_history(self, days_to_keep: int = 365) -> int:
        """Remove history records older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with self.history_lock:
            original_count = len(self.history)
            self.history = [
                record for record in self.history
                if record.result.started_at >= cutoff_date
            ]
            removed_count = original_count - len(self.history)
            
            if removed_count > 0:
                self._save_history()
        
        self.logger.info(f"Cleaned up {removed_count} old history records")
        return removed_count
    
    def export_history(self, output_path: str, task_id: Optional[str] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> None:
        """Export history to a file."""
        # Filter history based on parameters
        filtered_history = self.history
        
        if task_id:
            filtered_history = [r for r in filtered_history if r.task_id == task_id]
        
        if start_date:
            filtered_history = [r for r in filtered_history if r.result.started_at >= start_date]
        
        if end_date:
            filtered_history = [r for r in filtered_history if r.result.started_at <= end_date]
        
        # Prepare export data
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'filters': {
                'task_id': task_id,
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat() if end_date else None
            },
            'record_count': len(filtered_history),
            'history': [self._serialize_history_record(record) for record in filtered_history]
        }
        
        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported {len(filtered_history)} history records to {output_path}")
    
    def _load_history(self) -> None:
        """Load history from storage."""
        if not self.storage_path.exists():
            self.logger.info("No existing history file found")
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            for record_data in data.get('history', []):
                record = self._deserialize_history_record(record_data)
                if record:
                    self.history.append(record)
            
            self.logger.info(f"Loaded {len(self.history)} history records from storage")
            
        except Exception as e:
            self.logger.error(f"Error loading history: {e}", exc_info=True)
    
    def _save_history(self) -> None:
        """Save history to storage."""
        try:
            data = {
                'version': '1.0',
                'saved_at': datetime.now().isoformat(),
                'record_count': len(self.history),
                'history': [self._serialize_history_record(record) for record in self.history]
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Error saving history: {e}", exc_info=True)
    
    def _serialize_history_record(self, record: MaintenanceHistory) -> Dict:
        """Serialize a history record to dictionary."""
        return {
            'id': record.id,
            'task_id': record.task_id,
            'execution_id': record.execution_id,
            'result': {
                'task_id': record.result.task_id,
                'status': record.result.status.value,
                'started_at': record.result.started_at.isoformat(),
                'completed_at': record.result.completed_at.isoformat() if record.result.completed_at else None,
                'success': record.result.success,
                'output': record.result.output,
                'error_message': record.result.error_message,
                'duration_seconds': record.result.duration_seconds,
                'files_modified': record.result.files_modified,
                'lines_changed': record.result.lines_changed,
                'issues_fixed': record.result.issues_fixed,
                'impact_score': record.result.impact_score,
                'quality_improvement': record.result.quality_improvement,
                'rollback_available': record.result.rollback_available,
                'rollback_data': record.result.rollback_data,
                'log_file': record.result.log_file,
                'artifacts': record.result.artifacts
            },
            'triggered_by': record.triggered_by,
            'environment': record.environment,
            'git_commit': record.git_commit,
            'rolled_back': record.rolled_back,
            'rollback_reason': record.rollback_reason,
            'rollback_timestamp': record.rollback_timestamp.isoformat() if record.rollback_timestamp else None
        }
    
    def _deserialize_history_record(self, data: Dict) -> Optional[MaintenanceHistory]:
        """Deserialize a history record from dictionary."""
        try:
            result_data = data['result']
            result = MaintenanceResult(
                task_id=result_data['task_id'],
                status=result_data['status'],
                started_at=datetime.fromisoformat(result_data['started_at']),
                completed_at=datetime.fromisoformat(result_data['completed_at']) if result_data.get('completed_at') else None,
                success=result_data.get('success', False),
                output=result_data.get('output', ''),
                error_message=result_data.get('error_message'),
                duration_seconds=result_data.get('duration_seconds', 0.0),
                files_modified=result_data.get('files_modified', 0),
                lines_changed=result_data.get('lines_changed', 0),
                issues_fixed=result_data.get('issues_fixed', 0),
                impact_score=result_data.get('impact_score', 0.0),
                quality_improvement=result_data.get('quality_improvement', 0.0),
                rollback_available=result_data.get('rollback_available', False),
                rollback_data=result_data.get('rollback_data'),
                log_file=result_data.get('log_file'),
                artifacts=result_data.get('artifacts', [])
            )
            
            return MaintenanceHistory(
                id=data['id'],
                task_id=data['task_id'],
                execution_id=data['execution_id'],
                result=result,
                triggered_by=data.get('triggered_by', 'scheduler'),
                environment=data.get('environment', 'development'),
                git_commit=data.get('git_commit'),
                rolled_back=data.get('rolled_back', False),
                rollback_reason=data.get('rollback_reason'),
                rollback_timestamp=datetime.fromisoformat(data['rollback_timestamp']) if data.get('rollback_timestamp') else None
            )
            
        except Exception as e:
            self.logger.error(f"Error deserializing history record: {e}", exc_info=True)
            return None
    
    def _get_current_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None