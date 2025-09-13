"""
Maintenance operation logging and audit trail system.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import getpass
import os

try:
    from tools.maintenance-reporter.models import (
        MaintenanceOperation, MaintenanceAuditTrail, MaintenanceOperationType,
        MaintenanceStatus, ImpactLevel
    )
except ImportError:
    from models import (
        MaintenanceOperation, MaintenanceAuditTrail, MaintenanceOperationType,
        MaintenanceStatus, ImpactLevel
    )


class OperationLogger:
    """Logs and tracks maintenance operations with detailed audit trails."""
    
    def __init__(self, data_dir: str = "data/maintenance-operations"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.operations_file = self.data_dir / "operations.json"
        self.audit_trail_file = self.data_dir / "audit_trail.json"
        
        self.operations: Dict[str, MaintenanceOperation] = {}
        self.audit_trail: List[MaintenanceAuditTrail] = []
        
        self._load_data()
    
    def _load_data(self):
        """Load existing operations and audit trail."""
        # Load operations
        if self.operations_file.exists():
            try:
                with open(self.operations_file) as f:
                    data = json.load(f)
                
                for op_data in data.get('operations', []):
                    operation = MaintenanceOperation(
                        id=op_data['id'],
                        operation_type=MaintenanceOperationType(op_data['operation_type']),
                        title=op_data['title'],
                        description=op_data['description'],
                        status=MaintenanceStatus(op_data['status']),
                        impact_level=ImpactLevel(op_data['impact_level']),
                        started_at=datetime.fromisoformat(op_data['started_at']),
                        completed_at=datetime.fromisoformat(op_data['completed_at']) if op_data.get('completed_at') else None,
                        duration_seconds=op_data.get('duration_seconds'),
                        success_metrics=op_data.get('success_metrics', {}),
                        error_details=op_data.get('error_details'),
                        rollback_info=op_data.get('rollback_info'),
                        files_affected=op_data.get('files_affected', []),
                        components_affected=op_data.get('components_affected', [])
                    )
                    self.operations[operation.id] = operation
            
            except Exception as e:
                print(f"Error loading operations: {e}")
        
        # Load audit trail
        if self.audit_trail_file.exists():
            try:
                with open(self.audit_trail_file) as f:
                    data = json.load(f)
                
                for trail_data in data.get('audit_trail', []):
                    trail = MaintenanceAuditTrail(
                        operation_id=trail_data['operation_id'],
                        timestamp=datetime.fromisoformat(trail_data['timestamp']),
                        action=trail_data['action'],
                        user=trail_data['user'],
                        details=trail_data['details'],
                        before_state=trail_data.get('before_state'),
                        after_state=trail_data.get('after_state')
                    )
                    self.audit_trail.append(trail)
            
            except Exception as e:
                print(f"Error loading audit trail: {e}")
    
    def _save_data(self):
        """Save operations and audit trail to storage."""
        # Save operations
        operations_data = {
            'operations': [op.to_dict() for op in self.operations.values()],
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.operations_file, 'w') as f:
            json.dump(operations_data, f, indent=2)
        
        # Save audit trail
        audit_data = {
            'audit_trail': [trail.to_dict() for trail in self.audit_trail],
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.audit_trail_file, 'w') as f:
            json.dump(audit_data, f, indent=2)
    
    def start_operation(
        self,
        operation_type: MaintenanceOperationType,
        title: str,
        description: str,
        impact_level: ImpactLevel = ImpactLevel.MEDIUM,
        files_affected: Optional[List[str]] = None,
        components_affected: Optional[List[str]] = None
    ) -> str:
        """Start a new maintenance operation."""
        operation_id = str(uuid.uuid4())
        
        operation = MaintenanceOperation(
            id=operation_id,
            operation_type=operation_type,
            title=title,
            description=description,
            status=MaintenanceStatus.IN_PROGRESS,
            impact_level=impact_level,
            started_at=datetime.now(),
            files_affected=files_affected or [],
            components_affected=components_affected or []
        )
        
        self.operations[operation_id] = operation
        
        # Log audit trail
        self._log_audit_trail(
            operation_id=operation_id,
            action="operation_started",
            details={
                'operation_type': operation_type.value,
                'title': title,
                'impact_level': impact_level.value
            }
        )
        
        self._save_data()
        return operation_id
    
    def complete_operation(
        self,
        operation_id: str,
        success_metrics: Optional[Dict[str, Any]] = None,
        files_affected: Optional[List[str]] = None,
        components_affected: Optional[List[str]] = None
    ) -> bool:
        """Mark an operation as completed successfully."""
        if operation_id not in self.operations:
            return False
        
        operation = self.operations[operation_id]
        before_state = operation.to_dict()
        
        operation.status = MaintenanceStatus.COMPLETED
        operation.completed_at = datetime.now()
        operation.duration_seconds = int((operation.completed_at - operation.started_at).total_seconds())
        
        if success_metrics:
            operation.success_metrics.update(success_metrics)
        
        if files_affected:
            operation.files_affected.extend(files_affected)
        
        if components_affected:
            operation.components_affected.extend(components_affected)
        
        # Log audit trail
        self._log_audit_trail(
            operation_id=operation_id,
            action="operation_completed",
            details={
                'duration_seconds': operation.duration_seconds,
                'success_metrics': success_metrics or {}
            },
            before_state=before_state,
            after_state=operation.to_dict()
        )
        
        self._save_data()
        return True
    
    def fail_operation(
        self,
        operation_id: str,
        error_details: str,
        rollback_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Mark an operation as failed."""
        if operation_id not in self.operations:
            return False
        
        operation = self.operations[operation_id]
        before_state = operation.to_dict()
        
        operation.status = MaintenanceStatus.FAILED
        operation.completed_at = datetime.now()
        operation.duration_seconds = int((operation.completed_at - operation.started_at).total_seconds())
        operation.error_details = error_details
        operation.rollback_info = rollback_info
        
        # Log audit trail
        self._log_audit_trail(
            operation_id=operation_id,
            action="operation_failed",
            details={
                'error_details': error_details,
                'rollback_info': rollback_info or {}
            },
            before_state=before_state,
            after_state=operation.to_dict()
        )
        
        self._save_data()
        return True
    
    def rollback_operation(
        self,
        operation_id: str,
        rollback_details: Dict[str, Any]
    ) -> bool:
        """Rollback a completed or failed operation."""
        if operation_id not in self.operations:
            return False
        
        operation = self.operations[operation_id]
        before_state = operation.to_dict()
        
        operation.status = MaintenanceStatus.ROLLBACK
        operation.rollback_info = rollback_details
        
        # Log audit trail
        self._log_audit_trail(
            operation_id=operation_id,
            action="operation_rollback",
            details=rollback_details,
            before_state=before_state,
            after_state=operation.to_dict()
        )
        
        self._save_data()
        return True
    
    def update_operation_progress(
        self,
        operation_id: str,
        progress_details: Dict[str, Any]
    ) -> bool:
        """Update operation progress with additional details."""
        if operation_id not in self.operations:
            return False
        
        operation = self.operations[operation_id]
        
        # Update success metrics with progress
        operation.success_metrics.update(progress_details)
        
        # Log audit trail
        self._log_audit_trail(
            operation_id=operation_id,
            action="progress_update",
            details=progress_details
        )
        
        self._save_data()
        return True
    
    def _log_audit_trail(
        self,
        operation_id: str,
        action: str,
        details: Dict[str, Any],
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None
    ):
        """Log an audit trail entry."""
        user = getpass.getuser()
        
        trail = MaintenanceAuditTrail(
            operation_id=operation_id,
            timestamp=datetime.now(),
            action=action,
            user=user,
            details=details,
            before_state=before_state,
            after_state=after_state
        )
        
        self.audit_trail.append(trail)
    
    def get_operation(self, operation_id: str) -> Optional[MaintenanceOperation]:
        """Get a specific operation by ID."""
        return self.operations.get(operation_id)
    
    def get_operations_by_status(self, status: MaintenanceStatus) -> List[MaintenanceOperation]:
        """Get all operations with a specific status."""
        return [op for op in self.operations.values() if op.status == status]
    
    def get_operations_by_type(self, operation_type: MaintenanceOperationType) -> List[MaintenanceOperation]:
        """Get all operations of a specific type."""
        return [op for op in self.operations.values() if op.operation_type == operation_type]
    
    def get_operations_in_period(self, start_date: datetime, end_date: datetime) -> List[MaintenanceOperation]:
        """Get all operations within a specific time period."""
        return [
            op for op in self.operations.values()
            if start_date <= op.started_at <= end_date
        ]
    
    def get_audit_trail_for_operation(self, operation_id: str) -> List[MaintenanceAuditTrail]:
        """Get audit trail entries for a specific operation."""
        return [trail for trail in self.audit_trail if trail.operation_id == operation_id]
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get statistics about all operations."""
        total_operations = len(self.operations)
        
        if total_operations == 0:
            return {
                'total_operations': 0,
                'by_status': {},
                'by_type': {},
                'by_impact_level': {},
                'average_duration_minutes': 0,
                'success_rate': 0
            }
        
        # Count by status
        status_counts = {}
        for status in MaintenanceStatus:
            status_counts[status.value] = len(self.get_operations_by_status(status))
        
        # Count by type
        type_counts = {}
        for op_type in MaintenanceOperationType:
            type_counts[op_type.value] = len(self.get_operations_by_type(op_type))
        
        # Count by impact level
        impact_counts = {}
        for impact in ImpactLevel:
            impact_counts[impact.value] = len([
                op for op in self.operations.values() 
                if op.impact_level == impact
            ])
        
        # Calculate average duration
        completed_ops = [op for op in self.operations.values() if op.duration_seconds is not None]
        avg_duration_minutes = 0
        if completed_ops:
            avg_duration_minutes = sum(op.duration_seconds for op in completed_ops) / len(completed_ops) / 60
        
        # Calculate success rate
        completed_count = status_counts.get('completed', 0)
        failed_count = status_counts.get('failed', 0)
        total_finished = completed_count + failed_count
        success_rate = (completed_count / total_finished * 100) if total_finished > 0 else 0
        
        return {
            'total_operations': total_operations,
            'by_status': status_counts,
            'by_type': type_counts,
            'by_impact_level': impact_counts,
            'average_duration_minutes': avg_duration_minutes,
            'success_rate': success_rate,
            'total_audit_entries': len(self.audit_trail)
        }
    
    def cleanup_old_operations(self, days: int = 90) -> int:
        """Clean up old completed operations and audit trail entries."""
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Remove old completed operations
        old_operations = [
            op_id for op_id, op in self.operations.items()
            if op.status in [MaintenanceStatus.COMPLETED, MaintenanceStatus.FAILED, MaintenanceStatus.CANCELLED]
            and op.started_at < cutoff_date
        ]
        
        for op_id in old_operations:
            del self.operations[op_id]
        
        # Remove old audit trail entries
        old_trail_entries = [
            trail for trail in self.audit_trail
            if trail.timestamp < cutoff_date
        ]
        
        for trail in old_trail_entries:
            self.audit_trail.remove(trail)
        
        if old_operations or old_trail_entries:
            self._save_data()
        
        return len(old_operations) + len(old_trail_entries)
