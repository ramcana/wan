"""
Technical Debt Tracking and Prioritization System

This module provides comprehensive technical debt tracking, analysis, and prioritization
to help teams manage and reduce technical debt systematically.
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from enum import Enum
import hashlib
import statistics


class DebtCategory(Enum):
    """Categories of technical debt"""
    CODE_QUALITY = "code_quality"
    ARCHITECTURE = "architecture"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    SCALABILITY = "scalability"


class DebtSeverity(Enum):
    """Severity levels for technical debt"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DebtStatus(Enum):
    """Status of technical debt items"""
    IDENTIFIED = "identified"
    ACKNOWLEDGED = "acknowledged"
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    DEFERRED = "deferred"
    WONT_FIX = "wont_fix"


@dataclass
class TechnicalDebtItem:
    """Represents a technical debt item"""
    id: str
    title: str
    description: str
    file_path: str
    line_start: int
    line_end: int
    category: DebtCategory
    severity: DebtSeverity
    status: DebtStatus
    created_date: datetime
    updated_date: datetime
    estimated_effort_hours: float
    business_impact: str
    technical_impact: str
    priority_score: float
    assignee: Optional[str] = None
    due_date: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    related_items: List[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.related_items is None:
            self.related_items = []
        if self.tags is None:
            self.tags = []


@dataclass
class DebtMetrics:
    """Technical debt metrics"""
    total_items: int
    total_estimated_hours: float
    items_by_category: Dict[str, int]
    items_by_severity: Dict[str, int]
    items_by_status: Dict[str, int]
    average_age_days: float
    oldest_item_days: float
    debt_trend: str  # increasing, decreasing, stable
    resolution_rate: float  # items resolved per week


@dataclass
class DebtRecommendation:
    """Recommendation for addressing technical debt"""
    debt_item_id: str
    recommendation_type: str
    description: str
    rationale: str
    estimated_impact: str
    suggested_timeline: str
    prerequisites: List[str]
    resources_needed: List[str]


class TechnicalDebtTracker:
    """Main technical debt tracking system"""
    
    def __init__(self, project_root: str = ".", db_path: str = "technical_debt.db"):
        self.project_root = Path(project_root)
        self.db_path = db_path
        self.debt_items: List[TechnicalDebtItem] = []
        self._init_database()
        self._load_debt_items()
    
    def _init_database(self):
        """Initialize SQLite database for debt tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS technical_debt (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                file_path TEXT,
                line_start INTEGER,
                line_end INTEGER,
                category TEXT,
                severity TEXT,
                status TEXT,
                created_date TEXT,
                updated_date TEXT,
                estimated_effort_hours REAL,
                business_impact TEXT,
                technical_impact TEXT,
                priority_score REAL,
                assignee TEXT,
                due_date TEXT,
                resolution_notes TEXT,
                related_items TEXT,
                tags TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS debt_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                debt_item_id TEXT,
                action TEXT,
                old_status TEXT,
                new_status TEXT,
                timestamp TEXT,
                notes TEXT,
                FOREIGN KEY (debt_item_id) REFERENCES technical_debt (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_debt_items(self):
        """Load debt items from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM technical_debt')
        rows = cursor.fetchall()
        
        self.debt_items = []
        for row in rows:
            item = TechnicalDebtItem(
                id=row[0],
                title=row[1],
                description=row[2],
                file_path=row[3],
                line_start=row[4],
                line_end=row[5],
                category=DebtCategory(row[6]),
                severity=DebtSeverity(row[7]),
                status=DebtStatus(row[8]),
                created_date=datetime.fromisoformat(row[9]),
                updated_date=datetime.fromisoformat(row[10]),
                estimated_effort_hours=row[11],
                business_impact=row[12],
                technical_impact=row[13],
                priority_score=row[14],
                assignee=row[15],
                due_date=datetime.fromisoformat(row[16]) if row[16] else None,
                resolution_notes=row[17],
                related_items=json.loads(row[18]) if row[18] else [],
                tags=json.loads(row[19]) if row[19] else []
            )
            self.debt_items.append(item)
        
        conn.close()
    
    def add_debt_item(self, item: TechnicalDebtItem) -> str:
        """Add a new technical debt item"""
        # Generate ID if not provided
        if not item.id:
            item.id = self._generate_debt_id(item)
        
        # Set timestamps
        now = datetime.now()
        item.created_date = now
        item.updated_date = now
        
        # Calculate priority score if not set
        if item.priority_score == 0:
            item.priority_score = self._calculate_priority_score(item)
        
        # Save to database
        self._save_debt_item(item)
        
        # Add to memory
        self.debt_items.append(item)
        
        # Log action
        self._log_debt_action(item.id, "created", None, item.status.value, "Debt item created")
        
        return item.id
    
    def update_debt_item(self, item_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing debt item"""
        item = self.get_debt_item(item_id)
        if not item:
            return False
        
        old_status = item.status.value
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(item, key):
                if key in ['category', 'severity', 'status'] and isinstance(value, str):
                    # Convert string to enum
                    enum_class = {
                        'category': DebtCategory,
                        'severity': DebtSeverity,
                        'status': DebtStatus
                    }[key]
                    value = enum_class(value)
                setattr(item, key, value)
        
        # Update timestamp
        item.updated_date = datetime.now()
        
        # Recalculate priority score
        item.priority_score = self._calculate_priority_score(item)
        
        # Save to database
        self._save_debt_item(item)
        
        # Log action if status changed
        if old_status != item.status.value:
            self._log_debt_action(item_id, "status_changed", old_status, item.status.value, 
                                f"Status changed from {old_status} to {item.status.value}")
        
        return True
    
    def resolve_debt_item(self, item_id: str, resolution_notes: str) -> bool:
        """Mark a debt item as resolved"""
        return self.update_debt_item(item_id, {
            'status': DebtStatus.RESOLVED,
            'resolution_notes': resolution_notes
        })
    
    def get_debt_item(self, item_id: str) -> Optional[TechnicalDebtItem]:
        """Get a debt item by ID"""
        for item in self.debt_items:
            if item.id == item_id:
                return item
        return None
    
    def get_debt_items_by_file(self, file_path: str) -> List[TechnicalDebtItem]:
        """Get all debt items for a specific file"""
        return [item for item in self.debt_items if item.file_path == file_path]
    
    def get_debt_items_by_category(self, category: DebtCategory) -> List[TechnicalDebtItem]:
        """Get all debt items in a specific category"""
        return [item for item in self.debt_items if item.category == category]
    
    def get_debt_items_by_severity(self, severity: DebtSeverity) -> List[TechnicalDebtItem]:
        """Get all debt items with specific severity"""
        return [item for item in self.debt_items if item.severity == severity]
    
    def get_prioritized_debt_items(self, limit: int = None) -> List[TechnicalDebtItem]:
        """Get debt items sorted by priority score"""
        sorted_items = sorted(self.debt_items, key=lambda x: x.priority_score, reverse=True)
        return sorted_items[:limit] if limit else sorted_items
    
    def calculate_debt_metrics(self) -> DebtMetrics:
        """Calculate comprehensive debt metrics"""
        if not self.debt_items:
            return DebtMetrics(0, 0, {}, {}, {}, 0, 0, "stable", 0)
        
        # Basic counts
        total_items = len(self.debt_items)
        total_hours = sum(item.estimated_effort_hours for item in self.debt_items)
        
        # Category breakdown
        category_counts = {}
        for item in self.debt_items:
            category_counts[item.category.value] = category_counts.get(item.category.value, 0) + 1
        
        # Severity breakdown
        severity_counts = {}
        for item in self.debt_items:
            severity_counts[item.severity.value] = severity_counts.get(item.severity.value, 0) + 1
        
        # Status breakdown
        status_counts = {}
        for item in self.debt_items:
            status_counts[item.status.value] = status_counts.get(item.status.value, 0) + 1
        
        # Age calculations
        now = datetime.now()
        ages = [(now - item.created_date).days for item in self.debt_items]
        average_age = statistics.mean(ages) if ages else 0
        oldest_age = max(ages) if ages else 0
        
        # Trend analysis (simplified)
        debt_trend = self._analyze_debt_trend()
        
        # Resolution rate (items resolved in last 30 days)
        thirty_days_ago = now - timedelta(days=30)
        recent_resolutions = len([
            item for item in self.debt_items 
            if item.status == DebtStatus.RESOLVED and item.updated_date >= thirty_days_ago
        ])
        resolution_rate = recent_resolutions / 4.3  # per week
        
        return DebtMetrics(
            total_items=total_items,
            total_estimated_hours=total_hours,
            items_by_category=category_counts,
            items_by_severity=severity_counts,
            items_by_status=status_counts,
            average_age_days=average_age,
            oldest_item_days=oldest_age,
            debt_trend=debt_trend,
            resolution_rate=resolution_rate
        )
    
    def generate_recommendations(self) -> List[DebtRecommendation]:
        """Generate recommendations for addressing technical debt"""
        recommendations = []
        
        # High-priority items
        high_priority_items = [
            item for item in self.debt_items 
            if item.priority_score > 8 and item.status in [DebtStatus.IDENTIFIED, DebtStatus.ACKNOWLEDGED]
        ]
        
        for item in high_priority_items[:5]:  # Top 5
            recommendations.append(DebtRecommendation(
                debt_item_id=item.id,
                recommendation_type="immediate_action",
                description=f"Address high-priority {item.category.value} issue",
                rationale=f"High priority score ({item.priority_score:.1f}) and {item.severity.value} severity",
                estimated_impact="High - will significantly improve code quality",
                suggested_timeline="Within 2 weeks",
                prerequisites=["Code review", "Testing plan"],
                resources_needed=["Senior developer", f"{item.estimated_effort_hours} hours"]
            ))
        
        # Category-based recommendations
        category_counts = {}
        for item in self.debt_items:
            if item.status not in [DebtStatus.RESOLVED, DebtStatus.WONT_FIX]:
                category_counts[item.category] = category_counts.get(item.category, 0) + 1
        
        # Recommend focusing on categories with most items
        if category_counts:
            top_category = max(category_counts.items(), key=lambda x: x[1])
            recommendations.append(DebtRecommendation(
                debt_item_id="category_focus",
                recommendation_type="category_focus",
                description=f"Focus on {top_category[0].value} issues",
                rationale=f"Highest concentration of debt items ({top_category[1]} items)",
                estimated_impact="Medium - will improve overall code quality in this area",
                suggested_timeline="Next sprint",
                prerequisites=["Team discussion", "Priority alignment"],
                resources_needed=["Team effort", "Dedicated time"]
            ))
        
        return recommendations
    
    def _generate_debt_id(self, item: TechnicalDebtItem) -> str:
        """Generate unique ID for debt item"""
        content = f"{item.file_path}:{item.line_start}:{item.title}:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _calculate_priority_score(self, item: TechnicalDebtItem) -> float:
        """Calculate priority score for debt item"""
        score = 0.0
        
        # Severity weight
        severity_weights = {
            DebtSeverity.CRITICAL: 10,
            DebtSeverity.HIGH: 7,
            DebtSeverity.MEDIUM: 4,
            DebtSeverity.LOW: 1
        }
        score += severity_weights.get(item.severity, 1)
        
        # Category weight
        category_weights = {
            DebtCategory.SECURITY: 3,
            DebtCategory.PERFORMANCE: 2.5,
            DebtCategory.ARCHITECTURE: 2,
            DebtCategory.MAINTAINABILITY: 1.5,
            DebtCategory.CODE_QUALITY: 1.2,
            DebtCategory.TESTING: 1,
            DebtCategory.DOCUMENTATION: 0.8,
            DebtCategory.SCALABILITY: 1.5
        }
        score *= category_weights.get(item.category, 1)
        
        # Age factor (older items get higher priority)
        age_days = (datetime.now() - item.created_date).days
        age_factor = min(2.0, 1 + (age_days / 365))  # Max 2x multiplier for items over 1 year
        score *= age_factor
        
        # Effort factor (prefer items with reasonable effort)
        if item.estimated_effort_hours <= 8:  # 1 day or less
            score *= 1.5
        elif item.estimated_effort_hours <= 40:  # 1 week or less
            score *= 1.2
        elif item.estimated_effort_hours > 160:  # More than 1 month
            score *= 0.8
        
        return round(score, 2)
    
    def _save_debt_item(self, item: TechnicalDebtItem):
        """Save debt item to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO technical_debt VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        ''', (
            item.id,
            item.title,
            item.description,
            item.file_path,
            item.line_start,
            item.line_end,
            item.category.value,
            item.severity.value,
            item.status.value,
            item.created_date.isoformat(),
            item.updated_date.isoformat(),
            item.estimated_effort_hours,
            item.business_impact,
            item.technical_impact,
            item.priority_score,
            item.assignee,
            item.due_date.isoformat() if item.due_date else None,
            item.resolution_notes,
            json.dumps(item.related_items),
            json.dumps(item.tags)
        ))
        
        conn.commit()
        conn.close()
    
    def _log_debt_action(self, debt_item_id: str, action: str, old_status: str, new_status: str, notes: str):
        """Log debt item action to history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO debt_history (debt_item_id, action, old_status, new_status, timestamp, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (debt_item_id, action, old_status, new_status, datetime.now().isoformat(), notes))
        
        conn.commit()
        conn.close()
    
    def _analyze_debt_trend(self) -> str:
        """Analyze debt trend over time"""
        # Simplified trend analysis
        now = datetime.now()
        thirty_days_ago = now - timedelta(days=30)
        
        recent_items = len([
            item for item in self.debt_items 
            if item.created_date >= thirty_days_ago
        ])
        
        recent_resolutions = len([
            item for item in self.debt_items 
            if item.status == DebtStatus.RESOLVED and item.updated_date >= thirty_days_ago
        ])
        
        if recent_items > recent_resolutions * 1.5:
            return "increasing"
        elif recent_resolutions > recent_items * 1.5:
            return "decreasing"
        else:
            return "stable"
    
    def export_debt_report(self, output_path: str = "technical_debt_report.json") -> Dict[str, Any]:
        """Export comprehensive debt report"""
        metrics = self.calculate_debt_metrics()
        recommendations = self.generate_recommendations()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "metrics": asdict(metrics),
            "recommendations": [asdict(rec) for rec in recommendations],
            "debt_items": [
                {
                    "id": item.id,
                    "title": item.title,
                    "description": item.description,
                    "file_path": item.file_path,
                    "line_start": item.line_start,
                    "line_end": item.line_end,
                    "category": item.category.value,
                    "severity": item.severity.value,
                    "status": item.status.value,
                    "created_date": item.created_date.isoformat(),
                    "updated_date": item.updated_date.isoformat(),
                    "estimated_effort_hours": item.estimated_effort_hours,
                    "business_impact": item.business_impact,
                    "technical_impact": item.technical_impact,
                    "priority_score": item.priority_score,
                    "assignee": item.assignee,
                    "due_date": item.due_date.isoformat() if item.due_date else None,
                    "resolution_notes": item.resolution_notes,
                    "related_items": item.related_items,
                    "tags": item.tags
                }
                for item in self.debt_items
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
