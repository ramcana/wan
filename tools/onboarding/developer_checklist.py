#!/usr/bin/env python3
"""
Developer Checklist and Progress Tracking

This module provides a comprehensive checklist for new developers
and tracks their onboarding progress.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

@dataclass
class ChecklistItem:
    """Individual checklist item"""
    id: str
    title: str
    description: str
    category: str
    priority: str  # 'critical', 'important', 'optional'
    estimated_time: str  # e.g., "15 minutes", "1 hour"
    prerequisites: List[str]  # List of item IDs that must be completed first
    validation_command: Optional[str] = None
    documentation_link: Optional[str] = None
    completed: bool = False
    completed_at: Optional[str] = None
    notes: Optional[str] = None

@dataclass
class ChecklistProgress:
    """Overall checklist progress"""
    developer_name: str
    started_at: str
    last_updated: str
    total_items: int
    completed_items: int
    critical_completed: int
    critical_total: int
    completion_percentage: float
    estimated_time_remaining: str
    items: Dict[str, ChecklistItem]

class DeveloperChecklist:
    """Manages developer onboarding checklist and progress tracking"""
    
    def __init__(self, project_root: Optional[Path] = None, developer_name: Optional[str] = None):
        self.project_root = project_root or Path.cwd()
        self.developer_name = developer_name or os.getenv('USER', 'developer')
        self.logger = logging.getLogger(__name__)
        
        # Progress file location
        self.progress_file = self.project_root / ".kiro" / "onboarding" / f"{self.developer_name}_progress.json"
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize checklist items
        self.checklist_items = self._create_checklist_items()
        
        # Load existing progress
        self.progress = self._load_progress()
    
    def _create_checklist_items(self) -> Dict[str, ChecklistItem]:
        """Create the comprehensive developer checklist"""
        items = {}
        
        # Environment Setup (Critical)
        items['env_python'] = ChecklistItem(
            id='env_python',
            title='Install Python 3.8+',
            description='Install Python 3.8 or higher for backend development',
            category='Environment Setup',
            priority='critical',
            estimated_time='10 minutes',
            prerequisites=[],
            validation_command='python --version',
            documentation_link='tools/onboarding/docs/development-setup.md#python-environment'
        )
        
        items['env_nodejs'] = ChecklistItem(
            id='env_nodejs',
            title='Install Node.js 16+',
            description='Install Node.js 16 or higher for frontend development',
            category='Environment Setup',
            priority='critical',
            estimated_time='10 minutes',
            prerequisites=[],
            validation_command='node --version',
            documentation_link='tools/onboarding/docs/development-setup.md#nodejs-environment'
        )
        
        items['env_git'] = ChecklistItem(
            id='env_git',
            title='Install and Configure Git',
            description='Install Git and configure user name and email',
            category='Environment Setup',
            priority='critical',
            estimated_time='5 minutes',
            prerequisites=[],
            validation_command='git --version',
            documentation_link='tools/onboarding/docs/development-setup.md#git-configuration'
        )
        
        items['env_editor'] = ChecklistItem(
            id='env_editor',
            title='Set Up Code Editor',
            description='Install and configure VS Code or preferred editor with extensions',
            category='Environment Setup',
            priority='important',
            estimated_time='15 minutes',
            prerequisites=[],
            documentation_link='tools/onboarding/docs/development-setup.md#ide-configuration'
        )
        
        # Project Setup (Critical)
        items['project_clone'] = ChecklistItem(
            id='project_clone',
            title='Clone Repository',
            description='Clone the WAN22 repository to your local machine',
            category='Project Setup',
            priority='critical',
            estimated_time='5 minutes',
            prerequisites=['env_git'],
            validation_command='ls -la .git'
        )
        
        items['project_venv'] = ChecklistItem(
            id='project_venv',
            title='Create Virtual Environment',
            description='Create and activate Python virtual environment',
            category='Project Setup',
            priority='critical',
            estimated_time='5 minutes',
            prerequisites=['env_python', 'project_clone'],
            validation_command='python -c "import sys; print(sys.prefix)"'
        )
        
        items['project_backend_deps'] = ChecklistItem(
            id='project_backend_deps',
            title='Install Backend Dependencies',
            description='Install Python dependencies from requirements.txt',
            category='Project Setup',
            priority='critical',
            estimated_time='10 minutes',
            prerequisites=['project_venv'],
            validation_command='pip list | grep fastapi'
        )
        
        items['project_frontend_deps'] = ChecklistItem(
            id='project_frontend_deps',
            title='Install Frontend Dependencies',
            description='Install Node.js dependencies with npm install',
            category='Project Setup',
            priority='critical',
            estimated_time='5 minutes',
            prerequisites=['env_nodejs', 'project_clone'],
            validation_command='ls -la frontend/node_modules'
        )
        
        items['project_env_files'] = ChecklistItem(
            id='project_env_files',
            title='Create Environment Files',
            description='Create .env files for backend and frontend configuration',
            category='Project Setup',
            priority='critical',
            estimated_time='5 minutes',
            prerequisites=['project_clone'],
            validation_command='ls -la backend/.env frontend/.env'
        )
        
        # Development Tools (Important)
        items['tools_precommit'] = ChecklistItem(
            id='tools_precommit',
            title='Install Pre-commit Hooks',
            description='Install and configure pre-commit hooks for code quality',
            category='Development Tools',
            priority='important',
            estimated_time='5 minutes',
            prerequisites=['project_backend_deps'],
            validation_command='pre-commit --version'
        )
        
        items['tools_health_check'] = ChecklistItem(
            id='tools_health_check',
            title='Run Environment Health Check',
            description='Validate development environment setup',
            category='Development Tools',
            priority='important',
            estimated_time='5 minutes',
            prerequisites=['project_backend_deps'],
            validation_command='python tools/dev-environment/environment_validator.py --validate'
        )
        
        # Testing and Validation (Critical)
        items['test_backend'] = ChecklistItem(
            id='test_backend',
            title='Run Backend Tests',
            description='Execute backend test suite to verify setup',
            category='Testing',
            priority='critical',
            estimated_time='10 minutes',
            prerequisites=['project_backend_deps'],
            validation_command='python tools/test-runner/orchestrator.py --category unit'
        )
        
        items['test_frontend'] = ChecklistItem(
            id='test_frontend',
            title='Run Frontend Tests',
            description='Execute frontend test suite to verify setup',
            category='Testing',
            priority='critical',
            estimated_time='5 minutes',
            prerequisites=['project_frontend_deps'],
            validation_command='cd frontend && npm test -- --run'
        )
        
        items['test_servers'] = ChecklistItem(
            id='test_servers',
            title='Start Development Servers',
            description='Start both backend and frontend servers and verify they work',
            category='Testing',
            priority='critical',
            estimated_time='10 minutes',
            prerequisites=['project_backend_deps', 'project_frontend_deps', 'project_env_files'],
            validation_command='curl -f http://localhost:8000/health && curl -f http://localhost:3000'
        )
        
        # Learning and Documentation (Important)
        items['docs_overview'] = ChecklistItem(
            id='docs_overview',
            title='Read Project Overview',
            description='Read and understand the project architecture and structure',
            category='Learning',
            priority='important',
            estimated_time='30 minutes',
            prerequisites=['project_clone'],
            documentation_link='tools/onboarding/docs/project-overview.md'
        )
        
        items['docs_standards'] = ChecklistItem(
            id='docs_standards',
            title='Review Coding Standards',
            description='Read and understand the coding standards and best practices',
            category='Learning',
            priority='important',
            estimated_time='20 minutes',
            prerequisites=['docs_overview'],
            documentation_link='tools/onboarding/docs/coding-standards.md'
        )
        
        items['docs_workflow'] = ChecklistItem(
            id='docs_workflow',
            title='Learn Development Workflow',
            description='Understand the development workflow and contribution process',
            category='Learning',
            priority='important',
            estimated_time='15 minutes',
            prerequisites=['docs_standards'],
            documentation_link='tools/onboarding/docs/development-setup.md#development-workflow'
        )
        
        # First Contribution (Optional)
        items['contrib_explore'] = ChecklistItem(
            id='contrib_explore',
            title='Explore Codebase',
            description='Browse the codebase to understand the structure and components',
            category='First Contribution',
            priority='optional',
            estimated_time='45 minutes',
            prerequisites=['docs_overview', 'test_servers']
        )
        
        items['contrib_small_change'] = ChecklistItem(
            id='contrib_small_change',
            title='Make Small Change',
            description='Make a small change (fix typo, update comment, add test)',
            category='First Contribution',
            priority='optional',
            estimated_time='30 minutes',
            prerequisites=['contrib_explore', 'docs_standards']
        )
        
        items['contrib_first_pr'] = ChecklistItem(
            id='contrib_first_pr',
            title='Submit First Pull Request',
            description='Create and submit your first pull request',
            category='First Contribution',
            priority='optional',
            estimated_time='20 minutes',
            prerequisites=['contrib_small_change', 'tools_precommit']
        )
        
        # Advanced Setup (Optional)
        items['advanced_debug'] = ChecklistItem(
            id='advanced_debug',
            title='Set Up Debugging Tools',
            description='Configure debugging tools and learn how to use them',
            category='Advanced Setup',
            priority='optional',
            estimated_time='20 minutes',
            prerequisites=['test_servers'],
            validation_command='python tools/dev-feedback/debug_tools.py --help'
        )
        
        items['advanced_watchers'] = ChecklistItem(
            id='advanced_watchers',
            title='Configure File Watchers',
            description='Set up test and config watchers for fast feedback',
            category='Advanced Setup',
            priority='optional',
            estimated_time='15 minutes',
            prerequisites=['test_backend', 'test_frontend'],
            validation_command='python tools/dev-feedback/test_watcher.py --help'
        )
        
        items['advanced_performance'] = ChecklistItem(
            id='advanced_performance',
            title='Learn Performance Tools',
            description='Understand performance monitoring and optimization tools',
            category='Advanced Setup',
            priority='optional',
            estimated_time='25 minutes',
            prerequisites=['advanced_debug'],
            validation_command='python tools/health-checker/health_checker.py --help'
        )
        
        return items
    
    def _load_progress(self) -> ChecklistProgress:
        """Load existing progress or create new progress tracking"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                
                # Convert items back to ChecklistItem objects
                items = {}
                for item_id, item_data in data['items'].items():
                    items[item_id] = ChecklistItem(**item_data)
                
                # Update with any new items
                for item_id, item in self.checklist_items.items():
                    if item_id not in items:
                        items[item_id] = item
                
                progress = ChecklistProgress(
                    developer_name=data['developer_name'],
                    started_at=data['started_at'],
                    last_updated=data['last_updated'],
                    total_items=data['total_items'],
                    completed_items=data['completed_items'],
                    critical_completed=data['critical_completed'],
                    critical_total=data['critical_total'],
                    completion_percentage=data['completion_percentage'],
                    estimated_time_remaining=data['estimated_time_remaining'],
                    items=items
                )
                
                # Update progress metrics
                self._update_progress_metrics(progress)
                return progress
                
            except Exception as e:
                self.logger.warning(f"Failed to load progress file: {e}")
        
        # Create new progress
        return ChecklistProgress(
            developer_name=self.developer_name,
            started_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            total_items=len(self.checklist_items),
            completed_items=0,
            critical_completed=0,
            critical_total=len([item for item in self.checklist_items.values() if item.priority == 'critical']),
            completion_percentage=0.0,
            estimated_time_remaining="Unknown",
            items=self.checklist_items.copy()
        )
    
    def _save_progress(self):
        """Save current progress to file"""
        self.progress.last_updated = datetime.now().isoformat()
        self._update_progress_metrics(self.progress)
        
        # Convert to serializable format
        data = asdict(self.progress)
        
        with open(self.progress_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _update_progress_metrics(self, progress: ChecklistProgress):
        """Update progress metrics"""
        completed_items = len([item for item in progress.items.values() if item.completed])
        critical_completed = len([
            item for item in progress.items.values() 
            if item.completed and item.priority == 'critical'
        ])
        
        progress.completed_items = completed_items
        progress.critical_completed = critical_completed
        progress.completion_percentage = (completed_items / progress.total_items) * 100 if progress.total_items > 0 else 0
        
        # Estimate remaining time
        remaining_items = [item for item in progress.items.values() if not item.completed]
        if remaining_items:
            # Simple estimation based on average time per item
            total_minutes = sum(self._parse_time_estimate(item.estimated_time) for item in remaining_items)
            if total_minutes < 60:
                progress.estimated_time_remaining = f"{total_minutes} minutes"
            else:
                hours = total_minutes // 60
                minutes = total_minutes % 60
                progress.estimated_time_remaining = f"{hours}h {minutes}m"
        else:
            progress.estimated_time_remaining = "Complete!"
    
    def _parse_time_estimate(self, time_str: str) -> int:
        """Parse time estimate string to minutes"""
        try:
            if 'hour' in time_str:
                hours = int(time_str.split()[0])
                return hours * 60
            elif 'minute' in time_str:
                minutes = int(time_str.split()[0])
                return minutes
            else:
                return 30  # Default estimate
        except:
            return 30
    
    def complete_item(self, item_id: str, notes: Optional[str] = None) -> bool:
        """Mark an item as completed"""
        if item_id not in self.progress.items:
            self.logger.error(f"Item '{item_id}' not found in checklist")
            return False
        
        item = self.progress.items[item_id]
        
        # Check prerequisites
        for prereq_id in item.prerequisites:
            if not self.progress.items.get(prereq_id, ChecklistItem('', '', '', '', '', '', [])).completed:
                self.logger.error(f"Prerequisite '{prereq_id}' not completed for item '{item_id}'")
                return False
        
        # Mark as completed
        item.completed = True
        item.completed_at = datetime.now().isoformat()
        if notes:
            item.notes = notes
        
        self._save_progress()
        self.logger.info(f"Completed: {item.title}")
        return True
    
    def uncomplete_item(self, item_id: str) -> bool:
        """Mark an item as not completed"""
        if item_id not in self.progress.items:
            self.logger.error(f"Item '{item_id}' not found in checklist")
            return False
        
        item = self.progress.items[item_id]
        item.completed = False
        item.completed_at = None
        
        self._save_progress()
        self.logger.info(f"Uncompleted: {item.title}")
        return True
    
    def validate_item(self, item_id: str) -> bool:
        """Validate an item using its validation command"""
        if item_id not in self.progress.items:
            return False
        
        item = self.progress.items[item_id]
        if not item.validation_command:
            return True  # No validation command means it's valid
        
        try:
            import subprocess
            result = subprocess.run(
                item.validation_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"Validation failed for {item_id}: {e}")
            return False
    
    def get_next_items(self) -> List[ChecklistItem]:
        """Get next items that can be completed (prerequisites met)"""
        next_items = []
        
        for item in self.progress.items.values():
            if item.completed:
                continue
            
            # Check if all prerequisites are met
            prerequisites_met = all(
                self.progress.items.get(prereq_id, ChecklistItem('', '', '', '', '', '', [])).completed
                for prereq_id in item.prerequisites
            )
            
            if prerequisites_met:
                next_items.append(item)
        
        # Sort by priority and category
        priority_order = {'critical': 0, 'important': 1, 'optional': 2}
        next_items.sort(key=lambda x: (priority_order.get(x.priority, 3), x.category, x.title))
        
        return next_items
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary"""
        categories = {}
        for item in self.progress.items.values():
            if item.category not in categories:
                categories[item.category] = {
                    'total': 0,
                    'completed': 0,
                    'critical': 0,
                    'critical_completed': 0
                }
            
            categories[item.category]['total'] += 1
            if item.completed:
                categories[item.category]['completed'] += 1
            if item.priority == 'critical':
                categories[item.category]['critical'] += 1
                if item.completed:
                    categories[item.category]['critical_completed'] += 1
        
        return {
            'overall': {
                'completion_percentage': self.progress.completion_percentage,
                'completed_items': self.progress.completed_items,
                'total_items': self.progress.total_items,
                'critical_completed': self.progress.critical_completed,
                'critical_total': self.progress.critical_total,
                'estimated_time_remaining': self.progress.estimated_time_remaining
            },
            'categories': categories,
            'next_items': [
                {
                    'id': item.id,
                    'title': item.title,
                    'priority': item.priority,
                    'estimated_time': item.estimated_time
                }
                for item in self.get_next_items()[:5]  # Next 5 items
            ]
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive progress report"""
        status = self.get_status_summary()
        
        report = [
            f"# Developer Onboarding Progress Report",
            f"",
            f"**Developer:** {self.progress.developer_name}",
            f"**Started:** {self.progress.started_at}",
            f"**Last Updated:** {self.progress.last_updated}",
            f"",
            f"## Overall Progress",
            f"",
            f"- **Completion:** {status['overall']['completion_percentage']:.1f}% ({status['overall']['completed_items']}/{status['overall']['total_items']} items)",
            f"- **Critical Items:** {status['overall']['critical_completed']}/{status['overall']['critical_total']} completed",
            f"- **Estimated Time Remaining:** {status['overall']['estimated_time_remaining']}",
            f"",
            f"## Progress by Category",
            f""
        ]
        
        for category, stats in status['categories'].items():
            completion = (stats['completed'] / stats['total']) * 100 if stats['total'] > 0 else 0
            report.append(f"### {category}")
            report.append(f"- Progress: {completion:.1f}% ({stats['completed']}/{stats['total']} items)")
            if stats['critical'] > 0:
                report.append(f"- Critical: {stats['critical_completed']}/{stats['critical']} completed")
            report.append("")
        
        if status['next_items']:
            report.append("## Next Steps")
            report.append("")
            for item in status['next_items']:
                priority_icon = {'critical': '🔴', 'important': '🟡', 'optional': '🟢'}
                icon = priority_icon.get(item['priority'], '⚪')
                report.append(f"- {icon} **{item['title']}** ({item['estimated_time']})")
            report.append("")
        
        # Completed items by category
        report.append("## Completed Items")
        report.append("")
        
        for category in status['categories'].keys():
            completed_items = [
                item for item in self.progress.items.values()
                if item.category == category and item.completed
            ]
            
            if completed_items:
                report.append(f"### {category}")
                for item in completed_items:
                    completed_date = item.completed_at[:10] if item.completed_at else "Unknown"
                    report.append(f"- ✅ {item.title} ({completed_date})")
                report.append("")
        
        return "\n".join(report)

def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Developer onboarding checklist and progress tracking")
    parser.add_argument('--status', action='store_true', help='Show current progress status')
    parser.add_argument('--next', action='store_true', help='Show next items to complete')
    parser.add_argument('--complete', type=str, help='Mark item as completed by ID')
    parser.add_argument('--uncomplete', type=str, help='Mark item as not completed by ID')
    parser.add_argument('--validate', type=str, help='Validate item by ID')
    parser.add_argument('--list', action='store_true', help='List all checklist items')
    parser.add_argument('--report', type=str, help='Generate progress report to file')
    parser.add_argument('--developer', type=str, help='Developer name for progress tracking')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    # Create checklist manager
    checklist = DeveloperChecklist(developer_name=args.developer)
    
    if args.status or not any([args.next, args.complete, args.uncomplete, args.validate, args.list, args.report]):
        # Show status
        status = checklist.get_status_summary()
        
        print(f"\n📋 ONBOARDING PROGRESS - {checklist.progress.developer_name}")
        print("=" * 60)
        print(f"Overall Completion: {status['overall']['completion_percentage']:.1f}%")
        print(f"Items Completed: {status['overall']['completed_items']}/{status['overall']['total_items']}")
        print(f"Critical Items: {status['overall']['critical_completed']}/{status['overall']['critical_total']}")
        print(f"Time Remaining: {status['overall']['estimated_time_remaining']}")
        
        print(f"\n📊 PROGRESS BY CATEGORY:")
        for category, stats in status['categories'].items():
            completion = (stats['completed'] / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"  {category}: {completion:.0f}% ({stats['completed']}/{stats['total']})")
        
        if status['next_items']:
            print(f"\n🎯 NEXT STEPS:")
            for item in status['next_items']:
                priority_icon = {'critical': '🔴', 'important': '🟡', 'optional': '🟢'}
                icon = priority_icon.get(item['priority'], '⚪')
                print(f"  {icon} {item['title']} ({item['estimated_time']})")
    
    if args.next:
        next_items = checklist.get_next_items()
        
        print(f"\n🎯 NEXT ITEMS TO COMPLETE:")
        print("-" * 40)
        
        for item in next_items[:10]:  # Show first 10
            priority_icon = {'critical': '🔴', 'important': '🟡', 'optional': '🟢'}
            icon = priority_icon.get(item.priority, '⚪')
            
            print(f"{icon} {item.id}: {item.title}")
            print(f"   Category: {item.category}")
            print(f"   Time: {item.estimated_time}")
            if item.prerequisites:
                print(f"   Prerequisites: {', '.join(item.prerequisites)}")
            print()
    
    if args.complete:
        if checklist.complete_item(args.complete):
            print(f"✅ Completed: {args.complete}")
        else:
            print(f"❌ Failed to complete: {args.complete}")
    
    if args.uncomplete:
        if checklist.uncomplete_item(args.uncomplete):
            print(f"↩️ Uncompleted: {args.uncomplete}")
        else:
            print(f"❌ Failed to uncomplete: {args.uncomplete}")
    
    if args.validate:
        if checklist.validate_item(args.validate):
            print(f"✅ Validation passed: {args.validate}")
        else:
            print(f"❌ Validation failed: {args.validate}")
    
    if args.list:
        print(f"\n📋 ALL CHECKLIST ITEMS:")
        print("-" * 50)
        
        categories = {}
        for item in checklist.progress.items.values():
            if item.category not in categories:
                categories[item.category] = []
            categories[item.category].append(item)
        
        for category, items in categories.items():
            print(f"\n{category}:")
            for item in items:
                status_icon = "✅" if item.completed else "⬜"
                priority_icon = {'critical': '🔴', 'important': '🟡', 'optional': '🟢'}
                p_icon = priority_icon.get(item.priority, '⚪')
                
                print(f"  {status_icon} {p_icon} {item.id}: {item.title}")
                print(f"      {item.description}")
                print(f"      Time: {item.estimated_time}")
                if item.prerequisites:
                    print(f"      Prerequisites: {', '.join(item.prerequisites)}")
                print()
    
    if args.report:
        report = checklist.generate_report()
        output_file = Path(args.report)
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"📄 Progress report saved to: {output_file}")

if __name__ == "__main__":
    main()