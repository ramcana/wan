#!/usr/bin/env python3
"""
Interactive Training System CLI

Provides interactive training materials and progress tracking for the WAN22 project's
cleanup and quality improvement tools.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import subprocess
import time

from .training_manager import TrainingManager
from .progress_tracker import ProgressTracker
from .interactive_tutorial import InteractiveTutorial
from .feedback_system import FeedbackSystem


class TrainingSystemCLI:
    """Interactive training system command-line interface."""
    
    def __init__(self):
        self.training_manager = TrainingManager()
        self.progress_tracker = ProgressTracker()
        self.feedback_system = FeedbackSystem()
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser."""
        parser = argparse.ArgumentParser(
            description="WAN22 Interactive Training System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s start-onboarding          # Start interactive onboarding
  %(prog)s tutorial test-management  # Run test management tutorial
  %(prog)s progress                  # Show learning progress
  %(prog)s troubleshoot             # Interactive troubleshooting
  %(prog)s feedback                 # Provide feedback
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Onboarding command
        onboarding_parser = subparsers.add_parser(
            'start-onboarding',
            help='Start interactive onboarding process'
        )
        onboarding_parser.add_argument(
            '--role',
            choices=['developer', 'quality-engineer', 'team-lead', 'admin'],
            help='Your role for customized onboarding'
        )
        onboarding_parser.add_argument(
            '--experience',
            choices=['beginner', 'intermediate', 'advanced'],
            default='intermediate',
            help='Your experience level'
        )
        
        # Tutorial command
        tutorial_parser = subparsers.add_parser(
            'tutorial',
            help='Run interactive tutorials'
        )
        tutorial_parser.add_argument(
            'topic',
            choices=[
                'test-management', 'configuration', 'code-quality',
                'documentation', 'cleanup', 'monitoring', 'troubleshooting'
            ],
            help='Tutorial topic'
        )
        tutorial_parser.add_argument(
            '--interactive',
            action='store_true',
            help='Run in interactive mode with hands-on exercises'
        )
        
        # Progress command
        progress_parser = subparsers.add_parser(
            'progress',
            help='Show learning progress and achievements'
        )
        progress_parser.add_argument(
            '--detailed',
            action='store_true',
            help='Show detailed progress information'
        )
        
        # Practice command
        practice_parser = subparsers.add_parser(
            'practice',
            help='Practice exercises and scenarios'
        )
        practice_parser.add_argument(
            'exercise',
            choices=[
                'health-check', 'test-fixing', 'config-consolidation',
                'quality-improvement', 'documentation-generation'
            ],
            help='Practice exercise'
        )
        
        # Troubleshoot command
        troubleshoot_parser = subparsers.add_parser(
            'troubleshoot',
            help='Interactive troubleshooting wizard'
        )
        troubleshoot_parser.add_argument(
            '--issue-type',
            choices=['test-failure', 'config-error', 'quality-issue', 'performance'],
            help='Type of issue you\'re experiencing'
        )
        
        # Feedback command
        feedback_parser = subparsers.add_parser(
            'feedback',
            help='Provide feedback on training materials'
        )
        feedback_parser.add_argument(
            '--topic',
            help='Specific topic to provide feedback on'
        )
        
        # Assessment command
        assessment_parser = subparsers.add_parser(
            'assessment',
            help='Take knowledge assessment'
        )
        assessment_parser.add_argument(
            '--topic',
            choices=['all', 'test-management', 'configuration', 'code-quality'],
            default='all',
            help='Assessment topic'
        )
        
        # Resources command
        resources_parser = subparsers.add_parser(
            'resources',
            help='Access training resources'
        )
        resources_parser.add_argument(
            '--type',
            choices=['videos', 'documentation', 'examples', 'checklists'],
            help='Type of resources to show'
        )
        
        return parser
    
    def run(self, args: List[str]) -> int:
        """Run the training system CLI."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        if not parsed_args.command:
            parser.print_help()
            return 1
        
        try:
            if parsed_args.command == 'start-onboarding':
                return self.start_onboarding(parsed_args)
            elif parsed_args.command == 'tutorial':
                return self.run_tutorial(parsed_args)
            elif parsed_args.command == 'progress':
                return self.show_progress(parsed_args)
            elif parsed_args.command == 'practice':
                return self.run_practice(parsed_args)
            elif parsed_args.command == 'troubleshoot':
                return self.run_troubleshooting(parsed_args)
            elif parsed_args.command == 'feedback':
                return self.collect_feedback(parsed_args)
            elif parsed_args.command == 'assessment':
                return self.run_assessment(parsed_args)
            elif parsed_args.command == 'resources':
                return self.show_resources(parsed_args)
            else:
                print(f"Unknown command: {parsed_args.command}")
                return 1
                
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")
            return 1
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    def start_onboarding(self, args) -> int:
        """Start interactive onboarding process."""
        print("🎓 Welcome to WAN22 Interactive Training!")
        print("=" * 50)
        
        # Get user information
        if not args.role:
            print("\nWhat's your primary role?")
            print("1. Developer")
            print("2. Quality Engineer") 
            print("3. Team Lead")
            print("4. System Administrator")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            role_map = {'1': 'developer', '2': 'quality-engineer', '3': 'team-lead', '4': 'admin'}
            args.role = role_map.get(choice, 'developer')
        
        print(f"\n🎯 Starting onboarding for: {args.role.replace('-', ' ').title()}")
        print(f"📊 Experience level: {args.experience.title()}")
        
        # Create personalized learning path
        learning_path = self.training_manager.create_learning_path(args.role, args.experience)
        
        print(f"\n📚 Your personalized learning path has {len(learning_path.modules)} modules:")
        for i, module in enumerate(learning_path.modules, 1):
            print(f"  {i}. {module.title} ({module.estimated_time} minutes)")
        
        # Start onboarding
        if self._confirm("Ready to start your onboarding journey?"):
            return self._run_learning_path(learning_path)
        
        return 0
    
    def run_tutorial(self, args) -> int:
        """Run interactive tutorial."""
        print(f"🎯 Starting {args.topic.replace('-', ' ').title()} Tutorial")
        print("=" * 50)
        
        tutorial = InteractiveTutorial(args.topic)
        
        if args.interactive:
            return tutorial.run_interactive()
        else:
            return tutorial.run_guided()
    
    def show_progress(self, args) -> int:
        """Show learning progress."""
        progress = self.progress_tracker.get_progress()
        
        print("📊 Your Learning Progress")
        print("=" * 30)
        
        print(f"Overall Completion: {progress.overall_completion:.1f}%")
        print(f"Modules Completed: {progress.completed_modules}/{progress.total_modules}")
        print(f"Exercises Completed: {progress.completed_exercises}/{progress.total_exercises}")
        print(f"Time Invested: {progress.total_time_minutes} minutes")
        
        if progress.achievements:
            print(f"\n🏆 Achievements ({len(progress.achievements)}):")
            for achievement in progress.achievements:
                print(f"  • {achievement.title} - {achievement.description}")
        
        if args.detailed:
            print(f"\n📈 Detailed Progress:")
            for module in progress.module_progress:
                status = "✅" if module.completed else "⏳"
                print(f"  {status} {module.name}: {module.completion_percentage:.1f}%")
        
        # Show recommendations
        recommendations = self.training_manager.get_recommendations(progress)
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"  • {rec}")
        
        return 0
    
    def run_practice(self, args) -> int:
        """Run practice exercises."""
        print(f"🏋️ Practice Exercise: {args.exercise.replace('-', ' ').title()}")
        print("=" * 50)
        
        exercise = self.training_manager.get_practice_exercise(args.exercise)
        
        print(f"📝 {exercise.description}")
        print(f"⏱️  Estimated time: {exercise.estimated_time} minutes")
        print(f"🎯 Learning objectives:")
        for objective in exercise.objectives:
            print(f"  • {objective}")
        
        if self._confirm("Ready to start the exercise?"):
            return self._run_practice_exercise(exercise)
        
        return 0
    
    def run_troubleshooting(self, args) -> int:
        """Run interactive troubleshooting wizard."""
        print("🔧 Interactive Troubleshooting Wizard")
        print("=" * 40)
        
        if not args.issue_type:
            print("What type of issue are you experiencing?")
            print("1. Test failures")
            print("2. Configuration errors")
            print("3. Code quality issues")
            print("4. Performance problems")
            print("5. Other")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            issue_map = {
                '1': 'test-failure',
                '2': 'config-error', 
                '3': 'quality-issue',
                '4': 'performance',
                '5': 'other'
            }
            args.issue_type = issue_map.get(choice, 'other')
        
        # Run troubleshooting wizard
        wizard = self.training_manager.get_troubleshooting_wizard(args.issue_type)
        return wizard.run()
    
    def collect_feedback(self, args) -> int:
        """Collect user feedback."""
        print("💬 Training Feedback")
        print("=" * 20)
        
        feedback = self.feedback_system.collect_feedback(args.topic)
        
        print("Thank you for your feedback! 🙏")
        print("Your input helps us improve the training experience.")
        
        return 0
    
    def run_assessment(self, args) -> int:
        """Run knowledge assessment."""
        print(f"📋 Knowledge Assessment: {args.topic.title()}")
        print("=" * 40)
        
        assessment = self.training_manager.get_assessment(args.topic)
        
        print(f"📝 This assessment has {len(assessment.questions)} questions")
        print(f"⏱️  Estimated time: {assessment.estimated_time} minutes")
        
        if self._confirm("Ready to start the assessment?"):
            result = assessment.run()
            
            print(f"\n📊 Assessment Results:")
            print(f"Score: {result.score:.1f}%")
            print(f"Grade: {result.grade}")
            
            if result.recommendations:
                print(f"\n💡 Recommendations:")
                for rec in result.recommendations:
                    print(f"  • {rec}")
        
        return 0
    
    def show_resources(self, args) -> int:
        """Show training resources."""
        print("📚 Training Resources")
        print("=" * 20)
        
        resources = self.training_manager.get_resources(args.type)
        
        for category, items in resources.items():
            print(f"\n{category.title()}:")
            for item in items:
                print(f"  • {item.title}")
                if item.description:
                    print(f"    {item.description}")
                if item.url:
                    print(f"    🔗 {item.url}")
        
        return 0
    
    def _run_learning_path(self, learning_path) -> int:
        """Run a complete learning path."""
        start_time = datetime.now()
        
        for i, module in enumerate(learning_path.modules, 1):
            print(f"\n📖 Module {i}/{len(learning_path.modules)}: {module.title}")
            print("-" * 50)
            
            if not self._confirm(f"Start {module.title}?"):
                continue
            
            module_start = datetime.now()
            
            # Run module content
            if module.type == 'tutorial':
                tutorial = InteractiveTutorial(module.topic)
                result = tutorial.run_interactive()
            elif module.type == 'exercise':
                exercise = self.training_manager.get_practice_exercise(module.topic)
                result = self._run_practice_exercise(exercise)
            elif module.type == 'assessment':
                assessment = self.training_manager.get_assessment(module.topic)
                result = assessment.run()
            
            module_time = (datetime.now() - module_start).total_seconds() / 60
            
            # Track progress
            self.progress_tracker.complete_module(module.id, module_time)
            
            if result != 0:
                print(f"⚠️  Module completed with issues. Continue anyway?")
                if not self._confirm("Continue to next module?"):
                    break
        
        total_time = (datetime.now() - start_time).total_seconds() / 60
        
        print(f"\n🎉 Onboarding Complete!")
        print(f"⏱️  Total time: {total_time:.1f} minutes")
        print(f"🏆 You've completed {len(learning_path.modules)} modules!")
        
        # Generate completion certificate
        self.training_manager.generate_certificate(learning_path.id, total_time)
        
        return 0
    
    def _run_practice_exercise(self, exercise) -> int:
        """Run a practice exercise."""
        print(f"\n🎯 Exercise Steps:")
        
        for i, step in enumerate(exercise.steps, 1):
            print(f"\n{i}. {step.title}")
            print(f"   {step.description}")
            
            if step.command:
                print(f"   💻 Command: {step.command}")
                
                if self._confirm("Run this command?"):
                    try:
                        result = subprocess.run(
                            step.command,
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                        
                        if result.returncode == 0:
                            print("   ✅ Command executed successfully")
                            if result.stdout:
                                print(f"   📄 Output: {result.stdout[:200]}...")
                        else:
                            print("   ❌ Command failed")
                            print(f"   📄 Error: {result.stderr}")
                            
                    except subprocess.TimeoutExpired:
                        print("   ⏰ Command timed out")
                    except Exception as e:
                        print(f"   ❌ Error running command: {e}")
            
            if step.verification:
                print(f"   ✅ Verification: {step.verification}")
                if not self._confirm("Did this step complete successfully?"):
                    print("   💡 Check the troubleshooting guide for help")
            
            input("   Press Enter to continue...")
        
        print(f"\n🎉 Exercise completed!")
        return 0
    
    def _confirm(self, message: str) -> bool:
        """Get user confirmation."""
        response = input(f"{message} (y/N): ").strip().lower()
        return response in ['y', 'yes']


def main():
    """Main entry point."""
    cli = TrainingSystemCLI()
    return cli.run(sys.argv[1:])


if __name__ == '__main__':
    sys.exit(main())