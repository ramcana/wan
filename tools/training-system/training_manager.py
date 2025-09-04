"""
Training Manager

Manages training content, learning paths, and educational resources for the WAN22 project.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import yaml

from tools..models import (
    LearningPath, TrainingModule, PracticeExercise, Assessment,
    TroubleshootingWizard, TrainingResource, Certificate
)


@dataclass
class UserProfile:
    """User profile for personalized training."""
    name: str
    role: str
    experience_level: str
    completed_modules: List[str]
    preferences: Dict[str, Any]
    created_at: datetime
    last_active: datetime


class TrainingManager:
    """Manages training content and personalized learning paths."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/training-config.yaml")
        self.content_dir = Path("docs/training")
        self.user_data_dir = Path("data/training/users")
        self.certificates_dir = Path("data/training/certificates")
        
        # Ensure directories exist
        self.user_data_dir.mkdir(parents=True, exist_ok=True)
        self.certificates_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = self._load_config()
        self.content = self._load_training_content()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'learning_paths': {
                'developer': {
                    'modules': [
                        'project-overview',
                        'environment-setup',
                        'test-management',
                        'code-quality',
                        'daily-workflow'
                    ],
                    'estimated_time': 120
                },
                'quality-engineer': {
                    'modules': [
                        'project-overview',
                        'test-management',
                        'code-quality',
                        'configuration-management',
                        'monitoring',
                        'ci-cd-integration'
                    ],
                    'estimated_time': 150
                },
                'team-lead': {
                    'modules': [
                        'project-overview',
                        'team-collaboration',
                        'monitoring',
                        'troubleshooting',
                        'best-practices'
                    ],
                    'estimated_time': 90
                },
                'admin': {
                    'modules': [
                        'project-overview',
                        'configuration-management',
                        'deployment',
                        'monitoring',
                        'maintenance'
                    ],
                    'estimated_time': 100
                }
            },
            'assessments': {
                'passing_score': 80,
                'max_attempts': 3,
                'time_limit_minutes': 30
            },
            'certificates': {
                'template_path': 'templates/certificate-template.html',
                'signature': 'WAN22 Training Team'
            }
        }
    
    def _load_training_content(self) -> Dict[str, Any]:
        """Load training content from documentation."""
        content = {
            'modules': {},
            'exercises': {},
            'assessments': {},
            'resources': {}
        }
        
        # Load modules
        modules_dir = self.content_dir / "modules"
        if modules_dir.exists():
            for module_file in modules_dir.glob("*.yaml"):
                with open(module_file, 'r') as f:
                    module_data = yaml.safe_load(f)
                    content['modules'][module_file.stem] = module_data
        
        # Load exercises
        exercises_dir = self.content_dir / "exercises"
        if exercises_dir.exists():
            for exercise_file in exercises_dir.glob("*.yaml"):
                with open(exercise_file, 'r') as f:
                    exercise_data = yaml.safe_load(f)
                    content['exercises'][exercise_file.stem] = exercise_data
        
        return content
    
    def create_learning_path(self, role: str, experience_level: str) -> LearningPath:
        """Create personalized learning path based on role and experience."""
        path_config = self.config['learning_paths'].get(role, self.config['learning_paths']['developer'])
        
        modules = []
        for module_id in path_config['modules']:
            module_data = self.content['modules'].get(module_id, {})
            
            # Adjust for experience level
            estimated_time = module_data.get('estimated_time', 20)
            if experience_level == 'beginner':
                estimated_time = int(estimated_time * 1.5)
            elif experience_level == 'advanced':
                estimated_time = int(estimated_time * 0.7)
            
            module = TrainingModule(
                id=module_id,
                title=module_data.get('title', module_id.replace('-', ' ').title()),
                description=module_data.get('description', ''),
                type=module_data.get('type', 'tutorial'),
                topic=module_id,
                estimated_time=estimated_time,
                prerequisites=module_data.get('prerequisites', []),
                objectives=module_data.get('objectives', [])
            )
            modules.append(module)
        
        return LearningPath(
            id=f"{role}-{experience_level}",
            title=f"{role.replace('-', ' ').title()} Learning Path",
            description=f"Personalized learning path for {role} at {experience_level} level",
            modules=modules,
            estimated_time=sum(m.estimated_time for m in modules),
            target_role=role,
            experience_level=experience_level
        )
    
    def get_practice_exercise(self, exercise_id: str) -> PracticeExercise:
        """Get practice exercise by ID."""
        exercise_data = self.content['exercises'].get(exercise_id, {})
        
        if not exercise_data:
            # Create default exercise
            exercise_data = {
                'title': exercise_id.replace('-', ' ').title(),
                'description': f'Practice exercise for {exercise_id}',
                'objectives': [f'Learn {exercise_id} concepts'],
                'steps': [
                    {
                        'title': 'Getting Started',
                        'description': 'Follow the instructions to complete this exercise',
                        'command': None,
                        'verification': 'Check that you understand the concepts'
                    }
                ],
                'estimated_time': 15
            }
        
from tools..models import ExerciseStep
        steps = []
        for step_data in exercise_data.get('steps', []):
            step = ExerciseStep(
                title=step_data.get('title', ''),
                description=step_data.get('description', ''),
                command=step_data.get('command'),
                verification=step_data.get('verification'),
                expected_output=step_data.get('expected_output')
            )
            steps.append(step)
        
        return PracticeExercise(
            id=exercise_id,
            title=exercise_data.get('title', ''),
            description=exercise_data.get('description', ''),
            objectives=exercise_data.get('objectives', []),
            steps=steps,
            estimated_time=exercise_data.get('estimated_time', 15),
            difficulty=exercise_data.get('difficulty', 'intermediate'),
            prerequisites=exercise_data.get('prerequisites', [])
        )
    
    def get_assessment(self, topic: str) -> Assessment:
        """Get assessment for topic."""
from tools..models import AssessmentQuestion, AssessmentResult
        
        # Load assessment questions
        questions_file = self.content_dir / "assessments" / f"{topic}-questions.yaml"
        if questions_file.exists():
            with open(questions_file, 'r') as f:
                questions_data = yaml.safe_load(f)
        else:
            # Default questions
            questions_data = {
                'questions': [
                    {
                        'question': f'What is the primary purpose of {topic} tools?',
                        'type': 'multiple_choice',
                        'options': ['Option A', 'Option B', 'Option C', 'Option D'],
                        'correct_answer': 0,
                        'explanation': 'Explanation of the correct answer'
                    }
                ]
            }
        
        questions = []
        for q_data in questions_data.get('questions', []):
            question = AssessmentQuestion(
                question=q_data.get('question', ''),
                type=q_data.get('type', 'multiple_choice'),
                options=q_data.get('options', []),
                correct_answer=q_data.get('correct_answer'),
                explanation=q_data.get('explanation', ''),
                points=q_data.get('points', 1)
            )
            questions.append(question)
        
        return Assessment(
            id=f"{topic}-assessment",
            title=f"{topic.replace('-', ' ').title()} Assessment",
            description=f"Knowledge assessment for {topic}",
            questions=questions,
            passing_score=self.config['assessments']['passing_score'],
            time_limit=self.config['assessments']['time_limit_minutes'],
            max_attempts=self.config['assessments']['max_attempts']
        )
    
    def get_troubleshooting_wizard(self, issue_type: str) -> TroubleshootingWizard:
        """Get troubleshooting wizard for issue type."""
from tools..troubleshooting_wizard import TroubleshootingWizardImpl
        return TroubleshootingWizardImpl(issue_type)
    
    def get_resources(self, resource_type: Optional[str] = None) -> Dict[str, List[TrainingResource]]:
        """Get training resources by type."""
        resources = {
            'videos': [
                TrainingResource(
                    id='project-intro-video',
                    title='Project Introduction Video',
                    description='15-minute overview of the WAN22 project',
                    type='video',
                    url='docs/training/video-tutorials/project-introduction.md',
                    duration=15,
                    difficulty='beginner'
                ),
                TrainingResource(
                    id='test-management-video',
                    title='Test Management Tools Video',
                    description='Comprehensive guide to test management tools',
                    type='video',
                    url='docs/training/video-tutorials/test-management-tools.md',
                    duration=20,
                    difficulty='intermediate'
                )
            ],
            'documentation': [
                TrainingResource(
                    id='tool-docs',
                    title='Tool Documentation',
                    description='Comprehensive documentation for all tools',
                    type='documentation',
                    url='docs/training/tools/README.md',
                    difficulty='all'
                ),
                TrainingResource(
                    id='best-practices',
                    title='Best Practices Guide',
                    description='Development and maintenance best practices',
                    type='documentation',
                    url='docs/training/best-practices/README.md',
                    difficulty='intermediate'
                )
            ],
            'examples': [
                TrainingResource(
                    id='code-examples',
                    title='Code Examples',
                    description='Real-world code examples and patterns',
                    type='examples',
                    url='examples/',
                    difficulty='all'
                )
            ],
            'checklists': [
                TrainingResource(
                    id='onboarding-checklist',
                    title='Onboarding Checklist',
                    description='Complete checklist for new team members',
                    type='checklist',
                    url='docs/training/onboarding/team-onboarding-guide.md',
                    difficulty='beginner'
                )
            ]
        }
        
        if resource_type:
            return {resource_type: resources.get(resource_type, [])}
        
        return resources
    
    def get_recommendations(self, progress) -> List[str]:
        """Get personalized recommendations based on progress."""
        recommendations = []
        
        if progress.overall_completion < 25:
            recommendations.append("Continue with the onboarding modules to build foundational knowledge")
        elif progress.overall_completion < 50:
            recommendations.append("Practice hands-on exercises to reinforce your learning")
        elif progress.overall_completion < 75:
            recommendations.append("Take assessments to validate your knowledge")
        else:
            recommendations.append("Explore advanced topics and help mentor other team members")
        
        # Check for specific gaps
        if not any('test' in module.name.lower() for module in progress.module_progress if module.completed):
            recommendations.append("Focus on test management tools - they're fundamental to the project")
        
        if not any('quality' in module.name.lower() for module in progress.module_progress if module.completed):
            recommendations.append("Learn code quality tools to improve your development workflow")
        
        return recommendations
    
    def generate_certificate(self, learning_path_id: str, completion_time: float) -> Certificate:
        """Generate completion certificate."""
        certificate = Certificate(
            id=f"cert-{learning_path_id}-{datetime.now().strftime('%Y%m%d')}",
            recipient_name="Training Participant",
            learning_path=learning_path_id,
            completion_date=datetime.now(),
            completion_time=completion_time,
            signature=self.config['certificates']['signature']
        )
        
        # Save certificate
        cert_file = self.certificates_dir / f"{certificate.id}.json"
        with open(cert_file, 'w') as f:
            json.dump(asdict(certificate), f, indent=2, default=str)
        
        print(f"🏆 Certificate generated: {cert_file}")
        return certificate
    
    def save_user_profile(self, profile: UserProfile) -> None:
        """Save user profile."""
        profile_file = self.user_data_dir / f"{profile.name.lower().replace(' ', '_')}.json"
        with open(profile_file, 'w') as f:
            json.dump(asdict(profile), f, indent=2, default=str)
    
    def load_user_profile(self, name: str) -> Optional[UserProfile]:
        """Load user profile."""
        profile_file = self.user_data_dir / f"{name.lower().replace(' ', '_')}.json"
        if profile_file.exists():
            with open(profile_file, 'r') as f:
                data = json.load(f)
                return UserProfile(**data)
        return None