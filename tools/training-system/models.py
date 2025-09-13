"""
Training System Models

Data models for the training and documentation system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class DifficultyLevel(Enum):
    """Training difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ModuleType(Enum):
    """Training module types."""
    TUTORIAL = "tutorial"
    EXERCISE = "exercise"
    ASSESSMENT = "assessment"
    VIDEO = "video"
    DOCUMENTATION = "documentation"


class QuestionType(Enum):
    """Assessment question types."""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    CODE_COMPLETION = "code_completion"


@dataclass
class TrainingModule:
    """Training module definition."""
    id: str
    title: str
    description: str
    type: str
    topic: str
    estimated_time: int  # minutes
    prerequisites: List[str] = field(default_factory=list)
    objectives: List[str] = field(default_factory=list)
    content_path: Optional[str] = None
    difficulty: str = "intermediate"


@dataclass
class LearningPath:
    """Personalized learning path."""
    id: str
    title: str
    description: str
    modules: List[TrainingModule]
    estimated_time: int  # minutes
    target_role: str
    experience_level: str
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class ExerciseStep:
    """Step in a practice exercise."""
    title: str
    description: str
    command: Optional[str] = None
    verification: Optional[str] = None
    expected_output: Optional[str] = None
    hints: List[str] = field(default_factory=list)


@dataclass
class PracticeExercise:
    """Hands-on practice exercise."""
    id: str
    title: str
    description: str
    objectives: List[str]
    steps: List[ExerciseStep]
    estimated_time: int  # minutes
    difficulty: str = "intermediate"
    prerequisites: List[str] = field(default_factory=list)
    tools_required: List[str] = field(default_factory=list)


@dataclass
class AssessmentQuestion:
    """Assessment question."""
    question: str
    type: str
    options: List[str] = field(default_factory=list)
    correct_answer: Any = None
    explanation: str = ""
    points: int = 1
    hints: List[str] = field(default_factory=list)


@dataclass
class AssessmentResult:
    """Assessment result."""
    score: float  # percentage
    grade: str
    correct_answers: int
    total_questions: int
    time_taken: int  # minutes
    recommendations: List[str] = field(default_factory=list)
    detailed_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Assessment:
    """Knowledge assessment."""
    id: str
    title: str
    description: str
    questions: List[AssessmentQuestion]
    passing_score: float = 80.0
    time_limit: int = 30  # minutes
    max_attempts: int = 3
    
    def run(self) -> AssessmentResult:
        """Run the assessment interactively."""
        print(f"\n📋 {self.title}")
        print(f"📝 {self.description}")
        print(f"⏱️  Time limit: {self.time_limit} minutes")
        print(f"🎯 Passing score: {self.passing_score}%")
        
        start_time = datetime.now()
        correct_answers = 0
        
        for i, question in enumerate(self.questions, 1):
            print(f"\n--- Question {i}/{len(self.questions)} ---")
            print(f"📝 {question.question}")
            
            if question.type == "multiple_choice":
                for j, option in enumerate(question.options):
                    print(f"  {j + 1}. {option}")
                
                while True:
                    try:
                        answer = int(input("Your answer (number): ")) - 1
                        if 0 <= answer < len(question.options):
                            break
                        else:
                            print("Invalid option. Please try again.")
                    except ValueError:
                        print("Please enter a number.")
                
                if answer == question.correct_answer:
                    correct_answers += 1
                    print("✅ Correct!")
                else:
                    print(f"❌ Incorrect. {question.explanation}")
            
            elif question.type == "true_false":
                while True:
                    answer = input("True or False (T/F): ").strip().upper()
                    if answer in ['T', 'F', 'TRUE', 'FALSE']:
                        break
                    print("Please enter T/F or True/False.")
                
                correct = (answer in ['T', 'TRUE']) == question.correct_answer
                if correct:
                    correct_answers += 1
                    print("✅ Correct!")
                else:
                    print(f"❌ Incorrect. {question.explanation}")
        
        end_time = datetime.now()
        time_taken = int((end_time - start_time).total_seconds() / 60)
        score = (correct_answers / len(self.questions)) * 100
        
        # Determine grade
        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        elif score >= 60:
            grade = "D"
        else:
            grade = "F"
        
        # Generate recommendations
        recommendations = []
        if score < self.passing_score:
            recommendations.append("Review the training materials and try again")
            recommendations.append("Focus on areas where you missed questions")
        else:
            recommendations.append("Great job! Consider helping others learn")
            if score < 90:
                recommendations.append("Review advanced topics for deeper understanding")
        
        return AssessmentResult(
            score=score,
            grade=grade,
            correct_answers=correct_answers,
            total_questions=len(self.questions),
            time_taken=time_taken,
            recommendations=recommendations
        )


@dataclass
class TroubleshootingStep:
    """Step in troubleshooting wizard."""
    question: str
    options: List[str]
    actions: Dict[str, str]  # option -> action
    next_steps: Dict[str, str] = field(default_factory=dict)  # option -> next step id


@dataclass
class TroubleshootingWizard:
    """Interactive troubleshooting wizard."""
    id: str
    title: str
    description: str
    steps: Dict[str, TroubleshootingStep]
    start_step: str
    
    def run(self) -> int:
        """Run the troubleshooting wizard."""
        current_step_id = self.start_step
        
        print(f"\n🔧 {self.title}")
        print(f"📝 {self.description}")
        
        while current_step_id:
            step = self.steps.get(current_step_id)
            if not step:
                break
            
            print(f"\n❓ {step.question}")
            for i, option in enumerate(step.options, 1):
                print(f"  {i}. {option}")
            
            while True:
                try:
                    choice = int(input("Your choice: ")) - 1
                    if 0 <= choice < len(step.options):
                        break
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a number.")
            
            selected_option = step.options[choice]
            
            # Execute action
            if selected_option in step.actions:
                action = step.actions[selected_option]
                print(f"\n💡 {action}")
            
            # Move to next step
            current_step_id = step.next_steps.get(selected_option)
            
            if current_step_id:
                input("\nPress Enter to continue...")
        
        print("\n✅ Troubleshooting complete!")
        return 0


@dataclass
class TrainingResource:
    """Training resource (video, document, etc.)."""
    id: str
    title: str
    description: str
    type: str  # video, documentation, example, checklist
    url: str
    duration: Optional[int] = None  # minutes for videos
    difficulty: str = "intermediate"
    tags: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class Achievement:
    """Training achievement/badge."""
    id: str
    title: str
    description: str
    icon: str
    criteria: Dict[str, Any]
    earned_date: Optional[datetime] = None


@dataclass
class ModuleProgress:
    """Progress for a specific module."""
    name: str
    completed: bool
    completion_percentage: float
    time_spent: int  # minutes
    last_accessed: Optional[datetime] = None


@dataclass
class UserProgress:
    """User's overall training progress."""
    overall_completion: float
    completed_modules: int
    total_modules: int
    completed_exercises: int
    total_exercises: int
    total_time_minutes: int
    achievements: List[Achievement] = field(default_factory=list)
    module_progress: List[ModuleProgress] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class Certificate:
    """Training completion certificate."""
    id: str
    recipient_name: str
    learning_path: str
    completion_date: datetime
    completion_time: float  # minutes
    signature: str
    verification_code: Optional[str] = None


@dataclass
class FeedbackItem:
    """User feedback item."""
    id: str
    topic: Optional[str]
    rating: int  # 1-5
    comments: str
    suggestions: str
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    category: str = "general"  # general, content, technical, ui


@dataclass
class TrainingMetrics:
    """Training system metrics."""
    total_users: int
    active_users: int
    completion_rate: float
    average_completion_time: float
    popular_modules: List[str]
    common_issues: List[str]
    satisfaction_score: float
    last_updated: datetime = field(default_factory=datetime.now)
