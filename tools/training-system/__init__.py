"""
WAN22 Training and Documentation System

A comprehensive training and documentation system for the WAN22 project's
cleanup and quality improvement tools.
"""

from .training_manager import TrainingManager
from .progress_tracker import ProgressTracker
from .feedback_system import FeedbackSystem
from .models import (
    LearningPath, TrainingModule, PracticeExercise, Assessment,
    TrainingResource, Certificate, UserProgress, Achievement
)

__version__ = "1.0.0"
__author__ = "WAN22 Development Team"

__all__ = [
    'TrainingManager',
    'ProgressTracker', 
    'FeedbackSystem',
    'LearningPath',
    'TrainingModule',
    'PracticeExercise',
    'Assessment',
    'TrainingResource',
    'Certificate',
    'UserProgress',
    'Achievement'
]