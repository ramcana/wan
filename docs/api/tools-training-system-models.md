---
title: tools.training-system.models
category: api
tags: [api, tools]
---

# tools.training-system.models

Training System Models

Data models for the training and documentation system.

## Classes

### DifficultyLevel

Training difficulty levels.

### ModuleType

Training module types.

### QuestionType

Assessment question types.

### TrainingModule

Training module definition.

### LearningPath

Personalized learning path.

### ExerciseStep

Step in a practice exercise.

### PracticeExercise

Hands-on practice exercise.

### AssessmentQuestion

Assessment question.

### AssessmentResult

Assessment result.

### Assessment

Knowledge assessment.

#### Methods

##### run(self: Any) -> AssessmentResult

Run the assessment interactively.

### TroubleshootingStep

Step in troubleshooting wizard.

### TroubleshootingWizard

Interactive troubleshooting wizard.

#### Methods

##### run(self: Any) -> int

Run the troubleshooting wizard.

### TrainingResource

Training resource (video, document, etc.).

### Achievement

Training achievement/badge.

### ModuleProgress

Progress for a specific module.

### UserProgress

User's overall training progress.

### Certificate

Training completion certificate.

### FeedbackItem

User feedback item.

### TrainingMetrics

Training system metrics.

## Constants

### BEGINNER

Type: `str`

Value: `beginner`

### INTERMEDIATE

Type: `str`

Value: `intermediate`

### ADVANCED

Type: `str`

Value: `advanced`

### TUTORIAL

Type: `str`

Value: `tutorial`

### EXERCISE

Type: `str`

Value: `exercise`

### ASSESSMENT

Type: `str`

Value: `assessment`

### VIDEO

Type: `str`

Value: `video`

### DOCUMENTATION

Type: `str`

Value: `documentation`

### MULTIPLE_CHOICE

Type: `str`

Value: `multiple_choice`

### TRUE_FALSE

Type: `str`

Value: `true_false`

### SHORT_ANSWER

Type: `str`

Value: `short_answer`

### CODE_COMPLETION

Type: `str`

Value: `code_completion`

