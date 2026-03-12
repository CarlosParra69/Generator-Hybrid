"""
Synthetic French Exam Generator - Training Data Generator for Language Models

A hybrid synthetic exam generator that automatically generates thousands of training examples 
for French language evaluation models using Groq API.

Version: 1.0.0
Author: ML Engineering Team
Date: March 2026
"""

__version__ = "1.0.0"
__author__ = "ML Engineering Team"

# Import main components for easy access
try:
    from .logger import logger
    from config.config import validate_config
    from .exam_builder import ExamBuilder
    from .trainer_client import TrainerClient
    from .generator import SyntheticTrainingGenerator
except ImportError:
    # Allow relative imports to fail when package not properly initialized
    pass
