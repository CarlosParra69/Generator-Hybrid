"""
Synthetic French Exam Generator — Training Data Generator for Language Models

Hybrid generator (Python + Groq LLM) that produces complete /train JSON batches
for French language evaluation models following the DELF schema.

Version: 2.0.0
Author: ML Engineering Team
Date: March 2026
"""

__version__ = "2.0.0"
__author__ = "ML Engineering Team"

try:
    from .logger import logger
    from config.config import validate_config
    from .exam_builder import TrainBatchBuilder
    from .generator import SyntheticTrainingGenerator
    from .train_validator import validate_training_batch
except ImportError:
    pass
