"""
Configuration module for the synthetic training generator.
Contains all configurable parameters for exam generation and API communication.
"""

import os
from typing import Literal

# API Configuration
TRAINER_API_URL = os.getenv("TRAINER_API_URL", "http://localhost:8000/train")
TRAINER_API_TIMEOUT = int(os.getenv("TRAINER_API_TIMEOUT", "30"))

# LLM Configuration
# Groq Configuration (Free API)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"  # Fast, free model

# Generation Configuration
NUM_EXAMS_TO_GENERATE = int(os.getenv("NUM_EXAMS_TO_GENERATE", "10"))
INFINITE_MODE = os.getenv("INFINITE_MODE", "false").lower() == "true"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))

# Error Injection Configuration
ERROR_INJECTION_PROBABILITY = float(os.getenv("ERROR_INJECTION_PROBABILITY", "0.4"))
MCQ_CORRECT_PROBABILITY = float(os.getenv("MCQ_CORRECT_PROBABILITY", "0.70"))

# Time Simulation (in seconds)
OPEN_QUESTION_TIME_MIN = 120
OPEN_QUESTION_TIME_MAX = 240
MCQ_TIME_MIN = 4
MCQ_TIME_MAX = 15

# Training Data
TRAINING_DATA_FILE = os.getenv("TRAINING_DATA_FILE", "train_french_priority.json")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = "synthetic_training_generator.log"

# Retry Configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# CEFR Level Word Count Expectations
CEFR_WORD_EXPECTATIONS = {
    "A1": {"min": 10, "max": 30},
    "A2": {"min": 30, "max": 60},
    "B1": {"min": 60, "max": 120},
    "B2": {"min": 100, "max": 160},
    "C1": {"min": 150, "max": 250},
    "C2": {"min": 200, "max": 300},
}

def validate_config():
    """Validates that required configurations are set."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is required")
    
    return True
