"""
Configuration module for the synthetic training generator.
Contains all configurable parameters for exam generation and API communication.
"""

import os

# API Configuration
TRAINER_API_URL = os.getenv("TRAINER_API_URL", "http://localhost:8000/train")
TRAINER_API_TIMEOUT = int(os.getenv("TRAINER_API_TIMEOUT", "30"))

# LLM Configuration — Groq (Free API)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Generation Configuration
NUM_EXAMS_TO_GENERATE = int(os.getenv("NUM_EXAMS_TO_GENERATE", "10"))
INFINITE_MODE = os.getenv("INFINITE_MODE", "false").lower() == "true"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))

# Batch Generation Configuration
EXAMPLES_PER_BATCH = int(os.getenv("EXAMPLES_PER_BATCH", "10"))

BATCH_TOPICS = [
    "vie quotidienne",
    "famille et relations",
    "travail et profession",
    "loisirs et activites",
    "voyage et transport",
    "alimentation et cuisine",
    "sante et bien-etre",
    "environnement et nature",
    "culture et arts",
    "education et apprentissage",
    "technologie et societe",
    "ville et logement",
    "auxiliaire etre - present et passe compose",
    "auxiliaire avoir - present et passe compose",
    "prepositions de lieu",
    "articles definis et indefinis",
    "adjectifs qualificatifs",
    "verbes pronominaux",
    "expressions de temps",
    "description de personnes et objets",
]

BATCH_LEVELS = ["A1", "A2", "B1", "B2"]

# Sample Files (reference JSON for format validation)
SAMPLE_DIR = os.getenv("SAMPLE_DIR", "sample")
SAMPLE_FILES = [
    "train_french_priority.json",
    "train_french_priority_2.json",
]

# Error Injection Configuration
ERROR_INJECTION_PROBABILITY = float(os.getenv("ERROR_INJECTION_PROBABILITY", "0.4"))
MCQ_CORRECT_PROBABILITY = float(os.getenv("MCQ_CORRECT_PROBABILITY", "0.70"))

# Time Simulation (in seconds)
OPEN_QUESTION_TIME_MIN = 120
OPEN_QUESTION_TIME_MAX = 240
MCQ_TIME_MIN = 4
MCQ_TIME_MAX = 15
SHORT_QUESTION_TIME_MIN = 10
SHORT_QUESTION_TIME_MAX = 45

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
