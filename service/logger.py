"""
Logger module for structured logging throughout the application.
"""

import logging
import logging.handlers
import json
from datetime import datetime
from pathlib import Path

from config.config import LOG_LEVEL, LOG_FILE

class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, "exam_id"):
            log_data["exam_id"] = record.exam_id
        if hasattr(record, "candidate_id"):
            log_data["candidate_id"] = record.candidate_id
        if hasattr(record, "status"):
            log_data["status"] = record.status
        if hasattr(record, "error"):
            log_data["error"] = record.error
            
        return json.dumps(log_data, ensure_ascii=False)

def setup_logger(name: str = "synthetic_generator") -> logging.Logger:
    """
    Setup and return a configured logger.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    json_formatter = JsonFormatter()
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with JSON format
    log_file = Path(LOG_FILE)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)
    
    return logger

# Global logger instance
logger = setup_logger()

def log_exam_generation(exam_id: str, status: str, details: dict = None):
    """Log exam generation event."""
    extra = {
        "exam_id": exam_id,
        "status": status,
    }
    logger.info(f"Exam generated: {exam_id}", extra=extra)

def log_api_request(exam_id: str, status: str, response_code: int = None, error: str = None):
    """Log API request event."""
    extra = {
        "exam_id": exam_id,
        "status": status,
    }
    if response_code:
        extra["response_code"] = response_code
    if error:
        extra["error"] = error
    
    logger.info(f"API request - {status}", extra=extra)

def log_error(message: str, error: Exception = None, exam_id: str = None):
    """Log error event."""
    extra = {}
    if exam_id:
        extra["exam_id"] = exam_id
    if error:
        extra["error"] = str(error)
    
    logger.error(message, extra=extra, exc_info=error is not None)
