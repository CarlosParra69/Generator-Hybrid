"""
Main generator module - orchestrates the synthetic exam generation and training process.
"""

import time
import sys
from typing import Optional

from config.config import (
    NUM_EXAMS_TO_GENERATE,
    INFINITE_MODE,
    validate_config,
)
from service.logger import logger, log_exam_generation
from service.exam_builder import get_or_create_exam_builder
from client.trainer_client import get_or_create_trainer_client


class SyntheticTrainingGenerator:
    """Main orchestrator for synthetic exam generation and training."""
    
    def __init__(self):
        """Initialize the synthetic training generator."""
        logger.info("Initializing synthetic training generator...")
        
        # Validate configuration
        try:
            validate_config()
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            raise
        
        self.exam_builder = get_or_create_exam_builder()
        self.trainer_client = get_or_create_trainer_client()
        
        self.generated_count = 0
        self.successful_count = 0
        self.failed_count = 0
        
        logger.info("Synthetic training generator initialized successfully")
    
    def run(self, num_exams: Optional[int] = None, infinite: bool = False):
        """
        Main execution loop for generating and sending exams.
        
        Args:
            num_exams: Number of exams to generate (uses config if None)
            infinite: If True, runs infinite loop (uses config if None)
        """
        num_exams = num_exams or NUM_EXAMS_TO_GENERATE
        infinite = infinite or INFINITE_MODE
        
        logger.info(f"Starting exam generation - num_exams: {num_exams}, infinite: {infinite}")
        
        # Verify API connectivity
        if not self.trainer_client.verify_connection():
            logger.warning("API endpoint unreachable - will retry during generation")
        
        batch_count = 0
        
        try:
            if infinite:
                self._run_infinite_loop()
            else:
                self._run_fixed_count(num_exams)
        
        except KeyboardInterrupt:
            logger.info("Generator interrupted by user")
            self._print_summary()
            sys.exit(0)
        
        except Exception as e:
            logger.error(f"Fatal error in generator: {e}", exc_info=True)
            self._print_summary()
            sys.exit(1)
    
    def _run_fixed_count(self, num_exams: int):
        """Run generator for a fixed number of exams."""
        logger.info(f"Running generator for {num_exams} exams")
        
        for i in range(num_exams):
            self._generate_and_send_exam(i + 1, num_exams)
        
        self._print_summary()
    
    def _run_infinite_loop(self):
        """Run generator in infinite loop mode."""
        logger.info("Running generator in infinite loop mode")
        
        iteration = 0
        while True:
            iteration += 1
            self._generate_and_send_exam(iteration, None)
            
            # Print summary every 100 exams
            if iteration % 100 == 0:
                self._print_summary()
    
    def _generate_and_send_exam(self, current: int, total: Optional[int] = None):
        """
        Generate a single exam and send it to the training endpoint.
        
        Args:
            current: Current exam number
            total: Total exams to generate (None for infinite)
        """
        try:
            # Generate exam
            start_time = time.time()
            exam = self.exam_builder.generate_exam(
                num_questions=self._get_random_question_count(),
                adaptive=True
            )
            generation_time = time.time() - start_time
            
            self.generated_count += 1
            
            # Send exam
            exam_id = exam.get("exam_id")
            logger.info(f"Generated exam {exam_id} in {generation_time:.2f}s ({current}/{total or '∞'})")
            
            send_start = time.time()
            success = self.trainer_client.send_exam(exam)
            send_time = time.time() - send_start
            
            if success:
                self.successful_count += 1
                logger.info(f"Successfully sent exam {exam_id} in {send_time:.2f}s")
            else:
                self.failed_count += 1
                logger.error(f"Failed to send exam {exam_id}")
        
        except Exception as e:
            self.failed_count += 1
            logger.error(f"Error generating/sending exam: {e}", exc_info=True)
    
    def _get_random_question_count(self) -> int:
        """Get random number of questions for exam (5-10)."""
        import random
        return random.randint(5, 10)
    
    def _print_summary(self):
        """Print generation summary."""
        logger.info("=" * 60)
        logger.info("GENERATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total generated: {self.generated_count}")
        logger.info(f"Successfully sent: {self.successful_count}")
        logger.info(f"Failed: {self.failed_count}")
        
        if self.generated_count > 0:
            success_rate = (self.successful_count / self.generated_count) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        
        logger.info("=" * 60)
