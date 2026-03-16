"""
Main generator module — orchestrates training batch generation and submission.

Flow per batch:
  1. TrainBatchBuilder asks Groq to generate a complete /train JSON.
  2. JSON is validated internally before being returned.
  3. Batch is saved locally to output/generated_exams.json.
  4. Batch is sent to POST /train via TrainerClient.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.config import INFINITE_MODE, NUM_EXAMS_TO_GENERATE, EXAMPLES_PER_BATCH, BATCH_COOLDOWN_SEC, validate_config
from service.exam_builder import get_or_create_exam_builder
from service.logger import logger
from client.trainer_client import get_or_create_trainer_client


class SyntheticTrainingGenerator:
    """Main orchestrator for synthetic training batch generation and submission."""

    def __init__(self):
        logger.info("Initializing synthetic training generator...")

        try:
            validate_config()
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            raise

        self.exam_builder = get_or_create_exam_builder()
        self.trainer_client = get_or_create_trainer_client()

        self.output_dir = Path(__file__).parent.parent / "output"
        self.output_dir.mkdir(exist_ok=True)
        self.exams_file = self.output_dir / "generated_exams.json"

        if not self.exams_file.exists():
            self._initialize_exams_file()

        self.generated_count = 0
        self.successful_count = 0
        self.failed_count = 0

        logger.info("Synthetic training generator initialized successfully")

    # ------------------------------------------------------------------ #
    # Public run API                                                        #
    # ------------------------------------------------------------------ #

    def run(self, num_exams: Optional[int] = None, infinite: bool = False):
        """
        Main execution loop.

        Args:
            num_exams: Number of batches to generate (falls back to config).
            infinite: If True, run forever.
        """
        num_exams = num_exams or NUM_EXAMS_TO_GENERATE
        infinite = infinite or INFINITE_MODE

        logger.info(f"Starting batch generation — num_batches: {num_exams}, infinite: {infinite}")

        if not self.trainer_client.verify_connection():
            logger.warning("API endpoint unreachable — will retry during generation")

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

    # ------------------------------------------------------------------ #
    # Internal loops                                                        #
    # ------------------------------------------------------------------ #

    def _run_fixed_count(self, num_batches: int):
        logger.info(f"Running generator for {num_batches} batches")
        for i in range(num_batches):
            self._generate_and_send_batch(i + 1, num_batches)
            if i < num_batches - 1:
                self._cooldown()
        self._print_summary()

    def _run_infinite_loop(self):
        logger.info("Running generator in infinite loop mode")
        iteration = 0
        while True:
            iteration += 1
            self._generate_and_send_batch(iteration, None)
            if iteration % 100 == 0:
                self._print_summary()
            self._cooldown()

    # ------------------------------------------------------------------ #
    # Cooldown                                                              #
    # ------------------------------------------------------------------ #

    def _cooldown(self):
        """
        Wait BATCH_COOLDOWN_SEC seconds between batches.

        Prevents rate-limiting on the Groq API and avoids saturating /train.
        The full cycle (generate → validate → send) completes before the
        cooldown starts, so each batch is always processed to completion.
        Skipped if BATCH_COOLDOWN_SEC == 0.
        """
        if BATCH_COOLDOWN_SEC <= 0:
            return
        logger.info(f"Cooldown — waiting {BATCH_COOLDOWN_SEC}s before next batch...")
        time.sleep(BATCH_COOLDOWN_SEC)

    # ------------------------------------------------------------------ #
    # Core: generate one batch and send it                                  #
    # ------------------------------------------------------------------ #

    def _generate_and_send_batch(self, current: int, total: Optional[int] = None):
        try:
            start_time = time.time()
            batch = self.exam_builder.generate_exam(
                num_questions=EXAMPLES_PER_BATCH,
                adaptive=True,
            )
            generation_time = time.time() - start_time

            if batch is None:
                self.failed_count += 1
                logger.error(
                    f"Batch generation returned None ({current}/{total or '∞'}) — skipping"
                )
                return

            self.generated_count += 1
            train_id = batch.get("train_id", "unknown")
            examples_count = len(batch.get("examples", []))

            logger.info(
                f"Batch {train_id} generated in {generation_time:.2f}s "
                f"({examples_count} examples, {current}/{total or '∞'})"
            )

            self._save_batch_to_json(batch)

            send_start = time.time()
            success = self.trainer_client.send_exam(batch)
            send_time = time.time() - send_start

            if success:
                self.successful_count += 1
                logger.info(f"Batch {train_id} sent successfully in {send_time:.2f}s")
            else:
                self.failed_count += 1
                logger.error(f"Failed to send batch {train_id}")

        except Exception as e:
            self.failed_count += 1
            logger.error(f"Error generating/sending batch: {e}", exc_info=True)

    # ------------------------------------------------------------------ #
    # Persistence                                                           #
    # ------------------------------------------------------------------ #

    def _initialize_exams_file(self):
        with open(self.exams_file, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        logger.info(f"Created exams file at {self.exams_file}")

    def _save_batch_to_json(self, batch: dict):
        """Append generated batch to the local output JSON array."""
        try:
            with open(self.exams_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            entry = {
                "train_id":  batch.get("train_id"),
                "examples":  batch.get("examples", []),
                "metadata":  batch.get("metadata", {
                    "source": "synthetic_generator_llm",
                    "version": "3.0.0",
                    "language": "fr",
                    "includes_errors": "true",
                    "question_types": "",
                }),
                "_generated_at": datetime.now().isoformat(),
            }

            data.append(entry)

            with open(self.exams_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(
                f"Batch {batch.get('train_id')} saved to {self.exams_file} (total: {len(data)})"
            )
        except Exception as e:
            logger.error(f"Error saving batch to JSON: {e}")

    # ------------------------------------------------------------------ #
    # Summary                                                               #
    # ------------------------------------------------------------------ #

    def _print_summary(self):
        logger.info("=" * 60)
        logger.info("GENERATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total generated:   {self.generated_count}")
        logger.info(f"Successfully sent: {self.successful_count}")
        logger.info(f"Failed:            {self.failed_count}")
        if self.generated_count > 0:
            rate = (self.successful_count / self.generated_count) * 100
            logger.info(f"Success rate:      {rate:.1f}%")
        logger.info("=" * 60)
