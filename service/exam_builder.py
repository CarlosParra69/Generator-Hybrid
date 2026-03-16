"""
Train batch builder — orchestrates complete /train JSON generation via Groq LLM.

Pipeline (promt_generator.md §4):
  1. Pick random topic, level and type distribution for the batch.
  2. Build dynamic master prompt and call Groq.
  3. Validate generated JSON against schema (rules_generator.md §7).
  4. If validation fails, retry up to MAX_BATCH_RETRIES times.
  5. Return the validated batch ready to send to POST /train.

Sample files in sample/ are loaded on startup as format reference and
logged so the operator can confirm the builder is aware of them.
"""

import json
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.config import (
    BATCH_LEVELS,
    BATCH_TOPICS,
    EXAMPLES_PER_BATCH,
    MAX_RETRIES,
    SAMPLE_DIR,
    SAMPLE_FILES,
)
from service.llm_generator import get_or_create_llm
from service.logger import logger
from service.train_validator import validate_training_batch

MAX_BATCH_RETRIES = MAX_RETRIES


class TrainBatchBuilder:
    """
    Generates complete training JSON batches via LLM following
    the architecture defined in promt_generator.md and rules_generator.md.
    """

    def __init__(self):
        self.llm = get_or_create_llm()
        self._load_sample_files()

    # ------------------------------------------------------------------ #
    # Initialisation                                                        #
    # ------------------------------------------------------------------ #

    def _load_sample_files(self) -> None:
        """Load sample JSON files as format reference (logged, not used in prompt to save tokens)."""
        sample_dir = Path(SAMPLE_DIR)
        loaded: List[str] = []

        for fname in SAMPLE_FILES:
            path = sample_dir / fname
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    examples_count = len(data.get("examples", []))
                    loaded.append(f"{fname} ({examples_count} examples)")
                except Exception as e:
                    logger.warning(f"Could not load sample file '{fname}': {e}")
            else:
                logger.warning(f"Sample file not found: {path}")

        if loaded:
            logger.info(f"TrainBatchBuilder — reference samples loaded: {', '.join(loaded)}")
        else:
            logger.warning("TrainBatchBuilder — no sample files found in sample/")

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def generate_exam(
        self,
        num_questions: int = EXAMPLES_PER_BATCH,
        adaptive: bool = True,  # kept for compatibility with generator.py
    ) -> Optional[Dict[str, Any]]:
        """
        Generate one complete /train batch via Groq LLM with validation.

        Args:
            num_questions: Target number of examples in the batch.
            adaptive: Ignored (kept for interface compatibility).

        Returns:
            Validated batch dict, or None if all retries failed.
        """
        topic = random.choice(BATCH_TOPICS)
        level = random.choice(BATCH_LEVELS)
        train_id = self._make_train_id()
        type_dist = self._build_type_distribution(num_questions)
        type_dist_str = ", ".join(f"{t}: {n}" for t, n in type_dist.items())

        logger.info(
            f"Generating batch {train_id} — topic: '{topic}', level: {level}, "
            f"examples: {num_questions}, distribution: {type_dist_str}"
        )

        # Delay between batch-level retries (on top of the 429 handling inside llm_generator).
        # Gives the TPM window more room to recover before a completely new generation attempt.
        _RETRY_DELAY_SEC = 8

        for attempt in range(1, MAX_BATCH_RETRIES + 1):
            batch = self.llm.generate_training_batch(
                topic=topic,
                level=level,
                num_examples=num_questions,
                type_distribution=type_dist_str,
                train_id=train_id,
            )

            if batch is None:
                logger.warning(
                    f"Batch {train_id}: LLM returned None (attempt {attempt}/{MAX_BATCH_RETRIES})"
                )
                if attempt < MAX_BATCH_RETRIES:
                    logger.info(f"Waiting {_RETRY_DELAY_SEC}s before next batch attempt...")
                    time.sleep(_RETRY_DELAY_SEC)
                continue

            is_valid, errors = validate_training_batch(batch)

            if is_valid:
                actual_count = len(batch.get("examples", []))
                logger.info(
                    f"Batch {train_id} validated OK — {actual_count} examples generated"
                )
                return batch

            logger.warning(
                f"Batch {train_id} validation failed (attempt {attempt}/{MAX_BATCH_RETRIES}): "
                f"{errors[:5]}"
            )
            if attempt < MAX_BATCH_RETRIES:
                logger.info(f"Waiting {_RETRY_DELAY_SEC}s before re-generating...")
                time.sleep(_RETRY_DELAY_SEC)

        logger.error(f"Batch {train_id}: could not generate valid batch after {MAX_BATCH_RETRIES} attempts")
        return None

    # ------------------------------------------------------------------ #
    # Helpers                                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_type_distribution(total: int) -> Dict[str, int]:
        """
        Build per-type example counts following rules_generator.md §5.
            writing_text / speaking_record : 22–35 %
            single_choice                  : 18–28 %
            fill_blank                     : 13–22 %
            image                          :  8–15 %
            ordering                       : remainder (10–20 %)
        """
        writing  = max(1, round(total * random.uniform(0.22, 0.35)))
        single   = max(1, round(total * random.uniform(0.18, 0.28)))
        fill     = max(1, round(total * random.uniform(0.13, 0.22)))
        image    = max(1, round(total * random.uniform(0.08, 0.15)))
        ordering = max(1, total - writing - single - fill - image)

        return {
            "writing_text": writing,
            "single_choice": single,
            "fill_blank": fill,
            "image": image,
            "ordering": ordering,
        }

    @staticmethod
    def _make_train_id() -> str:
        date_str = datetime.now().strftime("%Y-%m-%d")
        suffix = uuid.uuid4().hex[:6]
        return f"t{date_str}-fr-llm-{suffix}"


# ------------------------------------------------------------------ #
# Singleton                                                            #
# ------------------------------------------------------------------ #

_exam_builder: Optional[TrainBatchBuilder] = None


def get_or_create_exam_builder() -> TrainBatchBuilder:
    """Get or create the TrainBatchBuilder singleton."""
    global _exam_builder
    if _exam_builder is None:
        _exam_builder = TrainBatchBuilder()
    return _exam_builder
