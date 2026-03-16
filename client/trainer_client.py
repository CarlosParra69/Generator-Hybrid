"""
Trainer client — sends generated /train batches to the backend endpoint.
Handles retries, error handling and API communication.
"""

import json
import time
from typing import Any, Dict, Optional

import requests

from config.config import MAX_RETRIES, RETRY_DELAY, TRAINER_API_TIMEOUT, TRAINER_API_URL
from service.logger import log_api_request, log_error, logger


class TrainerClient:
    """Client for communicating with the training API (POST /train)."""

    def __init__(
        self,
        api_url: str = TRAINER_API_URL,
        timeout: int = TRAINER_API_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        retry_delay: int = RETRY_DELAY,
    ):
        self.api_url = api_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info(f"TrainerClient initialized — endpoint: {api_url}")

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def send_exam(self, batch: Dict[str, Any]) -> bool:
        """
        Send a training batch to POST /train with retry logic.

        The batch is expected to be in the exact format produced by
        TrainBatchBuilder (train_id, examples, metadata).

        Args:
            batch: Validated training batch dict.

        Returns:
            True if accepted (2xx), False otherwise.
        """
        train_id = batch.get("train_id", "unknown")

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._make_request(batch)

                if response.status_code in (200, 201):
                    log_api_request(train_id, "success", response.status_code)
                    logger.info(
                        f"Batch {train_id} accepted by /train "
                        f"(attempt {attempt}/{self.max_retries})"
                    )
                    return True

                if response.status_code in (400, 422):
                    # Client error — do not retry, log body for debugging
                    log_api_request(train_id, "client_error", response.status_code)
                    logger.error(
                        f"Client error {response.status_code} for batch {train_id}: "
                        f"{response.text[:500]}"
                    )
                    return False

                if response.status_code == 500:
                    if attempt < self.max_retries:
                        log_api_request(train_id, "server_error", response.status_code)
                        logger.warning(
                            f"Server error 500 for batch {train_id} — "
                            f"retrying in {self.retry_delay}s "
                            f"(attempt {attempt}/{self.max_retries})"
                        )
                        time.sleep(self.retry_delay)
                        continue
                    log_api_request(train_id, "max_retries_exceeded", response.status_code)
                    logger.error(f"Max retries exceeded for batch {train_id}")
                    return False

                # Unexpected status
                if attempt < self.max_retries:
                    logger.warning(
                        f"Unexpected status {response.status_code} for batch {train_id} — "
                        f"retrying in {self.retry_delay}s (attempt {attempt}/{self.max_retries})"
                    )
                    time.sleep(self.retry_delay)
                    continue

                log_api_request(train_id, "failed", response.status_code)
                logger.error(
                    f"Failed to send batch {train_id} after {self.max_retries} attempts "
                    f"(status {response.status_code})"
                )
                return False

            except requests.exceptions.Timeout:
                if attempt < self.max_retries:
                    logger.warning(
                        f"Timeout sending batch {train_id} — "
                        f"retrying in {self.retry_delay}s (attempt {attempt}/{self.max_retries})"
                    )
                    time.sleep(self.retry_delay)
                    continue
                log_error(f"Timeout sending batch {train_id} after {self.max_retries} attempts", exam_id=train_id)
                return False

            except requests.exceptions.ConnectionError as e:
                if attempt < self.max_retries:
                    logger.warning(
                        f"Connection error sending batch {train_id} — "
                        f"retrying in {self.retry_delay}s (attempt {attempt}/{self.max_retries})"
                    )
                    time.sleep(self.retry_delay)
                    continue
                log_error(f"Connection error sending batch {train_id}", error=e, exam_id=train_id)
                return False

            except Exception as e:
                log_error(f"Unexpected error sending batch {train_id}", error=e, exam_id=train_id)
                return False

        return False

    def verify_connection(self) -> bool:
        """
        Verify API reachability by sending a minimal valid POST /train payload.
        Uses single_choice (current type, not legacy mcq).

        Returns:
            True if endpoint responds (any HTTP status), False if unreachable.
        """
        test_payload = {
            "train_id": "test-connection-check",
            "examples": [
                {
                    "question_id": "q_test_001",
                    "type": "single_choice",
                    "text": "Quelle est la couleur du ciel?",
                    "language": "fr",
                    "difficulty": 1,
                    "options": ["bleu", "vert", "rouge", "jaune"],
                    "answer": "bleu",
                }
            ],
            "metadata": {
                "source": "connection_check",
                "version": "3.0.0",
                "language": "fr",
                "includes_errors": "false",
                "question_types": "single_choice",
            },
        }
        try:
            response = requests.post(self.api_url, json=test_payload, timeout=5)
            logger.info(
                f"API connection verified — {self.api_url} responded with {response.status_code}"
            )
            return True
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to API endpoint: {self.api_url}")
            return False
        except Exception as e:
            logger.warning(f"Error verifying API connection: {e}")
            return True  # Don't block generation if check is inconclusive

    # ------------------------------------------------------------------ #
    # Internal                                                              #
    # ------------------------------------------------------------------ #

    def _make_request(self, batch: Dict[str, Any]) -> requests.Response:
        """
        POST the batch to /train.

        Sends the full batch (train_id + examples + metadata) exactly as
        generated — metadata is already populated by TrainBatchBuilder/LLM.
        """
        payload = {
            "train_id": batch.get("train_id"),
            "examples": batch.get("examples", []),
            "metadata": batch.get("metadata", {
                "source": "synthetic_generator_llm",
                "version": "3.0.0",
                "language": "fr",
                "includes_errors": "true",
                "question_types": "",
            }),
        }

        return requests.post(
            self.api_url,
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=self.timeout,
        )


# Singleton instance
_trainer_client: Optional[TrainerClient] = None


def get_or_create_trainer_client() -> TrainerClient:
    """Get or create the TrainerClient singleton."""
    global _trainer_client
    if _trainer_client is None:
        _trainer_client = TrainerClient()
    return _trainer_client
