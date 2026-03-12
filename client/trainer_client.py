"""
Trainer client module for sending exams to the training API endpoint.
Handles retries, error handling, and API communication.
"""

import json
import time
from typing import Dict, Any, Optional
import requests

from config.config import (
    TRAINER_API_URL,
    TRAINER_API_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
)
from service.logger import logger, log_api_request, log_error


class TrainerClient:
    """Client for communicating with the training API."""
    
    def __init__(
        self,
        api_url: str = TRAINER_API_URL,
        timeout: int = TRAINER_API_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        retry_delay: int = RETRY_DELAY,
    ):
        """
        Initialize trainer client.
        
        Args:
            api_url: URL of the training API endpoint
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.api_url = api_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        logger.info(f"Trainer client initialized - API URL: {api_url}")
    
    def send_exam(self, exam: Dict[str, Any]) -> bool:
        """
        Send exam to training endpoint with retry logic.
        
        Args:
            exam: Complete exam dictionary
            
        Returns:
            True if successful, False otherwise
        """
        exam_id = exam.get("exam_id", "unknown")
        
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._make_request(exam)
                
                if response.status_code in [200, 201]:
                    log_api_request(exam_id, "success", response.status_code)
                    logger.info(
                        f"Exam {exam_id} sent successfully (attempt {attempt}/{self.max_retries})"
                    )
                    return True
                
                elif response.status_code in [400, 422]:
                    # Client error - don't retry
                    log_api_request(exam_id, "client_error", response.status_code)
                    logger.error(
                        f"Client error ({response.status_code}) sending exam {exam_id}: {response.text}"
                    )
                    return False
                
                elif response.status_code == 500:
                    # Server error - retry
                    if attempt < self.max_retries:
                        log_api_request(exam_id, "server_error", response.status_code)
                        logger.warning(
                            f"Server error ({response.status_code}) for exam {exam_id}. "
                            f"Retrying in {self.retry_delay}s (attempt {attempt}/{self.max_retries})"
                        )
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        log_api_request(exam_id, "max_retries_exceeded", response.status_code)
                        logger.error(
                            f"Max retries exceeded for exam {exam_id} (status: {response.status_code})"
                        )
                        return False
                
                else:
                    # Other status codes
                    if attempt < self.max_retries:
                        logger.warning(
                            f"Unexpected status {response.status_code} for exam {exam_id}. "
                            f"Retrying in {self.retry_delay}s (attempt {attempt}/{self.max_retries})"
                        )
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        log_api_request(exam_id, "failed", response.status_code)
                        logger.error(f"Failed to send exam {exam_id} after {self.max_retries} attempts")
                        return False
            
            except requests.exceptions.Timeout:
                if attempt < self.max_retries:
                    logger.warning(
                        f"Timeout sending exam {exam_id}. "
                        f"Retrying in {self.retry_delay}s (attempt {attempt}/{self.max_retries})"
                    )
                    time.sleep(self.retry_delay)
                    continue
                else:
                    log_error(f"Timeout sending exam {exam_id} after {self.max_retries} attempts", exam_id=exam_id)
                    return False
            
            except requests.exceptions.ConnectionError as e:
                if attempt < self.max_retries:
                    logger.warning(
                        f"Connection error sending exam {exam_id}. "
                        f"Retrying in {self.retry_delay}s (attempt {attempt}/{self.max_retries})"
                    )
                    time.sleep(self.retry_delay)
                    continue
                else:
                    log_error(f"Connection error sending exam {exam_id}", error=e, exam_id=exam_id)
                    return False
            
            except Exception as e:
                log_error(f"Unexpected error sending exam {exam_id}", error=e, exam_id=exam_id)
                return False
        
        return False
    
    def _make_request(self, exam: Dict[str, Any]) -> requests.Response:
        """
        Make HTTP POST request to training endpoint.
        
        Args:
            exam: Exam dictionary
            
        Returns:
            Response object
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        response = requests.post(
            self.api_url,
            json=exam,
            headers=headers,
            timeout=self.timeout,
        )
        
        return response
    
    def verify_connection(self) -> bool:
        """
        Verify that API endpoint is reachable.
        
        Returns:
            True if endpoint is reachable, False otherwise
        """
        try:
            # Try to make a simple request to check connectivity
            response = requests.head(
                self.api_url,
                timeout=5,
            )
            logger.info(f"API connection verified (status: {response.status_code})")
            return True
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to API endpoint: {self.api_url}")
            return False
        except Exception as e:
            logger.warning(f"Error verifying API connection: {e}")
            return True  # Assume connection OK for now


# Singleton instance
_trainer_client: Optional[TrainerClient] = None

def get_or_create_trainer_client() -> TrainerClient:
    """Get or create the trainer client singleton."""
    global _trainer_client
    if _trainer_client is None:
        _trainer_client = TrainerClient()
    return _trainer_client
