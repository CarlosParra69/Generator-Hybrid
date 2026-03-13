"""
Exam builder module for constructing complete synthetic exams.
Reads questions from training data and generates corresponding student answers.
"""

import json
import random
import uuid
from typing import Dict, List, Any, Optional
from pathlib import Path

from config.config import (
    TRAINING_DATA_FILE,
    CEFR_WORD_EXPECTATIONS,
    MCQ_CORRECT_PROBABILITY,
    OPEN_QUESTION_TIME_MIN,
    OPEN_QUESTION_TIME_MAX,
    MCQ_TIME_MIN,
    MCQ_TIME_MAX,
)
from service.llm_generator import get_or_create_llm
from service.error_injector import get_or_create_error_injector
from service.logger import logger


class ExamBuilder:
    """Builds complete synthetic exam JSON structures."""
    
    def __init__(self, training_data_path: str = TRAINING_DATA_FILE):
        """
        Initialize exam builder.
        
        Args:
            training_data_path: Path to training data JSON file
        """
        self.training_data = self._load_training_data(training_data_path)
        self.questions = self.training_data.get("examples", [])
        self.llm = get_or_create_llm()
        self.error_injector = get_or_create_error_injector()
        
        logger.info(f"Loaded {len(self.questions)} questions from training data")
    
    def _load_training_data(self, path: str) -> Dict:
        """Load training data from JSON file."""
        try:
            if not Path(path).exists():
                raise FileNotFoundError(f"Training data file not found: {path}")
            
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            logger.info(f"Loaded training data from {path}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in training data: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
    
    def generate_exam(
        self,
        num_questions: int = 5,
        adaptive: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a complete synthetic exam in train_french_priority.json format.
        
        Args:
            num_questions: Number of questions to include
            adaptive: Whether exam is adaptive (true in current schema)
            
        Returns:
            Complete exam JSON structure matching train_french_priority.json format
        """
        exam_id = self._generate_exam_id()
        candidate_id = self._generate_candidate_id()
        
        # Select random questions
        selected_questions = random.sample(self.questions, min(num_questions, len(self.questions)))
        
        # Build examples with integrated answers
        examples_list = []
        
        for question_data in selected_questions:
            # Build enriched example with answers integrated
            example_obj = self._build_example_with_answer(question_data)
            examples_list.append(example_obj)
        
        # Create exam in train_french_priority.json format
        exam = {
            "train_id": f"train_{exam_id}",
            "exam_id": exam_id,
            "candidate_id": candidate_id,
            "adaptive": adaptive,
            "examples": examples_list,
        }
        
        logger.info(f"Generated exam {exam_id} with {len(examples_list)} questions")
        return exam
    
    def _build_example_with_answer(self, question_data: Dict) -> Dict[str, Any]:
        """
        Build example with integrated answer (examples_answers).
        Matches train_french_priority.json format:
        - Open questions: include examples_answers with student responses
        - MCQ: NO examples_answers, just question and correct answer
        """
        example = {
            "question_id": question_data.get("question_id"),
            "text": question_data.get("text"),
            "type": question_data.get("type"),  # "open" or "mcq"
            "language": question_data.get("language"),
            "difficulty": question_data.get("difficulty"),
        }
        
        # Add expected_keywords if present
        if "expected_keywords" in question_data:
            example["expected_keywords"] = question_data["expected_keywords"]
        
        # Add rubric if present (for open questions)
        if question_data.get("type") == "open" and "rubric" in question_data:
            example["rubric"] = question_data["rubric"]
        
        # Add MCQ options if present
        if question_data.get("type") == "mcq" and "options" in question_data:
            example["options"] = question_data["options"]
            # For MCQ, add the correct answer
            if "answer" in question_data:
                example["answer"] = question_data["answer"]
        
        # Only add examples_answers for OPEN questions, NOT for MCQ
        if question_data.get("type") == "open":
            # Generate exactly 2 student answers with varying quality (matching train_french_priority.json)
            # train_french_priority.json sempre has 2 examples_answers per open question
            min_words = question_data.get("rubric", {}).get("expected_min_words", 30)
            max_words = question_data.get("rubric", {}).get("expected_min_words", 100) * 1.5
            
            examples_answers = []
            
            # Generate EXACTLY 2 different answers with varying quality
            for answer_idx in range(2):
                student_answer = self._generate_student_answer(question_data)
                word_count = len(student_answer.split())
                
                # Score varies based on word count and some randomness
                if min_words <= word_count <= max_words:
                    score = random.uniform(0.65, 0.85)
                else:
                    score = random.uniform(0.40, 0.60)
                
                examples_answers.append({
                    "text": student_answer,
                    "score": round(score, 2)
                })
            
            example["examples_answers"] = examples_answers
        
        # MCQ questions do NOT have examples_answers (no student responses needed)
        
        return example
    
    def _build_question(self, question_data: Dict) -> Dict[str, Any]:
        """Build individual question object."""
        question = {
            "question_id": question_data.get("question_id"),
            "text": question_data.get("text"),
            "type": question_data.get("type"),  # "open" or "mcq"
            "language": question_data.get("language"),
            "difficulty": question_data.get("difficulty"),
        }
        
        # Add MCQ options if present
        if question_data.get("type") == "mcq" and "options" in question_data:
            question["options"] = question_data["options"]
        
        # Add rubric if present (for open questions)
        if question_data.get("type") == "open" and "rubric" in question_data:
            question["rubric"] = question_data["rubric"]
        
        return question
    
    def _build_answer(self, question_data: Dict, question_obj: Dict) -> Dict[str, Any]:
        """Build corresponding answer object."""
        question_id = question_data.get("question_id")
        question_type = question_data.get("type")
        
        answer = {
            "question_id": question_id,
            "time_spent_sec": self._generate_time_spent(question_type),
        }
        
        if question_type == "open":
            # Generate student answer using LLM
            student_answer = self._generate_student_answer(question_data)
            answer["student_answer"] = student_answer
        
        elif question_type == "mcq":
            # Simulate MCQ response
            correct_answer = question_data.get("answer")
            options = question_data.get("options", [])
            
            if random.random() < MCQ_CORRECT_PROBABILITY:
                answer["selected_option"] = correct_answer
                answer["is_correct"] = True
            else:
                wrong_options = [o for o in options if o != correct_answer]
                answer["selected_option"] = random.choice(wrong_options) if wrong_options else correct_answer
                answer["is_correct"] = False
        
        return answer
    
    def _generate_student_answer(self, question_data: Dict) -> str:
        """Generate a student answer using LLM and apply error injection."""
        question_text = question_data.get("text", "")
        rubric = question_data.get("rubric", {})
        cefr_level = rubric.get("level", "A1")
        expected_keywords = question_data.get("expected_keywords", rubric.get("expected_keywords", []))
        expected_min_words = rubric.get("expected_min_words", CEFR_WORD_EXPECTATIONS.get(cefr_level, {}).get("min", 30))
        expected_max_words = rubric.get("expected_min_words", CEFR_WORD_EXPECTATIONS.get(cefr_level, {}).get("max", 100))
        difficulty = question_data.get("difficulty", 1)
        
        try:
            # Generate answer using LLM
            answer = self.llm.generate_answer(
                question=question_text,
                cefr_level=cefr_level,
                difficulty=difficulty,
                expected_keywords=expected_keywords,
                expected_min_words=expected_min_words,
                expected_max_words=expected_max_words,
            )
            
            # Inject errors
            answer = self.error_injector.inject_errors(answer)
            
            # Adjust word count
            target_words = (expected_min_words + expected_max_words) // 2
            answer = self.error_injector.vary_word_count(answer, target_words, tolerance=10)
            
            return answer
        
        except Exception as e:
            logger.warning(f"Error generating answer for question {question_data.get('question_id')}: {e}")
            # Return fallback answer
            return self._get_fallback_answer(cefr_level)
    
    def _get_fallback_answer(self, cefr_level: str) -> str:
        """Get a fallback answer if LLM generation fails."""
        fallback_answers = {
            "A1": "C'est bien aujourd'hui. J'ai travaillé.",
            "A2": "Aujourd'hui j'ai eu une bonne journée. J'ai travaillé et j'ai appris beaucoup de choses nouvelles.",
            "B1": "Aujourd'hui a été une journée intéressante. J'ai travaillé sur plusieurs projets importants et j'ai collaboré avec mes collègues.",
            "B2": "Aujourd'hui a été une journée très productive et enrichissante. J'ai eu l'occasion de travailler sur des tâches significatives et d'interagir avec mon équipe.",
            "C1": "Aujourd'hui a représenté une journée particulièrement fructueuse du point de vue professionnel. J'ai pu accomplir diverses tâches complexes et contribuer efficacement aux objectifs collectifs.",
        }
        
        return fallback_answers.get(cefr_level, fallback_answers["B1"])
    
    def _generate_time_spent(self, question_type: str) -> int:
        """Generate realistic time spent on question."""
        if question_type == "open":
            return random.randint(
                OPEN_QUESTION_TIME_MIN,
                OPEN_QUESTION_TIME_MAX
            )
        else:  # mcq
            return random.randint(
                MCQ_TIME_MIN,
                MCQ_TIME_MAX
            )
    
    @staticmethod
    def _generate_exam_id() -> str:
        """Generate unique exam ID."""
        return f"exam_{uuid.uuid4().hex[:12]}"
    
    @staticmethod
    def _generate_candidate_id() -> str:
        """Generate synthetic candidate ID."""
        return f"candidate_{random.randint(1000, 999999)}"


# Singleton instance
_exam_builder: Optional[ExamBuilder] = None

def get_or_create_exam_builder() -> ExamBuilder:
    """Get or create the exam builder singleton."""
    global _exam_builder
    if _exam_builder is None:
        _exam_builder = ExamBuilder()
    return _exam_builder
