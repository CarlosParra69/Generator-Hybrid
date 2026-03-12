"""
LLM Generator module for creating realistic student answers using Groq API.
Groq is free, fast, and does not require a local setup.
"""

import json
from typing import Optional
import requests

from config.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
)
from service.logger import logger, log_error


class GroqProvider:
    """Groq API provider (free, fast)."""
    
    def __init__(self):
        """Initialize Groq provider."""
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required for Groq provider")
        
        self.api_key = GROQ_API_KEY
        self.model = GROQ_MODEL
        self.base_url = "https://api.groq.com/openai/v1"
    
    def generate_answer(
        self,
        question: str,
        cefr_level: str,
        difficulty: int,
        expected_keywords: list,
        expected_min_words: int,
        expected_max_words: int,
    ) -> str:
        """Generate answer using Groq API."""
        
        prompt = self._build_prompt(
            question,
            cefr_level,
            difficulty,
            expected_keywords,
            expected_min_words,
            expected_max_words,
        )
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 500,
                },
                timeout=30,
            )
            
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"].strip()
                return answer
            else:
                logger.warning("No choices in Groq response")
                return self._generate_fallback_answer(cefr_level, expected_keywords)
                
        except requests.exceptions.HTTPError as e:
            try:
                error_detail = response.json()
                log_error(f"Groq API error ({response.status_code}): {error_detail}", error=e)
            except:
                log_error(f"Groq API error: {str(e)}", error=e)
            return self._generate_fallback_answer(cefr_level, expected_keywords)
        except requests.exceptions.RequestException as e:
            log_error(f"Groq API error: {str(e)}", error=e)
            return self._generate_fallback_answer(cefr_level, expected_keywords)
    
    def _build_prompt(
        self,
        question: str,
        cefr_level: str,
        difficulty: int,
        expected_keywords: list,
        expected_min_words: int,
        expected_max_words: int,
    ) -> str:
        """Build the prompt for Groq."""
        
        keywords_str = ", ".join(expected_keywords)
        
        prompt = f"""Tu es un étudiant français de niveau {cefr_level} qui répond à une question d'examen.

QUESTION: {question}

INSTRUCTIONS:
- Réponds comme un étudiant de niveau {cefr_level} (pas parfait, mais compréhensible)
- Utilise entre {expected_min_words} et {expected_max_words} mots
- Essaie d'inclure ces mots clés: {keywords_str}
- Inclus occasionnellement de petites erreurs de grammaire ou d'orthographe (accents manquants, accord des verbes) - c'est naturel pour un étudiant
- Réponds uniquement avec la réponse à la question, sans explications additionnelles
- La réponse doit être naturelle et authentique

RÉPONSE:"""
        
        return prompt
    
    def _generate_fallback_answer(self, cefr_level: str, expected_keywords: list) -> str:
        """Generate a fallback answer if API fails."""
        # Simple fallback answers based on CEFR level
        fallbacks = {
            "A1": "C'est bien. J'ai aujourd'hui travaillé et c'est bon.",
            "A2": "Aujourd'hui j'ai travaillé et c'étaient bien. J'ai aussi lu et c'est facile.",
            "B1": "Aujourd'hui j'ai eu une bonne journée. J'ai travaillé sur différents projets et c'était intéressant. Les résultats sont satisfaisants.",
            "B2": "Aujourd'hui a été une journée productive. J'ai accomplies plusieurs tâches importantes et j'ai pu avancer significativement sur mes projets. Les résultats sont encourageants.",
            "C1": "Cette journée a été particulièrement fructueuse professionnellement. J'ai eu l'opportunité de collaborer efficacement avec mes collègues, et nous avons pu résoudre les problèmes majeurs.",
        }
        
        level = cefr_level if cefr_level in fallbacks else "B1"
        return fallbacks[level]




# Singleton instance
_llm_provider: Optional[GroqProvider] = None

def get_or_create_llm() -> GroqProvider:
    """Get or create the Groq LLM provider singleton."""
    global _llm_provider
    if _llm_provider is None:
        logger.info("Using Groq LLM provider (free, fast)")
        _llm_provider = GroqProvider()
    return _llm_provider
