"""
LLM Generator module — Groq API integration.

Two modes of operation:
1. generate_training_batch(): generates a complete /train JSON via the master prompt
   (new, primary mode — used by TrainBatchBuilder)
2. generate_answer(): generates a single student answer text
   (legacy helper, kept for fallback use)
"""

import json
import re
from typing import Any, Dict, Optional

import requests

from config.config import GROQ_API_KEY, GROQ_MODEL
from service.logger import log_error, logger


class GroqProvider:
    """Groq API provider (free, fast)."""

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required for Groq provider")
        self.api_key = GROQ_API_KEY
        self.model = GROQ_MODEL
        self.base_url = "https://api.groq.com/openai/v1"

    # ------------------------------------------------------------------ #
    # Primary: full training batch generation                              #
    # ------------------------------------------------------------------ #

    def generate_training_batch(
        self,
        topic: str,
        level: str,
        num_examples: int,
        type_distribution: str,
        train_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Ask Groq to generate a complete /train JSON batch.

        Uses the master prompt defined in promt_generator.md §2 and the
        rules in rules_generator.md.

        Returns:
            Parsed JSON dict, or None if generation or parsing fails.
        """
        prompt = self._build_batch_prompt(topic, level, num_examples, type_distribution, train_id)

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
                    "temperature": 0.75,
                    "max_tokens": 4096,
                },
                timeout=90,
            )
            response.raise_for_status()
            result = response.json()

            if "choices" not in result or not result["choices"]:
                logger.warning("No choices in Groq response for batch generation")
                return None

            content = result["choices"][0]["message"]["content"].strip()
            return self._parse_json_response(content)

        except requests.exceptions.HTTPError as e:
            try:
                error_detail = response.json()
                log_error(f"Groq API HTTP error ({response.status_code}): {error_detail}", error=e)
            except Exception:
                log_error(f"Groq API HTTP error: {e}", error=e)
            return None
        except requests.exceptions.RequestException as e:
            log_error(f"Groq API request error: {e}", error=e)
            return None

    def _build_batch_prompt(
        self,
        topic: str,
        level: str,
        num_examples: int,
        type_distribution: str,
        train_id: str,
    ) -> str:
        """
        Master prompt for complete training batch generation.
        Based on promt_generator.md §2 and rules_generator.md §1–§8.
        """
        return f"""Eres un generador estricto de datasets JSON para el endpoint POST /train de un modelo de evaluacion de frances DELF.

OBJETIVO
Generar un JSON de entrenamiento listo para enviar a /train, validable por Pydantic, sin texto extra.

SALIDA OBLIGATORIA
- Devuelve SOLO un JSON valido.
- No uses markdown, no uses bloques de codigo, no uses comillas triples.
- No agregues comentarios ni texto antes o despues del JSON.

FORMATO RAIZ
{{
  "train_id": "{train_id}",
  "examples": [ ... ],
  "metadata": {{
    "source": "french_priority_corpus_v3_llm",
    "version": "3.0.0",
    "language": "fr",
    "includes_errors": "true",
    "question_types": "<lista separada por comas de los tipos usados>"
  }}
}}

TIPOS PERMITIDOS (usar solo estos)
- single_choice
- writing_text
- fill_blank
- ordering
- speaking_record

NO INCLUIR: audio, video.

REGLAS POR TIPO

1) single_choice
   Campos obligatorios: question_id, type, text, language, difficulty, options, answer
   - options: minimo 3 opciones, ideal 4.
   - answer: debe existir literalmente dentro de options.

2) writing_text
   Campos obligatorios: question_id, type, text, language, difficulty, expected_keywords, rubric, examples_answers
   - rubric.level: uno de [A1-, A1, A1+, A2-, A2, A2+, B1-, B1, B1+, B2-, B2]
   - rubric.expected_min_words: entero >= 1
   - rubric.expected_keywords: lista de palabras clave
   - rubric.criteria_weights: usar SIEMPRE estos cinco criterios con estos pesos exactos:
       {{"task_realisation": 0.25, "coherence": 0.20, "sociolinguistic": 0.15, "lexicon": 0.20, "morphosyntax": 0.20}}
   - examples_answers: EXACTAMENTE 3 respuestas (ver VARIACION DE RESPUESTAS ABIERTAS).

3) fill_blank
   Campos obligatorios: question_id, type, text, language, difficulty, accepted_answers
   - text debe incluir exactamente "___" donde va el hueco.
   - accepted_answers: lista no vacia de respuestas validas.
   - Incluir variantes utiles (con y sin acento cuando aplique).

4) ordering
   Campos obligatorios: question_id, type, text, language, difficulty, elements, correct_order
   - elements y correct_order deben tener la misma longitud.
   - correct_order debe ser una permutacion exacta de elements.
   - Minimo 4 elementos.

5) speaking_record
   Campos obligatorios: question_id, type, text, language, difficulty, rubric, examples_answers
   - Misma logica de rubric y examples_answers que writing_text.

VARIACION DE RESPUESTAS ABIERTAS (OBLIGATORIA para writing_text y speaking_record)
Generar EXACTAMENTE 3 respuestas en examples_answers:
- 1 respuesta ALTA (score entre 0.80 y 0.95):
    buena coherencia, vocabulario rico, pocos o ningun error.
- 1 respuesta MEDIA (score entre 0.50 y 0.70):
    mensaje comprensible, errores frecuentes de concordancia o vocabulario, estructura basica.
- 1 respuesta BAJA (score entre 0.20 y 0.45):
    errores morfosintacticos claros, interferencia del idioma nativo, pero con intencion comunicativa.

Las respuestas deben simular estudiantes reales de frances:
- Errores tipicos: acentos faltantes, concordancia verbo-sujeto, tiempos verbales incorrectos.
- El score debe ser coherente con la calidad del texto.

REGLAS DE CONSISTENCIA
- language siempre "fr".
- difficulty entre 1 y 5.
- question_id unicos, formato: q_fr_llm_001, q_fr_llm_002, etc.
- No dejes campos requeridos vacios.
- No uses tipos legacy (mcq, open, essay, short_answer).

CONTEXTO DEL LOTE
- Tema principal: {topic}
- Nivel objetivo: {level}
- Numero de ejemplos total: {num_examples}
- Distribucion por tipo: {type_distribution}

Genera ahora el JSON completo."""

    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response, handling markdown fences if present."""
        # Direct parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Strip markdown code fences and retry
        cleaned = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Last resort: find first { ... } block
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        logger.error("Could not parse JSON from LLM response")
        logger.debug(f"Raw content (first 500 chars): {content[:500]}")
        return None

    # ------------------------------------------------------------------ #
    # Legacy: single student answer generation (kept for fallback use)    #
    # ------------------------------------------------------------------ #

    def generate_answer(
        self,
        question: str,
        cefr_level: str,
        difficulty: int,
        expected_keywords: list,
        expected_min_words: int,
        expected_max_words: int,
    ) -> str:
        """Generate a single student answer using Groq (legacy helper)."""
        prompt = self._build_answer_prompt(
            question, cefr_level, difficulty,
            expected_keywords, expected_min_words, expected_max_words,
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
            if "choices" in result and result["choices"]:
                return result["choices"][0]["message"]["content"].strip()
            return self._fallback_answer(cefr_level, expected_keywords)
        except requests.exceptions.RequestException as e:
            log_error(f"Groq API error (generate_answer): {e}", error=e)
            return self._fallback_answer(cefr_level, expected_keywords)

    def _build_answer_prompt(
        self,
        question: str,
        cefr_level: str,
        difficulty: int,
        expected_keywords: list,
        expected_min_words: int,
        expected_max_words: int,
    ) -> str:
        keywords_str = ", ".join(expected_keywords)
        return (
            f"Tu es un étudiant français de niveau {cefr_level} qui répond à une question d'examen.\n\n"
            f"QUESTION: {question}\n\n"
            f"INSTRUCTIONS:\n"
            f"- Réponds comme un étudiant de niveau {cefr_level} (pas parfait, mais compréhensible)\n"
            f"- Utilise entre {expected_min_words} et {expected_max_words} mots\n"
            f"- Essaie d'inclure ces mots clés: {keywords_str}\n"
            f"- Inclus occasionnellement de petites erreurs naturelles de grammaire ou d'orthographe\n"
            f"- Réponds uniquement avec la réponse, sans explications additionnelles\n\n"
            f"RÉPONSE:"
        )

    def _fallback_answer(self, cefr_level: str, _keywords: list) -> str:
        fallbacks = {
            "A1": "C'est bien. J'ai aujourd'hui travaillé et c'est bon.",
            "A2": "Aujourd'hui j'ai travaillé et c'étaient bien. J'ai aussi lu et c'est facile.",
            "B1": "Aujourd'hui j'ai eu une bonne journée. J'ai travaillé sur différents projets et c'était intéressant.",
            "B2": "Aujourd'hui a été une journée productive. J'ai accompli plusieurs tâches importantes.",
            "C1": "Cette journée a été particulièrement fructueuse professionnellement. J'ai collaboré efficacement avec mes collègues.",
        }
        return fallbacks.get(cefr_level, fallbacks["B1"])


# Singleton instance
_llm_provider: Optional[GroqProvider] = None


def get_or_create_llm() -> GroqProvider:
    """Get or create the Groq LLM provider singleton."""
    global _llm_provider
    if _llm_provider is None:
        logger.info("Initializing Groq LLM provider")
        _llm_provider = GroqProvider()
    return _llm_provider
