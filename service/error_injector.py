"""
Error injection module for realistic student mistakes.
Injects common French learner errors like missing accents, conjugation mistakes, etc.
"""

import random
from typing import List

from config.config import ERROR_INJECTION_PROBABILITY


class ErrorInjector:
    """Injects realistic French learner errors into text."""
    
    # Common accent mistakes
    ACCENT_REPLACEMENTS = {
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "à": "a",
        "â": "a",
        "ù": "u",
        "û": "u",
        "î": "i",
        "ï": "i",
        "ô": "o",
        "ö": "o",
        "ç": "c",
    }
    
    # Common conjugation errors (subject-verb agreement)
    CONJUGATION_ERRORS = {
        "je suis": ["je sont", "j'ai"],
        "tu es": ["tu sont", "tu est"],
        "il est": ["il sont"],
        "elle est": ["elle sont"],
        "nous sommes": ["nous est", "nous sommes pas"],
        "vous êtes": ["vous est"],
        "ils sont": ["ils est"],
        "elles sont": ["elles est"],
        "j'ai": ["j'ont", "j'est", "j'as"],
        "tu as": ["tu ont", "tu a"],
        "il a": ["il ont", "il as"],
        "elle a": ["elle ont"],
        "nous avons": ["nous avez", "nous a"],
        "vous avez": ["vous ont"],
        "ils ont": ["ils a"],
        "elles ont": ["elles a"],
    }
    
    # Common spelling mistakes
    SPELLING_ERRORS = {
        "aujourd'hui": "aujourd hui",
        "s'il vous plaît": "s il vous plait",
        "c'est": "c est",
        "n'aime": "n aime",
        "m'aime": "m aime",
        "t'aime": "t aime",
        "l'amour": "l amour",
    }
    
    # Plural mistakes
    PLURAL_ERRORS = {
        "les jours": "les jour",
        "les journées": "les journée",
        "les heures": "les heure",
        "les minutes": "les minute",
        "les jours importants": "les jour important",
    }
    
    def __init__(self, error_probability: float = ERROR_INJECTION_PROBABILITY):
        """
        Initialize the error injector.
        
        Args:
            error_probability: Probability of injecting an error (0.0 to 1.0)
        """
        self.error_probability = max(0.0, min(1.0, error_probability))
    
    def inject_errors(self, text: str) -> str:
        """
        Inject realistic errors into text with configured probability.
        
        Args:
            text: Original text from LLM
            
        Returns:
            Text with injected errors
        """
        if random.random() > self.error_probability:
            return text  # No errors injected
        
        errors_to_inject = random.randint(1, 3)  # Inject 1-3 errors per answer
        
        for _ in range(errors_to_inject):
            error_type = random.choice([
                "accent",
                "conjugation",
                "spelling",
                "plural",
            ])
            
            if error_type == "accent":
                text = self._inject_accent_error(text)
            elif error_type == "conjugation":
                text = self._inject_conjugation_error(text)
            elif error_type == "spelling":
                text = self._inject_spelling_error(text)
            elif error_type == "plural":
                text = self._inject_plural_error(text)
        
        return text
    
    def _inject_accent_error(self, text: str) -> str:
        """Remove accents from random words."""
        words = text.split()
        
        if len(words) == 0:
            return text
        
        # Pick random word with accents
        word_idx = random.randint(0, len(words) - 1)
        original_word = words[word_idx]
        
        # Try to remove an accent
        modified_word = original_word
        for accented, plain in self.ACCENT_REPLACEMENTS.items():
            if accented in modified_word:
                modified_word = modified_word.replace(accented, plain, 1)
                break
        
        if modified_word != original_word:
            words[word_idx] = modified_word
        
        return " ".join(words)
    
    def _inject_conjugation_error(self, text: str) -> str:
        """Inject conjugation errors."""
        for correct, errors in self.CONJUGATION_ERRORS.items():
            if correct in text.lower():
                error = random.choice(errors)
                # Case-insensitive replacement
                text = text.replace(correct, error, 1)
                break
        
        return text
    
    def _inject_spelling_error(self, text: str) -> str:
        """Inject spelling errors."""
        for correct, error in self.SPELLING_ERRORS.items():
            if correct in text.lower():
                text = text.replace(correct, error, 1)
                break
        
        return text
    
    def _inject_plural_error(self, text: str) -> str:
        """Inject plural agreement errors."""
        for correct, error in self.PLURAL_ERRORS.items():
            if correct in text.lower():
                text = text.replace(correct, error, 1)
                break
        
        return text
    
    def vary_word_count(self, text: str, target_word_count: int, tolerance: int = 5) -> str:
        """
        Vary word count to match target (with tolerance).
        Used to ensure responses match expected length for CEFR level.
        
        Args:
            text: Original text
            target_word_count: Target number of words
            tolerance: Acceptable variation
            
        Returns:
            Text with adjusted word count
        """
        words = text.split()
        current_count = len(words)
        
        if abs(current_count - target_word_count) <= tolerance:
            return text  # Already within tolerance
        
        if current_count < target_word_count:
            # Add words
            filler_phrases = [
                " Il est important de mentionner que c'est très bon.",
                " Je crois que c'est une bonne chose.",
                " C'est vraiment intéressant et utile.",
                " Je pense que c'est magnifique.",
                " C'est extraordinaire et merveilleux.",
            ]
            
            while len(text.split()) < target_word_count:
                text += random.choice(filler_phrases)
        
        elif current_count > target_word_count:
            # Remove words (be careful with sentence structure)
            words = text.split()
            while len(words) > target_word_count and len(words) > 5:
                idx = random.randint(0, len(words) - 1)
                words.pop(idx)
            text = " ".join(words)
        
        return text


# Singleton instance
_error_injector: ErrorInjector = None

def get_or_create_error_injector() -> ErrorInjector:
    """Get or create the error injector singleton."""
    global _error_injector
    if _error_injector is None:
        _error_injector = ErrorInjector()
    return _error_injector
