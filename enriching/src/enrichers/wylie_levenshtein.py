"""Wylie Levenshtein distance enricher."""

import logging
from typing import Any, Dict, List

import Levenshtein

from .base import BaseEnricher
from ..models import EnrichedParallel

logger = logging.getLogger(__name__)


def calculate_wylie_syllable_distance(sent1: str, sent2: str) -> int:
    """
    Calculate syllable-level Levenshtein distance between two Wylie/EWTS strings.

    Wylie/EWTS uses spaces or slashes to separate syllables.
    Each syllable is treated as a single token for edit-distance calculation.

    Args:
        sent1: First Wylie/EWTS text
        sent2: Second Wylie/EWTS text

    Returns:
        Edit distance (insertions, deletions, substitutions) at syllable level
    """
    # Split by space and slash to get syllables (tokens)
    s1_tokens: List[str] = sent1.replace("/", " ").split()
    s2_tokens: List[str] = sent2.replace("/", " ").split()

    if not s1_tokens and not s2_tokens:
        return 0
    if not s1_tokens:
        return len(s2_tokens)
    if not s2_tokens:
        return len(s1_tokens)

    # Map each unique syllable to a single character for Levenshtein.distance
    # (Levenshtein works on strings; encoding tokens as chars gives token-level distance)
    all_tokens = sorted(set(s1_tokens) | set(s2_tokens))
    token_to_char = {t: chr(0xE000 + i) for i, t in enumerate(all_tokens)}

    s1_encoded = "".join(token_to_char[t] for t in s1_tokens)
    s2_encoded = "".join(token_to_char[t] for t in s2_tokens)

    return Levenshtein.distance(s1_encoded, s2_encoded)


class WylieLevenshteinEnricher(BaseEnricher):
    """
    Enricher that calculates Levenshtein distance between parallels on Wylie/EWTS text.

    Adds:
    - wylie_syllable_distance: Syllable-level edit distance (insertions, deletions, substitutions)
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)

    def enrich(self, parallel: EnrichedParallel) -> EnrichedParallel:
        """
        Enrich the parallel with Wylie syllable-level Levenshtein distance.

        Uses parallel_a and parallel_b (EWTS transliteration) from the detection output.
        """
        text_a = (parallel.parallel_a or "").strip()
        text_b = (parallel.parallel_b or "").strip()

        syllable_dist = calculate_wylie_syllable_distance(text_a, text_b)

        parallel.enriched_fields["wylie_syllable_distance"] = syllable_dist

        return parallel

    @property
    def field_names(self) -> list:
        return ["wylie_syllable_distance"]
