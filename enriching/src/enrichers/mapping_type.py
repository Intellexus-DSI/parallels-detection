"""Mapping type enricher: classifies parallels as textual, semantic, or uncertain."""

import logging
from typing import Any, Dict, List, Set, Tuple

from .base import BaseEnricher
from .tibetan_particles import TIBETAN_PARTICLES
from .wylie_levenshtein import calculate_wylie_syllable_distance
from ..models import EnrichedParallel

logger = logging.getLogger(__name__)


def _tokenize_wylie(text: str) -> List[str]:
    """Split Wylie/EWTS text into syllable tokens."""
    return text.replace("/", " ").split()


def _get_content_tokens(tokens: List[str]) -> List[str]:
    """Filter out particles; return content syllables preserving order."""
    return [t for t in tokens if t and t not in TIBETAN_PARTICLES]


def _content_overlap_ratio(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard: |A ∩ B| / |A ∪ B|."""
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    inter = set_a & set_b
    return len(inter) / len(union) if union else 0.0


def _bigrams(tokens: List[str]) -> Set[Tuple[str, str]]:
    """Return set of (token[i], token[i+1]) for i in 0..len-2."""
    return {tuple(tokens[i : i + 2]) for i in range(len(tokens) - 1)}


def _bigram_overlap_ratio(content_a: List[str], content_b: List[str]) -> float:
    """Jaccard overlap of syllable bigrams (shared phrases)."""
    bg_a = _bigrams(content_a)
    bg_b = _bigrams(content_b)
    return _content_overlap_ratio(bg_a, bg_b)


def classify_mapping_type(
    parallel_a: str,
    parallel_b: str,
    syllable_distance: int,
    overlap_textual: float = 0.40,
    overlap_semantic: float = 0.25,
    norm_lev_textual: float = 0.25,
    norm_lev_semantic: float = 0.40,
    bigram_textual: float = 0.25,
    bigram_semantic: float = 0.15,
) -> str:
    """
    Classify parallel as 'textual', 'semantic', or 'uncertain'.

    Uses three signals:
    1. Unigram content overlap (after stripping particles)
    2. Normalized Levenshtein distance (distance / max_syllables)
    3. Bigram (phrase) overlap

    - **Textual**: Strong evidence (high overlap, low distance, or high phrase overlap).
    - **Semantic**: Clear semantic (low overlap, high distance, low phrase overlap).
    - **Uncertain**: Borderline, worth manual review.
    """
    tokens_a = _tokenize_wylie(parallel_a.strip())
    tokens_b = _tokenize_wylie(parallel_b.strip())

    content_a = _get_content_tokens(tokens_a)
    content_b = _get_content_tokens(tokens_b)

    unigram_overlap = _content_overlap_ratio(set(content_a), set(content_b))
    bigram_overlap = _bigram_overlap_ratio(content_a, content_b)

    max_syllables = max(len(tokens_a), len(tokens_b)) or 1
    normalized_lev = syllable_distance / max_syllables

    # Textual: any strong signal
    is_textual = (
        unigram_overlap >= overlap_textual
        or normalized_lev <= norm_lev_textual
        or bigram_overlap >= bigram_textual
    )

    # Semantic: all weak signals
    is_semantic = (
        unigram_overlap <= overlap_semantic
        and normalized_lev >= norm_lev_semantic
        and bigram_overlap <= bigram_semantic
    )

    if is_textual and not is_semantic:
        return "textual"
    if is_semantic and not is_textual:
        return "semantic"
    return "uncertain"


class MappingTypeEnricher(BaseEnricher):
    """
    Enricher that classifies parallels as textual, semantic, or uncertain.

    Combines:
    - Unigram content overlap (particles stripped)
    - Normalized Levenshtein distance
    - Bigram (phrase) overlap

    Output: 'textual' | 'semantic' | 'uncertain'
    """

    DEFAULT_PARAMS = {
        "overlap_textual": 0.40,
        "overlap_semantic": 0.25,
        "norm_lev_textual": 0.25,
        "norm_lev_semantic": 0.40,
        "bigram_textual": 0.25,
        "bigram_semantic": 0.15,
    }

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)
        self.overlap_textual = self.params.get(
            "overlap_textual", self.DEFAULT_PARAMS["overlap_textual"]
        )
        self.overlap_semantic = self.params.get(
            "overlap_semantic", self.DEFAULT_PARAMS["overlap_semantic"]
        )
        self.norm_lev_textual = self.params.get(
            "norm_lev_textual", self.DEFAULT_PARAMS["norm_lev_textual"]
        )
        self.norm_lev_semantic = self.params.get(
            "norm_lev_semantic", self.DEFAULT_PARAMS["norm_lev_semantic"]
        )
        self.bigram_textual = self.params.get(
            "bigram_textual", self.DEFAULT_PARAMS["bigram_textual"]
        )
        self.bigram_semantic = self.params.get(
            "bigram_semantic", self.DEFAULT_PARAMS["bigram_semantic"]
        )

    def enrich(self, parallel: EnrichedParallel) -> EnrichedParallel:
        """Add mapping_type: 'textual', 'semantic', or 'uncertain'."""
        text_a = (parallel.parallel_a or "").strip()
        text_b = (parallel.parallel_b or "").strip()

        # Use precomputed distance if available (from wylie_levenshtein), else compute
        syllable_dist = parallel.enriched_fields.get("wylie_syllable_distance")
        if syllable_dist is None:
            syllable_dist = calculate_wylie_syllable_distance(text_a, text_b)

        mapping_type = classify_mapping_type(
            text_a,
            text_b,
            syllable_distance=syllable_dist,
            overlap_textual=self.overlap_textual,
            overlap_semantic=self.overlap_semantic,
            norm_lev_textual=self.norm_lev_textual,
            norm_lev_semantic=self.norm_lev_semantic,
            bigram_textual=self.bigram_textual,
            bigram_semantic=self.bigram_semantic,
        )

        parallel.enriched_fields["mapping_type"] = mapping_type

        return parallel

    @property
    def field_names(self) -> list:
        return ["mapping_type"]
