"""Fuzzy matching enricher."""

import logging
from typing import Any, Dict

from fuzzywuzzy import fuzz

from .base import BaseEnricher
from ..models import EnrichedParallel

logger = logging.getLogger(__name__)


class FuzzyMatcherEnricher(BaseEnricher):
    """
    Enricher that checks if parallels are fuzzy matches.
    
    Adds a field 'is_fuzzy_match' with values:
    - 0: The texts are a fuzzy match (similar)
    - 1: The texts are not a fuzzy match (different)
    
    Uses fuzzy string matching to compare text_tibetan_a and text_tibetan_b.
    """
    
    DEFAULT_THRESHOLD = 90  # Default similarity threshold (0-100)
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the fuzzy matcher enricher.
        
        Args:
            params: Configuration parameters. Supported keys:
                - threshold (int): Similarity threshold (0-100). Default: 90.
                - use_ratio (bool): Use fuzz.ratio instead of fuzz.token_sort_ratio. Default: False.
        """
        super().__init__(params)
        self.threshold = self.params.get("threshold", self.DEFAULT_THRESHOLD)
        self.use_ratio = self.params.get("use_ratio", False)
        
        logger.info(f"Initialized FuzzyMatcherEnricher with threshold={self.threshold}")
    
    def enrich(self, parallel: EnrichedParallel) -> EnrichedParallel:
        """
        Enrich the parallel with fuzzy match information.
        
        Args:
            parallel: The parallel match to enrich.
            
        Returns:
            The enriched parallel match.
        """
        text_a = parallel.text_tibetan_a or ""
        text_b = parallel.text_tibetan_b or ""
        
        # Calculate fuzzy similarity score
        if self.use_ratio:
            similarity_score = fuzz.ratio(text_a, text_b)
        else:
            # token_sort_ratio is more robust to word order changes
            similarity_score = fuzz.token_sort_ratio(text_a, text_b)
        
        # Determine if it's a fuzzy match
        # 0 = fuzzy match (texts are similar)
        # 1 = not a fuzzy match (texts are different)
        is_fuzzy_match = 0 if similarity_score >= self.threshold else 1
        
        # Add enriched fields
        parallel.enriched_fields["is_fuzzy_match"] = is_fuzzy_match
        parallel.enriched_fields["fuzzy_score"] = similarity_score
        
        return parallel
    
    @property
    def field_names(self) -> list:
        """Return the names of fields this enricher adds."""
        return ["is_fuzzy_match", "fuzzy_score"]
