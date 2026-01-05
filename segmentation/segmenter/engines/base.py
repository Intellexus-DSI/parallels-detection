"""Base classes and constants for segmentation engines."""

import re
from abc import ABC, abstractmethod


# Tibetan Unicode Constants
TIBETAN_SHAD = "\u0F0D"  # །
TIBETAN_DOUBLE_SHAD = "\u0F0E"  # ༎
TSHEG = "\u0F0B"  # ་

# Linguistic patterns for sentence boundaries
TERMINATORS = {
    "གོ", "ངོ", "དོ", "ནོ", "བོ", "མོ", "འོ", "རོ", "ལོ", "སོ", "ཏོ", "ཐོ"
}

CONTINUATORS = {
    "དང་", "ནས", "ཏེ", "སྟེ", "དེ", "ཀྱང", "ཡང", "འང", "ཞིང", "ཤིང", "ཅིང"
}


class SegmentationEngine(ABC):
    """Base class for segmentation engines."""

    def __init__(self, min_syllables: int = 4):
        """Initialize segmentation engine.

        Args:
            min_syllables: Minimum number of syllables per segment
        """
        self.min_syllables = min_syllables
        self.english_pattern = re.compile(r"[a-zA-Z]")
        self.tibetan_pattern = re.compile(r"[\u0F00-\u0FFF]")
        self.number_pattern = re.compile(r"[\u0F20-\u0F29]")

    @abstractmethod
    def segment_with_indices(self, text: str) -> list[tuple[str, int, int]]:
        """Segment text and return segments with their indices.

        Args:
            text: Input text to segment

        Returns:
            List of (segment_text, start_index, end_index) tuples
        """
        pass

    def count_syllables(self, text: str) -> int:
        """Count syllables in Tibetan text.

        Args:
            text: Tibetan text

        Returns:
            Number of syllables (tsheg count)
        """
        return text.count(TSHEG)
