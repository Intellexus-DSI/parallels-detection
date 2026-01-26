"""Base classes and constants for segmentation engines."""

import re
from abc import ABC, abstractmethod


# Tibetan Unicode Constants
TIBETAN_SHAD = "\u0F0D"  # །
TIBETAN_DOUBLE_SHAD = "\u0F0E"  # ༎
TER_TSHEG = "\u0F14"  # ༔ (gter tsheg)
TSHEG = "\u0F0B"  # ་

# Sentence terminators (split after single shad when preceded by these)
TERMINATORS = {
    "གོ", "ངོ", "དོ", "ནོ", "བོ", "མོ", "འོ", "རོ", "ལོ", "སོ", "ཏོ", "ཐོ"
}

# Optative/imperative suffixes (split after single shad when preceded by these)
OPTATIVE_IMPERATIVE = {
    "ཅིག", "གྱུར་ཅིག", "ཞིག", "ཤིག", "ཤོག", "རོགས", "རོགས་གནང"
}

# Sentence-initial markers (split after single shad when followed by these)
SENTENCE_INITIAL_MARKERS = {
    "དེ་ནས", "དེ་བས་ན", "དེའི་རྗེས", "དེ་ལྟ་བས་ན", "དེ་ཡང", "དེར་ཡང", 
    "དེ་མ་ཡིན", "དེ་མ་ཐག", "དེ་མ་གཏོགས", "དེ་མིན", "དེ་ཕྱིར", "དེའི་ཕྱིར", 
    "དེ་བཞིན་དུ", "དེས་ན", "གཞན་དུ་ན", "གཞན་ཡང", "ཡང་ན", "ཡོང་ནི", 
    "འོ་ན", "འོན་ཏེ", "འོན་ཀྱང", "གལ་ཏེ", "གལ་སྲིད", "སླར་ཡང", 
    "སྤྱིར་ལ", "སྤྱིར་བཏང", "སྤྱིར་ཡང", "མདོར་ན", "མདོར་བསྡུས་ན"
}

# Gerund markers (do NOT split after single shad when preceded by these)
GERUND_MARKERS = {
    "ནས", "བཞིན", "བཞིན་དུ", "བཞིན་པ", "བཞིན་པར", "ཀྱིན"
}

# Particles (do NOT split after single shad when preceded by these)
PARTICLES = {
    "ལ", "སུ", "དུ", "ན", "རུ", "ཏུ", "ནས", "ལས", "གི", "གྱི", "ཀྱི", 
    "འི", "ཡི", "ཕྱིར", "དང", "ཞིང", "ཅིང", "སྟེ", "ཏེ", "ཀྱང", "ཡང", 
    "འང", "འམ", "པས", "བས", "ལྟར", "གིས", "ཀྱིས", "ཡིས"
}

# Adverbializer markers (do NOT split after single shad when preceded by these)
ADVERBIALIZER_MARKERS = {
    "པར", "བར"
}

# Topic marker (do NOT split after single shad when preceded by this)
TOPIC_MARKER = {
    "ནི"
}

# Correlative pairs - first part (if found, don't split until second part appears)
CORRELATIVE_FIRST = {
    "ཅི་ཞིག", "ཅི་སྟེ", "ཅི་འདྲ", "ཅི་ལྟར", "ཅི་བཞིན", "ཅི་ཙམ", 
    "ཇི་སྙེད", "ཇི་སྲིད", "ཇི་ལྟར", "ཇི་བཞིན", "ཇི་སྐད", "ཇི་ཙམ"
}

# Correlative pairs - second part (appears after first part)
CORRELATIVE_SECOND = {
    "དེ་བཞིན", "དེ་ལྟར", "དེ་ཙམ", "དེ་སྐད", "དེ་འདྲ", "དེ་སྲིད", "དེ་སྙེད"
}

# Legacy continuators (kept for backward compatibility with Botok engine)
# These are now superseded by GERUND_MARKERS, PARTICLES, etc. but kept for compatibility
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
