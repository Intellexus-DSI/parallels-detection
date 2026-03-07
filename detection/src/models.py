"""Data models for the parallels pipeline."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Segment:
    """Represents a text segment with its metadata."""

    id: int  # Row index in the CSV
    text_id: str  # File_ID - Wylie transliteration
    text_id_tibetan: str  # File_ID_Tibetan - Original Tibetan Unicode
    text: str  # Segmented_Text (Tibetan/Sanskrit)
    text_ewts: str  # Segmented_Text_EWTS (transliteration)
    length: int
    source_line_number: int
    sentence_order: int
    start_index: int
    end_index: int


@dataclass
class ParallelMatch:
    """Represents a parallel match between two segments."""

    segment_a_id: str  # Format: {line_number}_{segment_order}
    segment_b_id: str  # Format: {line_number}_{segment_order}
    similarity: float

    # Optional enriched fields (populated on output)
    file_id_a: Optional[str] = None  # Wylie version
    file_id_b: Optional[str] = None  # Wylie version
    source_file_id_a: Optional[str] = None  # Embedding file key, e.g. line_000114
    source_file_id_b: Optional[str] = None  # Embedding file key for lexical lookup
    file_id_tibetan_a: Optional[str] = None  # Original Tibetan
    file_id_tibetan_b: Optional[str] = None  # Original Tibetan
    text_a: Optional[str] = None
    text_b: Optional[str] = None
    text_tibetan_a: Optional[str] = None  # Tibetan Unicode text
    text_tibetan_b: Optional[str] = None  # Tibetan Unicode text
    parallel_a: Optional[str] = None  # EWTS transliteration
    parallel_b: Optional[str] = None  # EWTS transliteration

    def to_dict(self, include_text: bool = True) -> dict:
        """Convert to dictionary for output."""
        result = {
            "segment_a_id": self.segment_a_id,
            "segment_b_id": self.segment_b_id,
            "similarity": self.similarity,
            "file_id_a": self.file_id_a,
            "file_id_b": self.file_id_b,
            "source_file_id_a": self.source_file_id_a,
            "source_file_id_b": self.source_file_id_b,
            "file_id_tibetan_a": self.file_id_tibetan_a,
            "file_id_tibetan_b": self.file_id_tibetan_b,
        }
        if include_text:
            result["text_tibetan_a"] = self.text_tibetan_a
            result["text_tibetan_b"] = self.text_tibetan_b
            result["parallel_a"] = self.parallel_a
            result["parallel_b"] = self.parallel_b
        return result
