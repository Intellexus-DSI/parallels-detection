"""Data models for the parallels pipeline."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Segment:
    """Represents a text segment with its metadata."""

    id: int  # Row index in the CSV
    text_id: str  # File_Path - unique text identifier
    title: str
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
    title_a: Optional[str] = None
    title_b: Optional[str] = None
    text_a: Optional[str] = None
    text_b: Optional[str] = None
    parallel_a: Optional[str] = None
    parallel_b: Optional[str] = None
    file_path_a: Optional[str] = None
    file_path_b: Optional[str] = None

    def to_dict(self, include_text: bool = True) -> dict:
        """Convert to dictionary for output."""
        result = {
            "segment_a_id": self.segment_a_id,
            "segment_b_id": self.segment_b_id,
            "similarity": self.similarity,
            "title_a": self.title_a,
            "title_b": self.title_b,
            "file_path_a": self.file_path_a,
            "file_path_b": self.file_path_b,
        }
        if include_text:
            result["parallel_a"] = self.parallel_a
            result["parallel_b"] = self.parallel_b
        return result
