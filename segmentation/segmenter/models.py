"""Data models for the segmentation pipeline."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Segment:
    """Represents a segmented text with metadata."""

    text: str
    text_ewts: str  # EWTS transliteration
    length: int
    file_id: str  # Wylie transliteration
    file_id_tibetan: str  # Original Tibetan Unicode
    source_line_number: int
    sentence_order: int
    start_index: int
    end_index: int
    span_num_atoms: Optional[int] = None  # For overlapping mode
    span_type: Optional[str] = None  # "forward" or "centered"


@dataclass
class DocumentMetadata:
    """Metadata for a source document."""

    file_id: str  # Wylie transliteration
    file_id_tibetan: str  # Original Tibetan Unicode
    line_number: int


@dataclass
class SegmentationResult:
    """Result of segmenting a document."""

    segments: list[Segment]
    metadata: DocumentMetadata
    atom_count: int  # Number of atomic segments before overlapping
