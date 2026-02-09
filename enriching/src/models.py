"""Data models for the enriching pipeline."""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class EnrichedParallel:
    """Represents a parallel match with enriched fields."""

    # Original fields from detection stage
    segment_a_id: str
    segment_b_id: str
    similarity: float
    file_id_a: str
    file_id_b: str
    file_id_tibetan_a: str
    file_id_tibetan_b: str
    text_tibetan_a: str
    text_tibetan_b: str
    parallel_a: str
    parallel_b: str
    
    # Enriched fields
    enriched_fields: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize enriched_fields if not provided."""
        if self.enriched_fields is None:
            self.enriched_fields = {}
    
    @classmethod
    def from_dict(cls, data: dict) -> "EnrichedParallel":
        """Create an EnrichedParallel from a dictionary."""
        # Extract original fields
        original_fields = {
            "segment_a_id": data.get("segment_a_id", ""),
            "segment_b_id": data.get("segment_b_id", ""),
            "similarity": data.get("similarity", 0.0),
            "file_id_a": data.get("file_id_a", ""),
            "file_id_b": data.get("file_id_b", ""),
            "file_id_tibetan_a": data.get("file_id_tibetan_a", ""),
            "file_id_tibetan_b": data.get("file_id_tibetan_b", ""),
            "text_tibetan_a": data.get("text_tibetan_a", ""),
            "text_tibetan_b": data.get("text_tibetan_b", ""),
            "parallel_a": data.get("parallel_a", ""),
            "parallel_b": data.get("parallel_b", ""),
        }
        
        # Extract any additional fields as enriched fields
        enriched = {k: v for k, v in data.items() if k not in original_fields}
        
        return cls(**original_fields, enriched_fields=enriched)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for output."""
        result = {
            "segment_a_id": self.segment_a_id,
            "segment_b_id": self.segment_b_id,
            "similarity": self.similarity,
            "file_id_a": self.file_id_a,
            "file_id_b": self.file_id_b,
            "file_id_tibetan_a": self.file_id_tibetan_a,
            "file_id_tibetan_b": self.file_id_tibetan_b,
            "text_tibetan_a": self.text_tibetan_a,
            "text_tibetan_b": self.text_tibetan_b,
            "parallel_a": self.parallel_a,
            "parallel_b": self.parallel_b,
        }
        
        # Add enriched fields
        if self.enriched_fields:
            result.update(self.enriched_fields)
        
        return result
