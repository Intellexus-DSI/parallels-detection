"""Filters for post-processing parallel matches."""

from typing import Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.segment_store import SegmentStore


class CrossTextFilter:
    """
    Filter that ensures matches are from different source texts.
    
    Uses the File_Path field to determine if two segments
    are from the same text.
    """

    def __init__(self, store: "SegmentStore"):
        """
        Initialize the filter.

        Args:
            store: Segment store with text ID mappings.
        """
        self.store = store

    def is_cross_text(self, segment_a_id: int, segment_b_id: int) -> bool:
        """
        Check if two segments are from different texts.

        Args:
            segment_a_id: ID of first segment.
            segment_b_id: ID of second segment.

        Returns:
            True if segments are from different texts.
        """
        return self.store.get_text_id(segment_a_id) != self.store.get_text_id(segment_b_id)


class Deduplicator:
    """
    Ensures each parallel pair is only recorded once.
    
    Normalizes pairs to (min_id, max_id) to avoid
    recording both (A, B) and (B, A).
    """

    def __init__(self):
        """Initialize the deduplicator."""
        self._seen: Set[Tuple[int, int]] = set()

    def is_new(self, segment_a_id: int, segment_b_id: int) -> bool:
        """
        Check if this pair hasn't been seen before.

        Args:
            segment_a_id: ID of first segment.
            segment_b_id: ID of second segment.

        Returns:
            True if this is a new pair, False if already seen.
        """
        # Normalize to (smaller_id, larger_id)
        key = (min(segment_a_id, segment_b_id), max(segment_a_id, segment_b_id))
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def reset(self) -> None:
        """Clear all seen pairs."""
        self._seen.clear()

    @property
    def count(self) -> int:
        """Return the number of unique pairs seen."""
        return len(self._seen)
