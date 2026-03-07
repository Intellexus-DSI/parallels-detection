"""Utility functions."""

from .overlapping import make_overlapping_spans
from .syllable_constraint import constrain_exclusive_segments
from .word_constraint import constrain_exclusive_segments_by_words

__all__ = [
    "make_overlapping_spans",
    "constrain_exclusive_segments",
    "constrain_exclusive_segments_by_words",
]
