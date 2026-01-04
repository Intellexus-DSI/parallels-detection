"""Matching strategies for finding parallel segments."""

from .base import MatcherStrategy
from .threshold_matcher import ThresholdMatcher
from .knn_matcher import KNNMatcher
from .filters import CrossTextFilter, Deduplicator

__all__ = [
    "MatcherStrategy",
    "ThresholdMatcher",
    "KNNMatcher",
    "CrossTextFilter",
    "Deduplicator",
]
