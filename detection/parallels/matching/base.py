"""Abstract base class for matching strategies."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator

from ..models import ParallelMatch

if TYPE_CHECKING:
    from ..data.segment_store import SegmentStore
    from ..index.faiss_index import FAISSIndex


class MatcherStrategy(ABC):
    """
    Abstract base class for matching strategies.
    
    Implementations define how to find parallel matches
    between segments using different algorithms.
    """

    @abstractmethod
    def find_matches(
        self,
        index: "FAISSIndex",
        store: "SegmentStore",
    ) -> Iterator[ParallelMatch]:
        """
        Find parallel matches between segments.

        Args:
            index: FAISS index containing all segment embeddings.
            store: Segment store with metadata and embeddings.

        Yields:
            ParallelMatch objects for each matching pair.
        """
        pass
