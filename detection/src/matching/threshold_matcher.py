"""Threshold-based matching strategy."""

from typing import Iterator, TYPE_CHECKING

from tqdm import tqdm

from ..models import ParallelMatch
from .base import MatcherStrategy
from .filters import CrossTextFilter, Deduplicator

if TYPE_CHECKING:
    from ..data.segment_store import SegmentStore
    from ..index.faiss_index import FAISSIndex


class ThresholdMatcher(MatcherStrategy):
    """
    Finds all segment pairs with similarity above a threshold.
    
    Uses a high-k search to find candidates, then filters
    by the threshold. This is efficient because FAISS doesn't
    have a true range_search for inner product.
    """

    def __init__(
        self,
        threshold: float,
        batch_size: int = 1000,
        search_k: int = 100,
    ):
        """
        Initialize the threshold matcher.

        Args:
            threshold: Minimum cosine similarity (0.0 to 1.0).
            batch_size: Number of segments to process per batch.
            search_k: Number of candidates to retrieve per segment.
                      Should be large enough to capture all matches above threshold.
        """
        self.threshold = threshold
        self.batch_size = batch_size
        self.search_k = search_k

    def find_matches(
        self,
        index: "FAISSIndex",
        store: "SegmentStore",
    ) -> Iterator[ParallelMatch]:
        """
        Find all parallel matches above the similarity threshold.

        Args:
            index: FAISS index containing all segment embeddings.
            store: Segment store with metadata and embeddings.

        Yields:
            ParallelMatch objects for each matching pair.
        """
        cross_text_filter = CrossTextFilter(store)
        deduplicator = Deduplicator()

        n_segments = len(store)
        n_batches = (n_segments + self.batch_size - 1) // self.batch_size

        # Limit search_k to actual number of segments
        actual_k = min(self.search_k, n_segments)

        for batch_idx in tqdm(range(n_batches), desc="Finding parallels"):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, n_segments)

            # Search for candidates
            similarities, indices = index.search_batch(
                store.embeddings, actual_k, batch_start, batch_end
            )

            # Process each query in the batch
            for i, query_idx in enumerate(range(batch_start, batch_end)):
                for j in range(actual_k):
                    neighbor_idx = int(indices[i, j])
                    similarity = float(similarities[i, j])

                    # Skip self-matches
                    if neighbor_idx == query_idx:
                        continue

                    # Apply threshold filter
                    if similarity < self.threshold:
                        # Results are sorted by similarity, so we can break early
                        break

                    # Apply cross-text filter
                    if not cross_text_filter.is_cross_text(query_idx, neighbor_idx):
                        continue

                    # Apply deduplication
                    if not deduplicator.is_new(query_idx, neighbor_idx):
                        continue

                    yield ParallelMatch(
                        segment_a_id=query_idx,
                        segment_b_id=neighbor_idx,
                        similarity=similarity,
                    )
