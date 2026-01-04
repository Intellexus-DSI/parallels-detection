"""K-Nearest Neighbors matching strategy."""

from typing import Iterator, TYPE_CHECKING

from tqdm import tqdm

from ..models import ParallelMatch
from .base import MatcherStrategy
from .filters import CrossTextFilter, Deduplicator

if TYPE_CHECKING:
    from ..data.segment_store import SegmentStore
    from ..index.faiss_index import FAISSIndex


class KNNMatcher(MatcherStrategy):
    """
    Finds the top-k most similar segments for each segment.
    
    Optionally applies a minimum similarity threshold to
    filter out low-quality matches.
    """

    def __init__(
        self,
        k: int,
        min_threshold: float = 0.0,
        batch_size: int = 1000,
    ):
        """
        Initialize the KNN matcher.

        Args:
            k: Number of nearest neighbors to find per segment.
            min_threshold: Minimum similarity to include (0.0 to 1.0).
            batch_size: Number of segments to process per batch.
        """
        self.k = k
        self.min_threshold = min_threshold
        self.batch_size = batch_size

    def find_matches(
        self,
        index: "FAISSIndex",
        store: "SegmentStore",
    ) -> Iterator[ParallelMatch]:
        """
        Find k nearest neighbors for each segment.

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

        # We need extra candidates to account for same-text filtering
        # Request more than k to ensure we get k cross-text matches
        search_k = min(self.k * 3, n_segments)

        for batch_idx in tqdm(range(n_batches), desc="Finding parallels"):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, n_segments)

            # Search for candidates
            similarities, indices = index.search_batch(
                store.embeddings, search_k, batch_start, batch_end
            )

            # Process each query in the batch
            for i, query_idx in enumerate(range(batch_start, batch_end)):
                matches_found = 0

                for j in range(search_k):
                    if matches_found >= self.k:
                        break

                    neighbor_idx = int(indices[i, j])
                    similarity = float(similarities[i, j])

                    # Skip self-matches
                    if neighbor_idx == query_idx:
                        continue

                    # Apply minimum threshold filter
                    if similarity < self.min_threshold:
                        break  # Results are sorted, no need to continue

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
                    matches_found += 1
