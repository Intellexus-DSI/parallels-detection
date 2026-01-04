"""Main pipeline orchestrator for finding parallel segments."""

import logging
from typing import Optional

from .config import Config
from .data.segment_store import SegmentStore
from .data.output_writer import OutputWriter
from .index.faiss_index import FAISSIndex
from .matching.base import MatcherStrategy
from .matching.threshold_matcher import ThresholdMatcher
from .matching.knn_matcher import KNNMatcher
from .models import ParallelMatch

logger = logging.getLogger(__name__)


class ParallelsPipeline:
    """
    Main pipeline for finding parallel text segments.
    
    Orchestrates loading data, building the FAISS index,
    running the matching strategy, and writing results.
    """

    def __init__(self, config: Config):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.store: Optional[SegmentStore] = None
        self.index: Optional[FAISSIndex] = None
        self.matcher: Optional[MatcherStrategy] = None

    def run(self) -> int:
        """
        Execute the full pipeline.

        Returns:
            Number of parallel matches found.
        """
        logger.info("Starting parallels pipeline")

        # Step 1: Load data
        logger.info(f"Loading segments from {self.config.segments_csv}")
        logger.info(f"Loading embeddings from {self.config.embeddings_path}")
        self.store = SegmentStore(
            self.config.segments_csv,
            self.config.embeddings_path,
        )
        logger.info(
            f"Loaded {len(self.store)} segments from {self.store.num_texts} texts"
        )
        logger.info(f"Embedding dimension: {self.store.embedding_dim}")

        # Step 2: Build FAISS index
        logger.info("Building FAISS index")
        self.index = FAISSIndex(
            dimension=self.store.embedding_dim,
            normalize=self.config.processing.normalize_embeddings,
        )
        self.index.build(self.store.embeddings)
        logger.info(f"Index built with {self.index.ntotal} vectors")

        # Step 3: Create matcher strategy
        self.matcher = self._create_matcher()
        logger.info(f"Using {self.config.matching.strategy} matching strategy")

        # Step 4: Find matches and write output
        logger.info(f"Finding parallels (threshold={self.config.matching.threshold})")
        match_count = 0

        with OutputWriter(
            self.config.output_path,
            format=self.config.output.format,
            include_text=self.config.output.include_text,
        ) as writer:
            for match in self.matcher.find_matches(self.index, self.store):
                enriched = self._enrich_match(match)
                writer.write_match(enriched)
                match_count += 1

                # Log progress periodically
                if match_count % 10000 == 0:
                    logger.info(f"Found {match_count} matches so far...")

        logger.info(f"Pipeline complete. Found {match_count} parallel matches")
        logger.info(f"Results written to {self.config.output_path}")

        return match_count

    def _create_matcher(self) -> MatcherStrategy:
        """Create the appropriate matcher based on configuration."""
        if self.config.matching.strategy == "threshold":
            return ThresholdMatcher(
                threshold=self.config.matching.threshold,
                batch_size=self.config.processing.batch_size,
            )
        elif self.config.matching.strategy == "knn":
            return KNNMatcher(
                k=self.config.matching.k,
                min_threshold=self.config.matching.min_threshold,
                batch_size=self.config.processing.batch_size,
            )
        else:
            raise ValueError(f"Unknown strategy: {self.config.matching.strategy}")

    def _enrich_match(self, match: ParallelMatch) -> ParallelMatch:
        """Add metadata to a match for output."""
        meta_a = self.store.get_metadata_for_output(match.segment_a_id)
        meta_b = self.store.get_metadata_for_output(match.segment_b_id)

        match.title_a = meta_a["title"]
        match.title_b = meta_b["title"]
        match.ewts_a = meta_a["ewts"]
        match.ewts_b = meta_b["ewts"]
        match.file_path_a = meta_a["file_path"]
        match.file_path_b = meta_b["file_path"]

        return match
