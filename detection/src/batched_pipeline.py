"""Batched pipeline for memory-efficient parallel detection across files."""

import gc
import logging
from typing import Iterator, List, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .batched_config import BatchedConfig
from .data.corpus_loader import CorpusLoader
from .data.output_writer import OutputWriter
from .index.faiss_index import FAISSIndex
from .models import ParallelMatch

logger = logging.getLogger(__name__)


def chunks(lst: List, n: int) -> Iterator[List]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class BatchedPipeline:
    """
    Memory-efficient pipeline for finding parallels across per-file embeddings.

    Processes files in batches:
    1. Load k files into FAISS index
    2. Query each remaining file against the index
    3. Handle within-batch matching
    4. Move to next batch
    """

    def __init__(self, config: BatchedConfig):
        """
        Initialize the batched pipeline.

        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.corpus = CorpusLoader(config.data_dir)

    def run(self) -> int:
        """
        Execute the batched pipeline.

        Returns:
            Total number of parallel matches found.
        """
        logger.info("Starting batched parallels pipeline")
        logger.info(f"Data directory: {self.config.data_dir}")
        logger.info(f"Found {self.corpus.num_files} corpus files")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"FAISS device: {self.config.processing.device}")

        file_ids = self.corpus.file_ids
        batch_list = list(chunks(file_ids, self.config.batch_size))

        logger.info(f"Processing {len(batch_list)} batches")

        total_matches = 0

        with OutputWriter(
            self.config.output_path,
            format=self.config.output.format,
            include_text=self.config.output.include_text,
            max_lines_per_file=self.config.output.max_lines_per_file,
        ) as writer:

            for batch_idx, index_batch in enumerate(tqdm(batch_list, desc="Batches")):
                batch_start_idx = batch_idx * self.config.batch_size
                batch_end_idx = batch_start_idx + len(index_batch)

                logger.info(f"Processing batch {batch_idx + 1}/{len(batch_list)}: {index_batch}")

                # Load batch files and build index
                index_segments, index_embeddings, offsets = self.corpus.load_batch(index_batch)

                index = FAISSIndex(
                    dimension=self.corpus.embedding_dim,
                    normalize=self.config.processing.normalize_embeddings,
                    device=self.config.processing.device,
                )
                index.build(index_embeddings)

                logger.info(f"Index built with {index.ntotal} vectors from {len(index_batch)} files")

                # Query remaining files (after this batch) against the index
                remaining_file_ids = file_ids[batch_end_idx:]

                for query_file_id in remaining_file_ids:
                    matches = self._query_file_against_index(
                        query_file_id=query_file_id,
                        index=index,
                        index_segments=index_segments,
                        offsets=offsets,
                    )

                    for match in matches:
                        writer.write_match(match)
                        total_matches += 1

                # Within-batch matching (each file against later files in the batch)
                within_batch_matches = self._match_within_batch(
                    index_batch=index_batch,
                    index_segments=index_segments,
                    index_embeddings=index_embeddings,
                    offsets=offsets,
                )

                for match in within_batch_matches:
                    writer.write_match(match)
                    total_matches += 1

                # Free memory
                del index_segments, index_embeddings, index
                gc.collect()

                if total_matches % 10000 == 0 and total_matches > 0:
                    logger.info(f"Found {total_matches} matches so far...")

        logger.info(f"Pipeline complete. Found {total_matches} parallel matches")
        logger.info(f"Results written to {self.config.output_path}")

        return total_matches

    def _query_file_against_index(
        self,
        query_file_id: str,
        index: FAISSIndex,
        index_segments: pd.DataFrame,
        offsets: Dict[str, int],
    ) -> Iterator[ParallelMatch]:
        """
        Query a single file against the index.

        Args:
            query_file_id: ID of the file to query.
            index: FAISS index built from batch files.
            index_segments: Segments DataFrame for indexed files.
            offsets: Offset mapping for global IDs.

        Yields:
            ParallelMatch objects.
        """
        query_segments, query_embeddings = self.corpus.load_file(query_file_id)

        # Determine search_k based on strategy
        if self.config.matching.strategy == "knn":
            search_k = self.config.matching.k
        else:
            search_k = min(100, index.ntotal)  # Threshold strategy: get top candidates

        # Search
        similarities, indices = index.search(query_embeddings, search_k)

        # Process results
        for query_local_idx in range(len(query_embeddings)):
            for j in range(search_k):
                index_global_idx = int(indices[query_local_idx, j])
                similarity = float(similarities[query_local_idx, j])

                # Skip invalid indices
                if index_global_idx < 0:
                    continue

                # Apply threshold
                if self.config.matching.strategy == "threshold":
                    if similarity < self.config.matching.threshold:
                        break  # Sorted by similarity, so we can stop

                # Get metadata for the match
                index_row = index_segments.iloc[index_global_idx]
                query_row = query_segments.iloc[query_local_idx]

                # Skip matches from the same file_id
                if query_row.get("File_ID", "") == index_row.get("File_ID", ""):
                    continue

                yield ParallelMatch(
                    segment_a_id=query_row.get("segment_id", str(query_local_idx)),
                    segment_b_id=index_row.get("segment_id", str(index_global_idx)),
                    similarity=similarity,
                    file_id_a=query_row.get("File_ID", ""),
                    file_id_b=index_row.get("File_ID", ""),
                    source_file_id_a=query_row.get("source_file_id", ""),
                    source_file_id_b=index_row.get("source_file_id", ""),
                    file_id_tibetan_a=query_row.get("File_ID_Tibetan", ""),
                    file_id_tibetan_b=index_row.get("File_ID_Tibetan", ""),
                    text_tibetan_a=query_row.get("Segmented_Text", ""),
                    text_tibetan_b=index_row.get("Segmented_Text", ""),
                    parallel_a=query_row.get("Segmented_Text_EWTS", ""),
                    parallel_b=index_row.get("Segmented_Text_EWTS", ""),
                )

        # Free query data
        del query_segments, query_embeddings

    def _match_within_batch(
        self,
        index_batch: List[str],
        index_segments: pd.DataFrame,
        index_embeddings: np.ndarray,
        offsets: Dict[str, int],
    ) -> Iterator[ParallelMatch]:
        """
        Find matches within a batch (between files in the same batch).

        For files [A, B, C] in a batch, finds matches:
        - A vs B, A vs C, B vs C

        Args:
            index_batch: List of file IDs in this batch.
            index_segments: Combined segments DataFrame.
            index_embeddings: Combined embeddings array.
            offsets: Offset mapping.

        Yields:
            ParallelMatch objects.
        """
        if len(index_batch) < 2:
            return

        # For each file, match against later files in the batch
        for i, query_file_id in enumerate(index_batch[:-1]):
            target_file_ids = index_batch[i + 1:]

            # Get query file's embeddings
            query_start = offsets[query_file_id]
            query_end = offsets.get(
                index_batch[i + 1] if i + 1 < len(index_batch) else None,
                len(index_embeddings)
            )

            # Handle case where this is the last file in the offset order
            if i + 1 < len(index_batch):
                query_end = offsets[index_batch[i + 1]]
            else:
                query_end = len(index_embeddings)

            query_embeddings = index_embeddings[query_start:query_end]

            # Build index from target files only
            target_start = offsets[target_file_ids[0]]
            target_embeddings = index_embeddings[target_start:]
            target_segments = index_segments.iloc[target_start:].reset_index(drop=True)

            if len(target_embeddings) == 0:
                continue

            # Build temporary index
            temp_index = FAISSIndex(
                dimension=self.corpus.embedding_dim,
                normalize=self.config.processing.normalize_embeddings,
                device=self.config.processing.device,
            )
            temp_index.build(target_embeddings)

            # Determine search_k
            if self.config.matching.strategy == "knn":
                search_k = self.config.matching.k
            else:
                search_k = min(100, temp_index.ntotal)

            # Search
            similarities, indices = temp_index.search(query_embeddings, search_k)

            # Process results
            for query_local_idx in range(len(query_embeddings)):
                for j in range(search_k):
                    target_local_idx = int(indices[query_local_idx, j])
                    similarity = float(similarities[query_local_idx, j])

                    if target_local_idx < 0:
                        continue

                    if self.config.matching.strategy == "threshold":
                        if similarity < self.config.matching.threshold:
                            break

                    # Get metadata
                    query_row = index_segments.iloc[query_start + query_local_idx]
                    target_row = target_segments.iloc[target_local_idx]

                    # Skip matches from the same file_id
                    if query_row.get("File_ID", "") == target_row.get("File_ID", ""):
                        continue

                    yield ParallelMatch(
                        segment_a_id=query_row.get("segment_id", str(query_local_idx)),
                        segment_b_id=target_row.get("segment_id", str(target_local_idx)),
                        similarity=similarity,
                        file_id_a=query_row.get("File_ID", ""),
                        file_id_b=target_row.get("File_ID", ""),
                        source_file_id_a=query_row.get("source_file_id", ""),
                        source_file_id_b=target_row.get("source_file_id", ""),
                        file_id_tibetan_a=query_row.get("File_ID_Tibetan", ""),
                        file_id_tibetan_b=target_row.get("File_ID_Tibetan", ""),
                        text_tibetan_a=query_row.get("Segmented_Text", ""),
                        text_tibetan_b=target_row.get("Segmented_Text", ""),
                        parallel_a=query_row.get("Segmented_Text_EWTS", ""),
                        parallel_b=target_row.get("Segmented_Text_EWTS", ""),
                    )

            del temp_index
