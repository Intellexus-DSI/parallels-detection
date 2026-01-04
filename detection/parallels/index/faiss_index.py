"""FAISS index wrapper for efficient similarity search."""

from typing import Tuple

import faiss
import numpy as np


class FAISSIndex:
    """
    FAISS index wrapper for cosine similarity search.
    
    Uses IndexFlatIP (inner product) with L2 normalization
    to compute cosine similarity efficiently.
    """

    def __init__(self, dimension: int = 768, normalize: bool = True):
        """
        Initialize the FAISS index.

        Args:
            dimension: Embedding dimension (default 768).
            normalize: Whether to L2-normalize vectors for cosine similarity.
        """
        self.dimension = dimension
        self.normalize = normalize
        self.index: faiss.IndexFlatIP = None
        self._is_built = False

    def build(self, embeddings: np.ndarray) -> None:
        """
        Build the FAISS index from embeddings.

        Args:
            embeddings: 2D array of shape (n_vectors, dimension).
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        # Ensure float32
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Normalize for cosine similarity
        if self.normalize:
            embeddings = self._normalize(embeddings)

        # Create and populate the index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self._is_built = True

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)
        return vectors / norms

    def search(
        self, queries: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.

        Args:
            queries: Query vectors of shape (n_queries, dimension).
            k: Number of nearest neighbors to return.

        Returns:
            Tuple of (similarities, indices):
                - similarities: shape (n_queries, k), cosine similarities
                - indices: shape (n_queries, k), indices of neighbors
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build() first.")

        # Ensure float32
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)

        # Normalize queries for cosine similarity
        if self.normalize:
            queries = self._normalize(queries)

        similarities, indices = self.index.search(queries, k)
        return similarities, indices

    def search_batch(
        self, embeddings: np.ndarray, k: int, batch_start: int, batch_end: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for neighbors of a batch of embeddings.
        
        Convenience method that extracts a batch and searches.

        Args:
            embeddings: Full embedding matrix.
            k: Number of neighbors.
            batch_start: Start index of batch.
            batch_end: End index of batch.

        Returns:
            Tuple of (similarities, indices) for the batch.
        """
        batch = embeddings[batch_start:batch_end]
        return self.search(batch, k)

    @property
    def ntotal(self) -> int:
        """Return the number of vectors in the index."""
        return self.index.ntotal if self._is_built else 0
