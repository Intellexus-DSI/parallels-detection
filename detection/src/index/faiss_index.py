"""FAISS index wrapper for efficient similarity search."""

import logging
from typing import Literal, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def _resolve_device(device: Literal["auto", "cuda", "cpu"]) -> str:
    """Resolve 'auto' device to 'cuda' or 'cpu'."""
    if device == "auto":
        if faiss.get_num_gpus() > 0:
            return "cuda"
        return "cpu"
    return device


class FAISSIndex:
    """
    FAISS index wrapper for cosine similarity search.

    Uses IndexFlatIP (inner product) with L2 normalization
    to compute cosine similarity efficiently. Supports both
    CPU and GPU backends.
    """

    def __init__(
        self,
        dimension: int = 768,
        normalize: bool = True,
        device: Literal["auto", "cuda", "cpu"] = "cpu",
    ):
        """
        Initialize the FAISS index.

        Args:
            dimension: Embedding dimension (default 768).
            normalize: Whether to L2-normalize vectors for cosine similarity.
            device: Device to use ("auto", "cuda", or "cpu").
        """
        self.dimension = dimension
        self.normalize = normalize
        self.device = _resolve_device(device)
        self.index: faiss.IndexFlatIP = None
        self._gpu_resource = None
        self._is_built = False

        if self.device == "cuda":
            logger.info("FAISS will use GPU acceleration")

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

        # Create the base CPU index
        cpu_index = faiss.IndexFlatIP(self.dimension)

        # Move to GPU if requested
        if self.device == "cuda":
            self._gpu_resource = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self._gpu_resource, 0, cpu_index)
        else:
            self.index = cpu_index

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
