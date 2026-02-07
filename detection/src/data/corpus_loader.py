"""Loader for per-file corpus data (embeddings + segments)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class CorpusFile:
    """Metadata for a single corpus file."""

    file_id: str              # e.g., "line_000114"
    segments_path: Path
    embeddings_path: Path
    num_segments: int
    source_line_number: int


class CorpusLoader:
    """
    Loads per-file corpus data from a directory structure.

    Expects a directory with:
    - embeddings_metadata.json (index of all files)
    - For each file: {file_id}_segments.csv and {file_id}_embeddings.npy
    """

    def __init__(self, data_dir: Path):
        """
        Initialize the corpus loader.

        Args:
            data_dir: Directory containing per-file embeddings and segments.
        """
        self.data_dir = Path(data_dir)
        self.metadata_path = self.data_dir / "embeddings_metadata.json"

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self._files: Dict[str, CorpusFile] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load the embeddings metadata JSON."""
        with open(self.metadata_path, "r") as f:
            meta = json.load(f)

        self.embedding_dim = meta["embedding_dimension"]
        self.normalized = meta.get("normalized", True)

        for file_info in meta["files"]:
            file_id = Path(file_info["embeddings_file"]).stem.replace("_embeddings", "")

            self._files[file_id] = CorpusFile(
                file_id=file_id,
                segments_path=self.data_dir / file_info["segments_file"],
                embeddings_path=self.data_dir / file_info["embeddings_file"],
                num_segments=file_info["num_segments"],
                source_line_number=file_info["source_line_number"],
            )

    @property
    def file_ids(self) -> List[str]:
        """Return sorted list of file IDs."""
        return sorted(self._files.keys())

    @property
    def num_files(self) -> int:
        """Return the number of corpus files."""
        return len(self._files)

    def get_file(self, file_id: str) -> CorpusFile:
        """Get metadata for a specific file."""
        return self._files[file_id]

    def load_file(self, file_id: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load segments and embeddings for a single file.

        Args:
            file_id: The file identifier (e.g., "line_000114").

        Returns:
            Tuple of (segments DataFrame, embeddings ndarray).
        """
        file_info = self._files[file_id]

        # Load segments
        segments = pd.read_csv(file_info.segments_path)

        # Load embeddings
        embeddings = np.load(file_info.embeddings_path)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        return segments, embeddings

    def load_batch(
        self,
        file_ids: List[str]
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, int]]:
        """
        Load and concatenate multiple files.

        Args:
            file_ids: List of file IDs to load.

        Returns:
            Tuple of:
            - Combined segments DataFrame with 'source_file_id' column added
            - Combined embeddings ndarray
            - Offset mapping: {file_id: start_index} for global ID tracking
        """
        all_segments = []
        all_embeddings = []
        offsets = {}
        current_offset = 0

        for file_id in file_ids:
            segments, embeddings = self.load_file(file_id)

            # Track offset for this file
            offsets[file_id] = current_offset

            # Add source file identifier to segments
            segments = segments.copy()
            segments["source_file_id"] = file_id

            all_segments.append(segments)
            all_embeddings.append(embeddings)

            current_offset += len(embeddings)

        combined_segments = pd.concat(all_segments, ignore_index=True)
        combined_embeddings = np.vstack(all_embeddings)

        return combined_segments, combined_embeddings, offsets

    def get_file_id_for_index(
        self,
        global_index: int,
        offsets: Dict[str, int]
    ) -> Tuple[str, int]:
        """
        Get the file ID and local index for a global index.

        Args:
            global_index: Index in the combined array.
            offsets: Offset mapping from load_batch.

        Returns:
            Tuple of (file_id, local_index).
        """
        # Sort offsets by value to find the right file
        sorted_offsets = sorted(offsets.items(), key=lambda x: x[1], reverse=True)

        for file_id, offset in sorted_offsets:
            if global_index >= offset:
                return file_id, global_index - offset

        raise ValueError(f"Invalid global index: {global_index}")
