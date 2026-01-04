"""Segment data store for loading and accessing segments and embeddings."""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from ..models import Segment


class SegmentStore:
    """
    Manages segment metadata and embeddings.

    Loads segments from a CSV or XLSX file and embeddings from a NumPy file.
    Provides access to individual segments and their text IDs for filtering.
    """

    # Expected CSV columns
    REQUIRED_COLUMNS = [
        "Segmented_Text",
        "Segmented_Text_EWTS",
        "Length",
        "File_Path",
        "Title",
        "Source_Line_Number",
        "Sentence_Order",
        "Start_Index",
        "End_Index",
    ]

    def __init__(self, csv_path: Union[str, Path], embeddings_path: Union[str, Path]):
        """
        Initialize the segment store.

        Args:
            csv_path: Path to the segments CSV file.
            embeddings_path: Path to the embeddings .npy file.
        """
        self.csv_path = Path(csv_path)
        self.embeddings_path = Path(embeddings_path)

        self.metadata: pd.DataFrame = self._load_metadata()
        self.embeddings: np.ndarray = self._load_embeddings()
        self.text_ids: np.ndarray = self.metadata["File_Path"].values

        self._validate()

    def _load_metadata(self) -> pd.DataFrame:
        """Load segment metadata from CSV or XLSX."""
        if self.csv_path.suffix.lower() == '.xlsx':
            df = pd.read_excel(self.csv_path)
        else:
            df = pd.read_csv(self.csv_path, sep="\t")

        # Validate required columns
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {self.csv_path.suffix.upper()}: {missing}")

        return df

    def _load_embeddings(self) -> np.ndarray:
        """Load embeddings from NumPy file."""
        embeddings = np.load(self.embeddings_path)

        # Ensure float32 for FAISS
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        return embeddings

    def _validate(self) -> None:
        """Validate that metadata and embeddings are aligned."""
        if len(self.metadata) != len(self.embeddings):
            raise ValueError(
                f"Mismatch: CSV has {len(self.metadata)} rows, "
                f"embeddings has {len(self.embeddings)} vectors"
            )

    def __len__(self) -> int:
        """Return the number of segments."""
        return len(self.metadata)

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embeddings.shape[1]

    @property
    def num_texts(self) -> int:
        """Return the number of unique texts (File_Paths)."""
        return self.metadata["File_Path"].nunique()

    def get_text_id(self, segment_id: int) -> str:
        """
        Get the text ID (File_Path) for a segment.

        Args:
            segment_id: Row index of the segment.

        Returns:
            The File_Path of the segment's source text.
        """
        return self.text_ids[segment_id]

    def get_segment(self, segment_id: int) -> Segment:
        """
        Get a full Segment object by ID.

        Args:
            segment_id: Row index of the segment.

        Returns:
            A Segment object with all metadata.
        """
        row = self.metadata.iloc[segment_id]
        return Segment(
            id=segment_id,
            text_id=row["File_Path"],
            title=row["Title"],
            text=row["Segmented_Text"],
            text_ewts=row["Segmented_Text_EWTS"],
            length=row["Length"],
            source_line_number=row["Source_Line_Number"],
            sentence_order=row["Sentence_Order"],
            start_index=row["Start_Index"],
            end_index=row["End_Index"],
        )

    def get_embedding(self, segment_id: int) -> np.ndarray:
        """
        Get the embedding vector for a segment.

        Args:
            segment_id: Row index of the segment.

        Returns:
            The embedding vector as a 1D numpy array.
        """
        return self.embeddings[segment_id]

    def get_batch_embeddings(self, start: int, end: int) -> np.ndarray:
        """
        Get a batch of embeddings.

        Args:
            start: Start index (inclusive).
            end: End index (exclusive).

        Returns:
            A 2D numpy array of shape (end-start, embedding_dim).
        """
        return self.embeddings[start:end]

    def get_metadata_for_output(self, segment_id: int) -> dict:
        """
        Get metadata needed for output enrichment.

        Args:
            segment_id: Row index of the segment.

        Returns:
            Dictionary with title, ewts, and file_path.
        """
        row = self.metadata.iloc[segment_id]
        return {
            "title": row["Title"],
            "ewts": row["Segmented_Text_EWTS"],
            "file_path": row["File_Path"],
        }
