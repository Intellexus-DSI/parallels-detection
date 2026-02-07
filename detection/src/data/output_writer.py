"""Output writer for parallel matches."""

import json
import logging
from pathlib import Path
from typing import Iterator, List, Literal, Union

import pandas as pd

from ..models import ParallelMatch

logger = logging.getLogger(__name__)


class OutputWriter:
    """
    Writes parallel matches to various output formats.

    Supports CSV, Parquet, and JSON formats.
    Can be used as a context manager for streaming writes.
    Supports chunked output for large datasets.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        format: Literal["csv", "parquet", "json"] = "csv",
        include_text: bool = True,
        max_lines_per_file: int = 0,
    ):
        """
        Initialize the output writer.

        Args:
            output_path: Path to write output file.
            format: Output format (csv, parquet, or json).
            include_text: Whether to include segment text in output.
            max_lines_per_file: Max lines per file (0 = unlimited, single file).
        """
        self.output_path = Path(output_path)
        self.format = format
        self.include_text = include_text
        self.max_lines_per_file = max_lines_per_file
        self._buffer: List[dict] = []
        self._chunk_index = 0
        self._total_written = 0
        self._chunk_files: List[Path] = []

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_chunk_path(self, chunk_index: int) -> Path:
        """Get the path for a specific chunk file."""
        stem = self.output_path.stem
        suffix = self.output_path.suffix
        return self.output_path.parent / f"{stem}_{chunk_index:03d}{suffix}"

    def write_match(self, match: ParallelMatch) -> None:
        """
        Write a single match to the buffer.

        Args:
            match: A ParallelMatch to write.
        """
        self._buffer.append(match.to_dict(include_text=self.include_text))

        # Check if we need to flush due to chunk size limit
        if self.max_lines_per_file > 0 and len(self._buffer) >= self.max_lines_per_file:
            self._flush_chunk()

    def write_matches(self, matches: Iterator[ParallelMatch]) -> None:
        """
        Write multiple matches to the buffer.

        Args:
            matches: Iterator of ParallelMatch objects.
        """
        for match in matches:
            self.write_match(match)

    def _flush_chunk(self) -> None:
        """Write current buffer to a chunk file."""
        if not self._buffer:
            return

        chunk_path = self._get_chunk_path(self._chunk_index)
        self._write_to_file(chunk_path)
        self._chunk_files.append(chunk_path)

        logger.info(f"Wrote chunk {self._chunk_index}: {len(self._buffer)} matches to {chunk_path}")

        self._total_written += len(self._buffer)
        self._buffer = []
        self._chunk_index += 1

    def _write_to_file(self, path: Path) -> None:
        """Write buffer to a specific file."""
        df = pd.DataFrame(self._buffer)

        if self.format == "csv":
            df.to_csv(path, index=False)
        elif self.format == "parquet":
            df.to_parquet(path, index=False)
        elif self.format == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._buffer, f, ensure_ascii=False, indent=2)

    def flush(self) -> None:
        """Write any remaining buffered matches to file."""
        if not self._buffer:
            return

        if self.max_lines_per_file > 0:
            # Chunked mode: write remaining as final chunk
            self._flush_chunk()
        else:
            # Single file mode: write to output path
            self._write_to_file(self.output_path)
            self._total_written = len(self._buffer)

    def __enter__(self) -> "OutputWriter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - flush on close."""
        self.flush()

    @property
    def count(self) -> int:
        """Return the total number of matches written."""
        return self._total_written + len(self._buffer)

    @property
    def chunk_files(self) -> List[Path]:
        """Return list of chunk files written."""
        return self._chunk_files.copy()

    @property
    def is_chunked(self) -> bool:
        """Return whether output is chunked."""
        return self.max_lines_per_file > 0
