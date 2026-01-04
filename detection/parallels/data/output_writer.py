"""Output writer for parallel matches."""

import json
from pathlib import Path
from typing import Iterator, List, Literal, Union

import pandas as pd

from ..models import ParallelMatch


class OutputWriter:
    """
    Writes parallel matches to various output formats.
    
    Supports CSV, Parquet, and JSON formats.
    Can be used as a context manager for streaming writes.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        format: Literal["csv", "parquet", "json"] = "csv",
        include_text: bool = True,
    ):
        """
        Initialize the output writer.

        Args:
            output_path: Path to write output file.
            format: Output format (csv, parquet, or json).
            include_text: Whether to include segment text in output.
        """
        self.output_path = Path(output_path)
        self.format = format
        self.include_text = include_text
        self._buffer: List[dict] = []

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_match(self, match: ParallelMatch) -> None:
        """
        Write a single match to the buffer.

        Args:
            match: A ParallelMatch to write.
        """
        self._buffer.append(match.to_dict(include_text=self.include_text))

    def write_matches(self, matches: Iterator[ParallelMatch]) -> None:
        """
        Write multiple matches to the buffer.

        Args:
            matches: Iterator of ParallelMatch objects.
        """
        for match in matches:
            self.write_match(match)

    def flush(self) -> None:
        """Write buffered matches to file."""
        if not self._buffer:
            return

        df = pd.DataFrame(self._buffer)

        if self.format == "csv":
            df.to_csv(self.output_path, index=False)
        elif self.format == "parquet":
            df.to_parquet(self.output_path, index=False)
        elif self.format == "json":
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(self._buffer, f, ensure_ascii=False, indent=2)

    def __enter__(self) -> "OutputWriter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - flush on close."""
        self.flush()

    @property
    def count(self) -> int:
        """Return the number of matches in the buffer."""
        return len(self._buffer)
