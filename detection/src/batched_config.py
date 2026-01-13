"""Configuration for the parallels detection pipeline."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class MatchingConfig(BaseModel):
    """Configuration for matching strategy."""

    strategy: Literal["threshold", "knn"] = "threshold"
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    k: int = Field(default=10, ge=1)
    min_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class ProcessingConfig(BaseModel):
    """Configuration for processing options."""

    batch_size: int = Field(default=1000, ge=1)
    normalize_embeddings: bool = True


class OutputConfig(BaseModel):
    """Configuration for output options."""

    format: Literal["csv", "parquet", "json"] = "csv"
    include_text: bool = True


class BatchedConfig(BaseModel):
    """Main configuration for the parallels detection pipeline."""

    data_dir: Path                    # Directory with per-file embeddings
    output_path: Path = Path("output/parallels.csv")
    batch_size: int = Field(default=5, ge=1)  # Number of files in index at once

    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("data_dir", mode="before")
    @classmethod
    def convert_data_dir_to_path(cls, v):
        """Convert string to Path."""
        return Path(v) if isinstance(v, str) else v

    @field_validator("output_path", mode="before")
    @classmethod
    def convert_output_to_path(cls, v):
        """Convert string to Path."""
        return Path(v) if isinstance(v, str) else v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BatchedConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
