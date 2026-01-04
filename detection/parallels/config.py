"""Configuration management for the parallels pipeline."""

from pathlib import Path
from typing import Literal, Optional

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


class Config(BaseModel):
    """Main configuration for the parallels pipeline."""

    segments_csv: Path  # Can be .csv or .xlsx
    embeddings_path: Path
    output_path: Path = Path("output/parallels.csv")

    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("segments_csv", "embeddings_path", mode="before")
    @classmethod
    def convert_to_path(cls, v):
        """Convert string to Path."""
        return Path(v) if isinstance(v, str) else v

    @field_validator("output_path", mode="before")
    @classmethod
    def convert_output_to_path(cls, v):
        """Convert string to Path."""
        return Path(v) if isinstance(v, str) else v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        data = self.model_dump(mode="json")
        # Convert Path objects to strings for YAML
        data["segments_csv"] = str(data["segments_csv"])
        data["embeddings_path"] = str(data["embeddings_path"])
        data["output_path"] = str(data["output_path"])
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
