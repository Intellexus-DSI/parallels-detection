"""Configuration management for the segmentation pipeline."""

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class AzureConfig(BaseModel):
    """Configuration for Azure storage."""

    account_url: str = "https://intlxresearchstorage.file.core.windows.net"
    sas_token: str = ""
    share_name: str = "intlx-gpu-fs"
    file_path: str = ""


class SegmentationConfig(BaseModel):
    """Configuration for segmentation engine."""

    engine: Literal["botok", "regex"] = "regex"
    min_syllables: int = Field(default=4, ge=1)
    use_overlapping: bool = True
    overlap_max_atoms: int = Field(default=8, ge=1)
    overlap_min_chars: int = Field(default=8, ge=1)
    overlap_max_chars: int = Field(default=350, ge=1)
    max_spans_per_line: int = Field(default=300, ge=1)
    remove_spaces: bool = Field(default=False, description="Remove all spaces from Tibetan text")
    embedding_model: str = Field(
        default="Intellexus/Bi-Tib-mbert-v2",
        description="Model name for tokenizer (to calculate Wylie token lengths)"
    )


class OutputConfig(BaseModel):
    """Configuration for output options."""

    output_dir: Path = Path("data/segmented_output")
    save_single_lines: bool = True  # Create individual CSV files per line
    save_full_files: bool = True     # Create one CSV file with all segments combined


class Config(BaseModel):
    """Main configuration for the segmentation pipeline."""

    input_file: Optional[Path] = None
    azure: AzureConfig = Field(default_factory=AzureConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("input_file", mode="before")
    @classmethod
    def convert_to_path(cls, v):
        """Convert string to Path."""
        if v is None:
            return None
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
        if data.get("input_file"):
            data["input_file"] = str(data["input_file"])
        data["output"]["output_dir"] = str(data["output"]["output_dir"])
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
