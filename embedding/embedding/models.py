"""
Pydantic models for embedding configuration and data structures.
"""

from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


class InputConfig(BaseModel):
    """Input configuration for loading segmented files."""
    
    segments_dir: Path = Field(
        description="Directory containing segmented CSV files"
    )
    text_column: str = Field(
        default="Segmented_Text_EWTS",
        description="Column name containing text to embed"
    )
    id_column: Optional[str] = Field(
        default="Segment_ID",
        description="Column name for unique segment identifiers"
    )
    
    @field_validator('segments_dir')
    @classmethod
    def validate_segments_dir(cls, v: Path) -> Path:
        """Ensure segments directory exists."""
        if not v.exists():
            raise ValueError(f"Segments directory does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"Segments path is not a directory: {v}")
        return v


class EmbeddingConfig(BaseModel):
    """Configuration for embedding model and generation."""
    
    model_name: str = Field(
        default="Intellexus/Bi-Tib-mbert-v2",
        description="HuggingFace model name for embeddings"
    )
    batch_size: int = Field(
        default=32,
        gt=0,
        description="Batch size for embedding generation"
    )
    max_length: int = Field(
        default=512,
        gt=0,
        description="Maximum sequence length in tokens"
    )
    normalize: bool = Field(
        default=True,
        description="Normalize embeddings for cosine similarity"
    )
    device: Literal["auto", "cuda", "cpu"] = Field(
        default="auto",
        description="Device for model inference"
    )
    show_progress: bool = Field(
        default=True,
        description="Show progress bar during embedding generation"
    )


class OutputConfig(BaseModel):
    """Output configuration for saving embeddings and segments."""
    
    output_dir: Path = Field(
        description="Directory for output files"
    )
    mode: Literal["combined", "per_file", "per_line"] = Field(
        default="per_file",
        description="Output mode: 'combined' for one file, 'per_file' for separate files per source, 'per_line' for separate files per Source_Line_Number"
    )
    embeddings_file: str = Field(
        default="embeddings.npy",
        description="Filename for embeddings numpy array (combined mode only)"
    )
    segments_file: str = Field(
        default="full_segmentation.csv",
        description="Filename for consolidated segments (combined mode only)"
    )
    per_file_subdir: str = Field(
        default="embeddings_by_source",
        description="Subdirectory for per-file outputs (per_file mode only)"
    )
    per_line_subdir: str = Field(
        default="embeddings_by_line",
        description="Subdirectory for per-line outputs (per_line mode only)"
    )
    metadata_file: str = Field(
        default="embeddings_metadata.json",
        description="Filename for embedding metadata"
    )
    segments_format: Literal["csv"] = Field(
        default="csv",
        description="Format for segments file"
    )
    
    @field_validator('output_dir')
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        """Create output directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


class ProcessingConfig(BaseModel):
    """Configuration for data processing."""
    
    skip_existing: bool = Field(
        default=False,
        description="Skip processing if output files exist"
    )
    max_segments: Optional[int] = Field(
        default=None,
        description="Maximum number of segments to process (for testing)"
    )
    num_workers: int = Field(
        default=0,
        ge=0,
        description="Number of workers for data loading"
    )
    chunk_size: int = Field(
        default=10000,
        gt=0,
        description="Chunk size for memory-efficient processing"
    )


class IndexingConfig(BaseModel):
    """Complete configuration for embedding stage."""
    
    input: InputConfig
    embedding: EmbeddingConfig
    output: OutputConfig
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "IndexingConfig":
        """Load configuration from YAML file."""
        import yaml
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)


class EmbeddingMetadata(BaseModel):
    """Metadata about generated embeddings."""
    
    model_name: str
    num_segments: int
    embedding_dimension: int
    normalized: bool
    created_at: str
    config: dict

