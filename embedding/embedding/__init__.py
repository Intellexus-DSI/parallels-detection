"""Embedding - Generate embeddings for text segments."""

__version__ = "0.1.0"

from .pipeline import EmbeddingPipeline, run_pipeline
from .config import load_config, get_default_config
from .models import IndexingConfig, EmbeddingConfig, InputConfig, OutputConfig

__all__ = [
    "EmbeddingPipeline",
    "run_pipeline",
    "load_config",
    "get_default_config",
    "IndexingConfig",
    "EmbeddingConfig",
    "InputConfig",
    "OutputConfig",
]