"""
Configuration utilities for the embedding stage.
"""

from pathlib import Path
from typing import Optional
import yaml

from .models import IndexingConfig


def load_config(config_path: Optional[str | Path] = None) -> IndexingConfig:
    """
    Load embedding configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, looks for config.yaml in current directory.
        
    Returns:
        IndexingConfig object
        
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config is invalid
    """
    if config_path is None:
        config_path = Path("config.yaml")
    
    return IndexingConfig.from_yaml(config_path)


def get_default_config() -> IndexingConfig:
    """
    Get default configuration for embedding.
    
    Returns:
        IndexingConfig with default values
    """
    from .models import InputConfig, EmbeddingConfig, OutputConfig, ProcessingConfig
    
    return IndexingConfig(
        input=InputConfig(
            segments_dir=Path("../data/05_clean_data/00_tibetan/segmented_output/overlapping/Full_Files"),
            text_column="Segmented_Text_EWTS",
            id_column="Segment_ID"
        ),
        embedding=EmbeddingConfig(
            model_name="Intellexus/Bi-Tib-mbert-v1",
            batch_size=32,
            max_length=512,
            normalize=True,
            device="auto",
            show_progress=True
        ),
        output=OutputConfig(
            output_dir=Path("../detection/data"),
            embeddings_file="embeddings.npy",
            segments_file="full_segmentation.xlsx",
            metadata_file="embeddings_metadata.json",
            segments_format="xlsx"
        ),
        processing=ProcessingConfig(
            skip_existing=False,
            max_segments=None,
            num_workers=0,
            chunk_size=10000
        )
    )

