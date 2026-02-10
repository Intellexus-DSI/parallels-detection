"""
Tests for the embedding pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from embedding.models import (
    IndexingConfig,
    InputConfig,
    EmbeddingConfig,
    OutputConfig,
    ProcessingConfig
)
from embedding.pipeline import EmbeddingPipeline


def test_config_validation():
    """Test configuration validation."""
    # This should work with default values
    config = IndexingConfig(
        input=InputConfig(segments_dir=Path(".")),
        embedding=EmbeddingConfig(),
        output=OutputConfig(output_dir=Path(".")),
        processing=ProcessingConfig()
    )
    
    assert config.embedding.model_name == "Intellexus/Bi-Tib-mbert-v2"
    assert config.embedding.batch_size == 32
    assert config.embedding.normalize == True


def test_embedding_config_defaults():
    """Test embedding configuration defaults."""
    config = EmbeddingConfig()
    
    assert config.model_name == "Intellexus/Bi-Tib-mbert-v2"
    assert config.batch_size == 32
    assert config.max_length == 512
    assert config.normalize == True
    assert config.device == "auto"


def test_processing_config():
    """Test processing configuration."""
    config = ProcessingConfig(
        max_segments=100,
        skip_existing=True
    )
    
    assert config.max_segments == 100
    assert config.skip_existing == True


def test_yaml_config_loading():
    """Test loading config from YAML."""
    # This test requires a valid config.yaml file
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    if config_path.exists():
        config = IndexingConfig.from_yaml(config_path)
        assert isinstance(config, IndexingConfig)
        assert config.embedding.model_name == "Intellexus/Bi-Tib-mbert-v2"


# Integration tests (commented out - require actual data and model)
"""
def test_pipeline_run():
    # Test full pipeline run
    config = IndexingConfig(...)
    pipeline = EmbeddingPipeline(config)
    embeddings, segments = pipeline.run()
    
    assert isinstance(embeddings, np.ndarray)
    assert isinstance(segments, pd.DataFrame)
    assert len(embeddings) == len(segments)
    

def test_model_loading():
    # Test model loading
    config = EmbeddingConfig()
    pipeline = EmbeddingPipeline(...)
    model = pipeline.load_model()
    
    assert model is not None
    assert model.get_sentence_embedding_dimension() == 768
"""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

