"""End-to-end tests for the parallels pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from parallels.config import Config
from parallels.pipeline import ParallelsPipeline


def create_test_data(tmp_path: Path, n_segments: int = 100, n_texts: int = 5):
    """Create synthetic test data."""
    # Create segments CSV
    data = []
    for i in range(n_segments):
        text_id = f"text_{i % n_texts}.html"
        data.append({
            "Segmented_Text": f"སེགམེནཏ་{i}",
            "Segmented_Text_EWTS": f"segment_{i}",
            "Length": 10 + i % 20,
            "File_Path": text_id,
            "Title": f"Title {i % n_texts}",
            "Source_Line_Number": i,
            "Sentence_Order": 1,
            "Start_Index": 0,
            "End_Index": 10,
        })
    
    df = pd.DataFrame(data)
    csv_path = tmp_path / "segments.csv"
    df.to_csv(csv_path, sep="\t", index=False)
    
    # Create embeddings with some similar pairs
    embeddings = np.random.randn(n_segments, 768).astype(np.float32)
    
    # Make some segments similar (from different texts)
    # Segments 0 (text_0) and 6 (text_1) should be similar
    embeddings[6] = embeddings[0] + np.random.randn(768).astype(np.float32) * 0.1
    # Segments 2 (text_2) and 8 (text_3) should be similar
    embeddings[8] = embeddings[2] + np.random.randn(768).astype(np.float32) * 0.1
    
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    
    return csv_path, embeddings_path


class TestPipeline:
    """Tests for the main pipeline."""

    def test_threshold_matching(self, tmp_path):
        """Test threshold-based matching finds similar segments."""
        csv_path, embeddings_path = create_test_data(tmp_path)
        output_path = tmp_path / "parallels.csv"
        
        config = Config(
            segments_csv=csv_path,
            embeddings_path=embeddings_path,
            output_path=output_path,
        )
        config.matching.strategy = "threshold"
        config.matching.threshold = 0.5  # Low threshold to catch our synthetic pairs
        
        pipeline = ParallelsPipeline(config)
        match_count = pipeline.run()
        
        # Should find some matches
        assert match_count > 0
        assert output_path.exists()
        
        # Check output format
        results = pd.read_csv(output_path)
        assert "segment_a_id" in results.columns
        assert "segment_b_id" in results.columns
        assert "similarity" in results.columns
        
    def test_knn_matching(self, tmp_path):
        """Test KNN-based matching."""
        csv_path, embeddings_path = create_test_data(tmp_path)
        output_path = tmp_path / "parallels.csv"
        
        config = Config(
            segments_csv=csv_path,
            embeddings_path=embeddings_path,
            output_path=output_path,
        )
        config.matching.strategy = "knn"
        config.matching.k = 5
        
        pipeline = ParallelsPipeline(config)
        match_count = pipeline.run()
        
        # Should find matches
        assert match_count > 0
        assert output_path.exists()
        
    def test_cross_text_filtering(self, tmp_path):
        """Test that same-text matches are filtered out."""
        csv_path, embeddings_path = create_test_data(tmp_path, n_segments=20, n_texts=2)
        output_path = tmp_path / "parallels.csv"
        
        config = Config(
            segments_csv=csv_path,
            embeddings_path=embeddings_path,
            output_path=output_path,
        )
        config.matching.strategy = "knn"
        config.matching.k = 10
        
        pipeline = ParallelsPipeline(config)
        pipeline.run()
        
        results = pd.read_csv(output_path)
        
        # Load original data to check file paths
        segments = pd.read_csv(csv_path, sep="\t")
        
        # Verify no same-text matches
        for _, row in results.iterrows():
            file_a = segments.iloc[row["segment_a_id"]]["File_Path"]
            file_b = segments.iloc[row["segment_b_id"]]["File_Path"]
            assert file_a != file_b, "Found same-text match that should be filtered"


class TestConfig:
    """Tests for configuration."""

    def test_config_from_yaml(self, tmp_path):
        """Test loading config from YAML."""
        yaml_content = """
segments_csv: "data/segments.csv"
embeddings_path: "data/embeddings.npy"
output_path: "output/parallels.csv"

matching:
  strategy: "threshold"
  threshold: 0.9
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)
        
        config = Config.from_yaml(config_path)
        assert config.matching.threshold == 0.9
        assert config.matching.strategy == "threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
