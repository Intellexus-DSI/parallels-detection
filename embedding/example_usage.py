#!/usr/bin/env python3
"""
Example usage of the embedding pipeline.
Demonstrates how to use the embedding package programmatically.
"""

from pathlib import Path
from embedding import EmbeddingPipeline, IndexingConfig, load_config

def example_with_config_file():
    """Example: Run pipeline with config file."""
    print("Example 1: Using config file")
    print("-" * 60)
    
    config_path = Path("config.yaml")
    config = load_config(config_path)
    
    pipeline = EmbeddingPipeline(config)
    embeddings, segments = pipeline.run()
    
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Segments shape: {segments.shape}")


def example_with_code_config():
    """Example: Configure pipeline programmatically."""
    print("\nExample 2: Programmatic configuration (Per-File Mode)")
    print("-" * 60)
    
    from embedding.models import InputConfig, EmbeddingConfig, OutputConfig, ProcessingConfig
    
    config = IndexingConfig(
        input=InputConfig(
            segments_dir=Path("../data/05_clean_data/00_tibetan/segmented_output/overlapping/Full_Files"),
            text_column="Segmented_Text_EWTS",
            id_column="Segment_ID"
        ),
        embedding=EmbeddingConfig(
            model_name="Intellexus/Bi-Tib-mbert-v1",
            batch_size=32,
            normalize=True,
            device="auto"
        ),
        output=OutputConfig(
            output_dir=Path("../detection/data"),
            mode="per_file",  # ← Per-file mode
            per_file_subdir="embeddings_by_source"
        ),
        processing=ProcessingConfig(
            max_segments=100,  # Limit for testing
            skip_existing=False
        )
    )
    
    pipeline = EmbeddingPipeline(config)
    embeddings, segments = pipeline.run()
    
    print(f"Generated {len(embeddings)} embeddings in per-file mode (limited to 100 for testing)")


def example_load_per_file_embeddings():
    """Example: Load per-file embeddings."""
    print("\nExample 3: Loading per-file embeddings")
    print("-" * 60)
    
    import numpy as np
    import pandas as pd
    import json
    
    embeddings_dir = Path("../detection/data/embeddings_by_source")
    
    # Load metadata to see what files we have
    with open(embeddings_dir / "embeddings_metadata.json") as f:
        metadata = json.load(f)
    
    print(f"Found {len(metadata['files'])} source files:")
    for file_info in metadata['files']:
        print(f"  - {file_info['source_file']}: {file_info['num_segments']} segments")
    
    # Load first file
    first_file = metadata['files'][0]
    source = first_file['source_file']
    
    embeddings = np.load(embeddings_dir / first_file['embeddings_file'])
    segments = pd.read_excel(embeddings_dir / first_file['segments_file'])
    
    print(f"\nLoaded {source}:")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Segments: {len(segments)} rows")
    
    # Compute similarity between first two segments
    from sklearn.metrics.pairwise import cosine_similarity
    
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    print(f"  Similarity between segments 0 and 1: {sim:.4f}")


def example_load_combined_embeddings():
    """Example: Load combined embeddings (if using combined mode)."""
    print("\nExample 4: Loading combined embeddings")
    print("-" * 60)
    
    import numpy as np
    import pandas as pd
    
    # Load embeddings
    embeddings = np.load("../detection/data/embeddings.npy")
    segments = pd.read_excel("../detection/data/full_segmentation.xlsx")
    
    print(f"Loaded {len(embeddings)} embeddings")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Sample segments:")
    print(segments.head())
    
    # Compute similarity between first two segments
    from sklearn.metrics.pairwise import cosine_similarity
    
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    print(f"\nSimilarity between segments 0 and 1: {sim:.4f}")


if __name__ == "__main__":
    print("="*60)
    print("INDEXING PIPELINE - USAGE EXAMPLES")
    print("="*60)
    
    # Uncomment the example you want to run
    
    # example_with_config_file()
    # example_with_code_config()
    # example_load_per_file_embeddings()
    # example_load_combined_embeddings()
    
    print("\n" + "="*60)
    print("Note: Uncomment the example you want to run")
    print("="*60)

