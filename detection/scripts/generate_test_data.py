#!/usr/bin/env python3
"""Generate synthetic test data for testing the parallels pipeline."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_test_data(
    output_dir: Path,
    n_segments: int = 1000,
    n_texts: int = 20,
    embedding_dim: int = 768,
    seed: int = 42,
):
    """
    Generate synthetic test data.
    
    Creates some intentionally similar segments across different texts
    to verify the pipeline finds them.
    """
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate segment metadata
    print(f"Generating {n_segments} segments from {n_texts} texts...")
    data = []
    for i in range(n_segments):
        text_idx = i % n_texts
        text_id = f"text_{text_idx:03d}.html"
        data.append({
            "Segmented_Text": f"ཚིག་གྲུབ་{i}་ནི་དཔེ་མཚོན་ཡིན།",
            "Segmented_Text_EWTS": f"tshig grub {i} ni dpe mtshon yin/",
            "Length": 20 + np.random.randint(0, 50),
            "File_Path": text_id,
            "Title": f"Document Title {text_idx}",
            "Source_Line_Number": i // 10,
            "Sentence_Order": i % 10 + 1,
            "Start_Index": 0,
            "End_Index": 50,
        })
    
    df = pd.DataFrame(data)
    csv_path = output_dir / "segments.csv"
    df.to_csv(csv_path, sep="\t", index=False)
    print(f"Saved segments to {csv_path}")
    
    # Generate embeddings
    print(f"Generating {n_segments} x {embedding_dim} embeddings...")
    embeddings = np.random.randn(n_segments, embedding_dim).astype(np.float32)
    
    # Create intentional similar pairs across different texts
    # We'll make segments with similar content have similar embeddings
    n_similar_pairs = min(50, n_segments // 4)
    print(f"Creating {n_similar_pairs} intentionally similar cross-text pairs...")
    
    for i in range(n_similar_pairs):
        # Pick two segments from different texts
        seg_a = i * 2
        seg_b = i * 2 + n_texts  # Different text
        
        if seg_b < n_segments:
            # Make seg_b similar to seg_a with small noise
            noise = np.random.randn(embedding_dim).astype(np.float32) * 0.05
            embeddings[seg_b] = embeddings[seg_a] + noise
    
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")
    
    # Calculate expected file size
    file_size_mb = (n_segments * embedding_dim * 4) / (1024 * 1024)
    print(f"Embeddings file size: {file_size_mb:.2f} MB")
    
    print("\nTest data generation complete!")
    print(f"  - Segments: {csv_path}")
    print(f"  - Embeddings: {embeddings_path}")
    print(f"\nRun the pipeline with:")
    print(f"  python -m parallels.cli --segments {csv_path} --embeddings {embeddings_path} --threshold 0.9")


def main():
    parser = argparse.ArgumentParser(description="Generate test data for parallels pipeline")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--n-segments",
        type=int,
        default=1000,
        help="Number of segments to generate (default: 1000)",
    )
    parser.add_argument(
        "--n-texts",
        type=int,
        default=20,
        help="Number of unique texts (default: 20)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=768,
        help="Embedding dimension (default: 768)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()
    generate_test_data(
        args.output_dir,
        args.n_segments,
        args.n_texts,
        args.embedding_dim,
        args.seed,
    )


if __name__ == "__main__":
    main()
