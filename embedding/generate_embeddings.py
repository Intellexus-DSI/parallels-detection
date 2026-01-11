#!/usr/bin/env python3
"""
Standalone script to generate embeddings.
Can be run directly without installing the package.
"""

import sys
from pathlib import Path

# Add embedding to path
sys.path.insert(0, str(Path(__file__).parent))

from embedding.pipeline import run_pipeline


def main():
    """Run the embedding generation pipeline."""
    # Default to config.yaml in the same directory
    config_path = Path(__file__).parent / "config.yaml"
    
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Usage: python generate_embeddings.py [config.yaml]")
        sys.exit(1)
    
    try:
        embeddings, segments = run_pipeline(config_path)
        print(f"\n✓ Generated {len(embeddings)} embeddings successfully!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

