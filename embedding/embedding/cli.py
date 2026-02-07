"""
Command-line interface for the embedding stage.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .config import load_config, get_default_config
from .pipeline import EmbeddingPipeline


def main(args: Optional[list] = None):
    """
    Main entry point for the embedding CLI.
    
    Args:
        args: Command-line arguments (for testing)
    """
    parser = argparse.ArgumentParser(
        description="Generate embeddings for segmented Tibetan text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  embedding --config config.yaml
  
  # Run with default config
  embedding
  
  # Override specific settings
  embedding --config config.yaml --batch-size 64 --max-segments 1000
  
  # Use specific model
  embedding --model "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        help='Directory containing segmented CSV files (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for embeddings and segments (overrides config)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='HuggingFace model name (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        help='Batch size for embedding generation (overrides config)'
    )
    
    parser.add_argument(
        '--max-segments',
        type=int,
        help='Maximum number of segments to process (for testing)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        help='Device for model inference (overrides config)'
    )
    
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Do not normalize embeddings'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip processing if output files already exist'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['combined', 'per_file', 'per_line'],
        help='Output mode: "combined" for one file, "per_file" for separate files per source, "per_line" for separate files per Source_Line_Number'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress bars'
    )
    
    parsed_args = parser.parse_args(args)
    
    # Load configuration
    try:
        if parsed_args.config:
            config = load_config(parsed_args.config)
            print(f"Loaded config from: {parsed_args.config}")
        else:
            # Try to find config.yaml in current directory
            config_path = Path("config.yaml")
            if config_path.exists():
                config = load_config(config_path)
                print(f"Loaded config from: {config_path}")
            else:
                print("No config file found, using defaults")
                config = get_default_config()
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Apply command-line overrides
    if parsed_args.input_dir:
        config.input.segments_dir = Path(parsed_args.input_dir)
    
    if parsed_args.output_dir:
        config.output.output_dir = Path(parsed_args.output_dir)
    
    if parsed_args.model:
        config.embedding.model_name = parsed_args.model
    
    if parsed_args.batch_size:
        config.embedding.batch_size = parsed_args.batch_size
    
    if parsed_args.max_segments:
        config.processing.max_segments = parsed_args.max_segments
    
    if parsed_args.device:
        config.embedding.device = parsed_args.device
    
    if parsed_args.no_normalize:
        config.embedding.normalize = False
    
    if parsed_args.skip_existing:
        config.processing.skip_existing = True
    
    if parsed_args.mode:
        config.output.mode = parsed_args.mode
    
    if parsed_args.quiet:
        config.embedding.show_progress = False
    
    # Validate configuration
    try:
        if not config.input.segments_dir.exists():
            print(f"Error: Input directory does not exist: {config.input.segments_dir}", 
                  file=sys.stderr)
            sys.exit(1)
        
        # Create output directory if needed
        config.output.output_dir.mkdir(parents=True, exist_ok=True)
        
    except Exception as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run pipeline
    try:
        pipeline = EmbeddingPipeline(config)
        embeddings, segments = pipeline.run()
        
        print("\nSuccess!")
        if config.output.mode == "per_file":
            output_subdir = config.output.output_dir / config.output.per_file_subdir
            print(f"  Per-file embeddings saved to: {output_subdir}")
        elif config.output.mode == "per_line":
            output_subdir = config.output.output_dir / config.output.per_line_subdir
            print(f"  Per-line embeddings saved to: {output_subdir}")
        else:
            print(f"  Embeddings saved: {config.output.output_dir / config.output.embeddings_file}")
            print(f"  Segments saved: {config.output.output_dir / config.output.segments_file}")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

