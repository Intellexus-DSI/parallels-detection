"""Command-line interface for the parallels pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from .config import Config, MatchingConfig, ProcessingConfig, OutputConfig
from .pipeline import ParallelsPipeline


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Find parallel text segments across a corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using a config file
  python -m parallels.cli --config config.yaml

   # Direct arguments
   python -m parallels.cli \\
       --segments data/segments.xlsx \\
       --embeddings data/embeddings.npy \\
       --output output/parallels.csv \\
       --threshold 0.85

  # Using KNN strategy
  python -m parallels.cli --config config.yaml --strategy knn --k 10
        """,
    )

    # Config file option
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file",
    )

    # Input files
    parser.add_argument(
        "--segments",
        type=Path,
        help="Path to segments CSV or XLSX file",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        help="Path to embeddings .npy file",
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/parallels.csv"),
        help="Path to output file (default: output/parallels.csv)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet", "json"],
        default="csv",
        help="Output format (default: csv)",
    )

    # Matching options
    parser.add_argument(
        "--strategy",
        choices=["threshold", "knn"],
        help="Matching strategy (default: threshold)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Similarity threshold for matching (default: 0.85)",
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Number of neighbors for KNN strategy (default: 10)",
    )

    # Processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing (default: 1000)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip embedding normalization",
    )

    # Other options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Exclude segment text from output",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    """Build configuration from arguments."""
    # Start with config file if provided
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        # Require segments and embeddings if no config file
        if not args.segments or not args.embeddings:
            raise ValueError(
                "Either --config or both --segments (CSV/XLSX) and --embeddings are required"
            )
        config = Config(
            segments_csv=args.segments,
            embeddings_path=args.embeddings,
        )

    # Override with command-line arguments
    if args.segments:
        config.segments_csv = args.segments
    if args.embeddings:
        config.embeddings_path = args.embeddings
    if args.output:
        config.output_path = args.output

    # Matching config overrides
    if args.strategy:
        config.matching.strategy = args.strategy
    if args.threshold is not None:
        config.matching.threshold = args.threshold
    if args.k is not None:
        config.matching.k = args.k

    # Processing config overrides
    if args.batch_size:
        config.processing.batch_size = args.batch_size
    if args.no_normalize:
        config.processing.normalize_embeddings = False

    # Output config overrides
    if args.format:
        config.output.format = args.format
    if args.no_text:
        config.output.include_text = False

    return config


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    try:
        config = build_config(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Run the pipeline
    try:
        pipeline = ParallelsPipeline(config)
        match_count = pipeline.run()
        print(f"\nFound {match_count} parallel matches")
        print(f"Results saved to: {config.output_path}")
        return 0
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logging.exception("Pipeline failed")
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
