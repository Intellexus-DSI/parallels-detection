"""Command-line interface for the parallels detection pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from .batched_config import BatchedConfig
from .batched_pipeline import BatchedPipeline


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Find parallel text segments across corpus files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli --data-dir data/embeddings_by_line --threshold 0.85

  python -m src.cli \\
      --data-dir data/embeddings_by_line \\
      --batch-size 5 \\
      --threshold 0.80 \\
      --output output/parallels.csv
        """,
    )

    # Required
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory with per-file embeddings (contains embeddings_metadata.json)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/parallels.csv"),
        help="Output file path (default: output/parallels.csv)",
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
        default="threshold",
        help="Matching strategy (default: threshold)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Similarity threshold (default: 0.85)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of neighbors for KNN strategy (default: 10)",
    )

    # Processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of files to load into index per batch (default: 5)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip embedding normalization",
    )

    # Other options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Exclude segment text from output",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    config = BatchedConfig(
        data_dir=args.data_dir,
        output_path=args.output,
        batch_size=args.batch_size,
    )

    # Apply overrides
    config.matching.strategy = args.strategy
    config.matching.threshold = args.threshold
    config.matching.k = args.k
    config.processing.normalize_embeddings = not args.no_normalize
    config.output.format = args.format
    config.output.include_text = not args.no_text

    try:
        pipeline = BatchedPipeline(config)
        match_count = pipeline.run()
        print(f"\nFound {match_count} parallel matches")
        print(f"Results saved to: {config.output_path}")
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logging.exception("Pipeline failed")
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
