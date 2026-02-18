"""Command-line interface for the segmentation pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from .config import Config, SegmentationConfig, OutputConfig, AzureConfig
from .data import AzureDownloader
from .pipeline import SegmentationPipeline


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
        description="Segment Tibetan text files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using a config file
  segmenter --config config.yaml

  # Direct arguments
  segmenter --input data/input.jsonl --output data/output --engine regex

  # Using overlapping segmentation
  segmenter --input data/input.jsonl --overlapping --max-atoms 8

  # Download from Azure
  segmenter download --azure-path "data/file.jsonl" --output data/local.jsonl
        """,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Segment command (default)
    segment_parser = subparsers.add_parser("segment", help="Segment text files")
    setup_segment_parser(segment_parser)

    # Download command
    download_parser = subparsers.add_parser(
        "download", help="Download file from Azure"
    )
    setup_download_parser(download_parser)

    # If no command specified, treat as segment command
    args, unknown = parser.parse_known_args()
    if args.command is None:
        # Parse as segment command
        setup_segment_parser(parser)
        args = parser.parse_args()
        args.command = "segment"

    return args


def setup_segment_parser(parser: argparse.ArgumentParser) -> None:
    """Setup arguments for segment command."""
    # Config file option
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file",
    )

    # Input/Output
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for segmented files",
    )

    # Segmentation options
    parser.add_argument(
        "--engine",
        choices=["botok", "regex"],
        help="Segmentation engine (default: regex)",
    )
    parser.add_argument(
        "--min-syllables",
        type=int,
        help="Minimum syllables per segment (default: 4)",
    )
    parser.add_argument(
        "--max-syllables",
        type=int,
        help="Maximum syllables per segment in exclusive mode",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        help="Minimum words per segment in exclusive mode (uses Botok)",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        help="Maximum words per segment in exclusive mode (uses Botok)",
    )
    parser.add_argument(
        "--overlapping",
        action="store_true",
        help="Use overlapping segmentation mode",
    )
    parser.add_argument(
        "--exclusive",
        action="store_true",
        help="Use exclusive segmentation mode (default)",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        help="Maximum atoms per span in overlapping mode (default: 8)",
    )
    parser.add_argument(
        "--remove-spaces",
        action="store_true",
        help="Remove all spaces from Tibetan text during cleaning",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, use e.g. 8 for multi-core)",
    )

    # Output options
    parser.add_argument(
        "--no-single-lines",
        action="store_true",
        help="Skip saving individual line files",
    )
    parser.add_argument(
        "--no-full-files",
        action="store_true",
        help="Skip saving combined file with all segments",
    )

    # Other options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )


def setup_download_parser(parser: argparse.ArgumentParser) -> None:
    """Setup arguments for download command."""
    parser.add_argument(
        "--azure-path",
        type=str,
        required=True,
        help="Path to file in Azure file share",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Local path to save the downloaded file",
    )
    parser.add_argument(
        "--account-url",
        type=str,
        help="Azure storage account URL",
    )
    parser.add_argument(
        "--sas-token",
        type=str,
        help="SAS token for authentication",
    )
    parser.add_argument(
        "--share-name",
        type=str,
        help="Azure file share name",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )


def build_config(args: argparse.Namespace) -> Config:
    """Build configuration from arguments."""
    # Start with config file if provided
    if hasattr(args, "config") and args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Override with command-line arguments
    if hasattr(args, "input") and args.input:
        config.input_file = args.input
    if hasattr(args, "output") and args.output:
        config.output.output_dir = args.output

    # Segmentation config overrides
    if hasattr(args, "engine") and args.engine:
        config.segmentation.engine = args.engine
    if hasattr(args, "min_syllables") and args.min_syllables:
        config.segmentation.min_syllables = args.min_syllables
    if hasattr(args, "max_syllables") and args.max_syllables is not None:
        config.segmentation.max_syllables = args.max_syllables
    if hasattr(args, "min_words") and args.min_words is not None:
        config.segmentation.min_words = args.min_words
    if hasattr(args, "max_words") and args.max_words is not None:
        config.segmentation.max_words = args.max_words
    if hasattr(args, "overlapping") and args.overlapping:
        config.segmentation.use_overlapping = True
    if hasattr(args, "exclusive") and args.exclusive:
        config.segmentation.use_overlapping = False
    if hasattr(args, "max_atoms") and args.max_atoms:
        config.segmentation.overlap_max_atoms = args.max_atoms
    if hasattr(args, "remove_spaces") and args.remove_spaces:
        config.segmentation.remove_spaces = True
    if hasattr(args, "workers") and args.workers is not None:
        config.segmentation.workers = args.workers

    # Output config overrides
    if hasattr(args, "no_single_lines") and args.no_single_lines:
        config.output.save_single_lines = False
    if hasattr(args, "no_full_files") and args.no_full_files:
        config.output.save_full_files = False

    return config


def handle_download(args: argparse.Namespace) -> int:
    """Handle download command."""
    # Build Azure config
    azure_config = AzureConfig()
    if args.account_url:
        azure_config.account_url = args.account_url
    if args.sas_token:
        azure_config.sas_token = args.sas_token
    if args.share_name:
        azure_config.share_name = args.share_name

    if not azure_config.sas_token:
        print("Error: SAS token is required (use --sas-token)", file=sys.stderr)
        return 1

    try:
        downloader = AzureDownloader(
            account_url=azure_config.account_url,
            sas_token=azure_config.sas_token,
            share_name=azure_config.share_name,
        )
        downloader.download_file(args.azure_path, args.output)
        return 0
    except Exception as e:
        logging.exception("Download failed")
        print(f"Error: {e}", file=sys.stderr)
        return 1


def handle_segment(args: argparse.Namespace) -> int:
    """Handle segment command."""
    try:
        config = build_config(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not config.input_file:
        print("Error: Input file is required (use --input or --config)", file=sys.stderr)
        return 1

    # Run the pipeline
    try:
        pipeline = SegmentationPipeline(config)
        line_count = pipeline.run()
        print(f"\nProcessed {line_count} lines")
        return 0
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logging.exception("Segmentation failed")
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose if hasattr(args, "verbose") else False)

    if args.command == "download":
        return handle_download(args)
    else:
        return handle_segment(args)


if __name__ == "__main__":
    sys.exit(main())
