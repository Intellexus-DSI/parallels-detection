"""Command-line interface for the enriching pipeline."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from .config import EnrichingConfig, InputConfig, OutputConfig, EnricherConfig
from .pipeline import EnrichingPipeline


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
        description="Enrich parallel matches with additional fields",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file
  python -m src.cli --config enriching_config.yaml

  # Using command-line arguments
  python -m src.cli \\
      --input ../detection/output/parallels.csv \\
      --output output/parallels_enriched.csv \\
      --enricher wylie_levenshtein
        """,
    )

    # Config file or CLI arguments
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file",
    )

    # Input/Output
    parser.add_argument(
        "--input",
        type=Path,
        help="Input file path (CSV/Parquet/JSON from detection stage)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path",
    )
    parser.add_argument(
        "--input-format",
        choices=["csv", "parquet", "json"],
        default="csv",
        help="Input file format (default: csv)",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "parquet", "json"],
        default="csv",
        help="Output file format (default: csv)",
    )

    # Enrichers
    parser.add_argument(
        "--enricher",
        action="append",
        choices=["wylie_levenshtein", "mapping_type"],
        help="Enricher to apply (can be repeated)",
    )

    # Other options
    parser.add_argument(
        "--max-lines-per-file",
        type=int,
        default=0,
        help="Max lines per output file (0 = unlimited, single file)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> EnrichingConfig:
    """Create configuration from command-line arguments."""
    # Input configuration
    if not args.input:
        raise ValueError("--input is required when not using --config")
    
    input_config = InputConfig(
        input_path=args.input,
        format=args.input_format,
    )

    # Output configuration
    if not args.output:
        raise ValueError("--output is required when not using --config")
    
    output_config = OutputConfig(
        output_path=args.output,
        format=args.output_format,
        max_lines_per_file=args.max_lines_per_file,
    )

    # Enrichers configuration
    enrichers = []
    if args.enricher:
        for enricher_name in args.enricher:
            enrichers.append(EnricherConfig(
                name=enricher_name,
                enabled=True,
                params={},
            ))
    else:
        # Default: enable wylie_levenshtein and mapping_type
        enrichers.extend([
            EnricherConfig(name="wylie_levenshtein", enabled=True, params={}),
            EnricherConfig(name="mapping_type", enabled=True, params={}),
        ])

    return EnrichingConfig(
        input=input_config,
        output=output_config,
        enrichers=enrichers,
    )


def config_from_file(config_path: Path) -> EnrichingConfig:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    
    return EnrichingConfig.from_dict(data)


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    try:
        # Load configuration
        if args.config:
            config = config_from_file(args.config)
        else:
            config = config_from_args(args)
        
        # Run pipeline
        pipeline = EnrichingPipeline(config)
        count = pipeline.run()
        
        print(f"\nEnriched {count} parallels")
        print(f"Results saved to: {config.output.output_path}")
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logging.exception("Pipeline failed")
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
