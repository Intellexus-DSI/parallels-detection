#!/usr/bin/env python
"""Main orchestrator for the parallel text detection pipeline.

Runs all three stages in sequence:
1. Segmentation: Break texts into segments
2. Indexing: Generate embeddings
3. Detection: Find parallel matches
"""

import argparse
import logging
import sys
import subprocess
from pathlib import Path
from typing import Optional

import yaml


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_command(cmd: list[str], cwd: Optional[Path] = None) -> int:
    """Run a shell command and return exit code.
    
    Args:
        cmd: Command and arguments as list
        cwd: Working directory
        
    Returns:
        Exit code (0 = success)
    """
    logging.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def run_segmentation(config: dict) -> int:
    """Run the segmentation stage.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Exit code (0 = success)
    """
    logging.info("=" * 60)
    logging.info("STAGE 1: SEGMENTATION")
    logging.info("=" * 60)
    
    seg_config = config.get("segmentation", {})
    
    # Check if segmentation script exists
    script_path = Path("segmentation/tibet_segmentation.py")
    if not script_path.exists():
        logging.error(f"Segmentation script not found: {script_path}")
        return 1
    
    # For now, just inform user to configure and run manually
    logging.info("Segmentation stage needs to be configured manually.")
    logging.info(f"Please edit: {script_path}")
    logging.info("Then run: python segmentation/tibet_segmentation.py")
    logging.info("")
    logging.info("Configuration from pipeline_config.yaml:")
    for key, value in seg_config.items():
        logging.info(f"  {key}: {value}")
    
    # TODO: When segmentation gets CLI support, call it programmatically
    # cmd = ["python", "segmentation/tibet_segmentation.py"]
    # return run_command(cmd)
    
    return 0


def run_indexing(config: dict) -> int:
    """Run the indexing stage.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Exit code (0 = success)
    """
    logging.info("=" * 60)
    logging.info("STAGE 2: INDEXING")
    logging.info("=" * 60)
    
    idx_config = config.get("indexing", {})
    
    logging.info("Indexing stage is not yet implemented.")
    logging.info("This stage will:")
    logging.info("  1. Load segmented Excel files")
    logging.info("  2. Generate embeddings using transformer model")
    logging.info("  3. Save embeddings.npy and full_segmentation.xlsx")
    logging.info("")
    logging.info("Configuration from pipeline_config.yaml:")
    for key, value in idx_config.items():
        logging.info(f"  {key}: {value}")
    
    # TODO: Implement indexing stage
    # cmd = ["python", "-m", "indexer.cli", "--config", "indexing/config.yaml"]
    # return run_command(cmd, cwd=Path("indexing"))
    
    return 0


def run_detection(config: dict) -> int:
    """Run the detection stage.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Exit code (0 = success)
    """
    logging.info("=" * 60)
    logging.info("STAGE 3: DETECTION")
    logging.info("=" * 60)
    
    det_config = config.get("detection", {})
    
    # Build command for detection CLI
    cmd = ["python", "-m", "parallels.cli"]
    
    # Add arguments from config
    if "segments_csv" in det_config:
        cmd.extend(["--segments", str(det_config["segments_csv"])])
    if "embeddings_path" in det_config:
        cmd.extend(["--embeddings", str(det_config["embeddings_path"])])
    if "output_path" in det_config:
        cmd.extend(["--output", str(det_config["output_path"])])
    
    # Matching options
    matching = det_config.get("matching", {})
    if "strategy" in matching:
        cmd.extend(["--strategy", matching["strategy"]])
    if "threshold" in matching:
        cmd.extend(["--threshold", str(matching["threshold"])])
    if "k" in matching:
        cmd.extend(["--k", str(matching["k"])])
    
    # Output format
    output = det_config.get("output", {})
    if "format" in output:
        cmd.extend(["--format", output["format"]])
    
    return run_command(cmd, cwd=Path("detection"))


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the full parallel text detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all stages
  python run_pipeline.py --config pipeline_config.yaml
  
  # Run specific stages
  python run_pipeline.py --config pipeline_config.yaml --stage segmentation
  python run_pipeline.py --config pipeline_config.yaml --stage detection
  
  # Skip stages
  python run_pipeline.py --config pipeline_config.yaml --skip segmentation
        """,
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to pipeline configuration YAML file",
    )
    parser.add_argument(
        "--stage",
        choices=["segmentation", "indexing", "detection"],
        help="Run only this stage",
    )
    parser.add_argument(
        "--skip",
        action="append",
        choices=["segmentation", "indexing", "detection"],
        help="Skip this stage (can be used multiple times)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Load configuration
    if not args.config.exists():
        logging.error(f"Configuration file not found: {args.config}")
        return 1
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Loaded configuration from: {args.config}")
    
    # Determine which stages to run
    skip_stages = set(args.skip or [])
    
    if args.stage:
        stages = [args.stage]
    else:
        stages = ["segmentation", "indexing", "detection"]
    
    stages = [s for s in stages if s not in skip_stages]
    
    logging.info(f"Running stages: {', '.join(stages)}")
    logging.info("")
    
    # Run stages in order
    exit_code = 0
    
    if "segmentation" in stages:
        exit_code = run_segmentation(config)
        if exit_code != 0:
            logging.error("Segmentation stage failed")
            return exit_code
        logging.info("")
    
    if "indexing" in stages:
        exit_code = run_indexing(config)
        if exit_code != 0:
            logging.error("Indexing stage failed")
            return exit_code
        logging.info("")
    
    if "detection" in stages:
        exit_code = run_detection(config)
        if exit_code != 0:
            logging.error("Detection stage failed")
            return exit_code
        logging.info("")
    
    logging.info("=" * 60)
    logging.info("PIPELINE COMPLETE")
    logging.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
