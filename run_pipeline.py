#!/usr/bin/env python
"""Pipeline orchestrator for parallel text detection.

Runs all four stages in sequence:
1. Segmentation: Break texts into segments
2. Embedding: Generate vector embeddings
3. Detection: Find parallel matches
4. Enriching: Add additional fields to parallels

Data flow:
  data/*.jsonl -> segmentation/output/ -> embedding/output/ -> detection/output/ -> enriching/output/ -> output/
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import yaml


def get_python_executable(root_dir: Path) -> str:
    """Get the Python executable, preferring venv if available."""
    venv_python = root_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_command(cmd: List[str], cwd: Path = None) -> int:
    """Run a shell command and return exit code."""
    logging.info(f"Running: {' '.join(cmd)}")
    if cwd:
        logging.info(f"  in: {cwd}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def run_segmentation(config: dict, root_dir: Path) -> int:
    """Run the segmentation stage."""
    logging.info("=" * 60)
    logging.info("STAGE 1: SEGMENTATION")
    logging.info("=" * 60)

    seg_config = config.get("segmentation", {})
    emb_config = config.get("embedding", {})
    input_file = root_dir / seg_config.get("input_file", "data/input.jsonl")
    output_dir = root_dir / seg_config.get("output_dir", "segmentation/output")

    if not input_file.exists():
        logging.error(f"Input file not found: {input_file}")
        return 1

    # Build command for segmenter CLI
    python_exe = get_python_executable(root_dir)
    cmd = [
        python_exe, "-m", "segmenter.cli", "segment",
        "--input", str(input_file),
        "--output", str(output_dir),
        "--engine", seg_config.get("engine", "regex"),
        "--min-syllables", str(seg_config.get("min_syllables", 4)),
    ]

    if seg_config.get("use_overlapping", True):
        cmd.append("--overlapping")
        if seg_config.get("overlap_max_atoms"):
            cmd.extend(["--max-atoms", str(seg_config["overlap_max_atoms"])])
    else:
        cmd.append("--exclusive")
    
    if seg_config.get("remove_spaces", False):
        cmd.append("--remove-spaces")
    
    # Use model from embedding section for token length calculation
    if emb_config.get("model"):
        cmd.extend(["--embedding-model", emb_config["model"]])

    return run_command(cmd, cwd=root_dir / "segmentation")


def run_embedding(config: dict, root_dir: Path) -> int:
    """Run the embedding stage."""
    logging.info("=" * 60)
    logging.info("STAGE 2: EMBEDDING")
    logging.info("=" * 60)

    emb_config = config.get("embedding", {})
    input_dir = root_dir / emb_config.get("input_dir", "segmentation/output/overlapping/Full_Files")
    output_dir = root_dir / emb_config.get("output_dir", "embedding/output")

    if not input_dir.exists():
        logging.error(f"Input directory not found: {input_dir}")
        logging.error("Did segmentation stage complete?")
        return 1

    # Build command for embedding CLI
    python_exe = get_python_executable(root_dir)
    cmd = [
        python_exe, "-m", "embedding.cli",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--model", emb_config.get("model", "Intellexus/Bi-Tib-mbert-v2"),
        "--batch-size", str(emb_config.get("batch_size", 32)),
        "--mode", emb_config.get("mode", "per_line"),
    ]

    if emb_config.get("device"):
        cmd.extend(["--device", emb_config["device"]])

    return run_command(cmd, cwd=root_dir / "embedding")


def run_detection(config: dict, root_dir: Path) -> int:
    """Run the detection stage."""
    logging.info("=" * 60)
    logging.info("STAGE 3: DETECTION")
    logging.info("=" * 60)

    det_config = config.get("detection", {})
    data_dir = root_dir / det_config.get("data_dir", "embedding/output/embeddings_by_line")
    output_path = root_dir / det_config.get("output_path", "detection/output/parallels.csv")

    if not data_dir.exists():
        logging.error(f"Data directory not found: {data_dir}")
        logging.error("Did embedding stage complete?")
        return 1

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build command for detection CLI
    python_exe = get_python_executable(root_dir)
    cmd = [
        python_exe, "-m", "src.cli",
        "--data-dir", str(data_dir),
        "--output", str(output_path),
        "--threshold", str(det_config.get("threshold", 0.85)),
        "--batch-size", str(det_config.get("batch_size", 5)),
        "--strategy", det_config.get("strategy", "threshold"),
        "--format", det_config.get("format", "csv"),
    ]

    if det_config.get("device"):
        cmd.extend(["--device", det_config["device"]])

    if det_config.get("max_lines_per_file", 0) > 0:
        cmd.extend(["--max-lines-per-file", str(det_config["max_lines_per_file"])])

    return run_command(cmd, cwd=root_dir / "detection")


def run_enriching(config: dict, root_dir: Path) -> int:
    """Run the enriching stage."""
    logging.info("=" * 60)
    logging.info("STAGE 4: ENRICHING")
    logging.info("=" * 60)

    enr_config = config.get("enriching", {})
    input_path = root_dir / enr_config.get("input", {}).get("path", "detection/output/parallels.csv")
    output_path = root_dir / enr_config.get("output", {}).get("path", "enriching/output/parallels_enriched.csv")

    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        logging.error("Did detection stage complete?")
        return 1

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build command for enriching CLI
    python_exe = get_python_executable(root_dir)
    cmd = [
        python_exe, "-m", "src.cli",
        "--input", str(input_path),
        "--output", str(output_path),
        "--input-format", enr_config.get("input", {}).get("format", "csv"),
        "--output-format", enr_config.get("output", {}).get("format", "csv"),
    ]

    # Add enrichers
    enrichers = enr_config.get("enrichers", [])
    if enrichers:
        for enricher in enrichers:
            if enricher.get("enabled", True):
                cmd.extend(["--enricher", enricher["name"]])
    else:
        # Default: use wylie_levenshtein and mapping_type
        cmd.extend(["--enricher", "wylie_levenshtein", "--enricher", "mapping_type"])

    if enr_config.get("output", {}).get("max_lines_per_file", 0) > 0:
        cmd.extend(["--max-lines-per-file", str(enr_config["output"]["max_lines_per_file"])])

    return run_command(cmd, cwd=root_dir / "enriching")


def copy_final_output(config: dict, root_dir: Path) -> int:
    """Copy or combine final results to root output directory."""
    logging.info("=" * 60)
    logging.info("COPYING FINAL OUTPUT")
    logging.info("=" * 60)

    det_config = config.get("detection", {})
    output_path = Path(det_config.get("output_path", "detection/output/parallels.csv"))
    source_dir = root_dir / output_path.parent
    dest_dir = root_dir / "output"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if chunked output (max_lines_per_file > 0)
    max_lines = det_config.get("max_lines_per_file", 0)
    if max_lines > 0:
        return combine_chunk_files(source_dir, output_path.stem, output_path.suffix, dest_dir)
    else:
        # Single file mode
        source = root_dir / output_path
        if not source.exists():
            logging.error(f"Source file not found: {source}")
            return 1
        dest = dest_dir / source.name
        shutil.copy2(source, dest)
        logging.info(f"Copied: {source} -> {dest}")
        return 0


def combine_chunk_files(source_dir: Path, stem: str, suffix: str, dest_dir: Path) -> int:
    """Combine chunk files into a single output file."""
    import glob
    import pandas as pd

    # Find all chunk files (e.g., parallels_000.csv, parallels_001.csv, ...)
    pattern = str(source_dir / f"{stem}_[0-9][0-9][0-9]{suffix}")
    chunk_files = sorted(glob.glob(pattern))

    if not chunk_files:
        logging.error(f"No chunk files found matching pattern: {pattern}")
        return 1

    logging.info(f"Found {len(chunk_files)} chunk files to combine")

    dest_file = dest_dir / f"{stem}{suffix}"

    if suffix == ".csv":
        # Stream combine CSV files
        first = True
        with open(dest_file, "w", encoding="utf-8") as out:
            for chunk_file in chunk_files:
                logging.info(f"  Adding: {Path(chunk_file).name}")
                with open(chunk_file, "r", encoding="utf-8") as inp:
                    if first:
                        # Include header from first file
                        out.write(inp.read())
                        first = False
                    else:
                        # Skip header line for subsequent files
                        next(inp)  # Skip header
                        out.write(inp.read())
    elif suffix == ".parquet":
        # Combine parquet files
        dfs = [pd.read_parquet(f) for f in chunk_files]
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_parquet(dest_file, index=False)
    elif suffix == ".json":
        import json
        all_data = []
        for chunk_file in chunk_files:
            with open(chunk_file, "r", encoding="utf-8") as f:
                all_data.extend(json.load(f))
        with open(dest_file, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
    else:
        logging.error(f"Unsupported format: {suffix}")
        return 1

    logging.info(f"Combined output written to: {dest_file}")
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the parallel text detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --config pipeline_config.yaml
  python run_pipeline.py --config pipeline_config.yaml --stage segmentation
  python run_pipeline.py --config pipeline_config.yaml --stage embedding
  python run_pipeline.py --config pipeline_config.yaml --stage detection
  python run_pipeline.py --config pipeline_config.yaml --stage enriching
        """,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pipeline_config.yaml"),
        help="Path to pipeline configuration (default: pipeline_config.yaml)",
    )
    parser.add_argument(
        "--stage",
        choices=["segmentation", "embedding", "detection", "enriching"],
        help="Run only this stage",
    )
    parser.add_argument(
        "--skip",
        action="append",
        choices=["segmentation", "embedding", "detection", "enriching"],
        help="Skip this stage (can be repeated)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Determine root directory
    root_dir = Path(__file__).parent.absolute()

    # Load configuration
    config_path = root_dir / args.config if not args.config.is_absolute() else args.config
    if not config_path.exists():
        logging.error(f"Configuration file not found: {config_path}")
        return 1

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logging.info(f"Loaded configuration from: {config_path}")

    # Determine which stages to run
    skip_stages = set(args.skip or [])
    if args.stage:
        stages = [args.stage]
    else:
        stages = ["segmentation", "embedding", "detection", "enriching"]
    stages = [s for s in stages if s not in skip_stages]

    logging.info(f"Running stages: {', '.join(stages)}")
    logging.info("")

    # Run stages
    for stage in stages:
        if stage == "segmentation":
            exit_code = run_segmentation(config, root_dir)
            if exit_code == 0:
                logging.info(f"Segmentation output available in: {root_dir / 'segmentation/output'}")
        elif stage == "embedding":
            exit_code = run_embedding(config, root_dir)
            if exit_code == 0:
                logging.info(f"Embedding output available in: {root_dir / 'embedding/output'}")
        elif stage == "detection":
            exit_code = run_detection(config, root_dir)
            if exit_code == 0:
                logging.info(f"Detection output available in: {root_dir / 'detection/output'}")
        elif stage == "enriching":
            exit_code = run_enriching(config, root_dir)
            if exit_code == 0:
                logging.info(f"Enriching output available in: {root_dir / 'enriching/output'}")
        else:
            continue

        if exit_code != 0:
            logging.error(f"{stage.upper()} stage failed")
            return exit_code
        logging.info("")

    # Copy final output to root output/ directory
    if "enriching" in stages:
        # Copy enriched output
        enr_config = config.get("enriching", {})
        output_path = Path(enr_config.get("output", {}).get("path", "enriching/output/parallels_enriched.csv"))
        source = root_dir / output_path
        dest_dir = root_dir / "output"
        dest_dir.mkdir(parents=True, exist_ok=True)
        if source.exists():
            dest = dest_dir / source.name
            shutil.copy2(source, dest)
            logging.info(f"Copied: {source} -> {dest}")
    elif "detection" in stages:
        exit_code = copy_final_output(config, root_dir)
        if exit_code != 0:
            return exit_code

    logging.info("=" * 60)
    logging.info("PIPELINE COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Results: {root_dir / 'output'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
