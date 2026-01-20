#!/usr/bin/env python
"""Purge all pipeline output data.

Cleans output directories from all stages while preserving:
- Input data in data/
- Directory structure (.gitkeep files)
- Source code and configs
"""

import argparse
import shutil
import sys
from pathlib import Path


def get_root_dir() -> Path:
    """Get the pipeline root directory."""
    return Path(__file__).parent.parent.absolute()


def purge_directory(path: Path, keep_gitkeep: bool = True) -> int:
    """Remove all files in a directory.

    Args:
        path: Directory to purge
        keep_gitkeep: If True, preserve .gitkeep files

    Returns:
        Number of items removed
    """
    if not path.exists():
        return 0

    count = 0
    for item in path.iterdir():
        if keep_gitkeep and item.name == ".gitkeep":
            continue

        if item.is_dir():
            shutil.rmtree(item)
            print(f"  Removed directory: {item}")
        else:
            item.unlink()
            print(f"  Removed file: {item}")
        count += 1

    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Purge all pipeline output data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/purge.py              # Purge all stage outputs
  python scripts/purge.py --all        # Also purge input data
  python scripts/purge.py --stage segmentation  # Purge only segmentation
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Also purge input data in data/ (excluding .gitkeep)",
    )
    parser.add_argument(
        "--stage",
        choices=["segmentation", "embedding", "detection", "output"],
        action="append",
        help="Purge only specific stage(s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )

    args = parser.parse_args()
    root = get_root_dir()

    # Define directories to purge
    stage_dirs = {
        "segmentation": root / "segmentation" / "output",
        "embedding": root / "embedding" / "output",
        "detection": root / "detection" / "output",
        "detection_data": root / "detection" / "data",  # Previous embeddings location
        "output": root / "output",
    }

    # Determine which stages to purge
    if args.stage:
        stages = args.stage
    else:
        stages = list(stage_dirs.keys())

    if args.dry_run:
        print("DRY RUN - No files will be deleted\n")

    total_removed = 0

    # Purge stage outputs
    for stage in stages:
        path = stage_dirs[stage]
        print(f"Purging {stage}: {path}")

        if args.dry_run:
            if path.exists():
                items = [p for p in path.iterdir() if p.name != ".gitkeep"]
                for item in items:
                    print(f"  Would remove: {item}")
                total_removed += len(items)
            else:
                print(f"  Directory does not exist")
        else:
            total_removed += purge_directory(path)

    # Optionally purge input data
    if args.all:
        data_dir = root / "data"
        print(f"\nPurging input data: {data_dir}")

        if args.dry_run:
            if data_dir.exists():
                items = [p for p in data_dir.iterdir() if p.name != ".gitkeep"]
                for item in items:
                    print(f"  Would remove: {item}")
                total_removed += len(items)
        else:
            total_removed += purge_directory(data_dir)

    print(f"\n{'Would remove' if args.dry_run else 'Removed'} {total_removed} items")
    return 0


if __name__ == "__main__":
    sys.exit(main())
