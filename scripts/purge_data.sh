#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Project root is one level up
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "Purging pipeline output data in: $PROJECT_ROOT"

# Define directories to clean
DIRS=(
    "$PROJECT_ROOT/segmentation/output"
    "$PROJECT_ROOT/embedding/output"
    "$PROJECT_ROOT/detection/output"
    "$PROJECT_ROOT/detection/data"
    "$PROJECT_ROOT/output"
)

for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Cleaning: $dir"
        # Delete everything inside the directory EXCEPT .gitkeep
        find "$dir" -mindepth 1 -not -name ".gitkeep" -exec rm -rf {} +
    else
        echo "Skipping (not found): $dir"
    fi
done

echo "Purge complete."
