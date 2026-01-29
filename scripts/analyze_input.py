#!/usr/bin/env python3
"""
Analyze input JSONL file for data quality issues:
1. Missing metadata (file_path, title).
2. Text length and segmentation suitability (shad presence, total length).
"""

import json
import argparse
import re
from pathlib import Path
from tqdm import tqdm

# Tibetan constants
TIBETAN_SHAD = "\u0F0D"
TIBETAN_DOUBLE_SHAD = "\u0F0E"
TER_TSHEG = "\u0F14"

def analyze_file(input_path: str):
    print(f"Analyzing {input_path}...")
    
    stats = {
        "total_lines": 0,
        "missing_metadata": 0,
        "missing_filepath": 0,
        "missing_title": 0,
        "no_shads": 0,
        "too_long_no_shads": 0,  # Likely dropped by max_chars
        "empty_text": 0
    }
    
    # Thresholds from pipeline_config.yaml
    MAX_CHARS = 350
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(tqdm(f), 1):
            stats["total_lines"] += 1
            line = line.strip()
            if not line:
                continue
                
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Line {line_num}: Invalid JSON")
                continue
                
            text = record.get("text") or record.get("content") or ""
            metadata = record.get("metadata", {})
            
            # Check text
            if not text:
                stats["empty_text"] += 1
            else:
                has_shad = (TIBETAN_SHAD in text or 
                           TIBETAN_DOUBLE_SHAD in text or 
                           TER_TSHEG in text)
                
                if not has_shad:
                    stats["no_shads"] += 1
                    if len(text) > MAX_CHARS:
                        stats["too_long_no_shads"] += 1
                        if line_num <= 5: # Print first few examples
                            print(f"[WARN] Line {line_num} has {len(text)} chars but NO shads. Likely dropped.")

            # Check metadata
            if not metadata:
                stats["missing_metadata"] += 1
            else:
                if not metadata.get("file_path"):
                    stats["missing_filepath"] += 1
                    if line_num == 116:
                        print(f"[WARN] Line {line_num} is missing file_path!")
                        
                if not metadata.get("title"):
                    stats["missing_title"] += 1

    print("\nAnalysis Results:")
    print(f"Total lines: {stats['total_lines']}")
    print(f"Empty text: {stats['empty_text']}")
    print(f"Missing metadata block: {stats['missing_metadata']}")
    print(f"Missing file_path: {stats['missing_filepath']}")
    print(f"Missing title: {stats['missing_title']}")
    print("-" * 30)
    print(f"Records with NO shads: {stats['no_shads']}")
    print(f"Records too long (> {MAX_CHARS} chars) AND no shads: {stats['too_long_no_shads']}")
    print("  (These are likely treated as single segments and then dropped by the overlap filter)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to JSONL file")
    args = parser.parse_args()
    
    if Path(args.input_file).exists():
        analyze_file(args.input_file)
    else:
        print(f"File not found: {args.input_file}")
