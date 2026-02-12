#!/usr/bin/env python3
"""Convert Tibetan Unicode text files to Wylie transliteration.

Usage:
    python scripts/convert_tibetan.py INPUT_DIR OUTPUT_DIR
"""

import argparse
import os
import sys

from conversion import converter


def convert_files(input_dir, output_dir):
    if not os.path.isdir(input_dir):
        print(f"Error: input directory '{input_dir}' does not exist.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    cv = converter()

    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".txt"))
    if not files:
        print(f"No .txt files found in '{input_dir}'.")
        return

    for filename in files:
        file_path = os.path.join(input_dir, filename)
        print(f"Converting {filename} ...")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            converted = cv.convert(content, "Wylie")
        except Exception as e:
            print(f"  Error converting {filename}: {e}")
            continue

        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(converted)
        print(f"  -> {output_path}")

    print(f"Done. Converted {len(files)} file(s) to '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Tibetan Unicode text files to Wylie transliteration."
    )
    parser.add_argument("input_dir", help="Folder containing Tibetan .txt files")
    parser.add_argument("output_dir", help="Folder to write converted Wylie .txt files")
    args = parser.parse_args()

    convert_files(args.input_dir, args.output_dir)
