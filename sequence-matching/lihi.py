import argparse
import os
import sys

def count_words(file_path):
    """
    Counts the number of words in a file by splitting on whitespace.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Splitting by whitespace handles spaces, tabs, and newlines
            words = content.split()
            return len(words)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Count words in a text file.")
    parser.add_argument("input_file", help="Path to the input text file.")
    
    args = parser.parse_args()

    word_count = count_words(args.input_file)
    print(f"File: {args.input_file}")
    print(f"Word count: {word_count}")

if __name__ == "__main__":
    main()
