#!/usr/bin/env python3
import json
import os
from pathlib import Path

def convert_text_to_jsonl(text_folder, output_file):
    """
    Convert all text files in a folder to JSONL format.
    Each line in the JSONL has 'text' and 'file_id' fields.
    """
    text_path = Path(text_folder)
    
    if not text_path.exists():
        print(f"Error: {text_folder} does not exist")
        return
    
    # Get all .txt files in the folder
    text_files = sorted(text_path.glob("*.txt"))
    
    if not text_files:
        print(f"Warning: No .txt files found in {text_folder}")
        return
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for text_file in text_files:
            # Get file_id from the filename (without .txt extension)
            file_id = text_file.stem
            
            # Read the entire content of the text file
            with open(text_file, 'r', encoding='utf-8') as infile:
                text_content = infile.read()
            
            # Create JSON object
            json_obj = {
                "text": text_content,
                "file_id": file_id
            }
            
            # Write as single line JSON
            outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    
    print(f"Created {output_file} with {len(text_files)} entries")

if __name__ == "__main__":
    # Base directory (parent of parallels-detection)
    base_dir = Path(__file__).parent.parent.parent  # Go up to C:\Users\goody
    
    # Convert derge-kangyur
    kangyur_text = base_dir / "derge-kangyur" / "text"
    kangyur_output = Path(__file__).parent.parent / "derge-kangyur.jsonl"
    convert_text_to_jsonl(kangyur_text, kangyur_output)
    
    # Convert derge-tengyur
    tengyur_text = base_dir / "derge-tengyur" / "text"
    tengyur_output = Path(__file__).parent.parent / "derge-tengyur.jsonl"
    convert_text_to_jsonl(tengyur_text, tengyur_output)
    
    print("\nConversion complete!")
