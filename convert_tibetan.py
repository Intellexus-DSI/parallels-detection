import os
import sys
from conversion import converter

def convert_files(input_dir):
    cv = converter()
    files_to_convert = ["001_'dul_ba_ka.txt", "002_'dul_ba_kha.txt"]
    
    for filename in files_to_convert:
        file_path = os.path.join(input_dir, filename)
        if not os.path.exists(file_path):
            print(f"File {file_path} not found.")
            continue
            
        print(f"Converting {filename} to Wylie...")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        try:
            # The README says cv.convert(text, "Wylie") or "EWTS"
            # Unicode -> Wylie
            converted_content = cv.convert(content, "Wylie")
            
            output_path = os.path.join(input_dir, filename.replace('.txt', '_wylie.txt'))
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(converted_content)
            print(f"Saved to {output_path}")
        except Exception as e:
            print(f"Error converting {filename}: {e}")

if __name__ == "__main__":
    input_directory = "sequence-matching/input"
    convert_files(input_directory)
