import json
import re
import os
import pandas as pd
from tqdm import tqdm

# REQUIRED dependency: detect_and_convert for EWTS conversion
# Add submodule to path and import (submodule must be set up beforehand; see README)
import sys
from pathlib import Path

# Add detect_and_convert submodule to path (must be set up beforehand; see README)
_project_root = Path(__file__).parent.parent
_submodule_path = _project_root / "detect_and_convert"

# Check if submodule exists and has content
_submodule_ready = False
if _submodule_path.exists() and _submodule_path.is_dir():
    try:
        if any(_submodule_path.iterdir()):
            if str(_submodule_path) not in sys.path:
                sys.path.insert(0, str(_submodule_path))
            _submodule_ready = True
    except Exception:
        pass

if not _submodule_ready:
    raise ImportError(
        "detect_and_convert submodule not found. See README Setup (step 2) to clone/init and install it."
    )

try:
    from conversion import Converter, ConversionError
except ImportError as e:
    raise ImportError(
        f"Failed to import Converter from detect_and_convert: {e}\n"
        "See README Setup (step 2) to install the submodule."
    )

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "data/05_clean_data/00_tibetan/Tibetan_1.jsonl"
OUTPUT_DIR = "data/05_clean_data/00_tibetan/segmented_output"

# TOGGLE ENGINE HERE
# True  = Uses Botok (High Accuracy, Slower)
# False = Uses Regex (High Speed, Good Accuracy)
USE_BOTOK = False 

MIN_SYLLABLES = 4

# OVERLAPPING SEGMENTATION MODE
# False = Exclusive segmentation (current behavior)
# True  = Overlapping multi-scale segmentation
USE_OVERLAPPING_SEGMENTS = True

# Overlapping segmentation parameters
OVERLAP_MAX_ATOMS = 8      # Maximum atoms per span
OVERLAP_MIN_CHARS = 8      # Minimum characters per span
OVERLAP_MAX_CHARS = 350    # Maximum characters per span
MAX_SPANS_PER_LINE = 300   # Safety cap on spans per line

# ============================================================================
# CONSTANTS & SHARED PATTERNS
# ============================================================================

TIBETAN_SHAD = "\u0F0D"       # །
TIBETAN_DOUBLE_SHAD = "\u0F0E" # ༎
TSHEG = "\u0F0B"              # ་

TERMINATORS = {
    "གོ", "ངོ", "དོ", "ནོ", "བོ", "མོ", "འོ", "རོ", "ལོ", "སོ", "ཏོ", "ཐོ"
}
CONTINUATORS = {
    "དང་", "ནས", "ཏེ", "སྟེ", "དེ", "ཀྱང", "ཡང", "འང", "ཞིང", "ཤིང", "ཅིང"
}

# ============================================================================
# ENGINE 1: BOTOK SEGMENTER (ACCURATE)
# ============================================================================

class BotokSegmenter:
    def __init__(self, min_syllables=4):
        print("Initializing Botok Engine...")
        from botok import WordTokenizer
        self.tokenizer = WordTokenizer()
        self.min_syllables = min_syllables
        
        self.english_pattern = re.compile(r'[a-zA-Z]')
        self.tibetan_pattern = re.compile(r'[\u0F00-\u0FFF]')
        self.number_pattern = re.compile(r'[\u0F20-\u0F29]')

    def count_syllables(self, text):
        return text.count(TSHEG)

    def segment_with_indices(self, text):
        if not text: return []
        
        tokens = self.tokenizer.tokenize(text)
        final_sentences = []
        
        current_buffer = ""
        buffer_start_idx = 0
        current_cursor = 0 
        last_meaningful_word = ""

        for token in tokens:
            token_text = token.text
            current_buffer += token_text
            
            is_shad = TIBETAN_SHAD in token_text
            is_double_shad = TIBETAN_DOUBLE_SHAD in token_text
            
            should_split = False

            if is_shad or is_double_shad:
                should_split = True
                prev_word_clean = last_meaningful_word.strip().rstrip(TSHEG)

                if is_double_shad:
                    should_split = True
                elif self.number_pattern.search(prev_word_clean):
                    should_split = False
                elif prev_word_clean in CONTINUATORS:
                    should_split = False
                elif prev_word_clean in TERMINATORS:
                    should_split = True
                else:
                    syllables_in_buffer = self.count_syllables(current_buffer)
                    if syllables_in_buffer < self.min_syllables:
                        should_split = False

            if not (is_shad or is_double_shad or token_text.isspace()):
                last_meaningful_word = token_text

            if should_split:
                clean_sent = current_buffer.strip()
                has_tibetan = self.tibetan_pattern.search(clean_sent)
                has_english = self.english_pattern.search(clean_sent)
                
                if not (has_english and not has_tibetan):
                    end_idx = current_cursor + len(token_text)
                    final_sentences.append((clean_sent, buffer_start_idx, end_idx))
                    current_buffer = ""
                    buffer_start_idx = end_idx
            
            current_cursor += len(token_text)

        if current_buffer.strip():
            clean_sent = current_buffer.strip()
            final_sentences.append((clean_sent, buffer_start_idx, current_cursor))

        return final_sentences

# ============================================================================
# ENGINE 2: REGEX SEGMENTER (FAST)
# ============================================================================

class RegexSegmenter:
    def __init__(self, min_syllables=4):
        print("Initializing Fast Regex Engine...")
        self.split_pattern = re.compile(f"([{TIBETAN_SHAD}{TIBETAN_DOUBLE_SHAD}]+)")
        self.english_pattern = re.compile(r'[a-zA-Z]')
        self.tibetan_pattern = re.compile(r'[\u0F00-\u0FFF]')
        self.number_pattern = re.compile(r'[\u0F20-\u0F29]')
        self.min_syllables = min_syllables

    def get_last_syllable(self, text):
        text = text.rstrip()
        if not text: return ""
        last_tsheg_index = text.rfind(TSHEG)
        if last_tsheg_index == -1: return text
        return text[last_tsheg_index + 1:].strip()

    def segment_with_indices(self, text):
        if not text: return []

        parts = self.split_pattern.split(text)
        final_sentences = []
        
        current_buffer = [] 
        current_buffer_len = 0 
        buffer_start_idx = 0
        cursor = 0
        last_text_segment = ""

        for part in parts:
            if not part: continue
            
            part_len = len(part)
            is_delimiter = (TIBETAN_SHAD in part) or (TIBETAN_DOUBLE_SHAD in part)

            if is_delimiter:
                current_buffer.append(part)
                current_buffer_len += part_len
                
                should_split = True 
                
                if TIBETAN_DOUBLE_SHAD in part:
                    should_split = True
                else:
                    last_syllable = self.get_last_syllable(last_text_segment)
                    
                    if self.number_pattern.search(last_syllable):
                        should_split = False
                    elif last_syllable in CONTINUATORS:
                        should_split = False
                    elif last_syllable in TERMINATORS:
                        should_split = True
                    else:
                        full_buffer_str = "".join(current_buffer)
                        syllable_count = full_buffer_str.count(TSHEG)
                        if syllable_count < self.min_syllables:
                            should_split = False

                if should_split:
                    clean_sent = "".join(current_buffer).strip()
                    has_tibetan = self.tibetan_pattern.search(clean_sent)
                    has_english = self.english_pattern.search(clean_sent)
                    
                    if not (has_english and not has_tibetan):
                        final_sentences.append((clean_sent, buffer_start_idx, cursor + part_len))
                        current_buffer = []
                        current_buffer_len = 0
                        buffer_start_idx = cursor + part_len
                
                cursor += part_len
                
            else:
                if not current_buffer:
                    buffer_start_idx = cursor
                
                current_buffer.append(part)
                current_buffer_len += part_len
                last_text_segment = part
                cursor += part_len

        if current_buffer:
            clean_sent = "".join(current_buffer).strip()
            has_tibetan = self.tibetan_pattern.search(clean_sent)
            has_english = self.english_pattern.search(clean_sent)
            if not (has_english and not has_tibetan):
                final_sentences.append((clean_sent, buffer_start_idx, cursor))

        return final_sentences

# ============================================================================
# OVERLAPPING SEGMENTATION LOGIC
# ============================================================================

def make_overlapping_spans(atoms, original_text, *, max_atoms=8, min_chars=8, max_chars=350, max_spans=300):
    """
    Generate overlapping spans from atomic segments.
    
    Args:
        atoms: list of (text, start, end) tuples from segmenter
        original_text: original full text for accurate slicing
        max_atoms: maximum number of atoms per span
        min_chars: minimum characters per span (after strip)
        max_chars: maximum characters per span
        max_spans: maximum total spans to return (safety cap)
    
    Returns:
        list of (span_text, span_start, span_end, span_num_atoms, span_type)
    """
    if not atoms:
        return []
    
    spans = []
    seen_indices = set()  # For deduplication by (start, end)
    
    # Forward windows: for each start index i, windows of size 1..max_atoms
    for i in range(len(atoms)):
        for w in range(1, min(max_atoms + 1, len(atoms) - i + 1)):
            if i + w > len(atoms):
                break
            
            span_start = atoms[i][1]  # start index of first atom
            span_end = atoms[i + w - 1][2]  # end index of last atom
            
            # Deduplicate by indices
            if (span_start, span_end) in seen_indices:
                continue
            seen_indices.add((span_start, span_end))
            
            # Extract span from original text (preserves exact indices)
            span_text = original_text[span_start:span_end]
            
            # Apply filters
            if len(span_text.strip()) < min_chars:
                continue
            if len(span_text) > max_chars:
                continue
            
            spans.append((span_text.strip(), span_start, span_end, w, "forward"))
    
    # Centered windows: symmetric-ish patterns around each center
    center_patterns = [(1, 1), (2, 2), (3, 3), (2, 3), (3, 2)]
    
    for c in range(len(atoms)):
        for left_count, right_count in center_patterns:
            # For center at index c, include left_count atoms before (inclusive) and right_count atoms after (inclusive)
            start_idx = max(0, c - left_count + 1)  # +1 to make it inclusive of the leftmost atom
            end_idx = min(len(atoms), c + right_count + 1)  # +1 because end is exclusive in slicing
            
            if start_idx >= end_idx or start_idx < 0 or end_idx > len(atoms):
                continue
            
            span_start = atoms[start_idx][1]
            span_end = atoms[end_idx - 1][2]
            
            # Deduplicate by indices
            if (span_start, span_end) in seen_indices:
                continue
            seen_indices.add((span_start, span_end))
            
            # Extract span from original text
            span_text = original_text[span_start:span_end]
            
            # Apply filters
            if len(span_text.strip()) < min_chars:
                continue
            if len(span_text) > max_chars:
                continue
            
            num_atoms = end_idx - start_idx
            spans.append((span_text.strip(), span_start, span_end, num_atoms, "centered"))
    
    # Sanity checks
    for span_text, span_start, span_end, num_atoms, span_type in spans:
        assert 0 <= span_start < span_end <= len(original_text), \
            f"Invalid indices: {span_start}, {span_end} for text length {len(original_text)}"
        assert span_text == original_text[span_start:span_end].strip(), \
            f"Span text mismatch at ({span_start}, {span_end})"
    
    # Enforce max_spans cap (stable order: forward first, then centered)
    if len(spans) > max_spans:
        spans = spans[:max_spans]
    
    # Final deduplication check (should be redundant but safe)
    final_spans = []
    final_seen = set()
    for span in spans:
        span_start, span_end = span[1], span[2]
        if (span_start, span_end) not in final_seen:
            final_seen.add((span_start, span_end))
            final_spans.append(span)
    
    return final_spans

# ============================================================================
# MAIN PROCESSING LOGIC
# ============================================================================

def sanitize_filename(name):
    name = os.path.splitext(name)[0]
    return re.sub(r'[<>:"/\\|?*]', '_', name).strip()

def process_file(input_path, output_dir):
    # Setup Directories - add subfolder based on segmentation mode
    mode_subfolder = "overlapping" if USE_OVERLAPPING_SEGMENTS else "exclusive"
    full_output_dir = os.path.join(output_dir, mode_subfolder, "Full_Files")
    single_output_dir = os.path.join(output_dir, mode_subfolder, "Single_Lines")
    
    if not os.path.exists(full_output_dir): os.makedirs(full_output_dir)
    if not os.path.exists(single_output_dir): os.makedirs(single_output_dir)

    # Select Engine
    if USE_BOTOK:
        segmenter = BotokSegmenter(min_syllables=MIN_SYLLABLES)
        desc_text = "Botok Processing"
    else:
        segmenter = RegexSegmenter(min_syllables=MIN_SYLLABLES)
        desc_text = "Regex Processing"
    
    # Add overlapping mode indicator
    if USE_OVERLAPPING_SEGMENTS:
        desc_text += " (Overlapping Mode)"
    else:
        desc_text += " (Exclusive Mode)"

    file_groups = {} # To hold aggregated data for full files
    
    # Initialize converter - REQUIRED
    try:
        tib_converter = Converter()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Converter: {e}")
    
    print(f"Reading from: {input_path}")
    
    total_lines = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line_num, line in tqdm(enumerate(infile, 1), total=total_lines, desc=desc_text):
            line = line.strip()
            if not line: continue
            
            try:
                record = json.loads(line)
                text_content = record.get('text') or record.get('content') or ""
                
                if not text_content: continue

                metadata = record.get("metadata", {}) or {}
                raw_filename = metadata.get("file_name", "Unknown_Source")
                file_path = metadata.get("file_path", "")
                title = metadata.get("title", "")
                
                clean_name = sanitize_filename(raw_filename)
                
                # Segment - get atoms from segmenter
                atoms = segmenter.segment_with_indices(text_content)
                
                # Apply overlapping transformation if enabled
                if USE_OVERLAPPING_SEGMENTS:
                    segments = make_overlapping_spans(
                        atoms, 
                        text_content,
                        max_atoms=OVERLAP_MAX_ATOMS,
                        min_chars=OVERLAP_MIN_CHARS,
                        max_chars=OVERLAP_MAX_CHARS,
                        max_spans=MAX_SPANS_PER_LINE
                    )
                else:
                    # Exclusive mode: use atoms as-is
                    segments = [(sent, start, end) for sent, start, end in atoms]
                
                if segments:
                    # Collect rows for this specific line
                    single_line_rows = []
                    
                    for idx, segment_data in enumerate(segments, 1):
                        if USE_OVERLAPPING_SEGMENTS:
                            # Overlapping mode: (span_text, span_start, span_end, span_num_atoms, span_type)
                            sent, start, end, num_atoms, span_type = segment_data
                        else:
                            # Exclusive mode: (sent, start, end)
                            sent, start, end = segment_data
                            num_atoms = None
                            span_type = None
                        
                        # Segments are already Tibetan Unicode; bypass detection for speed
                        ewts_text = tib_converter.convert(
                            sent, "EWTS", text_scheme="Unicode", val_text_scheme=True
                        )
                        ewts_text = f" {ewts_text}"  # prepend single space to every EWTS cell
                        
                        row = {
                            "Segmented_Text": sent,
                            "Segmented_Text_EWTS": ewts_text,
                            "Length": len(sent), # New Length Column
                            "File_Path": file_path,
                            "Title": title,
                            "Source_Line_Number": line_num,
                            "Sentence_Order": idx,
                            "Start_Index": start,
                            "End_Index": end
                        }
                        
                        # Add overlapping-specific columns if in overlapping mode
                        if USE_OVERLAPPING_SEGMENTS:
                            row["Span_Num_Atoms"] = num_atoms
                            row["Span_Type"] = span_type
                        
                        # Add to Single List
                        single_line_rows.append(row)
                        
                        # Add to Full List (Grouped by Filename)
                        if clean_name not in file_groups:
                            file_groups[clean_name] = []
                        file_groups[clean_name].append(row)
                    
                    # SAVE INDIVIDUAL LINE CSV IMMEDIATELY
                    single_df = pd.DataFrame(single_line_rows)
                    single_filename = f"Line_{line_num}_{clean_name[:30]}.csv" # Truncate name to avoid OS errors
                    single_save_path = os.path.join(single_output_dir, single_filename)
                    single_df.to_csv(single_save_path, index=False)

            except json.JSONDecodeError:
                continue

    # SAVE AGGREGATED FILES
    print(f"\nSaving {len(file_groups)} Full Files to {full_output_dir}...")
    
    for filename, rows in tqdm(file_groups.items(), desc="Saving Full CSV"):
        if not rows: continue
        df = pd.DataFrame(rows)
        save_path = os.path.join(full_output_dir, f"{filename}.csv")
        try:
            df.to_csv(save_path, index=False)
        except Exception as e:
            print(f"Error saving {filename}: {e}")

    print(f"\n✓ Completed using {desc_text}.")
    print(f"Full files saved in: {full_output_dir}")
    print(f"Individual line files saved in: {single_output_dir}")

if __name__ == "__main__":
    if os.path.exists(INPUT_FILE):
        process_file(INPUT_FILE, OUTPUT_DIR)
    else:
        print("Input file not found.")