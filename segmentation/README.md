# Segmentation Stage

The first stage in the parallel text detection pipeline. Segments Tibetan text files into individual sentences/phrases for further processing.

## Purpose

Takes raw Tibetan text documents and breaks them into meaningful segments using linguistic rules and patterns. These segments become the atomic units for embedding generation and parallel detection.

## Input

- **Format**: JSONL files with Tibetan text
- **Source**: Azure File Share or local files
- **Structure**: Each line contains a JSON object with `text` or `content` field and optional `metadata`

Example input:
```json
{"text": "བོད་ཀྱི་ལོ་རྒྱུས་ནི་རྒྱལ་ཁབ་འདི་རིང་ལུགས་དང་།", "metadata": {"file_name": "example.txt", "title": "History"}}
```

## Output

Excel files (.xlsx) containing segmented text with metadata:
- `Segmented_Text`: Tibetan Unicode text segment
- `Segmented_Text_EWTS`: EWTS transliteration
- `Length`: Character count
- `File_Path`: Source document identifier
- `Title`: Document title
- `Source_Line_Number`: Line number in input file
- `Sentence_Order`: Order within the line
- `Start_Index`: Character position in original text
- `End_Index`: Character end position

## Scripts

### `download_to_disk_tibet.py`
Downloads files from Azure File Share to local disk.

**Configuration** (edit in script):
```python
ACCOUNT_URL = "https://intlxresearchstorage.file.core.windows.net"
SAS_TOKEN = ""  # Add your SAS token
SHARE_NAME = "intlx-gpu-fs"
AZURE_FILE_PATH = "data/05_clean_data/00_tibetan/Tibetan_1.jsonl"
LOCAL_OUTPUT_DIR = "data/05_clean_data/00_tibetan"
```

**Usage**:
```bash
python download_to_disk_tibet.py
```

### `tibet_segmentation.py`
Main segmentation script with two engines and two modes.

**Configuration** (edit in script):
```python
INPUT_FILE = "data/05_clean_data/00_tibetan/Tibetan_1.jsonl"
OUTPUT_DIR = "data/05_clean_data/00_tibetan/segmented_output"

# Engine Selection
USE_BOTOK = False  # True = Botok (accurate), False = Regex (fast)

# Segmentation Mode
USE_OVERLAPPING_SEGMENTS = True  # True = overlapping, False = exclusive
```

**Usage**:
```bash
python tibet_segmentation.py
```

## Segmentation Engines

### Regex Engine (Default - Fast)
- Pattern-based segmentation using Tibetan punctuation marks (shad ༎ and double-shad །)
- Linguistic rules for sentence boundaries (terminators, continuators)
- Minimum syllable threshold
- **Speed**: ~1000 lines/second
- **Best for**: Large datasets, production use

### Botok Engine (Accurate)
- Uses Botok tokenizer for word-level analysis
- More sophisticated linguistic understanding
- Better handling of edge cases
- **Speed**: ~100 lines/second
- **Best for**: High-quality segmentation, smaller datasets

## Segmentation Modes

### Exclusive Mode (Traditional)
- Each segment is independent, non-overlapping
- One segment = one sentence/phrase
- Clear boundaries between segments
- **Output**: ~50-200 segments per document

### Overlapping Mode (Multi-Scale)
- Generates multiple overlapping spans at different scales
- Captures text at 1-8 atom windows
- Both forward sliding windows and centered patterns
- **Output**: ~100-300 segments per document
- **Purpose**: Better parallel detection by capturing text at multiple granularities

## Output Structure

```
data/segmented_output/
├── overlapping/              # or 'exclusive' depending on mode
│   ├── Full_Files/
│   │   ├── Document1.xlsx    # All segments from Document1
│   │   ├── Document2.xlsx
│   │   └── ...
│   └── Single_Lines/
│       ├── Line_1_Document1.xlsx  # Segments from line 1
│       ├── Line_2_Document1.xlsx
│       └── ...
```

## Dependencies

```bash
pip install pandas openpyxl tqdm pyyaml
pip install azure-storage-file-share azure-core  # For Azure download
pip install botok  # For Botok engine (optional)
```

## Next Stage

Output Excel files are consumed by the **Indexing** stage to generate embeddings.

## Notes

- EWTS conversion requires the `conversion` library (detect_and_convert package)
- Segments are filtered to exclude English-only text
- Minimum syllable threshold prevents very short segments
- Progress bars show real-time processing status
