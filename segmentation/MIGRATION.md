# Migration Guide

This document shows how the original scripts have been refactored into the new modular structure.

## Original Scripts → New Structure

### download_to_disk_tibet.py → segmenter/data/azure_downloader.py

**Before:**
```python
# Monolithic script with hardcoded configuration
ACCOUNT_URL = "https://..."
SAS_TOKEN = ""
SHARE_NAME = "intlx-gpu-fs"

def download_file_to_disk():
    # All logic in one function
    ...
```

**After:**
```python
# Modular class with dependency injection
class AzureDownloader:
    def __init__(self, account_url, sas_token, share_name):
        ...
    
    def download_file(self, azure_path, local_path):
        ...
```

**Usage:**
```bash
# Old way
python download_to_disk_tibet.py

# New way
segmenter download \
    --azure-path "data/file.jsonl" \
    --output data/local.jsonl \
    --sas-token "YOUR_TOKEN"
```

### tibet_segmentation.py → Multiple Modules

The large 472-line script has been broken down into:

1. **segmenter/engines/base.py** (Lines 33-46)
   - Constants: `TIBETAN_SHAD`, `TSHEG`, `TERMINATORS`, `CONTINUATORS`
   - Base class: `SegmentationEngine`

2. **segmenter/engines/botok_engine.py** (Lines 52-123)
   - Class: `BotokSegmenter`
   - High accuracy tokenizer-based segmentation

3. **segmenter/engines/regex_engine.py** (Lines 129-215)
   - Class: `RegexSegmenter`
   - Fast pattern-based segmentation

4. **segmenter/utils/overlapping.py** (Lines 221-319)
   - Function: `make_overlapping_spans()`
   - Multi-scale span generation logic

5. **segmenter/pipeline.py** (Lines 323-471)
   - Class: `SegmentationPipeline`
   - Main processing logic
   - EWTS conversion
   - File I/O and Excel generation

6. **segmenter/config.py**
   - Configuration management (was hardcoded in lines 8-31)
   - Pydantic models for validation

7. **segmenter/models.py**
   - Data models for segments and metadata

8. **segmenter/cli.py**
   - Command-line interface
   - Argument parsing

## Configuration Changes

**Before:**
```python
# Hardcoded at top of script
INPUT_FILE = "data/05_clean_data/00_tibetan/Tibetan_1.jsonl"
OUTPUT_DIR = "data/05_clean_data/00_tibetan/segmented_output"
USE_BOTOK = False
MIN_SYLLABLES = 4
USE_OVERLAPPING_SEGMENTS = True
OVERLAP_MAX_ATOMS = 8
```

**After:**
```yaml
# config.yaml - externalized and validated
input_file: "data/05_clean_data/00_tibetan/Tibetan_1.jsonl"

segmentation:
  engine: "regex"
  min_syllables: 4
  use_overlapping: true
  overlap_max_atoms: 8

output:
  output_dir: "data/segmented_output"
```

## Usage Comparison

### Original Scripts

```bash
# Edit constants in script files
# Run each script separately
python download_to_disk_tibet.py
python tibet_segmentation.py
```

### New Modular System

```bash
# Install as package
cd segmentation
pip install -e .

# Use CLI with config file
segmenter --config config.yaml

# Or use direct arguments
segmenter --input data/input.jsonl --engine regex --overlapping

# Download and segment in one workflow
segmenter download --azure-path "..." --output data/file.jsonl
segmenter --input data/file.jsonl --output results/
```

## Benefits of Refactoring

1. **Modularity**: Each component has a single responsibility
2. **Testability**: Individual components can be tested in isolation
3. **Configurability**: No need to edit code to change settings
4. **Reusability**: Components can be imported and used in other projects
5. **Maintainability**: Clear separation makes updates easier
6. **Type Safety**: Pydantic validates configuration
7. **Documentation**: Clear module structure and docstrings
8. **CLI Integration**: Professional command-line interface
9. **Consistency**: Matches the detection module patterns

## File Mapping

| Original File | New Location | Lines |
|--------------|--------------|-------|
| download_to_disk_tibet.py | segmenter/data/azure_downloader.py | 49 → 63 |
| tibet_segmentation.py (constants) | segmenter/engines/base.py | 14 → 25 |
| tibet_segmentation.py (BotokSegmenter) | segmenter/engines/botok_engine.py | 71 → 96 |
| tibet_segmentation.py (RegexSegmenter) | segmenter/engines/regex_engine.py | 86 → 138 |
| tibet_segmentation.py (overlapping) | segmenter/utils/overlapping.py | 98 → 126 |
| tibet_segmentation.py (process_file) | segmenter/pipeline.py | 150 → 278 |
| N/A (new) | segmenter/config.py | 75 |
| N/A (new) | segmenter/models.py | 42 |
| N/A (new) | segmenter/cli.py | 301 |

**Total Lines:**
- Original: 521 lines (2 files)
- Refactored: ~1,047 lines (9 files + docs)
- Added: Configuration, models, CLI, documentation
