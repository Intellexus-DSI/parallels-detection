# Stage 1: Segmentation

Segments Tibetan text into phrases/sentences for embedding.

## Input

JSONL file with Tibetan text:
```json
{"text": "བོད་ཀྱི་ལོ་རྒྱུས...", "metadata": {"file_name": "doc.txt", "title": "History"}}
```

## Output

Excel files in `output/overlapping/Full_Files/` with columns:
- `Segmented_Text`: Tibetan Unicode segment
- `Segmented_Text_EWTS`: EWTS transliteration
- `File_Path`, `Title`, `Source_Line_Number`, `Start_Index`, `End_Index`

## Usage

### Via Pipeline (from project root)

The `run_pipeline.py` and `pipeline_config.yaml` files are located in the **project root directory** (not in this folder).

```bash
cd /path/to/parallels-detection
python run_pipeline.py --config pipeline_config.yaml --stage segmentation
```

### Standalone (from this directory)

```bash
cd segmentation
python -m segmenter.cli segment --input ../data/input.jsonl --output output
```

#### CLI Options

| Option | Description |
|--------|-------------|
| `--input` | Path to input JSONL file (required) |
| `--output` | Output directory for segmented files |
| `--engine` | `regex` (fast) or `botok` (accurate) |
| `--min-syllables` | Minimum syllables per segment (default: 4) |
| `--overlapping` | Use overlapping segmentation mode |
| `--max-atoms` | Max atoms per span in overlapping mode (default: 8) |
| `-v, --verbose` | Enable verbose logging |

#### Example with options

```bash
python -m segmenter.cli segment \
    --input ../data/input.jsonl \
    --output output \
    --engine regex \
    --overlapping \
    --max-atoms 8
```

## Configuration (in ../pipeline_config.yaml)

```yaml
segmentation:
  input_file: "data/input.jsonl"
  output_dir: "segmentation/output"
  engine: "regex"           # "regex" (fast) or "botok" (accurate)
  use_overlapping: true
```

## Dependencies

```bash
pip install pandas openpyxl tqdm botok
```
