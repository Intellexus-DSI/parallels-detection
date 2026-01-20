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

```bash
# Via pipeline
python run_pipeline.py --config pipeline_config.yaml --stage segmentation

# Standalone
cd segmentation
python tibet_segmentation.py
```

## Configuration (in pipeline_config.yaml)

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
