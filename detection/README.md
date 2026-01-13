# Detection Pipeline

Finds parallel text segments across corpus files using semantic similarity.

## Input

Directory with per-file embeddings:
```
data/embeddings_by_line/
├── embeddings_metadata.json
├── line_000114_segments.xlsx
├── line_000114_embeddings.npy
├── line_000115_segments.xlsx
├── line_000115_embeddings.npy
└── ...
```

Each file pair must have aligned rows (segment N in xlsx corresponds to embedding N in npy).

## Usage

```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run pipeline
python -m src.cli --data-dir data/embeddings_by_line --threshold 0.85
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | required | Directory with per-file embeddings |
| `--output` | `output/parallels.csv` | Output file path |
| `--threshold` | `0.85` | Minimum similarity (0.0-1.0) |
| `--batch-size` | `5` | Files to load per batch |
| `--strategy` | `threshold` | `threshold` or `knn` |
| `--k` | `10` | Neighbors for KNN strategy |
| `--format` | `csv` | Output format: `csv`, `parquet`, `json` |
| `--no-text` | false | Exclude segment text from output |
| `-v` | false | Verbose logging |

## Output

CSV with columns:
- `segment_a_id`, `segment_b_id` - Segment indices within their files
- `similarity` - Cosine similarity score
- `file_path_a`, `file_path_b` - Source file identifiers
- `title_a`, `title_b` - Document titles
- `ewts_a`, `ewts_b` - Segment text (EWTS transliteration)
