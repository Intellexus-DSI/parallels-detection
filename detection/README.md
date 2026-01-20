# Stage 3: Detection

Finds parallel text segments across the corpus using semantic similarity.

## Input

Embeddings from stage 2 in `embedding/output/embeddings_by_line/`:
- `embeddings_metadata.json`
- `line_NNNNNN_embeddings.npy` / `line_NNNNNN_segments.xlsx` pairs

## Output

`output/parallels.csv` with columns:
- `segment_a_id`, `segment_b_id`: Segment indices
- `similarity`: Cosine similarity score (0.0-1.0)
- `file_path_a`, `file_path_b`: Source identifiers
- `title_a`, `title_b`: Document titles
- `parallel_a`, `parallel_b`: The matched parallel text segments (Wylie transliteration)

## Usage

### Via Pipeline (from project root)

The `run_pipeline.py` and `pipeline_config.yaml` files are located in the **project root directory** (not in this folder).

```bash
cd /path/to/parallels-detection
python run_pipeline.py --config pipeline_config.yaml --stage detection
```

### Standalone (from this directory)

```bash
cd detection
python -m src.cli --data-dir ../embedding/output/embeddings_by_line --threshold 0.85
```

#### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | (required) | Directory with embeddings (contains `embeddings_metadata.json`) |
| `--output` | `output/parallels.csv` | Output file path |
| `--threshold` | `0.85` | Similarity threshold (0.0-1.0) |
| `--batch-size` | `5` | Number of files to load per batch |
| `--strategy` | `threshold` | Matching strategy: `threshold` or `knn` |
| `--k` | `10` | Number of neighbors for KNN strategy |
| `--format` | `csv` | Output format: `csv`, `parquet`, or `json` |
| `--no-normalize` | | Skip embedding normalization |
| `--no-text` | | Exclude segment text from output |
| `-v, --verbose` | | Enable verbose logging |

#### Example with options

```bash
python -m src.cli \
    --data-dir ../embedding/output/embeddings_by_line \
    --output output/parallels.csv \
    --threshold 0.80 \
    --batch-size 10 \
    --strategy threshold \
    --format csv \
    --verbose
```

## Configuration (in ../pipeline_config.yaml)

```yaml
detection:
  data_dir: "embedding/output/embeddings_by_line"
  output_path: "detection/output/parallels.csv"
  threshold: 0.85
  batch_size: 5
  strategy: "threshold"       # "threshold" or "knn"
  format: "csv"
```

## Dependencies

```bash
pip install faiss-cpu pandas numpy tqdm
```
