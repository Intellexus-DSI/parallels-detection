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
- `ewts_a`, `ewts_b`: Segment text

## Usage

```bash
# Via pipeline
python run_pipeline.py --config pipeline_config.yaml --stage detection

# Standalone
cd detection
python -m src.cli --data-dir ../embedding/output/embeddings_by_line --threshold 0.85
```

## Configuration (in pipeline_config.yaml)

```yaml
detection:
  data_dir: "embedding/output/embeddings_by_line"
  output_path: "output/parallels.csv"
  threshold: 0.85
  batch_size: 5
  strategy: "threshold"       # "threshold" or "knn"
```

## Dependencies

```bash
pip install faiss-cpu pandas numpy tqdm
```
