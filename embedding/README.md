# Stage 2: Embedding

Generates vector embeddings from segmented text for semantic similarity search.

## Input

CSV files from segmentation stage in `segmentation/output/overlapping/Full_Files/`

## Output

In `embedding/output/embeddings_by_line/`:
- `line_NNNNNN_embeddings.npy`: Vector embeddings (N x 768)
- `line_NNNNNN_segments.csv`: Segment metadata
- `embeddings_metadata.json`: File listing

## Usage

### Via Pipeline (from project root)

The `run_pipeline.py` and `pipeline_config.yaml` files are located in the **project root directory** (not in this folder).

```bash
cd /path/to/parallels-detection
python run_pipeline.py --config pipeline_config.yaml --stage embedding
```

### Standalone (from this directory)

```bash
cd embedding
python -m embedding.cli --input-dir ../segmentation/output/overlapping/Full_Files --output-dir output
```

#### CLI Options

| Option | Description |
|--------|-------------|
| `--input-dir, -i` | Directory containing segmented CSV files (required) |
| `--output-dir, -o` | Output directory for embeddings |
| `--model, -m` | HuggingFace model name |
| `--batch-size, -b` | Batch size for embedding generation (default: 32) |
| `--device` | `auto`, `cuda`, or `cpu` |
| `--mode` | `per_line`, `per_file`, or `combined` |
| `--no-normalize` | Do not normalize embeddings |
| `--skip-existing` | Skip if output files already exist |
| `-q, --quiet` | Suppress progress bars |

#### Example with options

```bash
python -m embedding.cli \
    --input-dir ../segmentation/output/overlapping/Full_Files \
    --output-dir output \
    --model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \
    --batch-size 64 \
    --device auto \
    --mode per_line
```

## Configuration (in ../pipeline_config.yaml)

```yaml
embedding:
  input_dir: "segmentation/output/overlapping/Full_Files"
  output_dir: "embedding/output"
  model: "Intellexus/Bi-Tib-mbert-v2"
  batch_size: 32
  device: "auto"              # "cuda", "cpu", or "auto"
  mode: "per_line"            # "per_line", "per_file", or "combined"
```

## Dependencies

```bash
pip install sentence-transformers torch pandas numpy
```

## Model Setup

First-time setup requires HuggingFace authentication:
```bash
huggingface-cli login
# Then accept model terms at: https://huggingface.co/Intellexus/Bi-Tib-mbert-v2
```
