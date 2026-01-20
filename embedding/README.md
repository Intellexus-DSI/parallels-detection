# Stage 2: Embedding

Generates vector embeddings from segmented text for semantic similarity search.

## Input

Excel files from segmentation stage in `segmentation/output/overlapping/Full_Files/`

## Output

In `embedding/output/embeddings_by_line/`:
- `line_NNNNNN_embeddings.npy`: Vector embeddings (N x 768)
- `line_NNNNNN_segments.xlsx`: Segment metadata
- `embeddings_metadata.json`: File listing

## Usage

```bash
# Via pipeline
python run_pipeline.py --config pipeline_config.yaml --stage embedding

# Standalone
cd embedding
python generate_embeddings.py config.yaml
```

## Configuration (in pipeline_config.yaml)

```yaml
embedding:
  input_dir: "segmentation/output/overlapping/Full_Files"
  output_dir: "embedding/output"
  model: "Intellexus/Bi-Tib-mbert-v1"
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
# Then accept model terms at: https://huggingface.co/Intellexus/Bi-Tib-mbert-v1
```
