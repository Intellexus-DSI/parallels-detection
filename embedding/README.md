# Embedding Stage

The second stage in the parallel text detection pipeline. Generates vector embeddings for segmented Tibetan text to enable semantic similarity search.

## Quick Start

### Installation

```bash
cd embedding
pip install -e .
```

### Configuration

Edit `config.yaml`:

```yaml
input:
  segments_dir: "../data/05_clean_data/00_tibetan/segmented_output/overlapping/Full_Files"
  text_column: "Segmented_Text_EWTS"

embedding:
  model_name: "Intellexus/Bi-Tib-mbert-v1"  # Tibetan-specific model
  batch_size: 32
  device: "auto"  # Will use GPU if available
  normalize: true

output:
  output_dir: "../detection/data"
  mode: "per_line"  # "per_line", "per_file", or "combined"
  per_line_subdir: "embeddings_by_line"
  per_file_subdir: "embeddings_by_source"
```

### Usage

```bash
# Using config file
embedding --config config.yaml

# With overrides
embedding --config config.yaml --batch-size 64 --device cpu

# Test with limited segments
embedding --max-segments 100
```

## Output Modes

### Per-Line Mode
Creates separate embedding files for each `Source_Line_Number`:
```
detection/data/embeddings_by_line/
├── line_000114_embeddings.npy
├── line_000114_segments.xlsx
├── line_000115_embeddings.npy
├── line_000115_segments.xlsx
├── ...
└── embeddings_metadata.json
```

**Benefits:** Maximum granularity, track embeddings by source line, efficient updates

### Per-File Mode
Creates separate embedding files for each source file:
```
detection/data/embeddings_by_source/
├── I1PD97353_embeddings.npy
├── I1PD97353_segments.xlsx
├── source2_embeddings.npy
├── source2_segments.xlsx
└── embeddings_metadata.json
```

**Benefits:** Better organization, incremental updates, memory efficient

### Combined Mode
Set `mode: "combined"` to create single files:
```
detection/data/
├── embeddings.npy              # ALL embeddings (N, 768)
├── full_segmentation.xlsx      # ALL segments
└── embeddings_metadata.json
```

## Python API

```python
from embedding import EmbeddingPipeline, load_config

# Simple usage
from embedding import run_pipeline
embeddings, segments = run_pipeline("config.yaml")

# Advanced usage
config = load_config("config.yaml")
config.output.mode = "per_file"  # Override settings
pipeline = EmbeddingPipeline(config)
embeddings, segments = pipeline.run()
```

## Model Information

**Bi-Tib-mbert-v1** by Intellexus:
- Tibetan-specific sentence transformer
- 768-dimensional embeddings
- 0.854 Spearman correlation on Tibetan similarity tasks
- Requires HuggingFace account and model terms acceptance

**First run setup:**
1. Visit: https://huggingface.co/Intellexus/Bi-Tib-mbert-v1
2. Accept model terms
3. Authenticate: `huggingface-cli login`

## Performance

### GPU Acceleration
```bash
# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Auto-detects GPU
embedding --config config.yaml
```

### Large Datasets
- GPU: Increase batch size (`--batch-size 128`)
- CPU: Reduce batch size (`--batch-size 16`)
- Use `--skip-existing` to avoid reprocessing

### Expected Performance (1M segments)
- GPU (RTX 3090): 20-30 minutes
- CPU: 2-3 hours

## CLI Options

```bash
embedding --help

Options:
  --config, -c          Path to config YAML file
  --input-dir, -i       Input directory (overrides config)
  --output-dir, -o      Output directory (overrides config)
  --model, -m           Model name (overrides config)
  --batch-size, -b      Batch size (overrides config)
  --device              Device: auto, cuda, or cpu
  --max-segments        Limit segments for testing
  --mode                Output mode: combined, per_file, or per_line
  --skip-existing       Skip if outputs exist
  --quiet, -q           Suppress progress bars
```

## Architecture

```
embedding/
├── config.yaml              # Configuration file
├── pyproject.toml          # Package metadata
├── requirements.txt        # Dependencies
├── example_usage.py        # Usage examples
├── generate_embeddings.py  # Standalone script
├── embedding/
│   ├── __init__.py        # Package exports
│   ├── models.py          # Pydantic data models
│   ├── config.py          # Configuration loader
│   ├── pipeline.py        # Main pipeline implementation
│   └── cli.py             # CLI interface
└── tests/
    └── test_pipeline.py   # Unit tests
```

## Troubleshooting

### Model Download Fails
```
Error: You need to accept the model conditions
```
→ Visit HuggingFace model page and accept terms, then: `huggingface-cli login`

### Out of Memory
```
RuntimeError: CUDA out of memory
```
→ Reduce batch size: `--batch-size 16` or use CPU: `--device cpu`

### Column Not Found
```
ValueError: Text column not found
```
→ Check Excel files have correct column names. Update `text_column` in config.

### No Excel Files Found
```
ValueError: No Excel files found
```
→ Verify segmentation stage completed and path is correct

## Integration with Pipeline

```
Stage 1: Segmentation → Excel files
          ↓
Stage 2: Embedding → embeddings + segments  ← THIS STAGE
          ↓
Stage 3: Detection → parallels.csv
```

## Dependencies

- `sentence-transformers>=2.2.0` - Embedding generation
- `torch>=2.0.0` - Deep learning framework
- `numpy>=1.21.0` - Array operations
- `pandas>=1.3.0` - Data manipulation
- `pydantic>=2.0.0` - Configuration validation

See `requirements.txt` for complete list.

## Next Steps

After generating embeddings:

```bash
# Verify outputs
ls -lh ../detection/data/

# Proceed to detection stage
cd ../detection
python -m parallels.cli --config config.yaml
```
