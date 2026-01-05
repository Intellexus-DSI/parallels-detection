# Detection Stage

The third and final stage in the parallel text detection pipeline. Finds semantically similar text segments across different documents using FAISS similarity search.

## Purpose

Takes embeddings from the Indexing stage and identifies "parallel" segments - pieces of text from different sources that share similar meaning. Uses efficient vector similarity search to find these matches at scale.

## Input

### 1. Segments File (CSV or XLSX)
From Indexing stage, containing segment metadata:

| Column | Description |
|--------|-------------|
| `Segmented_Text` | Original Tibetan/Sanskrit text |
| `Segmented_Text_EWTS` | EWTS transliteration |
| `Length` | Character length of segment |
| `File_Path` | Source file path (used as text identifier) |
| `Title` | Document title |
| `Source_Line_Number` | Line number in source |
| `Sentence_Order` | Order within the line |
| `Start_Index` | Start character index |
| `End_Index` | End character index |

### 2. Embeddings File
From Indexing stage:
- **Format**: NumPy `.npy` file
- **Shape**: `(N, embedding_dim)` where N = number of segments
- **Type**: `float32`
- **Order**: Row order must match the segments file

## Output

CSV/Parquet/JSON file containing parallel matches:

| Column | Description |
|--------|-------------|
| `segment_a_id` | Row index of first segment |
| `segment_b_id` | Row index of second segment |
| `similarity` | Cosine similarity score (0.0 to 1.0) |
| `title_a` | Title of text A |
| `title_b` | Title of text B |
| `ewts_a` | EWTS text of segment A |
| `ewts_b` | EWTS text of segment B |
| `file_path_a` | Source file of segment A |
| `file_path_b` | Source file of segment B |

## Architecture

```
parallels/
├── config.py              # Configuration management (Pydantic)
├── models.py              # Data classes (Segment, ParallelMatch)
├── pipeline.py            # Main orchestrator
├── cli.py                 # Command-line interface
├── data/
│   ├── segment_store.py   # Load segments + embeddings
│   └── output_writer.py   # Write results
├── index/
│   └── faiss_index.py     # FAISS similarity search
└── matching/
    ├── base.py            # Abstract matcher strategy
    ├── threshold_matcher.py   # Threshold-based matching
    ├── knn_matcher.py     # K-nearest neighbors
    └── filters.py         # Cross-text filter, deduplication
```

## Configuration

Create a `config.yaml` file:

```yaml
# Input paths (supports .csv or .xlsx)
segments_csv: "data/full_segmentation.xlsx"
embeddings_path: "data/embeddings.npy"

# Output path
output_path: "output/parallels.csv"

# Matching configuration
matching:
  strategy: "threshold"    # Options: "threshold" or "knn"
  threshold: 0.85          # For threshold strategy (0.0 to 1.0)
  k: 10                    # For knn strategy
  min_threshold: 0.0       # Optional minimum threshold for knn

# Processing options
processing:
  batch_size: 1000         # Segments per batch
  normalize_embeddings: true

# Output options
output:
  format: "csv"            # Options: "csv", "parquet", "json"
  include_text: true       # Include segment text in output
```

## Usage

### Command Line

```bash
# Using config file
python -m parallels.cli --config config.yaml

# Override specific options
python -m parallels.cli --config config.yaml --threshold 0.9

# Direct arguments (no config file)
python -m parallels.cli \
    --segments data/full_segmentation.xlsx \
    --embeddings data/embeddings.npy \
    --output output/parallels.csv \
    --strategy threshold \
    --threshold 0.85
```

### Python API

```python
from parallels.config import Config
from parallels.pipeline import ParallelsPipeline

# Load configuration
config = Config.from_yaml("config.yaml")

# Run pipeline
pipeline = ParallelsPipeline(config)
match_count = pipeline.run()
print(f"Found {match_count} parallel matches")
```

## Matching Strategies

### Threshold Matcher (Default)
Finds all segment pairs with similarity above a threshold.

- **Use when**: You want all matches above a quality threshold
- **Pros**: Complete coverage, quality controlled
- **Cons**: May produce many results for low thresholds
- **Typical threshold**: 0.85 for high-quality matches

### KNN Matcher
Finds the top-k most similar segments for each segment.

- **Use when**: You want a fixed number of matches per segment
- **Pros**: Predictable output size, finds best matches
- **Cons**: May include low-quality matches if k is large
- **Typical k**: 5-10 neighbors per segment

## Features

- **Efficient Search**: FAISS IndexFlatIP for exact cosine similarity
- **Cross-Text Only**: Automatically filters matches from the same source
- **Deduplication**: Each pair stored once as (min_id, max_id)
- **Batch Processing**: Memory-efficient processing for large datasets
- **Multiple Formats**: Output to CSV, Parquet, or JSON

## Performance

For 200K segments with 768-dimensional embeddings:

| Metric | Value |
|--------|-------|
| Memory usage | ~2-3 GB |
| Index build time | ~2-3 seconds |
| Search time | ~5-10 minutes |
| Output size | Varies (threshold-dependent) |

## Algorithm

### Cosine Similarity via FAISS

1. L2-normalize all embedding vectors
2. Build FAISS `IndexFlatIP` with normalized vectors
3. Inner product of normalized vectors = cosine similarity
4. Search for matches using threshold or KNN strategy

### Filtering

1. **Cross-text filter**: Remove matches within same `File_Path`
2. **Deduplication**: Store pairs as `(min(a,b), max(a,b))`
3. **Threshold filter**: Apply minimum similarity threshold

## Dependencies

```bash
pip install faiss-cpu numpy pandas pydantic pyyaml tqdm openpyxl
```

## Previous Stage

Requires output from the **Indexing** stage:
- Embeddings file (`.npy`)
- Segments file (CSV/XLSX)

## Notes

- Cosine similarity ranges from 0.0 (unrelated) to 1.0 (identical)
- Higher thresholds (0.90+) find very similar segments
- Lower thresholds (0.70-0.85) find broader parallels
- Cross-text filtering ensures matches are between different documents
- Results can be large - consider using Parquet for compression
