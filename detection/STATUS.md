# Parallels Pipeline - Implementation Status

## ✅ Implementation Complete

The parallels pipeline is **fully implemented and ready to use**.

## What's Included

### Core Pipeline
- ✅ FAISS-based similarity search (cosine similarity)
- ✅ CSV and XLSX input support
- ✅ NPY embeddings loading
- ✅ Cross-text filtering (only finds matches across different texts)
- ✅ Deduplication (each pair stored once)
- ✅ Batch processing for memory efficiency
- ✅ Progress tracking with tqdm

### Matching Strategies
- ✅ **Threshold Matcher**: Find all pairs above similarity threshold
- ✅ **KNN Matcher**: Find top-K matches per segment

### Configuration
- ✅ YAML-based configuration
- ✅ Pydantic validation
- ✅ Command-line overrides
- ✅ Multiple output formats (CSV, Parquet, JSON)

### Output
- ✅ Configurable output format
- ✅ Optional text inclusion
- ✅ Enriched metadata (titles, file paths, EWTS text)

## How to Use

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

You need:
- **Segments file**: `full_segmentation.xlsx` (or CSV)
- **Embeddings file**: `embeddings.npy` with shape `(N, 768)`

**Critical:** Both files must have the same number of rows (N) in the same order.

### 3. Configure

Edit `config.yaml`:

```yaml
segments_csv: "data/full_segmentation.xlsx"
embeddings_path: "data/embeddings.npy"
output_path: "output/parallels.csv"

matching:
  strategy: "threshold"
  threshold: 0.85

processing:
  batch_size: 1000
  normalize_embeddings: true

output:
  format: "csv"
  include_text: true
```

### 4. Run

```bash
python -m parallels.cli --config config.yaml
```

## Current Data Status

The repository contains:
- ✅ `data/full_segmentation.xlsx` - **260,124 segments** (your real data)
- ⚠️ `data/embeddings.npy` - **500 test embeddings** (placeholder)

### ⚠️ Action Required

You need to generate embeddings for all 260,124 segments:

```python
# Pseudo-code for generating embeddings
import numpy as np
import pandas as pd

# Load your segments
df = pd.read_excel('data/full_segmentation.xlsx')

# Generate embeddings (use your embedding model)
embeddings = []
for text in df['Segmented_Text_EWTS']:
    embedding = your_embedding_model.encode(text)  # Your model here
    embeddings.append(embedding)

# Save as NPY
embeddings = np.array(embeddings, dtype=np.float32)
np.save('data/embeddings.npy', embeddings)

# Verify shape
print(f"Saved embeddings: {embeddings.shape}")  # Should be (260124, 768)
```

## Testing

Once you have the correct embeddings file:

```bash
# Run with default threshold (0.85)
python -m parallels.cli --config config.yaml

# Try lower threshold for more matches
python -m parallels.cli --config config.yaml --threshold 0.75

# Use KNN strategy
python -m parallels.cli --config config.yaml --strategy knn --k 10
```

## Expected Output

The pipeline will create `output/parallels.csv` with columns:
- `segment_a_id`, `segment_b_id` - Row indices
- `similarity` - Cosine similarity score
- `title_a`, `title_b` - Document titles
- `ewts_a`, `ewts_b` - Segment text (EWTS)
- `file_path_a`, `file_path_b` - Source files

## Performance Estimates

For 260K segments:
- **Index building**: ~5-10 seconds
- **Search time**: ~15-30 minutes (threshold=0.85)
- **Memory usage**: ~4-6 GB
- **Output size**: Varies (could be 100K-1M+ matches depending on threshold)

## Files Changed

### Modified Files
1. `parallels/data/segment_store.py` - Added XLSX support
2. `parallels/cli.py` - Updated help text for XLSX
3. `config.yaml` - Updated to use XLSX file
4. `requirements.txt` - Added openpyxl
5. `pyproject.toml` - Added openpyxl dependency
6. `README.md` - Updated documentation

### New Files
1. `QUICKSTART.md` - Quick start guide
2. `STATUS.md` - This file

## Architecture Summary

```
Input: full_segmentation.xlsx + embeddings.npy
  ↓
SegmentStore (loads & validates)
  ↓
FAISSIndex (builds index)
  ↓
Matcher (threshold or knn)
  ↓  ← CrossTextFilter (same file)
  ↓  ← Deduplicator (A,B = B,A)
  ↓
OutputWriter
  ↓
Output: parallels.csv
```

## Troubleshooting

### "Mismatch: CSV has X rows, embeddings has Y vectors"
- Generate embeddings for all segments
- Ensure both files have same row count

### "Found 0 parallel matches"
- Lower the threshold (try 0.7 or 0.75)
- Check embeddings are meaningful (not random)
- Verify you have segments from multiple texts

### Memory issues
- Reduce batch_size in config
- Use fewer segments for testing

## Next Steps

1. ✅ Generate embeddings for all 260K segments
2. ✅ Run pipeline with your embeddings
3. ✅ Tune threshold based on results
4. ✅ Export results for analysis

## Support

See:
- `README.md` - Full documentation
- `QUICKSTART.md` - Quick start guide
- `config.yaml` - Configuration example
