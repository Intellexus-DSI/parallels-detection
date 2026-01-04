# Parallels Pipeline - Quick Start Guide

## What This Does

Finds similar text segments (parallels) across your Tibetan/Sanskrit corpus using semantic embeddings and FAISS vector search.

## Prerequisites

You need two files:
1. **Segments file** (CSV or XLSX) - Your segmented text with metadata
2. **Embeddings file** (NPY) - Embedding vectors for each segment

**Important:** The row order must match between both files!
- Row 0 in segments → embeddings[0]
- Row 1 in segments → embeddings[1]
- etc.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Basic Usage

### 1. Prepare Your Config File

Edit `config.yaml`:

```yaml
# Input paths (supports .csv or .xlsx)
segments_csv: "data/full_segmentation.xlsx"
embeddings_path: "data/embeddings.npy"

# Output path
output_path: "output/parallels.csv"

# Matching configuration
matching:
  strategy: "threshold"    # Find all pairs above threshold
  threshold: 0.85          # Similarity threshold (0.0 to 1.0)

# Processing options
processing:
  batch_size: 1000
  normalize_embeddings: true

# Output options
output:
  format: "csv"            # Options: "csv", "parquet", "json"
  include_text: true
```

### 2. Run the Pipeline

```bash
python -m parallels.cli --config config.yaml
```

### 3. Check the Output

Results will be in `output/parallels.csv`:

| Column | Description |
|--------|-------------|
| segment_a_id | Row index of first segment |
| segment_b_id | Row index of second segment |
| similarity | Similarity score (0.85 to 1.0) |
| title_a, title_b | Document titles |
| ewts_a, ewts_b | Segment text (EWTS) |
| file_path_a, file_path_b | Source files |

## Required Columns in Your Segments File

Your CSV/XLSX must have these columns:
- `Segmented_Text` - Original text
- `Segmented_Text_EWTS` - EWTS transliteration
- `Length` - Character length
- `File_Path` - Source file (used to filter cross-text matches)
- `Title` - Document title
- `Source_Line_Number` - Line number in source
- `Sentence_Order` - Order within line
- `Start_Index` - Start character index
- `End_Index` - End character index

## Troubleshooting

### Row Mismatch Error

```
ValueError: Mismatch: CSV has 260124 rows, embeddings has 500 vectors
```

**Solution:** Your embeddings file doesn't match your segments file. They must have the same number of rows.

### Missing Columns Error

```
ValueError: Missing required columns in XLSX: {'File_Path', 'Title'}
```

**Solution:** Add the missing columns to your segments file.

### No Matches Found

```
Found 0 parallel matches
```

**Possible causes:**
- Threshold too high (try lowering to 0.75)
- All segments from same text (pipeline only finds cross-text matches)
- Embeddings are random/not meaningful

## Advanced Options

### Use KNN Strategy

Instead of threshold, find top-K matches per segment:

```yaml
matching:
  strategy: "knn"
  k: 10                    # Find 10 best matches per segment
  min_threshold: 0.7       # Optional: filter by minimum quality
```

### Command Line Overrides

```bash
# Override threshold
python -m parallels.cli --config config.yaml --threshold 0.9

# Use different output format
python -m parallels.cli --config config.yaml --format parquet

# Run without config file
python -m parallels.cli \
    --segments data/full_segmentation.xlsx \
    --embeddings data/embeddings.npy \
    --output output/parallels.csv \
    --threshold 0.85
```

## Performance

Expected runtime for 260K segments:
- **Index build:** ~5-10 seconds
- **Search:** ~10-30 minutes (depends on threshold/k)
- **Memory:** ~3-5 GB

Lower thresholds = more matches = longer runtime.

## Next Steps

1. Review the output CSV to verify quality
2. Adjust threshold based on results
3. Try different strategies (threshold vs knn)
4. Filter results further if needed
