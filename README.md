# Parallel Text Detection Pipeline

A three-stage pipeline for discovering semantically similar text segments across a corpus of Tibetan and Sanskrit texts.

## Overview

This pipeline processes raw text documents through three stages to identify "parallels" - segments from different texts that share similar meaning or content:

1. **Segmentation**: Break texts into meaningful segments
2. **Indexing**: Generate vector embeddings for each segment  
3. **Detection**: Find similar segments using semantic search

```
Raw Text Files → [Segmentation] → Segments → [Indexing] → Embeddings → [Detection] → Parallels
```

## Pipeline Stages

### 1. Segmentation Stage

**Purpose**: Convert raw text into discrete segments for analysis

**Input**: 
- JSONL files with Tibetan text from Azure or local storage
- Each line contains a JSON object with text and metadata

**Process**:
- Downloads files from Azure File Share (optional)
- Segments text using Botok (accurate) or Regex (fast) engine
- Supports exclusive or overlapping segmentation modes
- Converts to EWTS transliteration

**Output**:
- Excel files with segmented text and metadata
- Each segment has position info, text, EWTS, and source metadata

**Location**: `segmentation/`

[Read more →](segmentation/README.md)

### 2. Indexing Stage

**Purpose**: Generate vector embeddings for semantic comparison

**Input**:
- Excel files from Segmentation stage
- Contains segmented text and EWTS transliterations

**Process**:
- Loads all segments from Excel files
- Generates embeddings using transformer models
- Normalizes vectors for cosine similarity
- Consolidates segments and embeddings

**Output**:
- `embeddings.npy`: Numpy array of vectors (N × embedding_dim)
- `full_segmentation.xlsx`: All segments with metadata

**Location**: `indexing/`

[Read more →](indexing/README.md)

### 3. Detection Stage

**Purpose**: Find parallel segments across different texts

**Input**:
- `embeddings.npy`: Vector embeddings
- `full_segmentation.xlsx`: Segment metadata

**Process**:
- Builds FAISS index for efficient similarity search
- Applies threshold or KNN matching strategy
- Filters matches to cross-text only (different sources)
- Deduplicates results

**Output**:
- CSV/Parquet/JSON with parallel matches
- Each row: segment pair + similarity score + metadata

**Location**: `detection/`

[Read more →](detection/README.md)

## Quick Start

### Option 1: Run Full Pipeline

```bash
# Run all three stages
python run_pipeline.py --config pipeline_config.yaml
```

### Option 2: Run Stages Individually

```bash
# Stage 1: Segmentation
cd segmentation
python tibet_segmentation.py

# Stage 2: Indexing (to be implemented)
cd ../indexing
python generate_embeddings.py

# Stage 3: Detection
cd ../detection
python -m parallels.cli --config config.yaml
```

## Directory Structure

```
parallels-detection/
├── README.md                    # This file
├── run_pipeline.py             # Main orchestrator
├── pipeline_config.yaml        # Full pipeline configuration
│
├── segmentation/               # Stage 1: Text Segmentation
│   ├── README.md
│   ├── download_to_disk_tibet.py
│   ├── tibet_segmentation.py
│   └── segmenter/             # Modular package (optional)
│
├── indexing/                   # Stage 2: Embedding Generation
│   ├── README.md
│   ├── pyproject.toml
│   └── indexer/               # To be implemented
│
└── detection/                  # Stage 3: Parallel Detection
    ├── README.md
    ├── config.yaml
    ├── pyproject.toml
    └── parallels/             # Main package
        ├── cli.py
        ├── pipeline.py
        ├── data/
        ├── index/
        └── matching/
```

## Configuration

Each stage can be configured independently, or use a unified config:

### Full Pipeline Config (`pipeline_config.yaml`)

```yaml
# Stage 1: Segmentation
segmentation:
  input_file: "data/input.jsonl"
  output_dir: "data/segmented"
  engine: "regex"
  use_overlapping: true

# Stage 2: Indexing
indexing:
  input_dir: "data/segmented/overlapping/Full_Files"
  output_dir: "data/"
  model: "sentence-transformers/all-MiniLM-L6-v2"

# Stage 3: Detection
detection:
  segments_csv: "data/full_segmentation.xlsx"
  embeddings_path: "data/embeddings.npy"
  output_path: "output/parallels.csv"
  threshold: 0.85
```

## Installation

### Install All Stages

```bash
# Install segmentation dependencies
cd segmentation
pip install pandas openpyxl tqdm azure-storage-file-share botok

# Install indexing dependencies (to be finalized)
cd ../indexing
pip install -e .

# Install detection dependencies
cd ../detection
pip install -e .
```

### Or Install Individually

Each stage can be installed separately - see stage README files for details.

## Data Flow

### Example Workflow

**Input**: 1 JSONL file with 1000 lines of Tibetan text

**After Segmentation**:
- 1000 Excel files (one per line)
- ~50 aggregated Excel files (one per source document)
- ~150,000 total segments (with overlapping mode)

**After Indexing**:
- 1 numpy file: `embeddings.npy` (150,000 × 768)
- 1 consolidated file: `full_segmentation.xlsx` (150,000 rows)

**After Detection**:
- 1 results file: `parallels.csv` (~10,000-50,000 matches at 0.85 threshold)

## Use Cases

### Research Applications
- **Textual Reuse**: Find quotations and borrowings between texts
- **Translation Studies**: Identify parallel passages in different languages
- **Manuscript Studies**: Discover textual variants and versions
- **Literary Analysis**: Track motifs and themes across corpus

### Technical Features
- **Scale**: Process millions of segments efficiently
- **Precision**: Tune similarity thresholds for quality
- **Flexibility**: Each stage runs independently
- **Extensibility**: Modular design for customization

## Performance

Typical processing times for 1M segments:

| Stage | Time | Output Size |
|-------|------|-------------|
| Segmentation | 10-30 min | ~500 MB (Excel) |
| Indexing | 1-3 hours | ~3 GB (embeddings) |
| Detection | 5-15 min | 50-200 MB (matches) |

## Requirements

- Python 3.9+
- 16+ GB RAM (for large corpora)
- Optional: GPU for faster embedding generation

## Development Status

| Stage | Status |
|-------|--------|
| Segmentation | ✅ Complete |
| Indexing | 🚧 In Progress |
| Detection | ✅ Complete |
| Pipeline Orchestrator | 🚧 In Progress |

## Contributing

Each stage is independently developed. See individual README files for stage-specific development guidelines.

## License

[To be specified]

## Citation

[To be specified]
