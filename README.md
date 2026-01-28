# Parallel Text Detection Pipeline

A 3-stage pipeline for finding semantically similar text segments across Tibetan/Sanskrit corpora.

## Pipeline Overview

```
data/*.jsonl → [Segmentation] → [Embedding] → [Detection] → output/parallels.csv
```

| Stage | Input | Output |
|-------|-------|--------|
| 1. Segmentation | JSONL files | Excel files with segments |
| 2. Embedding | Excel files | NPY embeddings + metadata |
| 3. Detection | NPY embeddings | CSV with parallel matches |

## Quick Start

### First Time Setup

```bash
# Clone the repository
git clone <repo-url>
cd parallels-detection

# Install stage dependencies
cd segmentation && pip install -e . && cd ..
cd embedding && pip install -e . && cd ..
cd detection && pip install -e . && cd ..
```

**Note:** The `detect_and_convert` submodule (required for EWTS conversion) will be automatically installed when you run the pipeline. No manual setup needed!

### Running the Pipeline

```bash
# Run full pipeline
python run_pipeline.py --config pipeline_config.yaml

# Run individual stage
python run_pipeline.py --config pipeline_config.yaml --stage segmentation
python run_pipeline.py --config pipeline_config.yaml --stage embedding
python run_pipeline.py --config pipeline_config.yaml --stage detection
```

**Automatic Dependency Setup:** When you run the pipeline, if the `detect_and_convert` submodule is missing, it will automatically:
1. Clone the repository from GitHub
2. Install it as an editable package
3. Continue with the pipeline execution

You'll see progress messages during the automatic setup. If automatic setup fails, you can manually run `python setup_submodule.py`.

## Directory Structure

```
parallels-detection/
├── data/                    # Input: raw JSONL files
├── output/                  # Output: final parallels.csv
├── segmentation/            # Stage 1
├── embedding/               # Stage 2
├── detection/               # Stage 3
├── run_pipeline.py          # Main orchestrator
└── pipeline_config.yaml     # Configuration
```

## Configuration

Edit `pipeline_config.yaml` to configure all stages. See stage READMEs for details.

## Requirements

- Python 3.9+
- 16+ GB RAM (for large corpora)
- GPU recommended for embedding stage

## Installation

### Core Dependencies

Install dependencies for each stage:

```bash
# Segmentation stage
cd segmentation
pip install -e .

# Embedding stage  
cd ../embedding
pip install -e .

# Detection stage
cd ../detection
pip install -e .
```

### Required: EWTS Conversion Support

The segmentation stage **requires** `detect_and_convert` for Tibetan Unicode → EWTS conversion.

**Automatic Installation:** The pipeline automatically installs `detect_and_convert` when needed. When you run the pipeline for the first time, it will:
- Detect if the submodule is missing
- Clone the repository from GitHub (if git submodule fails, it falls back to direct clone)
- Install it as an editable package with `pip install -e .`
- Continue with pipeline execution

**Manual Setup (Optional):** If you prefer to set it up manually before running the pipeline:
```bash
python setup_submodule.py
```

This will initialize the submodule and install it if needed.
