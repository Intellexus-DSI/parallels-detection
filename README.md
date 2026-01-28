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

# Run setup script (automatically initializes submodule and installs dependencies)
python setup_submodule.py

# Install stage dependencies
cd segmentation && pip install -e . && cd ..
cd embedding && pip install -e . && cd ..
cd detection && pip install -e . && cd ..
```

**Alternative:** Clone with submodules manually:
```bash
git clone --recursive <repo-url>
cd parallels-detection
cd detect_and_convert && pip install -e . && cd ..
```

### Running the Pipeline

```bash
# Run full pipeline
python run_pipeline.py --config pipeline_config.yaml

# Run individual stage
python run_pipeline.py --config pipeline_config.yaml --stage segmentation
python run_pipeline.py --config pipeline_config.yaml --stage embedding
python run_pipeline.py --config pipeline_config.yaml --stage detection
```

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

**The setup script handles this automatically:**
```bash
python setup_submodule.py
```

This will initialize the submodule and install it if needed.
