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
