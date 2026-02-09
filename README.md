# Parallel Text Detection Pipeline

A 4-stage pipeline for finding semantically similar text segments across Tibetan/Sanskrit corpora.

## Pipeline Overview

```
data/*.jsonl → [Segmentation] → [Embedding] → [Detection] → [Enriching] → output/parallels_enriched.csv
```

| Stage | Input | Output |
|-------|-------|--------|
| 1. Segmentation | JSONL files | CSV files with segments |
| 2. Embedding | CSV files | NPY embeddings + metadata |
| 3. Detection | NPY embeddings | CSV with parallel matches |
| 4. Enriching | CSV from Detection | Enriched CSV with additional fields |

## Setup

Do these steps **once** before running the pipeline. The pipeline does **not** download or initialize the submodule automatically.

### 1. Clone and install stage dependencies

```bash
git clone <repo-url>
cd parallels-detection

# Install each stage
cd segmentation && pip install -e . && cd ..
cd embedding && pip install -e . && cd ..
cd detection && pip install -e . && cd ..
cd enriching && pip install -e . && cd ..
```

### 2. Set up the detect_and_convert submodule (required for segmentation)

The segmentation stage needs the **detect_and_convert** repo for Tibetan Unicode → EWTS conversion. It is a **private** GitHub repo, so run these commands **in your terminal** (Git will prompt for credentials).

**Option A – submodule already registered (e.g. you have `.gitmodules`):**

```bash
git submodule update --init --recursive
cd detect_and_convert && pip install -e . && cd ..
```

**Option B – submodule not populated (e.g. clone without `--recursive`), or Option A failed:**  
Clone the repo yourself so Git can prompt for your username and [Personal Access Token](https://github.com/settings/tokens) (not your password):

```bash
git clone https://github.com/Intellexus-DSI/detect_and_convert.git detect_and_convert
cd detect_and_convert && pip install -e . && cd ..
```

The pipeline does **not** fetch or install the submodule for you. If it is missing, segmentation will fail and point you to the README.

## Running the Pipeline

```bash
# Run full pipeline
python run_pipeline.py --config pipeline_config.yaml

# Run individual stage
python run_pipeline.py --config pipeline_config.yaml --stage segmentation
python run_pipeline.py --config pipeline_config.yaml --stage embedding
python run_pipeline.py --config pipeline_config.yaml --stage detection
python run_pipeline.py --config pipeline_config.yaml --stage enriching
```

## Directory Structure

```
parallels-detection/
├── data/                    # Input: raw JSONL files
├── output/                  # Output: final parallels_enriched.csv
├── segmentation/            # Stage 1
├── embedding/               # Stage 2
├── detection/               # Stage 3
├── enriching/               # Stage 4
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

# Enriching stage
cd ../enriching
pip install -e .
```

### Required: EWTS conversion (detect_and_convert submodule)

The segmentation stage **requires** the `detect_and_convert` submodule for Tibetan Unicode → EWTS conversion. See **Setup** (step 2) above for how to clone/init the submodule and install it. The pipeline does not fetch or install it for you.
