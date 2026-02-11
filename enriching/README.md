# Stage 4: Enriching

Enriches parallel matches from the detection stage with additional fields.

## Input

CSV/Parquet/JSON files from stage 3 in `detection/output/`:
- `parallels.csv` (or chunked versions like `parallels_000.csv`, etc.)

## Output

Enriched CSV/Parquet/JSON files in `output/` with additional fields:
- All original fields from detection stage
- `wylie_syllable_distance`: Syllable-level Levenshtein distance between parallel_a and parallel_b (EWTS/Wylie)
- `mapping_type`: `textual` (many common words/phrases), `semantic` (similar meaning, fewer shared words), or `uncertain` (borderline)
- Additional fields from other enrichers (as configured)

## Usage

### Via Pipeline (from project root)

The `run_pipeline.py` and `pipeline_config.yaml` files are located in the **project root directory** (not in this folder).

```bash
cd /path/to/parallels-detection
python run_pipeline.py --config pipeline_config.yaml --stage enriching
```

### Standalone (from this directory)

```bash
cd enriching
python -m src.cli --input ../detection/output/parallels.csv --output output/parallels_enriched.csv
```

#### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | | Path to YAML configuration file |
| `--input` | (required) | Input file path (CSV/Parquet/JSON from detection) |
| `--output` | (required) | Output file path |
| `--input-format` | `csv` | Input file format: `csv`, `parquet`, or `json` |
| `--output-format` | `csv` | Output file format: `csv`, `parquet`, or `json` |
| `--enricher` | `wylie_levenshtein`, `mapping_type` | Enricher to apply (can be repeated) |
| `--max-lines-per-file` | `0` | Max lines per output file (0 = unlimited) |
| `-v, --verbose` | | Enable verbose logging |

#### Example with options

```bash
python -m src.cli \
    --input ../detection/output/parallels.csv \
    --output output/parallels_enriched.csv \
    --enricher wylie_levenshtein \
    --verbose
```

#### Using config file

```bash
python -m src.cli --config enriching_config.yaml
```

Example `enriching_config.yaml`:

```yaml
input:
  path: "../detection/output/parallels.csv"
  format: "csv"

output:
  path: "output/parallels_enriched.csv"
  format: "csv"
  max_lines_per_file: 0

enrichers:
  - name: "wylie_levenshtein"
    enabled: true
    params: {}
  - name: "mapping_type"
    enabled: true
    params:
      overlap_threshold: 0.35
```

## Configuration (in ../pipeline_config.yaml)

```yaml
enriching:
  input:
    path: "detection/output/parallels.csv"
    format: "csv"
  output:
    path: "enriching/output/parallels_enriched.csv"
    format: "csv"
    max_lines_per_file: 0
  enrichers:
    - name: "wylie_levenshtein"
      enabled: true
      params: {}
```

## Available Enrichers

### wylie_levenshtein

Calculates syllable-level Levenshtein distance between parallel texts in Wylie/EWTS.

Splits text by spaces and slashes to get syllables, then computes edit distance (insertions, deletions, substitutions) at the syllable level. Lower distance means more similar parallels.

**Output fields:**
- `wylie_syllable_distance`: Integer edit distance at syllable level

### mapping_type

Classifies parallels as **textual**, **semantic**, or **uncertain** using three signals:

1. **Unigram content overlap** (particles stripped per Tibetan guideline)
2. **Normalized Levenshtein distance** (distance / max_syllables)
3. **Bigram (phrase) overlap** – shared two-syllable sequences

- **Textual**: Strong evidence (high overlap, low distance, or high phrase overlap)
- **Semantic**: Clear semantic (low overlap, high distance, low phrase overlap)
- **Uncertain**: Borderline – worth manual review

**Parameters:**
- `overlap_textual` (0.40): Unigram overlap >= this → textual
- `overlap_semantic` (0.25): Unigram overlap <= this (with others) → semantic
- `norm_lev_textual` (0.25): Normalized Levenshtein <= this → textual
- `norm_lev_semantic` (0.40): Normalized Levenshtein >= this → semantic
- `bigram_textual` (0.25): Bigram overlap >= this → textual
- `bigram_semantic` (0.15): Bigram overlap <= this → semantic

**Output fields:**
- `mapping_type`: `"textual"`, `"semantic"`, or `"uncertain"`

## Adding New Enrichers

To add a new enricher:

1. Create a new file in `src/enrichers/` (e.g., `my_enricher.py`)
2. Inherit from `BaseEnricher`:

```python
from .base import BaseEnricher
from ..models import EnrichedParallel

class MyEnricher(BaseEnricher):
    def enrich(self, parallel: EnrichedParallel) -> EnrichedParallel:
        # Add your logic here
        parallel.enriched_fields["my_field"] = compute_value(parallel)
        return parallel
    
    @property
    def field_names(self) -> list:
        return ["my_field"]
```

3. Register it in `src/pipeline.py`:

```python
ENRICHER_REGISTRY = {
    "wylie_levenshtein": WylieLevenshteinEnricher,
    "my_enricher": MyEnricher,  # Add this line
}
```

4. Use it in configuration:

```yaml
enrichers:
  - name: "my_enricher"
    enabled: true
    params:
      param1: value1
```

## Dependencies

Install from requirements file:

```bash
cd enriching
pip install -r requirements.txt
```
