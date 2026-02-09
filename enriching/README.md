# Stage 4: Enriching

Enriches parallel matches from the detection stage with additional fields.

## Input

CSV/Parquet/JSON files from stage 3 in `detection/output/`:
- `parallels.csv` (or chunked versions like `parallels_000.csv`, etc.)

## Output

Enriched CSV/Parquet/JSON files in `output/` with additional fields:
- All original fields from detection stage
- `is_fuzzy_match`: 0 if texts are fuzzy match, 1 if not
- `fuzzy_score`: Fuzzy similarity score (0-100)
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
| `--enricher` | `fuzzy_matcher` | Enricher to apply (can be repeated) |
| `--fuzzy-threshold` | `90` | Fuzzy matching threshold (0-100) |
| `--max-lines-per-file` | `0` | Max lines per output file (0 = unlimited) |
| `-v, --verbose` | | Enable verbose logging |

#### Example with options

```bash
python -m src.cli \
    --input ../detection/output/parallels.csv \
    --output output/parallels_enriched.csv \
    --enricher fuzzy_matcher \
    --fuzzy-threshold 85 \
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
  - name: "fuzzy_matcher"
    enabled: true
    params:
      threshold: 90
      use_ratio: false
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
    - name: "fuzzy_matcher"
      enabled: true
      params:
        threshold: 90
```

## Available Enrichers

### fuzzy_matcher

Checks if parallel texts are fuzzy matches using fuzzy string matching.

**Parameters:**
- `threshold` (int, default: 90): Similarity threshold (0-100). Texts with similarity >= threshold are considered fuzzy matches (value 0), otherwise not fuzzy matches (value 1).
- `use_ratio` (bool, default: false): Use simple ratio instead of token_sort_ratio for comparison.

**Output fields:**
- `is_fuzzy_match`: 0 if fuzzy match, 1 if not
- `fuzzy_score`: Similarity score (0-100)

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
    "fuzzy_matcher": FuzzyMatcherEnricher,
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
