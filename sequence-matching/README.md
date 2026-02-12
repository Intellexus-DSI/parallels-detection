# Sequence Matching

Finds parallel passages (shared text segments) between documents using a seed-and-extend alignment algorithm adapted from bioinformatics (Smith-Waterman).

## How It Works

1. **Preprocessing** — Strips whitespace, metadata markers, and configurable characters to produce a dense character stream for alignment.
2. **Seeding** — Finds exact k-mer (substring) matches between each pair of documents. These act as anchor points.
3. **Region Merging** — Groups nearby seeds into candidate regions using diagonal consistency to avoid drifting across unrelated text.
4. **Alignment** — Runs local Smith-Waterman alignment within each candidate region to find the precise matching segments.
5. **Output** — Collects all matches across all document pairs into a single CSV.

## Usage

```bash
python run.py                          # uses config.yaml, seed mode
python run.py --config my_config.yaml  # custom config file
python run.py --mode waterfall         # legacy full-text mode (slow, no seeding)
```

### CLI Arguments

| Argument   | Default        | Description                                                                 |
|------------|----------------|-----------------------------------------------------------------------------|
| `--config` | `config.yaml`  | Path to the YAML configuration file.                                        |
| `--mode`   | `seed`         | `seed` (recommended) uses seed-and-extend. Any other value uses the legacy full-text Smith-Waterman waterfall. |

## Configuration Reference (`config.yaml`)

### `algorithm` — Alignment scoring parameters

| Key                | Default | Description                                                                                      |
|--------------------|---------|--------------------------------------------------------------------------------------------------|
| `match_score`      | `1.0`   | Score awarded when two characters match.                                                         |
| `mismatch_score`   | `-1.5`  | Penalty applied when two characters differ. More negative = stricter matching.                    |
| `open_gap_score`   | `-1.0`  | Penalty for opening a new gap (insertion/deletion). More negative = fewer gaps allowed.           |
| `extend_gap_score` | `-1.0`  | Penalty for extending an existing gap by one position. More negative = shorter gaps preferred.    |
| `min_score`        | `15.0`  | Minimum alignment score for a match to be reported. Higher = only longer/better matches survive.  |
| `max_iterations`   | `100`   | Maximum number of alignment passes per region (or per pair in waterfall mode). Each pass finds one match and masks it before the next pass. |

### `seeding` — Seed-and-extend parameters (only used in `seed` mode)

| Key              | Default | Description                                                                                       |
|------------------|---------|---------------------------------------------------------------------------------------------------|
| `k`              | `15`    | Length of exact-match substrings (k-mers) used as seeds. Larger = fewer but more specific seeds.   |
| `max_gap`        | `100`   | Maximum gap (in characters) between consecutive seeds to merge them into one region. Also controls diagonal drift tolerance. |
| `extend`         | `200`   | Number of characters to pad around each merged seed region before alignment, so the aligner can find the full extent of a match. |
| `max_kmer_hits`  | `null`  | If set, k-mers appearing more than this many times in either text are skipped. Filters out repetitive/common substrings that produce noise. |

### `preprocessing`

| Key           | Default | Description                                                                                |
|---------------|---------|--------------------------------------------------------------------------------------------|
| `strip_chars` | `""`    | Additional characters to remove from text before alignment (e.g. `"\\_"` to strip backslashes and underscores). Spaces, newlines, and metadata markers are always stripped. |

### `input` — Document source (two modes, mutually exclusive)

#### Mode 1: Single directory (all pairs)

```yaml
input:
  dir: "path/to/folder"
```

Reads all `.txt` files from the directory and compares every pair (`N choose 2` combinations).

#### Mode 2: Cross-comparison (point of comparison vs. corpus)

```yaml
input:
  point_of_comparison: "path/to/poc_folder"
  corpus: "path/to/corpus_folder"
```

Reads `.txt` files from both directories and compares each file in `point_of_comparison` against each file in `corpus` (cross-product). Files within the same group are **not** compared to each other.

For example, 1 file in `point_of_comparison` and 1000 files in `corpus` produces exactly 1000 pairs.

In the output CSV, filenames are prefixed with `poc/` or `corpus/` to indicate which group they belong to.

### `output`

| Key    | Description                                      |
|--------|--------------------------------------------------|
| `path` | Path to the output CSV file. Directories are created automatically. |

## Output Format

The output CSV contains one row per match with the following columns:

| Column    | Description                                              |
|-----------|----------------------------------------------------------|
| `file_a`  | Filename of the first document in the pair.              |
| `file_b`  | Filename of the second document in the pair.             |
| `score`   | Alignment score of the match (higher = stronger match).  |
| `text_a`  | The matched text segment from document A (original text, with whitespace/markers preserved). |
| `text_b`  | The matched text segment from document B.                |
| `start_a` | Character offset where the match begins in document A.   |
| `end_a`   | Character offset where the match ends in document A.     |
| `start_b` | Character offset where the match begins in document B.   |
| `end_b`   | Character offset where the match ends in document B.     |

Results are sorted by score in descending order within each pair.

## Example Config

```yaml
algorithm:
  match_score: 1.0
  mismatch_score: -0.8
  open_gap_score: -0.8
  extend_gap_score: -0.8
  min_score: 20.0
  max_iterations: 100

seeding:
  k: 15
  max_gap: 100
  extend: 750
  max_kmer_hits: 3

preprocessing:
  strip_chars: "\\_"

input:
  dir: "test_data/my_documents"
  # OR:
  # point_of_comparison: "data/query_docs"
  # corpus: "data/reference_docs"

output:
  path: "output/results.csv"
```

## Dependencies

- Python 3.8+
- `biopython` (provides `Bio.Align.PairwiseAligner`)
- `pyyaml`
- `pandas`
- `tqdm`
