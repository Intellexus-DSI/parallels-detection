# Indexing Stage

The second stage in the parallel text detection pipeline. Generates vector embeddings for segmented text to enable semantic similarity search.

## Purpose

Converts text segments from the Segmentation stage into high-dimensional vector embeddings. These embeddings capture semantic meaning and enable efficient similarity comparison in the Detection stage.

## Input

- **Format**: Excel files (.xlsx) from Segmentation stage
- **Location**: `segmentation/data/segmented_output/{mode}/Full_Files/`
- **Required columns**:
  - `Segmented_Text`: Tibetan Unicode text
  - `Segmented_Text_EWTS`: EWTS transliteration (used for embedding)
  - Metadata columns (preserved in output)

## Output

- **Embeddings file**: `.npy` file containing numpy array of embeddings
  - Shape: `(N, embedding_dim)` where N = number of segments
  - Type: `float32`
  
- **Segments file**: CSV/Excel with all segment data aligned with embeddings
  - Each row corresponds to one embedding vector
  - Preserves all metadata from input

Example output:
```
data/
├── embeddings.npy           # Numpy array (10000, 768)
└── full_segmentation.xlsx   # Excel with 10000 rows
```

## Process

1. **Load Segments**: Read segmented Excel files
2. **Generate Embeddings**: Use embedding model on EWTS text
3. **Save Outputs**: 
   - Embeddings as `.npy` file
   - Consolidated segments as CSV/Excel

## Embedding Models

Common options:
- **Sentence Transformers**: `all-MiniLM-L6-v2`, `paraphrase-multilingual-mpnet-base-v2`
- **OpenAI**: `text-embedding-ada-002`
- **Custom**: Fine-tuned models for Tibetan/Sanskrit text

## Configuration

*To be implemented - will support config.yaml similar to other stages*

```yaml
input:
  segments_dir: "segmentation/data/segmented_output/overlapping/Full_Files"
  
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32
  normalize: true
  
output:
  embeddings_path: "data/embeddings.npy"
  segments_path: "data/full_segmentation.xlsx"
```

## Usage

*Scripts to be implemented*

```bash
# Using config file
python indexing_script.py --config config.yaml

# Direct arguments
python indexing_script.py \
  --input segmentation/data/segmented_output/overlapping/Full_Files \
  --output data/ \
  --model sentence-transformers/all-MiniLM-L6-v2
```

## Dependencies

```bash
pip install numpy pandas sentence-transformers torch
# Or
pip install numpy pandas openai  # For OpenAI embeddings
```

## Output Format

### Embeddings (.npy)
- Numpy array saved with `np.save()`
- Shape: `(num_segments, embedding_dimension)`
- Can be memory-mapped for large datasets

### Segments (CSV/Excel)
- All columns from input segments preserved
- Row order matches embedding array order
- Index column for easy reference

## Performance Considerations

- **Batch processing**: Process segments in batches to optimize GPU/CPU usage
- **Normalization**: L2-normalize embeddings for cosine similarity
- **Precision**: Use float32 to balance accuracy and memory
- **Caching**: Cache embeddings to avoid recomputation

## Next Stage

Output files are consumed by the **Detection** stage:
- `embeddings.npy` → Used for similarity search with FAISS
- `full_segmentation.xlsx` → Provides segment metadata for results

## Notes

- Embedding dimension depends on model (typically 384, 768, or 1536)
- EWTS text is used for embedding (better ASCII representation)
- Consider GPU acceleration for large datasets
- Embeddings should be generated only once and reused
