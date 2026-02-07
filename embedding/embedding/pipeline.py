"""
Main pipeline for generating embeddings from segmented text.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from .models import IndexingConfig, EmbeddingMetadata


class EmbeddingPipeline:
    """Pipeline for generating embeddings from segmented text files."""
    
    def __init__(self, config: IndexingConfig):
        """
        Initialize the embedding pipeline.
        
        Args:
            config: IndexingConfig object with pipeline settings
        """
        self.config = config
        self.model: Optional[SentenceTransformer] = None
        self.segments_df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.file_segments: dict = {}  # Store segments grouped by source file
        
    def load_model(self) -> SentenceTransformer:
        """
        Load the sentence transformer model.
        
        Returns:
            Loaded SentenceTransformer model
        """
        print(f"Loading model: {self.config.embedding.model_name}")
        
        device = self.config.embedding.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {device}")
        
        self.model = SentenceTransformer(
            self.config.embedding.model_name,
            device=device
        )
        
        # Set max sequence length
        self.model.max_seq_length = self.config.embedding.max_length
        
        print(f"Model loaded successfully (embedding dim: {self.model.get_sentence_embedding_dimension()})")
        return self.model
    
    def load_segments(self) -> pd.DataFrame:
        """
        Load all segment files from the input directory.
        Stores file information for per-file output mode.
        
        Returns:
            DataFrame with all segments
        """
        segments_dir = self.config.input.segments_dir
        print(f"\nLoading segments from: {segments_dir}")
        
        # Find all CSV files
        csv_files = list(segments_dir.glob("*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {segments_dir}")
        
        print(f"Found {len(csv_files)} CSV files")
        
        # Load all files and track source
        all_segments = []
        self.file_segments = {}  # Store segments grouped by source file
        
        for csv_file in tqdm(csv_files, desc="Loading CSV files"):
            try:
                df = pd.read_csv(csv_file)
                
                # Add source file column if not present
                if 'Source_File' not in df.columns:
                    df['Source_File'] = csv_file.stem
                
                all_segments.append(df)
                
                # Store for per-file processing
                if self.config.output.mode == "per_file":
                    self.file_segments[csv_file.stem] = df.copy()
                    
            except Exception as e:
                print(f"Warning: Failed to load {csv_file.name}: {e}")
        
        # Concatenate all segments
        self.segments_df = pd.concat(all_segments, ignore_index=True)
        
        print(f"Loaded {len(self.segments_df)} total segments from {len(csv_files)} files")
        
        # Apply max_segments limit if specified
        if self.config.processing.max_segments:
            original_count = len(self.segments_df)
            self.segments_df = self.segments_df.head(self.config.processing.max_segments)
            print(f"Limited to {len(self.segments_df)} segments (from {original_count})")
        
        # Verify required column exists
        text_col = self.config.input.text_column
        if text_col not in self.segments_df.columns:
            available_cols = ', '.join(self.segments_df.columns)
            raise ValueError(
                f"Text column '{text_col}' not found in segments. "
                f"Available columns: {available_cols}"
            )
        
        return self.segments_df
    
    def generate_embeddings(self) -> np.ndarray:
        """
        Generate embeddings for all segments.
        
        Returns:
            Numpy array of embeddings (num_segments, embedding_dim)
        """
        if self.model is None:
            self.load_model()
        
        if self.segments_df is None:
            self.load_segments()
        
        print(f"\nGenerating embeddings...")
        
        # Extract text to embed
        text_col = self.config.input.text_column
        texts = self.segments_df[text_col].fillna("").astype(str).tolist()
        
        # Generate embeddings
        self.embeddings = self.model.encode(
            texts,
            batch_size=self.config.embedding.batch_size,
            show_progress_bar=self.config.embedding.show_progress,
            normalize_embeddings=self.config.embedding.normalize,
            convert_to_numpy=True
        )
        
        print(f"Generated embeddings: {self.embeddings.shape}")
        
        return self.embeddings
    
    def save_outputs(self) -> Tuple[Path, Path, Path]:
        """
        Save embeddings, segments, and metadata to disk.
        Supports combined, per-file, and per-line output modes.
        
        Returns:
            Tuple of (embeddings_path, segments_path, metadata_path)
        """
        output_dir = self.config.output.output_dir
        
        if self.config.output.mode == "per_file":
            return self._save_per_file()
        elif self.config.output.mode == "per_line":
            return self._save_per_line_number()
        else:
            return self._save_combined()
    
    def _save_combined(self) -> Tuple[Path, Path, Path]:
        """Save all embeddings and segments in single files."""
        output_dir = self.config.output.output_dir
        
        # Save embeddings
        embeddings_path = output_dir / self.config.output.embeddings_file
        print(f"\nSaving combined embeddings to: {embeddings_path}")
        np.save(embeddings_path, self.embeddings)
        
        # Save segments
        segments_path = output_dir / self.config.output.segments_file
        print(f"Saving combined segments to: {segments_path}")
        
        self.segments_df.to_csv(segments_path, index=False)
        
        # Save metadata
        metadata = EmbeddingMetadata(
            model_name=self.config.embedding.model_name,
            num_segments=len(self.segments_df),
            embedding_dimension=self.embeddings.shape[1],
            normalized=self.config.embedding.normalize,
            created_at=datetime.now().isoformat(),
            config=self.config.model_dump()
        )
        
        metadata_path = output_dir / self.config.output.metadata_file
        print(f"Saving metadata to: {metadata_path}")
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.model_dump(), f, indent=2, default=str)
        
        return embeddings_path, segments_path, metadata_path
    
    def _save_per_file(self) -> Tuple[Path, Path, Path]:
        """Save embeddings and segments in separate files per source file."""
        output_dir = self.config.output.output_dir / self.config.output.per_file_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving per-file embeddings to: {output_dir}")
        
        # Get unique source files
        source_files = self.segments_df['Source_File'].unique()
        
        saved_files = []
        total_segments = 0
        
        for source_file in tqdm(source_files, desc="Saving per-file embeddings"):
            # Get segments and embeddings for this source file
            mask = self.segments_df['Source_File'] == source_file
            file_segments = self.segments_df[mask].copy()
            file_embeddings = self.embeddings[mask]
            
            # Save embeddings
            embeddings_filename = f"{source_file}_embeddings.npy"
            embeddings_path = output_dir / embeddings_filename
            np.save(embeddings_path, file_embeddings)
            
            # Save segments
            segments_filename = f"{source_file}_segments.{self.config.output.segments_format}"
            segments_path = output_dir / segments_filename
            
            file_segments.to_csv(segments_path, index=False)
            
            saved_files.append({
                'source_file': source_file,
                'embeddings_file': embeddings_filename,
                'segments_file': segments_filename,
                'num_segments': len(file_segments),
                'embedding_shape': file_embeddings.shape
            })
            
            total_segments += len(file_segments)
        
        print(f"Saved {len(saved_files)} separate embedding files ({total_segments} total segments)")
        
        # Save overall metadata
        metadata = EmbeddingMetadata(
            model_name=self.config.embedding.model_name,
            num_segments=total_segments,
            embedding_dimension=self.embeddings.shape[1],
            normalized=self.config.embedding.normalize,
            created_at=datetime.now().isoformat(),
            config=self.config.model_dump()
        )
        
        # Add per-file information
        metadata_dict = metadata.model_dump()
        metadata_dict['output_mode'] = 'per_file'
        metadata_dict['files'] = saved_files
        
        metadata_path = output_dir / self.config.output.metadata_file
        print(f"Saving metadata to: {metadata_path}")
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
        
        # Return first file paths as representative
        first_embeddings = output_dir / saved_files[0]['embeddings_file']
        first_segments = output_dir / saved_files[0]['segments_file']
        
        return first_embeddings, first_segments, metadata_path
    
    def _save_per_line_number(self) -> Tuple[Path, Path, Path]:
        """Save embeddings and segments in separate files per Source_Line_Number."""
        output_dir = self.config.output.output_dir / self.config.output.per_line_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving per-line embeddings to: {output_dir}")
        
        # Check if Source_Line_Number column exists
        if 'Source_Line_Number' not in self.segments_df.columns:
            raise ValueError("Source_Line_Number column not found in segments data. Cannot use per_line mode.")
        
        # Get unique source line numbers
        unique_lines = self.segments_df['Source_Line_Number'].unique()
        unique_lines = sorted(unique_lines)  # Ensure sorted order
        
        print(f"Found {len(unique_lines)} unique source line numbers")
        
        saved_files = []
        total_segments = 0
        
        for line_num in tqdm(unique_lines, desc="Saving per-line embeddings"):
            # Get segments and embeddings for this line number
            mask = self.segments_df['Source_Line_Number'] == line_num
            line_segments = self.segments_df[mask].copy()
            line_embeddings = self.embeddings[mask]

            # Add segment_id column: {line_number}_{1-indexed segment order}
            line_segments['segment_id'] = line_segments['Sentence_Order'].apply(
                lambda order: f"{int(line_num)}_{int(order) + 1}"
            )
            
            # Create filename with zero-padded line number
            line_num_str = f"{int(line_num):06d}"
            embeddings_filename = f"line_{line_num_str}_embeddings.npy"
            segments_filename = f"line_{line_num_str}_segments.{self.config.output.segments_format}"
            
            # Save embeddings
            embeddings_path = output_dir / embeddings_filename
            np.save(embeddings_path, line_embeddings)
            
            # Save segments
            segments_path = output_dir / segments_filename
            line_segments.to_csv(segments_path, index=False)
            
            saved_files.append({
                'source_line_number': int(line_num),
                'embeddings_file': embeddings_filename,
                'segments_file': segments_filename,
                'num_segments': len(line_segments),
                'embedding_shape': list(line_embeddings.shape)
            })
            
            total_segments += len(line_segments)
        
        print(f"Saved {len(saved_files)} separate line files ({total_segments} total segments)")
        
        # Save overall metadata
        metadata = EmbeddingMetadata(
            model_name=self.config.embedding.model_name,
            num_segments=total_segments,
            embedding_dimension=self.embeddings.shape[1],
            normalized=self.config.embedding.normalize,
            created_at=datetime.now().isoformat(),
            config=self.config.model_dump()
        )
        
        # Add per-line information
        metadata_dict = metadata.model_dump()
        metadata_dict['output_mode'] = 'per_line'
        metadata_dict['num_line_files'] = len(saved_files)
        metadata_dict['line_numbers'] = {
            'min': int(unique_lines[0]),
            'max': int(unique_lines[-1]),
            'count': len(unique_lines)
        }
        metadata_dict['files'] = saved_files
        
        metadata_path = output_dir / self.config.output.metadata_file
        print(f"Saving metadata to: {metadata_path}")
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
        
        # Return first file paths as representative
        first_embeddings = output_dir / saved_files[0]['embeddings_file']
        first_segments = output_dir / saved_files[0]['segments_file']
        
        return first_embeddings, first_segments, metadata_path
    
    def run(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Run the complete embedding pipeline.
        
        Returns:
            Tuple of (embeddings, segments_df)
        """
        print("="*60)
        print("EMBEDDING PIPELINE - Embedding Generation")
        print("="*60)
        
        # Check if outputs already exist
        embeddings_path = self.config.output.output_dir / self.config.output.embeddings_file
        segments_path = self.config.output.output_dir / self.config.output.segments_file
        
        if self.config.processing.skip_existing:
            if embeddings_path.exists() and segments_path.exists():
                print(f"\nOutputs already exist. Loading existing embeddings...")
                self.embeddings = np.load(embeddings_path)
                
                self.segments_df = pd.read_csv(segments_path)
                
                print(f"Loaded: {self.embeddings.shape[0]} embeddings")
                return self.embeddings, self.segments_df
        
        # Load model and segments
        self.load_model()
        self.load_segments()
        
        # Generate embeddings
        self.generate_embeddings()
        
        # Save outputs
        self.save_outputs()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"✓ Embeddings: {self.embeddings.shape}")
        print(f"✓ Segments: {len(self.segments_df)} rows")
        print(f"✓ Model: {self.config.embedding.model_name}")
        print(f"✓ Output mode: {self.config.output.mode}")
        if self.config.output.mode == "per_file":
            num_files = len(self.segments_df['Source_File'].unique())
            print(f"✓ Files created: {num_files} separate embedding files")
        print("="*60)
        
        return self.embeddings, self.segments_df


def run_pipeline(config_path: Optional[str | Path] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Convenience function to run the embedding pipeline.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Tuple of (embeddings, segments_df)
    """
    from .config import load_config
    
    config = load_config(config_path)
    pipeline = EmbeddingPipeline(config)
    return pipeline.run()

