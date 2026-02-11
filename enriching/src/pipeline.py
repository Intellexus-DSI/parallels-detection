"""Enriching pipeline for parallel matches."""

import json
import logging
from pathlib import Path
from typing import Iterator, List

import pandas as pd
from tqdm import tqdm

from .config import EnrichingConfig
from .enrichers import BaseEnricher, MappingTypeEnricher, WylieLevenshteinEnricher
from .models import EnrichedParallel

logger = logging.getLogger(__name__)


class EnrichingPipeline:
    """
    Pipeline for enriching parallel matches with additional fields.
    
    Reads parallels from the detection stage and applies a chain of
    enrichers to add new fields.
    """
    
    # Registry of available enrichers
    ENRICHER_REGISTRY = {
        "wylie_levenshtein": WylieLevenshteinEnricher,
        "mapping_type": MappingTypeEnricher,
    }
    
    def __init__(self, config: EnrichingConfig):
        """
        Initialize the enriching pipeline.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.enrichers: List[BaseEnricher] = []
        
        # Initialize enrichers
        self._initialize_enrichers()
    
    def _initialize_enrichers(self) -> None:
        """Initialize enrichers based on configuration."""
        for enricher_config in self.config.enrichers:
            if not enricher_config.enabled:
                logger.info(f"Skipping disabled enricher: {enricher_config.name}")
                continue
            
            if enricher_config.name not in self.ENRICHER_REGISTRY:
                logger.warning(f"Unknown enricher: {enricher_config.name}")
                continue
            
            enricher_class = self.ENRICHER_REGISTRY[enricher_config.name]
            enricher = enricher_class(params=enricher_config.params)
            self.enrichers.append(enricher)
            
            logger.info(f"Initialized enricher: {enricher}")
    
    def _read_input(self) -> Iterator[EnrichedParallel]:
        """
        Read input parallels from file.
        
        Yields:
            EnrichedParallel objects.
        """
        input_path = self.config.input.input_path
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        logger.info(f"Reading input from: {input_path}")
        
        if self.config.input.format == "csv":
            df = pd.read_csv(input_path)
        elif self.config.input.format == "parquet":
            df = pd.read_parquet(input_path)
        elif self.config.input.format == "json":
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported input format: {self.config.input.format}")
        
        logger.info(f"Loaded {len(df)} parallels")
        
        # Convert to EnrichedParallel objects
        for _, row in df.iterrows():
            yield EnrichedParallel.from_dict(row.to_dict())
    
    def _apply_enrichers(self, parallel: EnrichedParallel) -> EnrichedParallel:
        """
        Apply all enrichers to a parallel match.
        
        Args:
            parallel: The parallel to enrich.
            
        Returns:
            The enriched parallel.
        """
        for enricher in self.enrichers:
            parallel = enricher.enrich(parallel)
        
        return parallel
    
    def _write_output(self, enriched_parallels: List[dict]) -> None:
        """
        Write enriched parallels to output file.
        
        Args:
            enriched_parallels: List of enriched parallel dictionaries.
        """
        output_path = self.config.output.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(enriched_parallels)
        
        logger.info(f"Writing {len(df)} enriched parallels to: {output_path}")
        
        if self.config.output.format == "csv":
            df.to_csv(output_path, index=False)
        elif self.config.output.format == "parquet":
            df.to_parquet(output_path, index=False)
        elif self.config.output.format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(enriched_parallels, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {self.config.output.format}")
        
        logger.info(f"Successfully wrote output to: {output_path}")
    
    def _write_chunked_output(self, enriched_parallels: List[dict], chunk_index: int) -> None:
        """
        Write a chunk of enriched parallels to output file.
        
        Args:
            enriched_parallels: List of enriched parallel dictionaries.
            chunk_index: The index of this chunk.
        """
        output_path = self.config.output.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create chunk filename
        stem = output_path.stem
        suffix = output_path.suffix
        chunk_path = output_path.parent / f"{stem}_{chunk_index:03d}{suffix}"
        
        df = pd.DataFrame(enriched_parallels)
        
        logger.info(f"Writing chunk {chunk_index} ({len(df)} parallels) to: {chunk_path}")
        
        if self.config.output.format == "csv":
            df.to_csv(chunk_path, index=False)
        elif self.config.output.format == "parquet":
            df.to_parquet(chunk_path, index=False)
        elif self.config.output.format == "json":
            with open(chunk_path, "w", encoding="utf-8") as f:
                json.dump(enriched_parallels, f, ensure_ascii=False, indent=2)
    
    def run(self) -> int:
        """
        Execute the enriching pipeline.
        
        Returns:
            Total number of parallels processed.
        """
        logger.info("Starting enriching pipeline")
        logger.info(f"Input: {self.config.input.input_path}")
        logger.info(f"Output: {self.config.output.output_path}")
        logger.info(f"Active enrichers: {len(self.enrichers)}")
        
        if not self.enrichers:
            logger.warning("No enrichers configured. Output will be identical to input.")
        
        # Process parallels
        enriched_parallels = []
        total_processed = 0
        chunk_index = 0
        max_lines = self.config.output.max_lines_per_file
        
        for parallel in tqdm(self._read_input(), desc="Enriching"):
            # Apply enrichers
            enriched = self._apply_enrichers(parallel)
            enriched_parallels.append(enriched.to_dict())
            total_processed += 1
            
            # Check if we need to write a chunk
            if max_lines > 0 and len(enriched_parallels) >= max_lines:
                self._write_chunked_output(enriched_parallels, chunk_index)
                enriched_parallels = []
                chunk_index += 1
        
        # Write remaining data
        if enriched_parallels:
            if max_lines > 0:
                self._write_chunked_output(enriched_parallels, chunk_index)
            else:
                self._write_output(enriched_parallels)
        
        logger.info(f"Pipeline complete. Processed {total_processed} parallels")
        
        return total_processed
