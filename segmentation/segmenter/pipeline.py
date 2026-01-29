"""Main segmentation pipeline."""

import json
import os
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import Config
from .engines import BotokSegmenter, RegexSegmenter
from .models import Segment, DocumentMetadata, SegmentationResult
from .utils import make_overlapping_spans


def sanitize_filename(name: str) -> str:
    """Sanitize filename for safe filesystem usage.

    Args:
        name: Original filename

    Returns:
        Sanitized filename
    """
    name = os.path.splitext(name)[0]
    return re.sub(r'[<>:"/\\|?*]', "_", name).strip()


# Regex pattern for illegal Excel characters (control chars except tab, newline, carriage return)
ILLEGAL_EXCEL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')


def sanitize_for_excel(text: str) -> str:
    """Remove characters that are illegal in Excel worksheets.

    Args:
        text: Input text

    Returns:
        Sanitized text safe for Excel
    """
    if not text:
        return text
    return ILLEGAL_EXCEL_CHARS.sub('', text)


class SegmentationPipeline:
    """Pipeline for segmenting Tibetan text files."""

    def __init__(self, config: Config):
        """Initialize segmentation pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config

        # Initialize segmentation engine
        if config.segmentation.engine == "botok":
            self.segmenter = BotokSegmenter(
                min_syllables=config.segmentation.min_syllables
            )
        else:
            self.segmenter = RegexSegmenter(
                min_syllables=config.segmentation.min_syllables
            )

        # Initialize converter - REQUIRED dependency (submodule must be set up beforehand; see README)
        from .converter_utils import setup_converter_path, get_converter
        
        if not setup_converter_path():
            raise ImportError(
                "detect_and_convert submodule not found. See README Setup (step 2)."
            )
        
        Converter = get_converter()
        if Converter is None:
            raise ImportError(
                "Failed to import Converter from detect_and_convert.\n"
                "The submodule may need to be installed. Run: cd detect_and_convert && pip install -e ."
            )
        
        try:
            self.converter = Converter()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Converter: {e}")

    def _convert_to_ewts(self, text: str) -> str:
        """Convert Tibetan Unicode to EWTS.

        Args:
            text: Tibetan Unicode text

        Returns:
            EWTS transliteration

        Raises:
            RuntimeError: If conversion fails
        """
        if self.converter is None:
            raise RuntimeError("Converter not initialized. This should not happen.")

        ewts_text = self.converter.convert(
            text, "EWTS", text_scheme="Unicode", val_text_scheme=True
        )
        return f" {ewts_text}"  # Prepend space as per original

    def _setup_output_dirs(self) -> tuple[Path, Path]:
        """Create output directories based on configuration.

        Returns:
            Tuple of (full_files_dir, single_lines_dir)
        """
        mode_subfolder = (
            "overlapping" if self.config.segmentation.use_overlapping else "exclusive"
        )
        full_dir = self.config.output.output_dir / mode_subfolder / "Full_Files"
        single_dir = self.config.output.output_dir / mode_subfolder / "Single_Lines"

        if self.config.output.save_full_files:
            full_dir.mkdir(parents=True, exist_ok=True)
        if self.config.output.save_single_lines:
            single_dir.mkdir(parents=True, exist_ok=True)

        return full_dir, single_dir

    def process_line(
        self, text: str, metadata: DocumentMetadata
    ) -> SegmentationResult:
        """Process a single line of text.

        Args:
            text: Input text content
            metadata: Document metadata

        Returns:
            SegmentationResult with segments and metadata
        """
        # Get atomic segments from engine
        atoms = self.segmenter.segment_with_indices(text)

        # Apply overlapping transformation if enabled
        if self.config.segmentation.use_overlapping:
            raw_segments = make_overlapping_spans(
                atoms,
                text,
                max_atoms=self.config.segmentation.overlap_max_atoms,
                min_chars=self.config.segmentation.overlap_min_chars,
                max_chars=self.config.segmentation.overlap_max_chars,
                max_spans=self.config.segmentation.max_spans_per_line,
            )
        else:
            # Exclusive mode: use atoms as-is (convert to expected format)
            raw_segments = [(sent, start, end, None, None) for sent, start, end in atoms]

        # Convert to Segment objects
        segments = []
        for idx, segment_data in enumerate(raw_segments, 1):
            if self.config.segmentation.use_overlapping:
                sent, start, end, num_atoms, span_type = segment_data
            else:
                sent, start, end, num_atoms, span_type = segment_data
                num_atoms = None
                span_type = None

            ewts_text = self._convert_to_ewts(sent)

            segment = Segment(
                text=sent,
                text_ewts=ewts_text,
                length=len(sent),
                file_path=metadata.file_path,
                title=metadata.title,
                source_line_number=metadata.line_number,
                sentence_order=idx,
                start_index=start,
                end_index=end,
                span_num_atoms=num_atoms,
                span_type=span_type,
            )
            segments.append(segment)

        return SegmentationResult(
            segments=segments, metadata=metadata, atom_count=len(atoms)
        )

    def process_file(self, input_path: Path) -> int:
        """Process a JSONL file and generate segmented output.

        Args:
            input_path: Path to input JSONL file

        Returns:
            Number of lines processed
        """
        full_dir, single_dir = self._setup_output_dirs()

        file_groups = {}  # Aggregated data for full files

        print(f"Reading from: {input_path}")

        # Count total lines
        total_lines = sum(1 for _ in open(input_path, "r", encoding="utf-8"))

        desc_text = f"{self.config.segmentation.engine.title()} Processing"
        if self.config.segmentation.use_overlapping:
            desc_text += " (Overlapping Mode)"
        else:
            desc_text += " (Exclusive Mode)"

        lines_processed = 0

        with open(input_path, "r", encoding="utf-8") as infile:
            for line_num, line in tqdm(
                enumerate(infile, 1), total=total_lines, desc=desc_text
            ):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    text_content = record.get("text") or record.get("content") or ""

                    if not text_content:
                        continue

                    metadata_dict = record.get("metadata", {}) or {}
                    raw_filename = metadata_dict.get("file_name", "Unknown_Source")
                    file_path = metadata_dict.get("file_path", "")
                    title = metadata_dict.get("title", "")

                    metadata = DocumentMetadata(
                        file_name=raw_filename,
                        file_path=file_path,
                        title=title,
                        line_number=line_num,
                    )

                    result = self.process_line(text_content, metadata)

                    if result.segments:
                        # Convert segments to dictionaries
                        single_line_rows = []
                        clean_name = sanitize_filename(raw_filename)

                        for segment in result.segments:
                            row = {
                                "Segmented_Text": segment.text,
                                "Segmented_Text_EWTS": segment.text_ewts,
                                "Length": segment.length,
                                "File_Path": segment.file_path,
                                "Title": segment.title,
                                "Source_Line_Number": segment.source_line_number,
                                "Sentence_Order": segment.sentence_order,
                                "Start_Index": segment.start_index,
                                "End_Index": segment.end_index,
                            }

                            # Add overlapping-specific columns if enabled
                            if self.config.segmentation.use_overlapping:
                                row["Span_Num_Atoms"] = segment.span_num_atoms
                                row["Span_Type"] = segment.span_type

                            single_line_rows.append(row)

                            # Add to full file groups
                            if clean_name not in file_groups:
                                file_groups[clean_name] = []
                            file_groups[clean_name].append(row)

                        # Save individual line Excel
                        if self.config.output.save_single_lines:
                            single_df = pd.DataFrame(single_line_rows)
                            # Sanitize text columns for Excel
                            for col in single_df.select_dtypes(include=['object']).columns:
                                single_df[col] = single_df[col].apply(
                                    lambda x: sanitize_for_excel(x) if isinstance(x, str) else x
                                )
                            single_filename = (
                                f"Line_{line_num}_{clean_name[:30]}.xlsx"
                            )
                            single_save_path = single_dir / single_filename
                            single_df.to_excel(single_save_path, index=False)

                    lines_processed += 1

                except json.JSONDecodeError:
                    continue

        # Save aggregated full files
        if self.config.output.save_full_files:
            print(f"\nSaving {len(file_groups)} Full Files to {full_dir}...")

            for filename, rows in tqdm(file_groups.items(), desc="Saving Full Xlsx"):
                if not rows:
                    continue
                df = pd.DataFrame(rows)
                # Sanitize text columns for Excel
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].apply(
                        lambda x: sanitize_for_excel(x) if isinstance(x, str) else x
                    )
                save_path = full_dir / f"{filename}.xlsx"
                try:
                    df.to_excel(save_path, index=False)
                except Exception as e:
                    print(f"Error saving {filename}: {e}")

        print(f"\n✓ Completed using {desc_text}.")
        if self.config.output.save_full_files:
            print(f"Full files saved in: {full_dir}")
        if self.config.output.save_single_lines:
            print(f"Individual line files saved in: {single_dir}")

        return lines_processed

    def run(self) -> int:
        """Run the segmentation pipeline.

        Returns:
            Number of lines processed
        """
        if not self.config.input_file:
            raise ValueError("Input file not specified in configuration")

        if not self.config.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.config.input_file}")

        return self.process_file(self.config.input_file)
