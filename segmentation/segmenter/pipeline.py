"""Main segmentation pipeline."""

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import Config
from .engines import BotokSegmenter, RegexSegmenter
from .models import Segment, DocumentMetadata, SegmentationResult
from .utils import (
    make_overlapping_spans,
    constrain_exclusive_segments,
    constrain_exclusive_segments_by_words,
)
from .utils.text_normalizer import normalize_tibetan_text


def sanitize_filename(name: str) -> str:
    """Sanitize filename for safe filesystem usage.

    Args:
        name: Original filename

    Returns:
        Sanitized filename
    """
    name = os.path.splitext(name)[0]
    return re.sub(r'[<>:"/\\|?*]', "_", name).strip()


# Regex pattern for illegal control characters (except tab, newline, carriage return)
ILLEGAL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')


def sanitize_text(text: str) -> str:
    """Remove control characters that may interfere with CSV/Excel.

    Args:
        text: Input text

    Returns:
        Sanitized text
    """
    if not text:
        return text
    return ILLEGAL_CHARS.sub('', text)


def clean_non_tibetan_characters(text: str) -> str:
    """Remove all non-Tibetan characters from text.

    Keeps only characters in the Tibetan Unicode range (U+0F00-U+0FFF).
    Removes Latin characters, numbers, punctuation, and other scripts.
    Preserves only spaces between Tibetan text for word separation.

    Args:
        text: Input text

    Returns:
        Text with only Tibetan characters and minimal whitespace
    """
    if not text:
        return text
    
    # Tibetan Unicode range: U+0F00 to U+0FFF
    # Remove everything that's not Tibetan or space
    # This will remove: Latin letters, numbers (0-9), quotes, punctuation, etc.
    non_tibetan_pattern = re.compile(r'[^\u0F00-\u0FFF\s]')
    cleaned_text = non_tibetan_pattern.sub('', text)
    
    # Normalize multiple spaces to single space and strip
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


def _process_record_worker(args: tuple) -> tuple[int, list[dict]] | None:
    """Worker for parallel processing. Must be module-level for pickling.

    Args:
        args: (line_num, record, seg_config_dict)

    Returns:
        (line_num, list of row dicts) or None to skip
    """
    line_num, record, seg_config = args
    text_content = record.get("text") or record.get("content") or ""
    if not text_content:
        return None

    text_content = clean_non_tibetan_characters(text_content)
    if seg_config.get("remove_spaces", False):
        text_content = normalize_tibetan_text(text_content, remove_spaces=True)
    if not text_content.strip():
        return None

    file_id_tibetan = record.get("file_id", "Unknown_Source")

    # Setup converter in worker process
    from .converter_utils import setup_converter_path, get_converter
    setup_converter_path()
    Converter = get_converter()
    if Converter is None:
        return None
    converter = Converter()

    # Convert file_id to Wylie
    tibetan_pattern = re.compile(r'[\u0F00-\u0FFF]+')
    def _replace_tibetan(match):
        try:
            wylie = converter.convert(
                match.group(0), "EWTS", text_scheme="Unicode", val_text_scheme=True
            )
            return wylie.strip()
        except Exception:
            return match.group(0)
    file_id = tibetan_pattern.sub(_replace_tibetan, file_id_tibetan)

    # Create segmenter in worker
    engine = seg_config.get("engine", "regex")
    min_syllables = seg_config.get("min_syllables", 4)
    if engine == "botok":
        segmenter = BotokSegmenter(min_syllables=min_syllables)
    else:
        segmenter = RegexSegmenter(min_syllables=min_syllables)

    metadata = DocumentMetadata(
        file_id=file_id,
        file_id_tibetan=file_id_tibetan,
        line_number=line_num,
    )

    # Segment
    atoms = segmenter.segment_with_indices(text_content)

    use_overlapping = seg_config.get("use_overlapping", True)
    if use_overlapping:
        raw_segments = make_overlapping_spans(
            atoms,
            text_content,
            max_atoms=seg_config.get("overlap_max_atoms", 8),
            min_chars=seg_config.get("overlap_min_chars", 8),
            max_chars=seg_config.get("overlap_max_chars", 350),
            max_spans=seg_config.get("max_spans_per_line", 300),
        )
    else:
        min_words = seg_config.get("min_words")
        max_words = seg_config.get("max_words")
        if min_words is not None or max_words is not None:
            min_words = min_words if min_words is not None else 1
            max_words = max_words if max_words is not None else 9999
            atoms = constrain_exclusive_segments_by_words(
                atoms, text_content, min_words, max_words
            )
        else:
            min_syl = seg_config.get("min_syllables", 4)
            max_syl = seg_config.get("max_syllables")
            atoms = constrain_exclusive_segments(atoms, min_syl, max_syl)
        raw_segments = [(sent, start, end, None, None) for sent, start, end in atoms]

    if not raw_segments:
        return None

    rows = []
    for idx, segment_data in enumerate(raw_segments, 1):
        if use_overlapping:
            sent, start, end, num_atoms, span_type = segment_data
        else:
            sent, start, end, num_atoms, span_type = segment_data
            num_atoms = None
            span_type = None

        ewts_text = f" {converter.convert(sent, 'EWTS', text_scheme='Unicode', val_text_scheme=True)}"

        row = {
            "Segmented_Text": sent,
            "Segmented_Text_EWTS": ewts_text,
            "Length": len(sent),
            "File_ID": file_id,
            "File_ID_Tibetan": file_id_tibetan,
            "Source_Line_Number": line_num,
            "Sentence_Order": idx,
            "Start_Index": start,
            "End_Index": end,
        }
        if use_overlapping:
            row["Span_Num_Atoms"] = num_atoms
            row["Span_Type"] = span_type
        rows.append(row)

    return (line_num, rows)


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

    def _convert_file_id_to_wylie(self, file_id: str) -> str:
        """Convert Tibetan parts of file_id to Wylie transliteration.

        Args:
            file_id: File identifier that may contain Tibetan Unicode characters

        Returns:
            File identifier with Tibetan parts converted to Wylie
        """
        if self.converter is None:
            raise RuntimeError("Converter not initialized. This should not happen.")

        # Extract only Tibetan characters from the file_id
        tibetan_pattern = re.compile(r'[\u0F00-\u0FFF]+')
        
        def replace_tibetan(match):
            tibetan_text = match.group(0)
            try:
                # Convert to Wylie (EWTS)
                wylie = self.converter.convert(
                    tibetan_text, "EWTS", text_scheme="Unicode", val_text_scheme=True
                )
                return wylie.strip()
            except Exception:
                return tibetan_text  # Return original if conversion fails
        
        return tibetan_pattern.sub(replace_tibetan, file_id)

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
            # Exclusive mode: apply min/max constraints (words or syllables)
            min_words = self.config.segmentation.min_words
            max_words = self.config.segmentation.max_words
            if min_words is not None or max_words is not None:
                min_words = min_words if min_words is not None else 1
                max_words = max_words if max_words is not None else 9999
                atoms = constrain_exclusive_segments_by_words(
                    atoms, text, min_words, max_words
                )
            else:
                atoms = constrain_exclusive_segments(
                    atoms,
                    self.config.segmentation.min_syllables,
                    self.config.segmentation.max_syllables,
                )
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
                file_id=metadata.file_id,  # Wylie version
                file_id_tibetan=metadata.file_id_tibetan,  # Original Tibetan
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

    def _process_file_sequential(
        self, input_path: Path, full_dir: Path, single_dir: Path
    ) -> int:
        """Process file sequentially (original logic)."""
        file_groups = {}
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
                    text_content = clean_non_tibetan_characters(text_content)
                    if self.config.segmentation.remove_spaces:
                        text_content = normalize_tibetan_text(text_content, remove_spaces=True)
                    if not text_content.strip():
                        continue
                    file_id_tibetan = record.get("file_id", "Unknown_Source")
                    file_id = self._convert_file_id_to_wylie(file_id_tibetan)
                    metadata = DocumentMetadata(
                        file_id=file_id,
                        file_id_tibetan=file_id_tibetan,
                        line_number=line_num,
                    )
                    result = self.process_line(text_content, metadata)
                    if result.segments:
                        single_line_rows = []
                        clean_name = "All_Segments"
                        for segment in result.segments:
                            row = {
                                "Segmented_Text": segment.text,
                                "Segmented_Text_EWTS": segment.text_ewts,
                                "Length": segment.length,
                                "File_ID": segment.file_id,
                                "File_ID_Tibetan": segment.file_id_tibetan,
                                "Source_Line_Number": segment.source_line_number,
                                "Sentence_Order": segment.sentence_order,
                                "Start_Index": segment.start_index,
                                "End_Index": segment.end_index,
                            }
                            if self.config.segmentation.use_overlapping:
                                row["Span_Num_Atoms"] = segment.span_num_atoms
                                row["Span_Type"] = segment.span_type
                            single_line_rows.append(row)
                            if clean_name not in file_groups:
                                file_groups[clean_name] = []
                            file_groups[clean_name].extend(single_line_rows)
                        if self.config.output.save_single_lines:
                            single_df = pd.DataFrame(single_line_rows)
                            for col in single_df.select_dtypes(include=["object"]).columns:
                                single_df[col] = single_df[col].apply(
                                    lambda x: sanitize_text(x) if isinstance(x, str) else x
                                )
                            single_filename = f"Line_{line_num}_{clean_name[:30]}.csv"
                            single_df.to_csv(single_dir / single_filename, index=False)
                        lines_processed += 1
                except json.JSONDecodeError:
                    continue

        self._save_full_files(full_dir, file_groups)
        return lines_processed

    def _save_full_files(self, full_dir: Path, file_groups: dict) -> None:
        """Save aggregated full files."""
        if not self.config.output.save_full_files:
            return
        print(f"\nSaving {len(file_groups)} Full Files to {full_dir}...")
        for filename, rows in tqdm(file_groups.items(), desc="Saving Full CSV"):
            if not rows:
                continue
            df = pd.DataFrame(rows)
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].apply(
                    lambda x: sanitize_text(x) if isinstance(x, str) else x
                )
            save_path = full_dir / f"{filename}.csv"
            try:
                df.to_csv(save_path, index=False)
            except Exception as e:
                print(f"Error saving {filename}: {e}")

    def _process_file_parallel(
        self, input_path: Path, full_dir: Path, single_dir: Path
    ) -> int:
        """Process file using multiple worker processes."""
        workers = self.config.segmentation.workers
        seg_config = self.config.segmentation.model_dump()

        # Load all records
        tasks = []
        with open(input_path, "r", encoding="utf-8") as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    tasks.append((line_num, record, seg_config))
                except json.JSONDecodeError:
                    continue

        total = len(tasks)
        desc_text = f"{self.config.segmentation.engine.title()} Processing ({workers} workers)"
        if self.config.segmentation.use_overlapping:
            desc_text += " (Overlapping Mode)"
        else:
            desc_text += " (Exclusive Mode)"

        file_groups = {"All_Segments": []}
        lines_processed = 0
        results_by_line = {}

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_record_worker, task): task[0]
                for task in tasks
            }
            for future in tqdm(as_completed(futures), total=total, desc=desc_text):
                line_num = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        _, rows = result
                        results_by_line[line_num] = rows
                        file_groups["All_Segments"].extend(rows)
                        lines_processed += 1
                except Exception as e:
                    print(f"\nWorker error at line {line_num}: {e}")

        # Sort and write single-line CSVs
        if self.config.output.save_single_lines:
            clean_name = "All_Segments"
            for line_num in sorted(results_by_line.keys()):
                rows = results_by_line[line_num]
                single_df = pd.DataFrame(rows)
                for col in single_df.select_dtypes(include=["object"]).columns:
                    single_df[col] = single_df[col].apply(
                        lambda x: sanitize_text(x) if isinstance(x, str) else x
                    )
                single_filename = f"Line_{line_num}_{clean_name[:30]}.csv"
                single_df.to_csv(single_dir / single_filename, index=False)

        self._save_full_files(full_dir, file_groups)

        return lines_processed

    def process_file(self, input_path: Path) -> int:
        """Process a JSONL file and generate segmented output.

        Args:
            input_path: Path to input JSONL file

        Returns:
            Number of lines processed
        """
        full_dir, single_dir = self._setup_output_dirs()
        print(f"Reading from: {input_path}")

        workers = self.config.segmentation.workers
        if workers <= 1:
            lines_processed = self._process_file_sequential(
                input_path, full_dir, single_dir
            )
        else:
            lines_processed = self._process_file_parallel(
                input_path, full_dir, single_dir
            )

        desc_text = f"{self.config.segmentation.engine.title()} Processing"
        if workers > 1:
            desc_text += f" ({workers} workers)"
        if self.config.segmentation.use_overlapping:
            desc_text += " (Overlapping Mode)"
        else:
            desc_text += " (Exclusive Mode)"
        print(f"\nCompleted using {desc_text}.")
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
