"""Regex-based segmentation engine (high speed)."""

import re

from .base import (
    SegmentationEngine,
    TIBETAN_SHAD,
    TIBETAN_DOUBLE_SHAD,
    TSHEG,
    TERMINATORS,
    CONTINUATORS,
)


class RegexSegmenter(SegmentationEngine):
    """Fast regex-based segmentation engine."""

    def __init__(self, min_syllables: int = 4):
        """Initialize regex segmenter.

        Args:
            min_syllables: Minimum number of syllables per segment
        """
        super().__init__(min_syllables)
        print("Initializing Fast Regex Engine...")
        self.split_pattern = re.compile(
            f"([{TIBETAN_SHAD}{TIBETAN_DOUBLE_SHAD}]+)"
        )

    def get_last_syllable(self, text: str) -> str:
        """Extract the last syllable from text.

        Args:
            text: Tibetan text

        Returns:
            Last syllable (text after last tsheg)
        """
        text = text.rstrip()
        if not text:
            return ""
        last_tsheg_index = text.rfind(TSHEG)
        if last_tsheg_index == -1:
            return text
        return text[last_tsheg_index + 1 :].strip()

    def segment_with_indices(self, text: str) -> list[tuple[str, int, int]]:
        """Segment text using regex patterns.

        Args:
            text: Input text to segment

        Returns:
            List of (segment_text, start_index, end_index) tuples
        """
        if not text:
            return []

        parts = self.split_pattern.split(text)
        final_sentences = []

        current_buffer = []
        current_buffer_len = 0
        buffer_start_idx = 0
        cursor = 0
        last_text_segment = ""

        for part in parts:
            if not part:
                continue

            part_len = len(part)
            is_delimiter = (TIBETAN_SHAD in part) or (TIBETAN_DOUBLE_SHAD in part)

            if is_delimiter:
                current_buffer.append(part)
                current_buffer_len += part_len

                should_split = True

                if TIBETAN_DOUBLE_SHAD in part:
                    should_split = True
                else:
                    last_syllable = self.get_last_syllable(last_text_segment)

                    if self.number_pattern.search(last_syllable):
                        should_split = False
                    elif last_syllable in CONTINUATORS:
                        should_split = False
                    elif last_syllable in TERMINATORS:
                        should_split = True
                    else:
                        full_buffer_str = "".join(current_buffer)
                        syllable_count = full_buffer_str.count(TSHEG)
                        if syllable_count < self.min_syllables:
                            should_split = False

                if should_split:
                    clean_sent = "".join(current_buffer).strip()
                    has_tibetan = self.tibetan_pattern.search(clean_sent)
                    has_english = self.english_pattern.search(clean_sent)

                    # Exclude English-only segments
                    if not (has_english and not has_tibetan):
                        final_sentences.append(
                            (clean_sent, buffer_start_idx, cursor + part_len)
                        )
                        current_buffer = []
                        current_buffer_len = 0
                        buffer_start_idx = cursor + part_len

                cursor += part_len

            else:
                if not current_buffer:
                    buffer_start_idx = cursor

                current_buffer.append(part)
                current_buffer_len += part_len
                last_text_segment = part
                cursor += part_len

        # Handle remaining buffer
        if current_buffer:
            clean_sent = "".join(current_buffer).strip()
            has_tibetan = self.tibetan_pattern.search(clean_sent)
            has_english = self.english_pattern.search(clean_sent)
            if not (has_english and not has_tibetan):
                final_sentences.append((clean_sent, buffer_start_idx, cursor))

        return final_sentences
