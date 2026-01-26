"""Regex-based segmentation engine (high speed)."""

import re

from .base import (
    SegmentationEngine,
    TIBETAN_SHAD,
    TIBETAN_DOUBLE_SHAD,
    TER_TSHEG,
    TSHEG,
    TERMINATORS,
    OPTATIVE_IMPERATIVE,
    SENTENCE_INITIAL_MARKERS,
    GERUND_MARKERS,
    PARTICLES,
    ADVERBIALIZER_MARKERS,
    TOPIC_MARKER,
    CORRELATIVE_FIRST,
    CORRELATIVE_SECOND,
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
        # Include SHAD, DOUBLE_SHAD, and TER_TSHEG (gter tsheg)
        self.split_pattern = re.compile(
            f"([{TIBETAN_SHAD}{TIBETAN_DOUBLE_SHAD}{TER_TSHEG}]+)"
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

    def get_last_word(self, text: str) -> str:
        """Extract the last word from text (up to but not including shad/delimiter).

        Args:
            text: Tibetan text

        Returns:
            Last word (text after last tsheg, up to shad)
        """
        text = text.rstrip()
        if not text:
            return ""
        # Remove any trailing delimiters first
        while text and (text[-1] in [TIBETAN_SHAD, TIBETAN_DOUBLE_SHAD, TER_TSHEG]):
            text = text[:-1].rstrip()
        if not text:
            return ""
        last_tsheg_index = text.rfind(TSHEG)
        if last_tsheg_index == -1:
            return text
        return text[last_tsheg_index + 1 :].strip()

    def is_strong_boundary(self, delimiter: str) -> bool:
        """Check if delimiter is a strong boundary (double shad, quadruple shad, etc.).

        Args:
            delimiter: Delimiter string

        Returns:
            True if strong boundary (always split)
        """
        # Double shad (།།)
        if TIBETAN_DOUBLE_SHAD in delimiter:
            return True
        # Double gter tsheg (༔༔)
        if delimiter.count(TER_TSHEG) >= 2:
            return True
        # Quadruple shad pattern (།། །།) - check for multiple double shads
        if delimiter.count(TIBETAN_SHAD) >= 4:
            return True
        return False

    def check_correlative_pair(self, text_before: str) -> bool:
        """Check if text contains an incomplete correlative pair.

        Rule d: Do NOT split if a correlative pair is present but not complete.
        E.g., "ji ltar" should not be split before "de ltar" appears.

        Args:
            text_before: Text before the delimiter

        Returns:
            True if correlative pair is found but incomplete (don't split)
        """
        text_before = text_before.strip()
        
        # Check if any correlative first part exists in the text
        for first_part in CORRELATIVE_FIRST:
            first_idx = text_before.find(first_part)
            if first_idx != -1:
                # Found a first part, check if corresponding second part appears after it
                has_second = False
                for second_part in CORRELATIVE_SECOND:
                    second_idx = text_before.find(second_part, first_idx)
                    if second_idx != -1:
                        has_second = True
                        break
                # If first part exists but no second part found, don't split
                if not has_second:
                    return True
        return False

    def check_following_text(self, text_after: str) -> tuple[bool, str]:
        """Check if text following delimiter indicates a split.

        Rule: Split after single shad when followed by section markers or
        sentence-initial markers (unless rule c prevents it).

        Args:
            text_after: Text following the delimiter

        Returns:
            Tuple of (should_split, reason)
        """
        if not text_after:
            return False, ""
        
        # Strip leading whitespace but keep for checking
        text_after_stripped = text_after.strip()
        if not text_after_stripped:
            return False, ""
        
        # Check for section markers/enumerations
        # These may appear immediately or after whitespace/underscore
        section_patterns = [
            "༡", "༢", "༣", "༤", "༥", "༦", "༧", "༨", "༩", "༠",
            "དང་པོ", "གཉིས་པ", "གསུམ་པ", "བཞི་པ", "ལྔ་པ"
        ]
        # Check at start or after whitespace/underscore
        for pattern in section_patterns:
            if text_after_stripped.startswith(pattern):
                return True, "section_marker"
            # Check after space/underscore if present
            if len(text_after) > len(text_after_stripped):
                if text_after_stripped.startswith(pattern):
                    return True, "section_marker"
        
        # Check for sentence-initial markers
        for marker in SENTENCE_INITIAL_MARKERS:
            if text_after_stripped.startswith(marker):
                return True, "sentence_initial"
        
        return False, ""

    def should_split_after_single_shad(
        self, delimiter: str, text_before: str, text_after: str
    ) -> bool:
        """Determine if we should split after a single shad/gter tsheg.

        Implements the prose segmentation rules:
        - Split when preceded by terminators or optative/imperative
        - Split when followed by sentence-initial markers (unless rule c applies)
        - Do NOT split when preceded by gerund markers, particles, etc.
        - Do NOT split when correlative pair is incomplete

        Args:
            delimiter: The delimiter string
            text_before: Text before the delimiter
            text_after: Text after the delimiter

        Returns:
            True if should split, False otherwise
        """
        # Check for strong boundary first
        if self.is_strong_boundary(delimiter):
            return True

        # Get the last word before delimiter
        last_word = self.get_last_word(text_before)
        if not last_word:
            return False

        # Rule c: Do NOT split if preceded by certain markers
        # Check gerund markers
        if last_word in GERUND_MARKERS:
            return False
        
        # Check particles
        if last_word in PARTICLES:
            return False
        
        # Check adverbializer markers
        if last_word in ADVERBIALIZER_MARKERS:
            return False
        
        # Check topic marker
        if last_word in TOPIC_MARKER:
            return False
        
        # Check for correlative pairs (rule d)
        if self.check_correlative_pair(text_before):
            return False

        # Rule: Split when preceded by terminators
        if last_word in TERMINATORS:
            return True

        # Rule: Split when preceded by optative/imperative suffixes
        # Check if last word ends with optative/imperative or is equal to it
        for opt_imp in OPTATIVE_IMPERATIVE:
            if last_word == opt_imp or last_word.endswith(opt_imp):
                return True
        # Also check if optative/imperative appears in the buffer before delimiter
        # (for multi-word cases like "གྱུར་ཅིག")
        buffer_text = text_before.strip()
        for opt_imp in OPTATIVE_IMPERATIVE:
            if opt_imp in buffer_text:
                # Check if it's at the end (near the delimiter)
                opt_imp_idx = buffer_text.rfind(opt_imp)
                if opt_imp_idx != -1:
                    # If the optative/imperative is close to the end, consider it
                    remaining = buffer_text[opt_imp_idx + len(opt_imp):].strip()
                    # Remove any trailing delimiters
                    while remaining and remaining[-1] in [TIBETAN_SHAD, TIBETAN_DOUBLE_SHAD, TER_TSHEG]:
                        remaining = remaining[:-1].strip()
                    if not remaining or len(remaining) < 5:  # Close to delimiter
                        return True

        # Rule: Split when followed by sentence-initial markers or section markers
        # (But rule c already handled the "not when" cases above)
        should_split_after, reason = self.check_following_text(text_after)
        if should_split_after:
            return True

        # Default: don't split after single shad if no rules match
        return False

    def segment_with_indices(self, text: str) -> list[tuple[str, int, int]]:
        """Segment text using regex patterns according to Tibetan prose guidelines.

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
        buffer_start_idx = 0
        cursor = 0

        # Process parts with look-ahead capability
        i = 0
        while i < len(parts):
            part = parts[i]
            if not part:
                i += 1
                continue

            part_len = len(part)
            is_delimiter = (
                TIBETAN_SHAD in part
                or TIBETAN_DOUBLE_SHAD in part
                or TER_TSHEG in part
            )

            if is_delimiter:
                # Add delimiter to buffer
                current_buffer.append(part)

                # Get text before and after delimiter for rule checking
                text_before = "".join(current_buffer[:-1])  # Everything before delimiter
                
                # Look ahead to get text after delimiter (up to next delimiter or reasonable limit)
                text_after = ""
                j = i + 1
                max_lookahead = min(5, len(parts) - i - 1)  # Look at most 5 parts ahead
                lookahead_count = 0
                while j < len(parts) and lookahead_count < max_lookahead:
                    next_part = parts[j]
                    if not next_part:
                        j += 1
                        lookahead_count += 1
                        continue
                    # Check if this part is a delimiter
                    if (
                        TIBETAN_SHAD in next_part
                        or TIBETAN_DOUBLE_SHAD in next_part
                        or TER_TSHEG in next_part
                    ):
                        break
                    text_after += next_part
                    j += 1
                    lookahead_count += 1

                # Determine if we should split
                should_split = self.should_split_after_single_shad(
                    part, text_before, text_after
                )

                if should_split:
                    clean_sent = "".join(current_buffer).strip()
                    has_tibetan = self.tibetan_pattern.search(clean_sent)
                    has_english = self.english_pattern.search(clean_sent)

                    # Exclude English-only segments
                    if not (has_english and not has_tibetan):
                        final_sentences.append(
                            (clean_sent, buffer_start_idx, cursor + part_len)
                        )
                    
                    # Reset buffer for next segment
                    current_buffer = []
                    buffer_start_idx = cursor + part_len

                cursor += part_len
                i += 1

            else:
                # Regular text part
                if not current_buffer:
                    buffer_start_idx = cursor

                current_buffer.append(part)
                cursor += part_len
                i += 1

        # Handle remaining buffer (text without final delimiter)
        if current_buffer:
            clean_sent = "".join(current_buffer).strip()
            has_tibetan = self.tibetan_pattern.search(clean_sent)
            has_english = self.english_pattern.search(clean_sent)
            if not (has_english and not has_tibetan):
                final_sentences.append((clean_sent, buffer_start_idx, cursor))

        return final_sentences
