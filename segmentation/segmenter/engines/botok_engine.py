"""Botok-based segmentation engine (high accuracy)."""

from .base import (
    SegmentationEngine,
    TIBETAN_SHAD,
    TIBETAN_DOUBLE_SHAD,
    TSHEG,
    TERMINATORS,
    CONTINUATORS,
)


class BotokSegmenter(SegmentationEngine):
    """Segmentation engine using Botok tokenizer for high accuracy."""

    def __init__(self, min_syllables: int = 4):
        """Initialize Botok segmenter.

        Args:
            min_syllables: Minimum number of syllables per segment
        """
        super().__init__(min_syllables)
        print("Initializing Botok Engine...")
        from botok import WordTokenizer

        self.tokenizer = WordTokenizer()

    def segment_with_indices(self, text: str) -> list[tuple[str, int, int]]:
        """Segment text using Botok tokenizer.

        Args:
            text: Input text to segment

        Returns:
            List of (segment_text, start_index, end_index) tuples
        """
        if not text:
            return []

        tokens = self.tokenizer.tokenize(text)
        final_sentences = []

        current_buffer = ""
        buffer_start_idx = 0
        current_cursor = 0
        last_meaningful_word = ""

        for token in tokens:
            token_text = token.text
            current_buffer += token_text

            is_shad = TIBETAN_SHAD in token_text
            is_double_shad = TIBETAN_DOUBLE_SHAD in token_text

            should_split = False

            if is_shad or is_double_shad:
                should_split = True
                prev_word_clean = last_meaningful_word.strip().rstrip(TSHEG)

                if is_double_shad:
                    should_split = True
                elif self.number_pattern.search(prev_word_clean):
                    should_split = False
                elif prev_word_clean in CONTINUATORS:
                    should_split = False
                elif prev_word_clean in TERMINATORS:
                    should_split = True
                else:
                    syllables_in_buffer = self.count_syllables(current_buffer)
                    if syllables_in_buffer < self.min_syllables:
                        should_split = False

            if not (is_shad or is_double_shad or token_text.isspace()):
                last_meaningful_word = token_text

            if should_split:
                clean_sent = current_buffer.strip()
                has_tibetan = self.tibetan_pattern.search(clean_sent)
                has_english = self.english_pattern.search(clean_sent)

                # Exclude English-only segments
                if not (has_english and not has_tibetan):
                    end_idx = current_cursor + len(token_text)
                    final_sentences.append((clean_sent, buffer_start_idx, end_idx))
                    current_buffer = ""
                    buffer_start_idx = end_idx

            current_cursor += len(token_text)

        # Handle remaining buffer
        if current_buffer.strip():
            clean_sent = current_buffer.strip()
            final_sentences.append((clean_sent, buffer_start_idx, current_cursor))

        return final_sentences
