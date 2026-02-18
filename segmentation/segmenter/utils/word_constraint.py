"""Enforce min/max word constraints on exclusive-mode segments using Botok."""

from botok import WordTokenizer

# Tokenizer is module-level to avoid repeated init (lazy)
_tokenizer = None


def _get_tokenizer() -> WordTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = WordTokenizer()
    return _tokenizer


def _tokenize_with_positions(text: str) -> list[tuple[str, int, int]]:
    """Tokenize text and return (token_text, start, end) for each token."""
    tok = _get_tokenizer()
    tokens = tok.tokenize(text)
    result = []
    pos = 0
    for t in tokens:
        s = t.text
        result.append((s, pos, pos + len(s)))
        pos += len(s)
    return result


def _count_words(tokens_with_pos: list[tuple[str, int, int]]) -> int:
    """Count non-whitespace tokens as words."""
    return sum(1 for t, _, _ in tokens_with_pos if t.strip())


def constrain_exclusive_segments_by_words(
    atoms: list[tuple[str, int, int]],
    original_text: str,
    min_words: int,
    max_words: int,
) -> list[tuple[str, int, int]]:
    """Filter and split atoms to satisfy min/max word constraints.

    Uses Botok for word boundaries (no mid-word splits).
    Splits long atoms at word boundaries. Drops or merges short segments.

    Args:
        atoms: List of (text, start, end) from segmenter
        original_text: Full text (for accurate indexing)
        min_words: Minimum words per segment
        max_words: Maximum words per segment

    Returns:
        List of (text, start, end) satisfying constraints
    """
    if not atoms:
        return []

    result = []

    for text, atom_start, atom_end in atoms:
        tokens_with_pos = _tokenize_with_positions(text)
        n_words = _count_words(tokens_with_pos)
        if n_words < min_words:
            continue
        if n_words <= max_words:
            result.append((text, atom_start, atom_end))
            continue

        # Split long atom at word boundaries
        word_tokens = [(t, a, b) for t, a, b in tokens_with_pos if t.strip()]
        if not word_tokens:
            result.append((text, atom_start, atom_end))
            continue

        pos = 0
        chunk_tokens = []
        chunk_word_count = 0
        chunk_start_offset = 0

        for i, (tok_text, tok_start, tok_end) in enumerate(word_tokens):
            if chunk_word_count == 0:
                chunk_start_offset = tok_start

            chunk_tokens.append((tok_text, tok_start, tok_end))
            chunk_word_count += 1

            if chunk_word_count >= max_words:
                # Emit chunk: include all tokens from chunk_start to current
                # Reconstruct text from original (preserves exact spacing)
                chunk_text = text[chunk_start_offset:tok_end]
                global_start = atom_start + chunk_start_offset
                global_end = atom_start + tok_end
                result.append((chunk_text, global_start, global_end))
                chunk_tokens = []
                chunk_word_count = 0

        # Remainder
        if chunk_tokens:
            tok_end = chunk_tokens[-1][2]
            chunk_text = text[chunk_start_offset:tok_end]
            global_start = atom_start + chunk_start_offset
            global_end = atom_start + tok_end
            if chunk_word_count >= min_words:
                result.append((chunk_text, global_start, global_end))
            elif result:
                # Merge remainder into previous (include any text between them)
                prev_text, prev_start, prev_end = result[-1]
                gap_start = prev_end - atom_start
                gap_text = text[gap_start:chunk_start_offset] if gap_start < chunk_start_offset else ""
                result[-1] = (prev_text + gap_text + chunk_text, prev_start, global_end)
            else:
                result.append((chunk_text, global_start, global_end))

    return result
