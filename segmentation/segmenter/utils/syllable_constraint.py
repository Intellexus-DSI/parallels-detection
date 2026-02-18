"""Enforce min/max syllable constraints on exclusive-mode segments."""

from ..engines.base import TSHEG


def _count_syllables(text: str) -> int:
    """Count syllables (tsheg count, consistent with base engine)."""
    return text.count(TSHEG)


def constrain_exclusive_segments(
    atoms: list[tuple[str, int, int]],
    min_syllables: int,
    max_syllables: int | None,
) -> list[tuple[str, int, int]]:
    """Filter and split atoms to satisfy min/max syllable constraints.

    Used in exclusive mode only. Splits long atoms at tsheg boundaries.
    Drops or merges segments shorter than min_syllables.

    Args:
        atoms: List of (text, start, end) from segmenter
        min_syllables: Minimum syllables (tsheg count) per segment
        max_syllables: Maximum syllables per segment (None = no limit)

    Returns:
        List of (text, start, end) satisfying constraints
    """
    if not atoms:
        return []

    result = []

    for text, start, end in atoms:
        n = _count_syllables(text)
        if n < min_syllables:
            continue  # Drop too-short
        if max_syllables is None or n <= max_syllables:
            result.append((text, start, end))
            continue

        # Split long atom at tsheg boundaries
        tsheg_positions = [i for i, c in enumerate(text) if c == TSHEG]
        if not tsheg_positions:
            result.append((text, start, end))
            continue

        pos = 0
        while pos < len(text):
            # Find end: include up to max_syllables tshegs
            tshegs_in_chunk = 0
            chunk_end = pos
            for i in range(pos, len(text)):
                if text[i] == TSHEG:
                    tshegs_in_chunk += 1
                    if tshegs_in_chunk > max_syllables:
                        chunk_end = i  # End before this tsheg
                        break
                chunk_end = i + 1

            chunk = text[pos:chunk_end]
            chunk_syl = _count_syllables(chunk)

            if chunk_syl >= min_syllables:
                result.append((chunk, start + pos, start + chunk_end))
            elif result:
                # Remainder too short - merge into previous chunk
                prev_text, prev_start, _ = result[-1]
                result[-1] = (prev_text + chunk, prev_start, start + chunk_end)
            else:
                # First chunk too short (can happen at atom start) - include
                result.append((chunk, start + pos, start + chunk_end))

            pos = chunk_end
            if pos >= len(text):
                break

    return result
