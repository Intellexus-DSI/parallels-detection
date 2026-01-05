"""Utilities for creating overlapping text spans from atomic segments."""


def make_overlapping_spans(
    atoms: list[tuple[str, int, int]],
    original_text: str,
    *,
    max_atoms: int = 8,
    min_chars: int = 8,
    max_chars: int = 350,
    max_spans: int = 300,
) -> list[tuple[str, int, int, int, str]]:
    """Generate overlapping spans from atomic segments.

    Creates both forward sliding windows and centered windows around each
    atomic segment to capture text at multiple scales.

    Args:
        atoms: List of (text, start, end) tuples from segmenter
        original_text: Original full text for accurate slicing
        max_atoms: Maximum number of atoms per span
        min_chars: Minimum characters per span (after strip)
        max_chars: Maximum characters per span
        max_spans: Maximum total spans to return (safety cap)

    Returns:
        List of (span_text, span_start, span_end, span_num_atoms, span_type)
    """
    if not atoms:
        return []

    spans = []
    seen_indices = set()  # For deduplication by (start, end)

    # Forward windows: for each start index i, windows of size 1..max_atoms
    for i in range(len(atoms)):
        for w in range(1, min(max_atoms + 1, len(atoms) - i + 1)):
            if i + w > len(atoms):
                break

            span_start = atoms[i][1]  # start index of first atom
            span_end = atoms[i + w - 1][2]  # end index of last atom

            # Deduplicate by indices
            if (span_start, span_end) in seen_indices:
                continue
            seen_indices.add((span_start, span_end))

            # Extract span from original text (preserves exact indices)
            span_text = original_text[span_start:span_end]

            # Apply filters
            if len(span_text.strip()) < min_chars:
                continue
            if len(span_text) > max_chars:
                continue

            spans.append((span_text.strip(), span_start, span_end, w, "forward"))

    # Centered windows: symmetric-ish patterns around each center
    center_patterns = [(1, 1), (2, 2), (3, 3), (2, 3), (3, 2)]

    for c in range(len(atoms)):
        for left_count, right_count in center_patterns:
            # For center at index c, include left_count atoms before
            # (inclusive) and right_count atoms after (inclusive)
            start_idx = max(0, c - left_count + 1)
            end_idx = min(len(atoms), c + right_count + 1)

            if start_idx >= end_idx or start_idx < 0 or end_idx > len(atoms):
                continue

            span_start = atoms[start_idx][1]
            span_end = atoms[end_idx - 1][2]

            # Deduplicate by indices
            if (span_start, span_end) in seen_indices:
                continue
            seen_indices.add((span_start, span_end))

            # Extract span from original text
            span_text = original_text[span_start:span_end]

            # Apply filters
            if len(span_text.strip()) < min_chars:
                continue
            if len(span_text) > max_chars:
                continue

            num_atoms = end_idx - start_idx
            spans.append((span_text.strip(), span_start, span_end, num_atoms, "centered"))

    # Sanity checks
    for span_text, span_start, span_end, num_atoms, span_type in spans:
        assert 0 <= span_start < span_end <= len(original_text), \
            f"Invalid indices: {span_start}, {span_end} for text length {len(original_text)}"
        assert span_text == original_text[span_start:span_end].strip(), \
            f"Span text mismatch at ({span_start}, {span_end})"

    # Enforce max_spans cap (stable order: forward first, then centered)
    if len(spans) > max_spans:
        spans = spans[:max_spans]

    # Final deduplication check (should be redundant but safe)
    final_spans = []
    final_seen = set()
    for span in spans:
        span_start, span_end = span[1], span[2]
        if (span_start, span_end) not in final_seen:
            final_seen.add((span_start, span_end))
            final_spans.append(span)

    return final_spans
