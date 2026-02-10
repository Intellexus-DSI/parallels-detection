import argparse
import os
import re
import itertools
from collections import defaultdict
import yaml
import pandas as pd
from tqdm import tqdm
from Bio.Align import PairwiseAligner


# ---------------------------------------------------------------------------
# 1. TEXT PREPROCESSING
# ---------------------------------------------------------------------------

MARKER_PATTERN = re.compile(r'\[\\\[[^\]]*\\\]\]')
METADATA_PATTERN = re.compile(r'@##|@#|/_|/|\{[^}]*\}')


def build_alignment_map(text, strip_spaces=True, strip_newlines=True, strip_markers=True, strip_chars=""):
    """
    Build a dense string suitable for alignment and a coordinate map back to the original text.

    Returns:
        dense_text (str): The cleaned text with ignorable characters removed.
        coord_map (list[int]): coord_map[i] = original index of the i-th char in dense_text.
    """
    masked = [False] * len(text)

    if strip_markers:
        for m in MARKER_PATTERN.finditer(text):
            for i in range(m.start(), m.end()):
                masked[i] = True
        for m in METADATA_PATTERN.finditer(text):
            for i in range(m.start(), m.end()):
                masked[i] = True

    coord_map = []
    chars = []
    for i, ch in enumerate(text):
        if masked[i]:
            continue
        if strip_spaces and ch == ' ':
            continue
        if strip_newlines and ch == '\n':
            continue
        if ch in strip_chars:
            continue
        chars.append(ch)
        coord_map.append(i)

    dense_text = "".join(chars)
    return dense_text, coord_map


def dense_range_to_original(coord_map, dense_start, dense_end):
    """
    Convert a [start, end) range in dense-string coordinates back to
    original-text coordinates.
    """
    orig_start = coord_map[dense_start]
    orig_end = coord_map[dense_end - 1] + 1
    return orig_start, orig_end


# ---------------------------------------------------------------------------
# 2. SEEDING — find exact k-mer matches between two dense texts
# ---------------------------------------------------------------------------

def build_kmer_index(text, k):
    """
    Build a dictionary mapping each k-mer to a list of start positions in `text`.
    """
    index = defaultdict(list)
    for i in range(len(text) - k + 1):
        kmer = text[i:i + k]
        index[kmer].append(i)
    return index


def find_seeds(dense_a, dense_b, k=15):
    """
    Find all (pos_a, pos_b) pairs where dense_a[pos_a:pos_a+k] == dense_b[pos_b:pos_b+k].

    Returns:
        list of (pos_a, pos_b) tuples, sorted by pos_a.
    """
    # Index the shorter text, scan the longer one
    if len(dense_a) <= len(dense_b):
        index = build_kmer_index(dense_a, k)
        seeds = []
        for j in range(len(dense_b) - k + 1):
            kmer = dense_b[j:j + k]
            if kmer in index:
                for i in index[kmer]:
                    seeds.append((i, j))
    else:
        index = build_kmer_index(dense_b, k)
        seeds = []
        for i in range(len(dense_a) - k + 1):
            kmer = dense_a[i:i + k]
            if kmer in index:
                for j in index[kmer]:
                    seeds.append((i, j))

    seeds.sort()
    return seeds


# ---------------------------------------------------------------------------
# 3. MERGING — cluster nearby seeds into candidate regions
# ---------------------------------------------------------------------------

def merge_seeds_into_regions(seeds, k, max_gap=100, extend=200):
    """
    Group seeds that are close together (on approximately the same diagonal)
    into candidate regions for alignment.

    Two seeds are merged if:
      - They are on a similar diagonal (|diag_1 - diag_2| <= max_gap)
      - They are close along text A (gap in pos_a <= max_gap)

    Each merged cluster is then expanded by `extend` characters on each side
    to give Smith-Waterman room to find the full match.

    Returns:
        list of (start_a, end_a, start_b, end_b) tuples — dense-coordinate regions.
    """
    if not seeds:
        return []

    # Sort seeds by diagonal, then by pos_a
    diag_seeds = sorted(seeds, key=lambda s: (s[0] - s[1], s[0]))

    clusters = []
    curr_seeds = [diag_seeds[0]]

    for seed in diag_seeds[1:]:
        prev = curr_seeds[-1]
        diag_prev = prev[0] - prev[1]
        diag_curr = seed[0] - seed[1]

        # Same cluster if close diagonal and close position
        if abs(diag_curr - diag_prev) <= max_gap and (seed[0] - prev[0]) <= max_gap:
            curr_seeds.append(seed)
        else:
            clusters.append(curr_seeds)
            curr_seeds = [seed]

    clusters.append(curr_seeds)

    # Convert clusters to regions with extension
    regions = []
    for cluster in clusters:
        min_a = min(s[0] for s in cluster)
        max_a = max(s[0] for s in cluster) + k
        min_b = min(s[1] for s in cluster)
        max_b = max(s[1] for s in cluster) + k

        # Extend boundaries
        start_a = max(0, min_a - extend)
        end_a = max_a + extend  # will be clamped later
        start_b = max(0, min_b - extend)
        end_b = max_b + extend

        regions.append((start_a, end_a, start_b, end_b))

    # Merge overlapping regions (only if on similar diagonals)
    regions.sort()
    merged = [regions[0]]
    for r in regions[1:]:
        prev = merged[-1]
        a_overlaps = r[0] <= prev[1]
        b_overlaps = r[2] <= prev[3]
        diag_prev = (prev[0] + prev[1]) // 2 - (prev[2] + prev[3]) // 2
        diag_curr = (r[0] + r[1]) // 2 - (r[2] + r[3]) // 2
        similar_diag = abs(diag_curr - diag_prev) <= max_gap

        if a_overlaps and b_overlaps and similar_diag:
            merged[-1] = (
                min(prev[0], r[0]),
                max(prev[1], r[1]),
                min(prev[2], r[2]),
                max(prev[3], r[3]),
            )
        else:
            merged.append(r)

    return merged


# ---------------------------------------------------------------------------
# 4. SMITH-WATERMAN on a single region pair (reusable core)
# ---------------------------------------------------------------------------

def align_region(dense_a, dense_b, aligner, min_score=15.0, max_iterations=100):
    """
    Run iterative Smith-Waterman on a pair of dense text regions.

    Returns:
        list of (score, dense_start_a, dense_end_a, dense_start_b, dense_end_b)
    """
    work_a = list(dense_a)
    work_b = list(dense_b)
    MASK_CHAR = '\uFFF0'
    matches = []

    for _ in range(max_iterations):
        # Build iteration-local strings skipping masked chars
        active_a = [(j, c) for j, c in enumerate(work_a) if c != MASK_CHAR]
        active_b = [(j, c) for j, c in enumerate(work_b) if c != MASK_CHAR]

        if not active_a or not active_b:
            break

        local_map_a = [j for j, c in active_a]
        local_dense_a = "".join(c for j, c in active_a)

        local_map_b = [j for j, c in active_b]
        local_dense_b = "".join(c for j, c in active_b)

        alignments = aligner.align(local_dense_a, local_dense_b)
        if not alignments:
            break

        best = alignments[0]
        if best.score < min_score:
            break

        ranges_a = best.aligned[0]
        ranges_b = best.aligned[1]

        ds_a, de_a = ranges_a[0][0], ranges_a[-1][1]
        ds_b, de_b = ranges_b[0][0], ranges_b[-1][1]

        # Translate local indices -> work array indices
        work_start_a = local_map_a[ds_a]
        work_end_a = local_map_a[de_a - 1] + 1
        work_start_b = local_map_b[ds_b]
        work_end_b = local_map_b[de_b - 1] + 1

        matches.append((best.score, work_start_a, work_end_a, work_start_b, work_end_b))

        # Mask matched characters
        for j in range(work_start_a, work_end_a):
            work_a[j] = MASK_CHAR
        for j in range(work_start_b, work_end_b):
            work_b[j] = MASK_CHAR

    return matches


# ---------------------------------------------------------------------------
# 5. MAIN PIPELINE — shrink -> seed -> merge -> extend -> map back
# ---------------------------------------------------------------------------

def seed_and_extend(text_a, text_b,
                    match_score=1.0, mismatch_score=-1.5,
                    open_gap_score=-1.0, extend_gap_score=-1.0,
                    min_score=15.0, max_iterations=100,
                    seed_k=15, seed_max_gap=100, seed_extend=200,
                    strip_chars=""):
    """
    Full pipeline: preprocess -> seed -> merge -> align regions -> map back.
    """
    # --- Preprocess ---
    dense_a, map_a = build_alignment_map(text_a, strip_chars=strip_chars)
    dense_b, map_b = build_alignment_map(text_b, strip_chars=strip_chars)

    print(f"Text A: {len(text_a)} original -> {len(dense_a)} dense chars")
    print(f"Text B: {len(text_b)} original -> {len(dense_b)} dense chars")

    # --- Seed ---
    print("Finding seeds...")
    seeds = find_seeds(dense_a, dense_b, k=seed_k)
    print(f"Found {len(seeds)} seed hits")

    # Count unique positions touched in each text
    unique_a = len(set(s[0] for s in seeds))
    unique_b = len(set(s[1] for s in seeds))
    print(f"Found {len(seeds)} seed pairs ({unique_a} unique positions in A, {unique_b} in B)")

    if not seeds:
        return []

    # --- Merge ---
    regions = merge_seeds_into_regions(seeds, k=seed_k,
                                       max_gap=seed_max_gap,
                                       extend=seed_extend)
    # Clamp to text boundaries
    regions = [
        (sa, min(ea, len(dense_a)), sb, min(eb, len(dense_b)))
        for sa, ea, sb, eb in regions
    ]
    print(f"Merged into {len(regions)} candidate regions")

    total_a_chars = sum(ea - sa for sa, ea, _, _ in regions)
    total_b_chars = sum(eb - sb for _, _, sb, eb in regions)
    print(f"Total alignment work: {total_a_chars} + {total_b_chars} region chars "
          f"(vs {len(dense_a)} + {len(dense_b)} full)")

    # --- Align each region ---
    aligner = PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = match_score
    aligner.mismatch_score = mismatch_score
    aligner.open_gap_score = open_gap_score
    aligner.extend_gap_score = extend_gap_score

    all_matches = []

    for sa, ea, sb, eb in tqdm(regions, desc="Aligning regions"):
        region_a = dense_a[sa:ea]
        region_b = dense_b[sb:eb]

        region_matches = align_region(region_a, region_b, aligner,
                                      min_score=min_score,
                                      max_iterations=max_iterations)

        for score, rs_a, re_a, rs_b, re_b in region_matches:
            # Region-local -> dense-global -> original coordinates
            glob_start_a = sa + rs_a
            glob_end_a = sa + re_a
            glob_start_b = sb + rs_b
            glob_end_b = sb + re_b

            orig_start_a, orig_end_a = dense_range_to_original(map_a, glob_start_a, glob_end_a)
            orig_start_b, orig_end_b = dense_range_to_original(map_b, glob_start_b, glob_end_b)

            match_seg_a = text_a[orig_start_a:orig_end_a]
            match_seg_b = text_b[orig_start_b:orig_end_b]

            all_matches.append((score, match_seg_a, match_seg_b,
                                orig_start_a, orig_end_a, orig_start_b, orig_end_b))

    # Deduplicate matches (same position found by overlapping regions)
    seen = set()
    unique_matches = []
    for item in all_matches:
        key = (item[3], item[4], item[5], item[6])  # start_a, end_a, start_b, end_b
        if key not in seen:
            seen.add(key)
            unique_matches.append(item)
    all_matches = unique_matches

    print(f"Found {len(all_matches)} total matches across {len(regions)} regions")
    # Sort by score descending
    all_matches.sort(key=lambda x: -x[0])
    return all_matches


# ---------------------------------------------------------------------------
# 6. ORIGINAL FULL-TEXT FALLBACK (kept for comparison / small texts)
# ---------------------------------------------------------------------------

def smith_waterman_waterfall(text_a, text_b, match_score=1.0, mismatch_score=-1.5,
                              open_gap_score=-1.0, extend_gap_score=-1.0,
                              min_score=15.0, max_iterations=100,
                              strip_chars=""):
    aligner = PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = match_score
    aligner.mismatch_score = mismatch_score
    aligner.open_gap_score = open_gap_score
    aligner.extend_gap_score = extend_gap_score

    matches = []
    dense_a, map_a = build_alignment_map(text_a, strip_chars=strip_chars)
    dense_b, map_b = build_alignment_map(text_b, strip_chars=strip_chars)

    print(f"Text A: {len(text_a)} original chars -> {len(dense_a)} alignment chars")
    print(f"Text B: {len(text_b)} original chars -> {len(dense_b)} alignment chars")

    work_a = list(dense_a)
    work_b = list(dense_b)
    MASK_CHAR = '\uFFF0'

    pbar = tqdm(total=max_iterations, desc="Aligning", leave=False)
    for i in range(max_iterations):
        pbar.update(1)

        iter_map_a = [map_a[j] for j, c in enumerate(work_a) if c != MASK_CHAR]
        iter_dense_a = "".join(c for c in work_a if c != MASK_CHAR)
        iter_map_b = [map_b[j] for j, c in enumerate(work_b) if c != MASK_CHAR]
        iter_dense_b = "".join(c for c in work_b if c != MASK_CHAR)

        alignments = aligner.align(iter_dense_a, iter_dense_b)
        if not alignments:
            break
        best = alignments[0]
        if best.score < min_score:
            break

        pbar.set_postfix(score=f"{best.score:.1f}", matches=len(matches) + 1)

        ranges_a = best.aligned[0]
        ranges_b = best.aligned[1]
        dense_start_a, dense_end_a = ranges_a[0][0], ranges_a[-1][1]
        dense_start_b, dense_end_b = ranges_b[0][0], ranges_b[-1][1]

        orig_start_a, orig_end_a = dense_range_to_original(iter_map_a, dense_start_a, dense_end_a)
        orig_start_b, orig_end_b = dense_range_to_original(iter_map_b, dense_start_b, dense_end_b)

        match_seg_a = text_a[orig_start_a:orig_end_a]
        match_seg_b = text_b[orig_start_b:orig_end_b]
        matches.append((best.score, match_seg_a, match_seg_b,
                        orig_start_a, orig_end_a, orig_start_b, orig_end_b))

        idx = 0
        for j, c in enumerate(work_a):
            if c == MASK_CHAR:
                continue
            if dense_start_a <= idx < dense_end_a:
                work_a[j] = MASK_CHAR
            idx += 1
        idx = 0
        for j, c in enumerate(work_b):
            if c == MASK_CHAR:
                continue
            if dense_start_b <= idx < dense_end_b:
                work_b[j] = MASK_CHAR
            idx += 1

    pbar.close()
    return matches


# ---------------------------------------------------------------------------
# 7. CLI ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sequence matching between text documents")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--mode", choices=["seed", "full"], default="seed",
                        help="'seed' for seed-and-extend (fast), 'full' for original waterfall")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    algo = config["algorithm"]
    input_dir = config["input"]["dir"]
    output_path = config["output"]["path"]

    # Seed-and-extend parameters (with defaults)
    seed_cfg = config.get("seeding", {})
    seed_k = seed_cfg.get("k", 15)
    seed_max_gap = seed_cfg.get("max_gap", 100)
    seed_extend = seed_cfg.get("extend", 200)

    preproc_cfg = config.get("preprocessing", {})
    strip_chars = preproc_cfg.get("strip_chars", "")

    txt_files = sorted([
        fname for fname in os.listdir(input_dir)
        if fname.endswith(".txt")
    ])

    if len(txt_files) < 2:
        print(f"Need at least 2 .txt files in '{input_dir}', found {len(txt_files)}")
        return

    print(f"Found {len(txt_files)} files: {txt_files}")
    print(f"Mode: {args.mode}")

    documents = {}
    for fname in txt_files:
        with open(os.path.join(input_dir, fname), "r") as f:
            documents[fname] = f.read()

    all_matches = []
    pairs = list(itertools.combinations(txt_files, 2))

    for file_a, file_b in tqdm(pairs, desc="Comparing pairs"):
        print(f"\n--- {file_a} vs {file_b} ---")

        if args.mode == "seed":
            matches = seed_and_extend(
                documents[file_a],
                documents[file_b],
                match_score=algo["match_score"],
                mismatch_score=algo["mismatch_score"],
                open_gap_score=algo["open_gap_score"],
                extend_gap_score=algo["extend_gap_score"],
                min_score=algo["min_score"],
                max_iterations=algo["max_iterations"],
                seed_k=seed_k,
                seed_max_gap=seed_max_gap,
                seed_extend=seed_extend,
                strip_chars=strip_chars,
            )
        else:
            matches = smith_waterman_waterfall(
                documents[file_a],
                documents[file_b],
                match_score=algo["match_score"],
                mismatch_score=algo["mismatch_score"],
                open_gap_score=algo["open_gap_score"],
                extend_gap_score=algo["extend_gap_score"],
                min_score=algo["min_score"],
                max_iterations=algo["max_iterations"],
                strip_chars=strip_chars,
            )

        for score, text_a, text_b, start_a, end_a, start_b, end_b in matches:
            all_matches.append({
                "file_a": file_a,
                "file_b": file_b,
                "score": score,
                "text_a": text_a,
                "text_b": text_b,
                "start_a": start_a,
                "end_a": end_a,
                "start_b": start_b,
                "end_b": end_b,
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(all_matches, columns=["file_a", "file_b", "score",
                                             "text_a", "text_b",
                                             "start_a", "end_a",
                                             "start_b", "end_b"])
    df.to_csv(output_path, index=False)
    print(f"\nWrote {len(df)} results to {output_path}")


if __name__ == "__main__":
    main()