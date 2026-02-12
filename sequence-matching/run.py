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
# 2. SEEDING — find exact k-mer matches
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


def find_seeds(dense_a, dense_b, k=15, max_kmer_hits=None):
    """
    Find all (pos_a, pos_b) pairs where dense_a[pos_a:pos_a+k] == dense_b[pos_b:pos_b+k].
    If max_kmer_hits is set, skip k-mers that appear more than that many times.
    """
    # Optimization: Build index on the shorter text to save RAM
    if len(dense_a) <= len(dense_b):
        short_text, long_text = dense_a, dense_b
        swapped = False
    else:
        short_text, long_text = dense_b, dense_a
        swapped = True

    index = build_kmer_index(short_text, k)

    # Pre-filter: remove k-mers that appear too many times in the short text
    if max_kmer_hits:
        index = {kmer: positions for kmer, positions in index.items()
                 if len(positions) <= max_kmer_hits}

    # Count k-mer frequencies in the long text to filter from that side too
    if max_kmer_hits:
        long_kmer_counts = defaultdict(int)
        for j in range(len(long_text) - k + 1):
            kmer = long_text[j:j + k]
            if kmer in index:
                long_kmer_counts[kmer] += 1
        # Remove k-mers that appear too many times in the long text
        for kmer, count in long_kmer_counts.items():
            if count > max_kmer_hits:
                del index[kmer]

    seeds = []
    for j in range(len(long_text) - k + 1):
        kmer = long_text[j:j + k]
        if kmer in index:
            for i in index[kmer]:
                if swapped:
                    seeds.append((j, i))
                else:
                    seeds.append((i, j))
    
    seeds.sort()
    return seeds


# ---------------------------------------------------------------------------
# 3. MERGING — The Critical Fix (Diagonal Lock + Slicer)
# ---------------------------------------------------------------------------

def merge_seeds_into_regions(seeds, k, max_gap=100, extend=200):
    """
    Group seeds into candidate regions, enforcing diagonal consistency to prevent 
    drifting into massive noise regions, and enforcing a size limit.
    """
    if not seeds:
        return []

    # 1. Sort primarily by Diagonal (Start - End), secondarily by Position
    # This naturally groups matches that lie on the same narrative "Lane".
    seeds.sort(key=lambda s: (s[0] - s[1], s[0]))

    clusters = []
    curr_seeds = [seeds[0]]
    ref_diag = seeds[0][0] - seeds[0][1]

    # --- SAFETY LIMIT: Cut region if it gets too long ---
    # 5,000 chars = ~25 million cells = ~1 second to process.
    MAX_REGION_LENGTH = 5000 

    for i in range(1, len(seeds)):
        seed = seeds[i]
        prev = curr_seeds[-1]
        
        curr_diag = seed[0] - seed[1]
        
        # 1. Gap Check (Connectivity)
        dist_ok = (seed[0] - prev[0]) <= max_gap
        
        # 2. Diagonal Check (Drift prevention)
        # Prevents "Daisy Chaining" across the text structure
        diag_ok = abs(curr_diag - ref_diag) <= max_gap
        
        # 3. Size Check (Hard Limit)
        # Prevents infinite regions even if the text matches perfectly
        len_ok = (seed[0] - curr_seeds[0][0]) < MAX_REGION_LENGTH

        if dist_ok and diag_ok and len_ok:
            curr_seeds.append(seed)
        else:
            clusters.append(curr_seeds)
            curr_seeds = [seed]
            ref_diag = curr_diag # Reset lane

    clusters.append(curr_seeds)

    regions = []
    for cluster in clusters:
        min_a = min(s[0] for s in cluster)
        max_a = max(s[0] for s in cluster) + k
        min_b = min(s[1] for s in cluster)
        max_b = max(s[1] for s in cluster) + k

        start_a = max(0, min_a - extend)
        end_a = max_a + extend
        start_b = max(0, min_b - extend)
        end_b = max_b + extend

        regions.append((start_a, end_a, start_b, end_b))

    return regions


# ---------------------------------------------------------------------------
# 4. ALIGNMENT — The Overflow Fix
# ---------------------------------------------------------------------------

def align_region(dense_a, dense_b, aligner, min_score=15.0, max_iterations=100):
    """
    Run iterative Smith-Waterman on a pair of dense text regions.
    """
    work_a = list(dense_a)
    work_b = list(dense_b)
    MASK_CHAR = '\uFFF0'
    matches = []

    for _ in range(max_iterations):
        active_a = [(j, c) for j, c in enumerate(work_a) if c != MASK_CHAR]
        active_b = [(j, c) for j, c in enumerate(work_b) if c != MASK_CHAR]

        if not active_a or not active_b:
            break

        local_map_a = [j for j, c in active_a]
        local_dense_a = "".join(c for j, c in active_a)
        local_map_b = [j for j, c in active_b]
        local_dense_b = "".join(c for j, c in active_b)

        alignments = aligner.align(local_dense_a, local_dense_b)
        
        # --- FIX: Don't count alignments, just take the first ---
        # Prevents OverflowError on highly repetitive text
        iterator = iter(alignments)
        try:
            best = next(iterator)
        except StopIteration:
            break

        if best.score < min_score:
            break

        ranges_a = best.aligned[0]
        ranges_b = best.aligned[1]

        ds_a, de_a = ranges_a[0][0], ranges_a[-1][1]
        ds_b, de_b = ranges_b[0][0], ranges_b[-1][1]

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
# 5. MAIN PIPELINE
# ---------------------------------------------------------------------------

def seed_and_extend(text_a, text_b, match_score=1.0, mismatch_score=-1.5,
                    open_gap_score=-1.0, extend_gap_score=-1.0,
                    min_score=15.0, max_iterations=100,
                    seed_k=15, seed_max_gap=100, seed_extend=200,
                    seed_max_kmer_hits=None, strip_chars=""):
    
    dense_a, map_a = build_alignment_map(text_a, strip_chars=strip_chars)
    dense_b, map_b = build_alignment_map(text_b, strip_chars=strip_chars)

    print(f"Text A: {len(text_a)} original -> {len(dense_a)} dense chars")
    print(f"Text B: {len(text_b)} original -> {len(dense_b)} dense chars")

    print("Finding seeds...")
    seeds = find_seeds(dense_a, dense_b, k=seed_k, max_kmer_hits=seed_max_kmer_hits)
    print(f"Found {len(seeds)} seed hits")

    if not seeds:
        return []

    regions = merge_seeds_into_regions(seeds, k=seed_k, max_gap=seed_max_gap, extend=seed_extend)
    
    # Clamp to text boundaries
    regions = [
        (sa, min(ea, len(dense_a)), sb, min(eb, len(dense_b)))
        for sa, ea, sb, eb in regions
    ]
    print(f"Merged into {len(regions)} candidate regions")

    # DEBUG: Check sizes
    sizes = [(ea - sa) * (eb - sb) for sa, ea, sb, eb in regions]
    sizes.sort(reverse=True)
    if sizes:
        print(f"Top 5 region sizes (cells): {sizes[:5]}")

    aligner = PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = match_score
    aligner.mismatch_score = mismatch_score
    aligner.open_gap_score = open_gap_score
    aligner.extend_gap_score = extend_gap_score

    all_matches = []

    for sa, ea, sb, eb in tqdm(regions, desc="Aligning regions"):
        # SAFETY VALVE: Skip massive regions (just in case)
        area = (ea - sa) * (eb - sb)
        if area > 50_000_000: # 50 Million cells limit (approx 2-3 sec)
             # Optional: Log warning if skipping
             continue

        region_a = dense_a[sa:ea]
        region_b = dense_b[sb:eb]

        region_matches = align_region(region_a, region_b, aligner,
                                      min_score=min_score,
                                      max_iterations=max_iterations)

        for score, rs_a, re_a, rs_b, re_b in region_matches:
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

    # Deduplicate matches
    seen = set()
    unique_matches = []
    for item in all_matches:
        key = (item[3], item[4], item[5], item[6])
        if key not in seen:
            seen.add(key)
            unique_matches.append(item)
    all_matches = unique_matches

    print(f"Found {len(all_matches)} total matches")
    all_matches.sort(key=lambda x: -x[0])
    return all_matches


# ---------------------------------------------------------------------------
# 6. ORIGINAL FULL-TEXT FALLBACK (kept for legacy)
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
        
        # --- FIX: Overflow protection ---
        iterator = iter(alignments)
        try:
            best = next(iterator)
        except StopIteration:
            break
            
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

        # Simple masking (not index-mapped) for legacy waterfall
        idx = 0
        for j, c in enumerate(work_a):
            if c == MASK_CHAR: continue
            if dense_start_a <= idx < dense_end_a: work_a[j] = MASK_CHAR
            idx += 1
        idx = 0
        for j, c in enumerate(work_b):
            if c == MASK_CHAR: continue
            if dense_start_b <= idx < dense_end_b: work_b[j] = MASK_CHAR
            idx += 1

    pbar.close()
    return matches


# ---------------------------------------------------------------------------
# 7. CLI ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", default="seed")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    algo = config["algorithm"]
    output_path = config["output"]["path"]

    seed_cfg = config.get("seeding", {})
    seed_k = seed_cfg.get("k", 15)
    seed_max_gap = seed_cfg.get("max_gap", 100)
    seed_extend = seed_cfg.get("extend", 200)
    seed_max_kmer_hits = seed_cfg.get("max_kmer_hits", None)
    strip_chars = config.get("preprocessing", {}).get("strip_chars", "")

    input_cfg = config["input"]
    documents = {}

    if "dir" in input_cfg:
        # Original mode: all-pairs from a single directory
        input_dir = input_cfg["dir"]
        txt_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".txt")])
        for fname in txt_files:
            with open(os.path.join(input_dir, fname), "r") as f:
                documents[fname] = f.read()
        pairs = list(itertools.combinations(txt_files, 2))
        print(f"Single-directory mode: {len(txt_files)} files, {len(pairs)} pairs")
    elif "point_of_comparison" in input_cfg and "corpus" in input_cfg:
        # Cross-comparison mode: point_of_comparison × corpus (recursive)
        poc_dir = input_cfg["point_of_comparison"]
        corpus_dir = input_cfg["corpus"]
        poc_files = sorted([f for f in os.listdir(poc_dir) if f.endswith(".txt")])
        corpus_files = []
        for root, dirs, files in os.walk(corpus_dir):
            dirs.sort()
            for f in sorted(files):
                if f.endswith(".txt"):
                    corpus_files.append(os.path.relpath(os.path.join(root, f), corpus_dir))
        for fname in poc_files:
            key = f"poc/{fname}"
            with open(os.path.join(poc_dir, fname), "r") as f:
                documents[key] = f.read()
        for rel_path in corpus_files:
            key = f"corpus/{rel_path}"
            with open(os.path.join(corpus_dir, rel_path), "r") as f:
                documents[key] = f.read()
        pairs = list(itertools.product(
            [f"poc/{f}" for f in poc_files],
            [f"corpus/{p}" for p in corpus_files],
        ))
        print(f"Cross-comparison mode: {len(poc_files)} point_of_comparison × {len(corpus_files)} corpus = {len(pairs)} pairs")
    else:
        raise ValueError("Config 'input' must specify either 'dir' or both 'point_of_comparison' and 'corpus'")

    all_matches = []

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
                seed_max_kmer_hits=seed_max_kmer_hits,
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