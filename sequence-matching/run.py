import argparse
import os
import itertools
import yaml
import pandas as pd
from tqdm import tqdm
from Bio.Align import PairwiseAligner


def smith_waterman_waterfall(text_a, text_b, match_score=1.0, mismatch_score=-1.5,
                              open_gap_score=-1.0, extend_gap_score=-1.0,
                              min_score=15.0, max_iterations=100):
    aligner = PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = match_score
    aligner.mismatch_score = mismatch_score
    aligner.open_gap_score = open_gap_score
    aligner.extend_gap_score = extend_gap_score

    matches = []
    curr_a = list(text_a)
    curr_b = list(text_b)

    MASK_A_CHAR = '\uFFF0'
    MASK_B_CHAR = '\uFFF1'

    pbar = tqdm(total=max_iterations, desc="Aligning", leave=False)
    for i in range(max_iterations):
        pbar.update(1)
        str_a = "".join(curr_a)
        str_b = "".join(curr_b)

        alignments = aligner.align(str_a, str_b)

        if not alignments:
            break

        best = alignments[0]

        if best.score < min_score:
            break

        pbar.set_postfix(score=f"{best.score:.1f}", matches=len(matches) + 1)

        ranges_a = best.aligned[0]
        ranges_b = best.aligned[1]

        start_a, end_a = ranges_a[0][0], ranges_a[-1][1]
        start_b, end_b = ranges_b[0][0], ranges_b[-1][1]

        match_seg_a = text_a[start_a:end_a]
        match_seg_b = text_b[start_b:end_b]

        matches.append((best.score, match_seg_a, match_seg_b))

        for k in range(start_a, end_a):
            curr_a[k] = MASK_A_CHAR
        for k in range(start_b, end_b):
            curr_b[k] = MASK_B_CHAR

    pbar.close()
    return matches


def main():
    parser = argparse.ArgumentParser(description="Sequence matching between text documents")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    algo = config["algorithm"]
    input_dir = config["input"]["dir"]
    output_path = config["output"]["path"]

    # Discover .txt files
    txt_files = sorted([
        fname for fname in os.listdir(input_dir)
        if fname.endswith(".txt")
    ])

    if len(txt_files) < 2:
        print(f"Need at least 2 .txt files in '{input_dir}', found {len(txt_files)}")
        return

    print(f"Found {len(txt_files)} files: {txt_files}")

    # Read all files
    documents = {}
    for fname in txt_files:
        with open(os.path.join(input_dir, fname), "r") as f:
            documents[fname] = f.read()

    # Run all pairwise comparisons
    all_matches = []
    pairs = list(itertools.combinations(txt_files, 2))

    for file_a, file_b in tqdm(pairs, desc="Comparing pairs"):
        matches = smith_waterman_waterfall(
            documents[file_a],
            documents[file_b],
            match_score=algo["match_score"],
            mismatch_score=algo["mismatch_score"],
            open_gap_score=algo["open_gap_score"],
            extend_gap_score=algo["extend_gap_score"],
            min_score=algo["min_score"],
            max_iterations=algo["max_iterations"],
        )

        for score, text_a, text_b in matches:
            all_matches.append({
                "file_a": file_a,
                "file_b": file_b,
                "score": score,
                "text_a": text_a,
                "text_b": text_b,
            })

    # Write results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(all_matches, columns=["file_a", "file_b", "score", "text_a", "text_b"])
    df.to_csv(output_path, index=False)
    print(f"\nWrote {len(df)} results to {output_path}")


if __name__ == "__main__":
    main()
