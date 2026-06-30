"""Analyze match quality and length differences (syllable or character mode outputs)."""

import csv
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "output"

DATASETS = {
    "syllable": {
        "self": OUT / "newfiles_self_syllable.csv",
        "rongzom": OUT / "newfiles_vs_rongzom_syllable.csv",
        "derge": OUT / "newfiles_vs_derge_syllable.csv",
    },
    "char": {
        "self": OUT / "newfiles_self_strict.csv",
        "rongzom": OUT / "newfiles_vs_rongzom_strict.csv",
        "derge": OUT / "newfiles_vs_derge_strict.csv",
    },
}

BAD_JACCARD = 0.35
BAD_IDENTITY = 0.45
LOW_JACCARD = 0.55
LOW_IDENTITY = 0.65
LEN_MISMATCH = 0.5
LEN_VERY_DIFF = 0.25


def syllable_tokens(text: str) -> list[str]:
    return [t for t in text.replace("/", " ").split() if t]


def jaccard(a: str, b: str) -> float:
    sa, sb = set(syllable_tokens(a)), set(syllable_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def identity_ratio(score: float, na: int, nb: int, match_score: float = 1.0) -> float:
    return score / (max(min(na, nb), 1) * match_score)


def len_ratio(na: int, nb: int) -> float:
    if na == 0 or nb == 0:
        return 0.0
    return min(na, nb) / max(na, nb)


def is_completely_different(jac: float, ident: float) -> bool:
    return jac <= BAD_JACCARD or ident <= BAD_IDENTITY


def is_good(jac: float, ident: float) -> bool:
    return jac >= LOW_JACCARD and ident >= LOW_IDENTITY


def analyze(label: str, path: Path, mode: str) -> None:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    print(f"\n=== {mode.upper()} / {label.upper()} ({len(rows)} matches) ===")
    use_char_identity = mode == "char"

    good = weak = bad = 0
    syl_exact = syl_mismatch = syl_very_diff = 0
    char_exact = char_mismatch = char_very_diff = 0
    bad_and_syl_len_diff = bad_and_char_len_diff = 0
    by_file_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for r in rows:
        sa, sb = syllable_tokens(r["text_a"]), syllable_tokens(r["text_b"])
        nsa, nsb = len(sa), len(sb)
        ca, cb = int(r["len_a"]), int(r["len_b"])
        score = float(r["score"])
        jac = jaccard(r["text_a"], r["text_b"])
        ident_units_a = ca if use_char_identity else nsa
        ident_units_b = cb if use_char_identity else nsb
        ident = identity_ratio(score, ident_units_a, ident_units_b)
        syl_lr = len_ratio(nsa, nsb)
        char_lr = len_ratio(ca, cb)

        if is_completely_different(jac, ident):
            bad += 1
        elif is_good(jac, ident):
            good += 1
        else:
            weak += 1

        if syl_lr < LEN_VERY_DIFF:
            syl_very_diff += 1
        elif syl_lr < LEN_MISMATCH:
            syl_mismatch += 1
        if nsa == nsb:
            syl_exact += 1

        if char_lr < LEN_VERY_DIFF:
            char_very_diff += 1
        elif char_lr < LEN_MISMATCH:
            char_mismatch += 1
        if ca == cb:
            char_exact += 1

        if is_completely_different(jac, ident) and syl_lr < LEN_MISMATCH:
            bad_and_syl_len_diff += 1
        if is_completely_different(jac, ident) and char_lr < LEN_MISMATCH:
            bad_and_char_len_diff += 1

        by_file_pair[(r["file_a"], r["file_b"])].append(
            {
                "jac": jac,
                "ident": ident,
                "syl_lr": syl_lr,
                "char_lr": char_lr,
                "nsa": nsa,
                "nsb": nsb,
                "ca": ca,
                "cb": cb,
            }
        )

    n = len(rows)
    unit = "char" if use_char_identity else "syllable"
    print(f"\nMATCH QUALITY (Jaccard on syllables; identity on {unit} units):")
    print(f"  Good parallels:              {good:>5}  ({100*good/n:.1f}%)")
    print(f"  Borderline / partial:        {weak:>5}  ({100*weak/n:.1f}%)")
    print(f"  Completely different text:   {bad:>5}  ({100*bad/n:.1f}%)")

    print("\nLENGTH — syllable counts in matched segments:")
    print(f"  Exact same syllable count:   {syl_exact:>5}  ({100*syl_exact/n:.1f}%)")
    print(f"  Similar (ratio >= 0.5):      {n - syl_very_diff - syl_mismatch:>5}  ({100*(n-syl_very_diff-syl_mismatch)/n:.1f}%)")
    print(f"  Different (ratio 0.25-0.5):  {syl_mismatch:>5}  ({100*syl_mismatch/n:.1f}%)")
    print(f"  Very different (ratio < 0.25): {syl_very_diff:>5}  ({100*syl_very_diff/n:.1f}%)")

    print("\nLENGTH — character counts (len_a / len_b from CSV):")
    print(f"  Exact same char count:       {char_exact:>5}  ({100*char_exact/n:.1f}%)")
    print(f"  Similar (ratio >= 0.5):      {n - char_very_diff - char_mismatch:>5}  ({100*(n-char_very_diff-char_mismatch)/n:.1f}%)")
    print(f"  Different (ratio 0.25-0.5):  {char_mismatch:>5}  ({100*char_mismatch/n:.1f}%)")
    print(f"  Very different (ratio < 0.25): {char_very_diff:>5}  ({100*char_very_diff/n:.1f}%)")
    print(f"  Bad content + char len<0.5:  {bad_and_char_len_diff:>5}  ({100*bad_and_char_len_diff/n:.1f}%)")

    fp_total = len(by_file_pair)
    fp_all_bad = fp_all_good = fp_mixed = 0
    fp_any_syl_len_diff = fp_any_char_len_diff = 0
    fp_all_char_len_diff = 0
    for matches in by_file_pair.values():
        flags_good = [is_good(m["jac"], m["ident"]) for m in matches]
        flags_bad = [is_completely_different(m["jac"], m["ident"]) for m in matches]
        if all(flags_good):
            fp_all_good += 1
        elif all(flags_bad):
            fp_all_bad += 1
        else:
            fp_mixed += 1
        if any(m["syl_lr"] < LEN_MISMATCH for m in matches):
            fp_any_syl_len_diff += 1
        if any(m["char_lr"] < LEN_MISMATCH for m in matches):
            fp_any_char_len_diff += 1
        if all(m["char_lr"] < LEN_MISMATCH for m in matches):
            fp_all_char_len_diff += 1

    print(f"\nFILE PAIRS with >=1 match ({fp_total} pairs):")
    print(f"  All matches good:            {fp_all_good:>5}  ({100*fp_all_good/fp_total:.1f}%)")
    print(f"  Mixed good + weak/bad:       {fp_mixed:>5}  ({100*fp_mixed/fp_total:.1f}%)")
    print(f"  All matches completely diff: {fp_all_bad:>5}  ({100*fp_all_bad/fp_total:.1f}%)")
    print(f"  At least one syl len mismatch:{fp_any_syl_len_diff:>4}  ({100*fp_any_syl_len_diff/fp_total:.1f}%)")
    print(f"  At least one char len mismatch:{fp_any_char_len_diff:>3}  ({100*fp_any_char_len_diff/fp_total:.1f}%)")

    print("\nExamples — completely different:")
    shown = 0
    for r in rows:
        sa, sb = syllable_tokens(r["text_a"]), syllable_tokens(r["text_b"])
        jac = jaccard(r["text_a"], r["text_b"])
        ca, cb = int(r["len_a"]), int(r["len_b"])
        ident = identity_ratio(
            float(r["score"]),
            ca if use_char_identity else len(sa),
            cb if use_char_identity else len(sb),
        )
        if is_completely_different(jac, ident):
            ca, cb = int(r["len_a"]), int(r["len_b"])
            print(
                f"  score={float(r['score']):.1f} j={jac:.2f} id={ident:.2f} "
                f"syl={len(sa)}/{len(sb)} char={ca}/{cb} char_ratio={len_ratio(ca, cb):.2f}"
            )
            print(f"    A: {r['text_a'][:100]}...")
            print(f"    B: {r['text_b'][:100]}...")
            shown += 1
            if shown >= 3:
                break


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "syllable"
    if mode not in DATASETS and mode != "both":
        print(f"Usage: python analyze_syllable_results.py [syllable|char|both]")
        sys.exit(1)

    modes = ["syllable", "char"] if mode == "both" else [mode]
    print("Criteria:")
    print("  'Completely different' = syllable Jaccard <= 0.35 OR identity <= 45%")
    print("    (identity uses syllable units for syllable mode, char units for char mode)")
    print("  'Very different length' = min/max ratio < 0.25 (4x+ gap)")
    print("  'Different length' = ratio < 0.5 (2x+ gap)")

    for m in modes:
        for label, path in DATASETS[m].items():
            if path.exists():
                analyze(label, path, m)
            else:
                print(f"\n=== {m.upper()} / {label.upper()}: MISSING ({path.name}) ===")


if __name__ == "__main__":
    main()
