"""Granular length + quality analysis for char vs syllable match outputs."""

import csv
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


def syllable_tokens(text: str) -> list[str]:
    return [t for t in text.replace("/", " ").split() if t]


def jaccard(a: str, b: str) -> float:
    sa, sb = set(syllable_tokens(a)), set(syllable_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def identity_ratio(score: float, na: int, nb: int) -> float:
    return score / (max(min(na, nb), 1) * 1.0)


def len_ratio(na: int, nb: int) -> float:
    if na == 0 or nb == 0:
        return 0.0
    return min(na, nb) / max(na, nb)


def quality_tier(jac: float, ident: float) -> str:
    """Four-way content quality bucket."""
    if jac <= 0.35 or ident <= 0.45:
        return "completely_different"
    if jac >= 0.55 and ident >= 0.65:
        return "good"
    if jac >= 0.45 and ident >= 0.55:
        return "somewhat_different"
    return "partial_weak"


def length_tier(ratio: float) -> str:
    if ratio >= 0.95:
        return "nearly_equal (0.95-1.0)"
    if ratio >= 0.85:
        return "slight_diff (0.85-0.95)"
    if ratio >= 0.70:
        return "moderate_diff (0.70-0.85)"
    if ratio >= 0.50:
        return "noticeable_diff (0.50-0.70)"
    if ratio >= 0.25:
        return "large_diff (0.25-0.50)"
    return "extreme_diff (<0.25)"


def analyze_rows(rows: list[dict], mode: str) -> dict:
    use_char = mode == "char"
    stats = {
        "n": len(rows),
        "quality": defaultdict(int),
        "syl_length": defaultdict(int),
        "char_length": defaultdict(int),
        "cross": defaultdict(int),  # quality x length tier
    }

    for r in rows:
        nsa, nsb = len(syllable_tokens(r["text_a"])), len(syllable_tokens(r["text_b"]))
        ca, cb = int(r["len_a"]), int(r["len_b"])
        score = float(r["score"])
        jac = jaccard(r["text_a"], r["text_b"])
        ident = identity_ratio(score, ca if use_char else nsa, cb if use_char else nsb)
        q = quality_tier(jac, ident)
        syl_lt = length_tier(len_ratio(nsa, nsb))
        char_lt = length_tier(len_ratio(ca, cb))

        stats["quality"][q] += 1
        stats["syl_length"][syl_lt] += 1
        stats["char_length"][char_lt] += 1
        stats["cross"][(q, syl_lt)] += 1

    return stats


def pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "0%"


def print_quality(stats: dict) -> None:
    n = stats["n"]
    order = ["good", "somewhat_different", "partial_weak", "completely_different"]
    labels = {
        "good": "Good",
        "somewhat_different": "Somewhat different",
        "partial_weak": "Partial / weak",
        "completely_different": "Completely different",
    }
    print("  Content quality:")
    for k in order:
        c = stats["quality"][k]
        print(f"    {labels[k]:<22} {c:>5}  ({pct(c, n)})")


def print_lengths(stats: dict, key: str, title: str) -> None:
    n = stats["n"]
    order = [
        "nearly_equal (0.95-1.0)",
        "slight_diff (0.85-0.95)",
        "moderate_diff (0.70-0.85)",
        "noticeable_diff (0.50-0.70)",
        "large_diff (0.25-0.50)",
        "extreme_diff (<0.25)",
    ]
    print(f"  {title}:")
    for k in order:
        c = stats[key][k]
        if c:
            print(f"    {k:<28} {c:>5}  ({pct(c, n)})")


def print_cross_somewhat(stats: dict) -> None:
    """Somewhat-different broken down by length."""
    n = stats["quality"]["somewhat_different"]
    if not n:
        print("  Somewhat different + length (syllables): none")
        return
    print(f"  Somewhat different by syllable length ({n} matches):")
    order = [
        "nearly_equal (0.95-1.0)",
        "slight_diff (0.85-0.95)",
        "moderate_diff (0.70-0.85)",
        "noticeable_diff (0.50-0.70)",
        "large_diff (0.25-0.50)",
        "extreme_diff (<0.25)",
    ]
    for k in order:
        c = stats["cross"][("somewhat_different", k)]
        if c:
            print(f"    {k:<28} {c:>5}  ({pct(c, n)})")


def print_examples(rows: list[dict], mode: str, tier: str, limit: int = 2) -> None:
    use_char = mode == "char"
    shown = 0
    for r in rows:
        nsa, nsb = len(syllable_tokens(r["text_a"])), len(syllable_tokens(r["text_b"]))
        ca, cb = int(r["len_a"]), int(r["len_b"])
        jac = jaccard(r["text_a"], r["text_b"])
        ident = identity_ratio(float(r["score"]), ca if use_char else nsa, cb if use_char else nsb)
        if quality_tier(jac, ident) != tier:
            continue
        syl_r = len_ratio(nsa, nsb)
        char_r = len_ratio(ca, cb)
        print(
            f"    score={float(r['score']):.1f} j={jac:.2f} id={ident:.2f} "
            f"syl={nsa}/{nsb}({syl_r:.2f}) char={ca}/{cb}({char_r:.2f})"
        )
        print(f"      A: {r['text_a'][:95]}...")
        print(f"      B: {r['text_b'][:95]}...")
        shown += 1
        if shown >= limit:
            break


def main():
    print("Quality tiers:")
    print("  Good:               Jaccard >= 0.55 AND identity >= 65%")
    print("  Somewhat different: Jaccard 0.45-0.55 OR identity 55-65%")
    print("  Partial / weak:     below good, above somewhat-different thresholds")
    print("  Completely different: Jaccard <= 0.35 OR identity <= 45%")
    print()
    print("Length tiers (min/max ratio of syllable counts or char counts):")
    print("  nearly_equal 0.95+ | slight 0.85-0.95 | moderate 0.70-0.85")
    print("  noticeable 0.50-0.70 | large 0.25-0.50 | extreme <0.25")

    for mode in ("syllable", "char"):
        print(f"\n{'='*60}")
        print(f"  {mode.upper()} MODE")
        print(f"{'='*60}")
        for label, path in DATASETS[mode].items():
            rows = list(csv.DictReader(path.open(encoding="utf-8")))
            stats = analyze_rows(rows, mode)
            print(f"\n--- {label.upper()} ({stats['n']} matches) ---")
            print_quality(stats)
            print_lengths(stats, "syl_length", "Syllable length ratio")
            print_lengths(stats, "char_length", "Character length ratio")
            print_cross_somewhat(stats)
            if stats["quality"]["somewhat_different"]:
                print("  Examples — somewhat different:")
                print_examples(rows, mode, "somewhat_different", 2)

    # Side-by-side summary table
    print(f"\n{'='*60}")
    print("  SUMMARY TABLE (all runs)")
    print(f"{'='*60}")
    print(f"{'Run':<10} {'Mode':<9} {'N':>6} {'Good':>7} {'Somewhat':>9} {'Partial':>8} {'Bad':>7} | {'Syl≠equal':>10} {'Syl≥0.85':>9}")
    print("-" * 85)
    for mode in ("syllable", "char"):
        for label, path in DATASETS[mode].items():
            rows = list(csv.DictReader(path.open(encoding="utf-8")))
            s = analyze_rows(rows, mode)
            n = s["n"]
            ne = s["syl_length"]["nearly_equal (0.95-1.0)"]
            ge85 = ne + s["syl_length"]["slight_diff (0.85-0.95)"]
            not_equal = n - ne
            print(
                f"{label:<10} {mode:<9} {n:>6} "
                f"{pct(s['quality']['good'], n):>7} "
                f"{pct(s['quality']['somewhat_different'], n):>9} "
                f"{pct(s['quality']['partial_weak'], n):>8} "
                f"{pct(s['quality']['completely_different'], n):>7} | "
                f"{pct(not_equal, n):>10} "
                f"{pct(ge85, n):>9}"
            )
    print("\n  Syl≠equal = syllable ratio < 0.95 (any noticeable length gap)")
    print("  Syl≥0.85  = ratio >= 0.85 (mild or no length gap)")


if __name__ == "__main__":
    main()
