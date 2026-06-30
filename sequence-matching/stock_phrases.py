"""Classify Tibetan tantra stock phrases in sequence-matching output."""

import re

CONCLUDING_MAX_LEN = 400

_LEADING_STROKES = re.compile(r"^[\s/_]+")

_OPENING_SIGNALS = (
    re.compile(r"bod\s+skad\s+du", re.I),
    re.compile(r"rgya\s+gar\s+skad\s+du", re.I),
    re.compile(r"ces\s+bya\s+ba", re.I),
    re.compile(r"'di\s+skad\s+bdag", re.I),
)
_PHYAG_TSHAL_LO = re.compile(r"phyag\s+'tshal\s+lo", re.I)
_PHYAG_TSHAL_OTHER = re.compile(r"phyag\s+'tshal\s+(?:nas|'dud)\b", re.I)
_OPENING_FORMULA = re.compile(
    r"phyag\s+'tshal\s+lo.*?('di\s+skad\s+bdag|bcom\s+ldan\s+'das)",
    re.I | re.DOTALL,
)

_LEU = re.compile(r"le'u", re.I)
_LEU_PAO = re.compile(r"le'u(?:\s+\w+)*\s+pa'o", re.I)
_LEU_PAO_DE_NAS = re.compile(
    r"le'u.*pa'o.*de\s+nas\s+bcom\s+ldan",
    re.I | re.DOTALL,
)
_GSUNGS_LEU = re.compile(
    r"(?:gsungs\s+so|zhes\s+gsungs\s+so).*le'u.*pa'o",
    re.I | re.DOTALL,
)
_DE_NAS_BYANG_CHUB = re.compile(r"de\s+nas\s+byang\s+chub\s+sems\s+dpa'", re.I)

_RGYUD_RDZOGS = re.compile(r"(?:[\w+\-'.\s]+?\s+)?rgyud\s+rdzogs\s+so", re.I)
_CES_BYA_RDZOGS = re.compile(r"ces\s+bya\s+ba\s+rdzogs\s+so", re.I)
_LEU_RDZOGS = re.compile(r"le'u.*rdzogs\s+so", re.I | re.DOTALL)

STOCK_PHRASE_TYPES = ("opening", "chapter_ending", "concluding")


def normalize_for_stock_check(text: str) -> str:
    """Strip leading strokes/markers; keep trailing strokes in the match."""
    if not text:
        return ""
    return _LEADING_STROKES.sub("", text)


def _is_opening(text: str) -> bool:
    if not _PHYAG_TSHAL_LO.search(text):
        return False
    if _PHYAG_TSHAL_OTHER.search(text):
        return False
    if any(p.search(text) for p in _OPENING_SIGNALS):
        return True
    return bool(_OPENING_FORMULA.search(text))


def _is_chapter_ending(text: str) -> bool:
    if not _LEU.search(text):
        return False
    if _DE_NAS_BYANG_CHUB.search(text) and not re.search(
        r"de\s+nas\s+bcom\s+ldan", text, re.I
    ):
        return False
    if _GSUNGS_LEU.search(text):
        return True
    if _LEU_PAO.search(text):
        return True
    if _LEU_PAO_DE_NAS.search(text):
        return True
    return False


def _is_concluding(text: str) -> bool:
    if _RGYUD_RDZOGS.search(text):
        return True
    if _CES_BYA_RDZOGS.search(text):
        return True
    if len(text) <= CONCLUDING_MAX_LEN and _LEU_RDZOGS.search(text):
        return True
    return False


def classify_stock_phrase(text: str) -> tuple[bool, str | None]:
    """
    Return (is_stock_phrase, type) where type is one of opening,
    chapter_ending, concluding, or None.
    """
    normalized = normalize_for_stock_check(text)
    if not normalized:
        return False, None

    if _is_opening(normalized):
        return True, "opening"
    if _is_concluding(normalized):
        return True, "concluding"
    if _is_chapter_ending(normalized):
        return True, "chapter_ending"
    return False, None


def classify_match_row(text_a: str, text_b: str) -> tuple[bool, str | None]:
    """True if either side is a stock phrase; type from text_a then text_b."""
    for text in (text_a, text_b):
        is_stock, phrase_type = classify_stock_phrase(text)
        if is_stock:
            return True, phrase_type
    return False, None
