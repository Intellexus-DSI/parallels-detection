"""Tests for Tibetan tantra stock phrase classification."""

import pytest

from stock_phrases import classify_match_row, classify_stock_phrase, normalize_for_stock_check


OPENING_EXAMPLES = [
    "la phyag 'tshal lo/_/'di skad bdag gi thos pa'i dus gcig na/_/bcom ldan 'das",
    "rnams la phyag 'tshal lo/_/'di skad bdag gis thos pa'i dus gcig na/_bcom ldan 'das",
    "ma/bod skad du/_pad+ma dbang chen dregs pa gsang byed kyi rgyud ces bya ba//pad+ma dbang chen he ru ka pad+ma gsang ba 'i sku la phyag 'tshal lo/_/'di skad bdag gi thos pa'i dus gcig na/_/bcom ldan 'das",
    "/_la phyag 'tshal lo/_/'di skad bdag gis thos pa dus gcig na/_bcom ldan 'das",
]

CHAPTER_ENDING_EXAMPLES = [
    "le'u ste gnyis pa'o//_______//de nas bcom ldan 'das kyis",
    "pa'i le'u ste bcu drug pa'o//___//de nas bcom ldan 'das kyis",
    "gsungs so/_/pad+ma dbang chen dregs pa gsang ba'i rgyud las/_/gzugs brnyan dkyil 'khor gyi le'u ste gnyis pa'o//_______//de nas bcom ldan 'das kyis",
    "ba'i le'u ste gsum pa'o//____//de nas bcom ldan 'das",
    "le'u ste bzhi pa'o//____//de nas bcom ldan 'das pad+ma dbang chen gyis",
]

CONCLUDING_EXAMPLES = [
    "le'u bcu gnyis pa'o//___//pad+ma dbang chen dregs pa gzan 'bebs kyi rgyud rdzogs so",
    "rgyud gtad pa'i le'u ste bcu gcig pa'o//____//pad+ma dbang chen dregs pa gnad 'bebs kyi rgyud rdzogs so",
    "dbang chen dregs pa gzan 'bebs kyi rgyud rdzogs so/_/pad+ma 'byung gnas dang*/_kha che A nan tas bsgyur ba'o",
]

NEGATIVE_EXAMPLES = [
    "rdzogs sangs rgyas la phyag 'tshal nas// thal mo sbyar ba byas nas ni// spyan sngar 'dug ste 'di skad gsol",
    "khyod la 'dud// sangs rgyas thugs la phyag 'tshal 'dud",
    "de nas byang chub sems dpa' sems dpa' chen po",
    "byang chub sems dpa' sems dpa' chen po ting nge 'dzin",
    "pa'i/_byang chub sems dpa' sems dpa' chen po rnams",
]


@pytest.mark.parametrize("text", OPENING_EXAMPLES)
def test_opening_stock_phrases(text):
    is_stock, phrase_type = classify_stock_phrase(text)
    assert is_stock is True
    assert phrase_type == "opening"


@pytest.mark.parametrize("text", CHAPTER_ENDING_EXAMPLES)
def test_chapter_ending_stock_phrases(text):
    is_stock, phrase_type = classify_stock_phrase(text)
    assert is_stock is True
    assert phrase_type == "chapter_ending"


@pytest.mark.parametrize("text", CONCLUDING_EXAMPLES)
def test_concluding_stock_phrases(text):
    is_stock, phrase_type = classify_stock_phrase(text)
    assert is_stock is True
    assert phrase_type == "concluding"


@pytest.mark.parametrize("text", NEGATIVE_EXAMPLES)
def test_non_stock_phrases(text):
    is_stock, phrase_type = classify_stock_phrase(text)
    assert is_stock is False
    assert phrase_type is None


def test_long_doctrinal_text_with_embedded_rdzogs_so():
    text = (
        "shogs rig pa'i yig bris te/_/'byung chen bzhi yi sgra sgrags pas/_/"
        "gsang sngags chos kyi rgyun mi 'chad/_/"
        "chos kyi rgyun gyi ting nge 'dzin zhes bya ba yongs su rdzogs so/_/"
        "de nas yang bcom ldan 'das gsang ba mchog gi bdag po des bdud rtsi rgyun gyi ting nge 'dzin la snyoms par zhugs te 'di skad ces gsungs so/_/"
        "stong gsum b+han d+ha'i bzang khong du/_/"
        "srid gsum sems can dug chen las/_/"
        "dus gsum sangs rgyas rtsi sbyar bas/_/"
        "bdud rtsi dkyil 'khor lhun gyis rdzogs/_/"
        "zhes gsungs so/_/"
        "de nas yang bcom ldan 'das sems dpa'i rdo rje snying po de mchod pa rgyun gyi ting nge 'dzin la snyoms par zhugs te/_"
        "'di skad ces gsungs so/_/"
        "'khor ba rgya mtsho'i yo byad rnams/_/"
        "mchod pa rgya mtsho'i rgya?n yin te/_/"
        "srid pa rgya mtsho mtha' yas kun/_/"
        "kun bzang mchod sprin mchog tu rol/_/"
        "zhes gsungs so/_/"
        "de nas yang sems dpa'i rdo rje snying po de bzlas brjod rgyun gyi ting nge 'dzin la snyoms par zhugs te 'di skad c"
    )
    is_stock, phrase_type = classify_stock_phrase(text)
    assert is_stock is False
    assert phrase_type is None


def test_normalize_strips_leading_strokes_only():
    assert normalize_for_stock_check("/_la phyag 'tshal lo") == "la phyag 'tshal lo"
    assert normalize_for_stock_check("pa'o//___") == "pa'o//___"


def test_classify_match_row_uses_either_side():
    opening = OPENING_EXAMPLES[0]
    doctrinal = "byang chub sems dpa' sems dpa' chen po ting nge 'dzin"
    is_stock, phrase_type = classify_match_row(opening, doctrinal)
    assert is_stock is True
    assert phrase_type == "opening"

    is_stock, phrase_type = classify_match_row(doctrinal, doctrinal)
    assert is_stock is False
    assert phrase_type is None


def test_concluding_takes_priority_over_chapter_ending():
    text = "le'u ste bcu gcig pa'o//____//pad+ma dbang chen dregs pa gnad 'bebs kyi rgyud rdzogs so"
    is_stock, phrase_type = classify_stock_phrase(text)
    assert is_stock is True
    assert phrase_type == "concluding"
