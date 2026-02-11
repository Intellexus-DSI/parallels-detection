"""
Tibetan grammatical particles (EWTS/Wylie forms).

Particles can be added/omitted while preserving meaning.
Used to filter out non-substantial tokens when comparing content overlap.
"""

# Flattened set of particle syllables from the guideline categories.
# Includes all forms (e.g. kyi, gyi, gi, yi, 'i for genitive).
# EWTS uses ' for apostrophe.
TIBETAN_PARTICLES: frozenset[str] = frozenset({
    # Terminative
    "su", "ru", "du", "tu", "r",
    # Locative
    "na", "la",
    # Genitive
    "kyi", "gyi", "gi", "yi", "'i", "i",
    # Ergative
    "kyis", "gyis", "gis", "yis", "'is", "is",
    # Concessive
    "kyang", "yang", "'ang", "ang",
    # Semi-final
    "ste", "te", "de",
    # Final - Question
    "gam", "ngam", "dam", "nam", "bam", "mam", "ram", "lam", "sam", "tam", "'am", "am",
    # Ablative
    "nas", "las",
    # Final - Statement
    "go", "ngo", "do", "no", "bo", "mo", "ro", "lo", "so", "to", "'o", "o",
    # Conjunction
    "dang",
    # Isolation/Emphasis
    "ni", "nyid",
    # Negation (ma, mi) NOT in set - user: "omission/addition significant!"
    # Nominal (excluding ma - see negation below)
    "pa", "ba", "po", "bo", "mo", "pho", "ka", "ko", "kha", "ga",
    # Personal pronouns
    "nga", "nged", "khyod", "khyed", "kho", "gong", "bdag",
    "nge'u", "cag", "'u", "u", "bu", "skol", "'o",
    # Demonstrative/Definite
    "'di", "di", "de",
    # Final - imperative / Indefinite
    "cig", "zhig", "shig", "shog",
    # Indefinite/Interrogative
    "su", "gang", "ga", "la", "ba", "na", "ci", "ji", "ltar", "dag", "zhe", "yang",
    "kha", "'ga'", "la", "ge", "mo", "che",
    # Possessive
    "can", "ldan", "bcas",
    # Coordination
    "cing", "zhing", "shing",
    # Connecting
    "kyin", "gin", "gyin",
    # Auxiliary verbs / copula
    "yin", "yod", "'dug", "red", "min", "med", "'gyur", "gyur", "'byung", "byung",
    "myong", "song", "phyin", "'gro", "'ong", "yong",
    "yod", "kyang", "kyi", "gyi",
    # Modal verbs
    "thub", "nus", "dgos", "zin", "tshar", "'dod", "'os", "rigs", "sla", "dka'",
    "rung", "chog", "phod", "mod",
    # Plural
    "rnams", "tsho",
    # Postposition components
    "drung", "nang", "steng", "bar", "'og", "slad", "phyi", "mdun", "rgyab",
    "sgang", "ngang", "sngon", "ched", "rjes", "rting", "thog", "mthu", "dus",
    "don", "byang", "g.yas", "g.yon", "ring", "shar", "lho",
    # Other particles
    "tsam", "cog", "cha", "bzhin",
    # Function words/phrases (key syllables)
    "yongs", "bas", "da", "lta", "skad", "gzhan", "dper", "bzhin", "khyed", "par",
    "gal", "te", "phyir", "dbang", "spyir", "ta", "re", "kun", "thams", "cad",
    "mang", "phrag",
    # Reference markers
    "gsungs", "zhal", "snga", "na", "re", "'dris", "nyid", "mdo", "rgyud",
    "gzhung", "lung", "gsung", "'dir", "smras", "brgal", "lan", "slar",
    "ce", "zhe", "she", "zer", "gsungs", "bya", "sogs", "'byung", "rgya",
    "cher", "'dod", "brjod", "smra", "smra'o", "gnang", "bshad", "grags",
    "snyam", "skad", "ste", "ngo", "par", "gong", "bstan", "zin", "to",
    # Verbal prefixes
    "tu", "nas", "par", "su", "bar",
    # Postpositions (Schwieger) - key syllables
    "klong", "dkyil", "rkyen", "skabs", "skor", "skyin", "khar", "khongs",
    "khrod", "'khris", "gong", "rgyab", "sgang", "sgo", "ngang", "ngor",
    "ngos", "sngon", "ched", "rjes", "rting", "steng", "thog", "mthu",
    "dus", "don", "drung", "mdun", "nang", "nub", "phyi", "phyir", "bar",
    "byang", "dbang", "mod", "rtsa", "tshul", "tshe", "zhabs", "'og",
    "g.yas", "g.yon", "ring", "shar", "slad", "lho",
})
