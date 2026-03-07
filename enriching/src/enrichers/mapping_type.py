"""Mapping type enricher: classifies parallels as textual, semantic, or uncertain."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .base import BaseEnricher
from .tibetan_particles import TIBETAN_PARTICLES
from .wylie_levenshtein import calculate_wylie_syllable_distance
from ..models import EnrichedParallel

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _segment_id_to_index(segment_id: str) -> int:
    """Convert segment_id (e.g. '114_1', '114_2') to 0-based row index."""
    parts = str(segment_id).strip().split("_")
    if len(parts) >= 2:
        return int(parts[-1]) - 1
    return 0


def _load_lexical_similarity(
    embeddings_dir: Path,
    source_file_id_a: str,
    segment_a_id: str,
    source_file_id_b: str,
    segment_b_id: str,
    cache: Dict[str, Tuple[np.ndarray, int]],
) -> Optional[float]:
    """
    Load dual-layer embeddings and compute lexical (early-layer) cosine similarity.
    Returns None if embeddings are not available or not dual-layer.
    """
    if not source_file_id_a or not source_file_id_b:
        return None

    emb_dir = Path(embeddings_dir)
    meta_path = emb_dir / "embeddings_metadata.json"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)
    if not meta.get("dual_layer", False):
        return None

    dim = meta["embedding_dimension"]

    def load_lexical_vec(file_id: str, seg_id: str) -> Optional[np.ndarray]:
        key = file_id
        if key not in cache:
            emb_path = emb_dir / f"{file_id}_embeddings.npy"
            if not emb_path.exists():
                return None
            arr = np.load(emb_path)
            if arr.ndim != 2 or arr.shape[1] != 2 * dim:
                return None
            cache[key] = (arr, dim)
        arr, d = cache[key]
        idx = _segment_id_to_index(seg_id)
        if idx < 0 or idx >= len(arr):
            return None
        return arr[idx, :d].astype(np.float32)

    va = load_lexical_vec(source_file_id_a, segment_a_id)
    vb = load_lexical_vec(source_file_id_b, segment_b_id)
    if va is None or vb is None:
        return None
    return _cosine_similarity(va, vb)


def _tokenize_wylie(text: str) -> List[str]:
    """Split Wylie/EWTS text into syllable tokens."""
    return text.replace("/", " ").split()


def _get_content_tokens(tokens: List[str]) -> List[str]:
    """Filter out particles; return content syllables preserving order."""
    return [t for t in tokens if t and t not in TIBETAN_PARTICLES]


def _content_overlap_ratio(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard: |A ∩ B| / |A ∪ B|."""
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    inter = set_a & set_b
    return len(inter) / len(union) if union else 0.0


def _bigrams(tokens: List[str]) -> Set[Tuple[str, str]]:
    """Return set of (token[i], token[i+1]) for i in 0..len-2."""
    return {tuple(tokens[i : i + 2]) for i in range(len(tokens) - 1)}


def _bigram_overlap_ratio(content_a: List[str], content_b: List[str]) -> float:
    """Jaccard overlap of syllable bigrams (shared phrases)."""
    bg_a = _bigrams(content_a)
    bg_b = _bigrams(content_b)
    return _content_overlap_ratio(bg_a, bg_b)


def classify_mapping_type(
    parallel_a: str,
    parallel_b: str,
    syllable_distance: int,
    overlap_textual: float = 0.40,
    overlap_semantic: float = 0.25,
    norm_lev_textual: float = 0.25,
    norm_lev_semantic: float = 0.40,
    bigram_textual: float = 0.25,
    bigram_semantic: float = 0.15,
    sim_lexical: Optional[float] = None,
    sim_semantic: Optional[float] = None,
    sim_lexical_textual: float = 0.85,
    sim_lexical_semantic: float = 0.55,
) -> str:
    """
    Classify parallel as 'textual', 'semantic', or 'uncertain'.

    Uses surface signals (overlap, Levenshtein, bigrams) and optionally dual-layer
    embedding similarities when available:
    - sim_lexical: Cosine similarity of early-layer (lexical) embeddings
    - sim_semantic: Detection similarity (semantic embeddings)

    - **Textual**: High sim_lexical, or strong surface overlap.
    - **Semantic**: Low sim_lexical but high sim_semantic, or weak surface signals.
    - **Uncertain**: Borderline, worth manual review.
    """
    tokens_a = _tokenize_wylie(parallel_a.strip())
    tokens_b = _tokenize_wylie(parallel_b.strip())

    content_a = _get_content_tokens(tokens_a)
    content_b = _get_content_tokens(tokens_b)

    unigram_overlap = _content_overlap_ratio(set(content_a), set(content_b))
    bigram_overlap = _bigram_overlap_ratio(content_a, content_b)

    max_syllables = max(len(tokens_a), len(tokens_b)) or 1
    normalized_lev = syllable_distance / max_syllables

    # Lexical similarity from dual-layer embeddings (strongest signal when available)
    if sim_lexical is not None:
        lexical_textual = sim_lexical >= sim_lexical_textual
        lexical_semantic = sim_lexical <= sim_lexical_semantic and (
            sim_semantic is None or sim_semantic >= 0.88
        )
    else:
        lexical_textual = False
        lexical_semantic = False

    # Surface-based signals (fallback when no dual-layer)
    surface_textual = (
        unigram_overlap >= overlap_textual
        or normalized_lev <= norm_lev_textual
        or bigram_overlap >= bigram_textual
    )
    surface_semantic = (
        unigram_overlap <= overlap_semantic
        and normalized_lev >= norm_lev_semantic
        and bigram_overlap <= bigram_semantic
    )

    is_textual = lexical_textual or (surface_textual and not lexical_semantic)
    is_semantic = lexical_semantic or (surface_semantic and not lexical_textual)

    if is_textual and not is_semantic:
        return "textual"
    if is_semantic and not is_textual:
        return "semantic"
    return "uncertain"


class MappingTypeEnricher(BaseEnricher):
    """
    Enricher that classifies parallels as textual, semantic, or uncertain.

    Combines:
    - Unigram content overlap (particles stripped)
    - Normalized Levenshtein distance
    - Bigram (phrase) overlap
    - Optional: sim_lexical from dual-layer embeddings (when embeddings_dir set)

    Output: 'textual' | 'semantic' | 'uncertain'
    """

    DEFAULT_PARAMS = {
        "overlap_textual": 0.40,
        "overlap_semantic": 0.25,
        "norm_lev_textual": 0.25,
        "norm_lev_semantic": 0.40,
        "bigram_textual": 0.25,
        "bigram_semantic": 0.15,
        "embeddings_dir": None,
        "sim_lexical_textual": 0.85,
        "sim_lexical_semantic": 0.55,
    }

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)
        self.overlap_textual = self.params.get(
            "overlap_textual", self.DEFAULT_PARAMS["overlap_textual"]
        )
        self.overlap_semantic = self.params.get(
            "overlap_semantic", self.DEFAULT_PARAMS["overlap_semantic"]
        )
        self.norm_lev_textual = self.params.get(
            "norm_lev_textual", self.DEFAULT_PARAMS["norm_lev_textual"]
        )
        self.norm_lev_semantic = self.params.get(
            "norm_lev_semantic", self.DEFAULT_PARAMS["norm_lev_semantic"]
        )
        self.bigram_textual = self.params.get(
            "bigram_textual", self.DEFAULT_PARAMS["bigram_textual"]
        )
        self.bigram_semantic = self.params.get(
            "bigram_semantic", self.DEFAULT_PARAMS["bigram_semantic"]
        )
        emb_dir = self.params.get("embeddings_dir")
        self.embeddings_dir = Path(emb_dir) if emb_dir else None
        self.sim_lexical_textual = self.params.get(
            "sim_lexical_textual", self.DEFAULT_PARAMS["sim_lexical_textual"]
        )
        self.sim_lexical_semantic = self.params.get(
            "sim_lexical_semantic", self.DEFAULT_PARAMS["sim_lexical_semantic"]
        )
        self._lexical_cache: Dict[str, Tuple[np.ndarray, int]] = {}

    def enrich(self, parallel: EnrichedParallel) -> EnrichedParallel:
        """Add mapping_type: 'textual', 'semantic', or 'uncertain'."""
        text_a = (parallel.parallel_a or "").strip()
        text_b = (parallel.parallel_b or "").strip()

        # Use precomputed distance if available (from wylie_levenshtein), else compute
        syllable_dist = parallel.enriched_fields.get("wylie_syllable_distance")
        if syllable_dist is None:
            syllable_dist = calculate_wylie_syllable_distance(text_a, text_b)

        sim_lexical = None
        if self.embeddings_dir and getattr(parallel, "source_file_id_a", ""):
            sim_lexical = _load_lexical_similarity(
                self.embeddings_dir,
                getattr(parallel, "source_file_id_a", "") or "",
                parallel.segment_a_id,
                getattr(parallel, "source_file_id_b", "") or "",
                parallel.segment_b_id,
                self._lexical_cache,
            )
            if sim_lexical is not None:
                parallel.enriched_fields["sim_lexical"] = round(sim_lexical, 4)

        mapping_type = classify_mapping_type(
            text_a,
            text_b,
            syllable_distance=syllable_dist,
            overlap_textual=self.overlap_textual,
            overlap_semantic=self.overlap_semantic,
            norm_lev_textual=self.norm_lev_textual,
            norm_lev_semantic=self.norm_lev_semantic,
            bigram_textual=self.bigram_textual,
            bigram_semantic=self.bigram_semantic,
            sim_lexical=sim_lexical,
            sim_semantic=parallel.similarity,
            sim_lexical_textual=self.sim_lexical_textual,
            sim_lexical_semantic=self.sim_lexical_semantic,
        )

        parallel.enriched_fields["mapping_type"] = mapping_type

        return parallel

    @property
    def field_names(self) -> list:
        return ["mapping_type", "sim_lexical"]
