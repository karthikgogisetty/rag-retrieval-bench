"""
Embedding model wrappers for the 3 selected models.

Models chosen from the MTEB Leaderboard (2026) — CPU-feasible tier:
- all-MiniLM-L6-v2: lightweight baseline (MTEB 56.3)
- nomic-embed-text-v1.5: mid-tier, Matryoshka support (MTEB ~62+)
- BGE-M3: top open-source self-hostable model (MTEB 63.0)

Each wrapper handles model-specific query/passage prefix conventions.
"""

import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddingModel:
    name: str
    model_id: str
    dim: int
    query_prefix: str
    passage_prefix: str
    description: str
    mteb_score: float


MODEL_CONFIGS = [
    EmbeddingModel(
        name="all-MiniLM-L6-v2",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
        query_prefix="",
        passage_prefix="",
        description="Lightweight baseline (22M params, 384d, MTEB 56.3)",
        mteb_score=56.3,
    ),
    EmbeddingModel(
        name="nomic-embed-text-v1.5",
        model_id="nomic-ai/nomic-embed-text-v1.5",
        dim=768,
        query_prefix="search_query: ",
        passage_prefix="search_document: ",
        description="Mid-tier with Matryoshka dims (137M params, 768d, MTEB ~62+)",
        mteb_score=62.0,
    ),
    EmbeddingModel(
        name="BGE-M3",
        model_id="BAAI/bge-m3",
        dim=1024,
        query_prefix="",
        passage_prefix="",
        description="Top open-source self-hostable (568M params, 1024d, MTEB 63.0)",
        mteb_score=63.0,
    ),
]


class EmbeddingEngine:
    """Wraps a SentenceTransformer model with proper prefix handling."""

    def __init__(self, config: EmbeddingModel):
        self.config = config
        print(f"  Loading {config.name} ({config.model_id})...")
        self.model = SentenceTransformer(config.model_id, trust_remote_code=True)
        print(f"  Loaded. Embedding dim: {config.dim}")

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed document passages/chunks."""
        prefixed = [self.config.passage_prefix + t for t in texts]
        embeddings = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return np.array(embeddings)

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """Embed search queries."""
        prefixed = [self.config.query_prefix + q for q in queries]
        embeddings = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return np.array(embeddings)


def load_all_models() -> list[EmbeddingEngine]:
    """Load all 3 embedding models."""
    engines = []
    for config in MODEL_CONFIGS:
        engines.append(EmbeddingEngine(config))
    return engines


if __name__ == "__main__":
    # Quick sanity test
    for config in MODEL_CONFIGS:
        engine = EmbeddingEngine(config)
        q_emb = engine.embed_queries(["test query"])
        p_emb = engine.embed_passages(["test passage"])
        sim = np.dot(q_emb[0], p_emb[0])
        print(f"{config.name}: query shape={q_emb.shape}, passage shape={p_emb.shape}, sim={sim:.4f}")
