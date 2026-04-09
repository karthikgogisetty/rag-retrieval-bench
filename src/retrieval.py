"""
Retrieval engine: given embedded queries and passages, return top-K results.
Uses cosine similarity (embeddings are already L2-normalized).
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    query_idx: int
    chunk_id: int
    rank: int  # 1-indexed
    score: float


def retrieve_top_k(
    query_embeddings: np.ndarray,
    passage_embeddings: np.ndarray,
    chunk_ids: list[int],
    k: int = 5,
) -> list[list[RetrievalResult]]:
    """
    For each query, retrieve the top-K most similar passages.

    Args:
        query_embeddings: (num_queries, dim)
        passage_embeddings: (num_passages, dim)
        chunk_ids: list of chunk IDs corresponding to passages
        k: number of results to return per query

    Returns:
        List of lists of RetrievalResult, one per query.
    """
    # Cosine similarity = dot product (since embeddings are normalized)
    # Shape: (num_queries, num_passages)
    similarity_matrix = query_embeddings @ passage_embeddings.T

    all_results = []
    for q_idx in range(similarity_matrix.shape[0]):
        scores = similarity_matrix[q_idx]
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(RetrievalResult(
                query_idx=q_idx,
                chunk_id=chunk_ids[idx],
                rank=rank,
                score=float(scores[idx]),
            ))
        all_results.append(results)

    return all_results
