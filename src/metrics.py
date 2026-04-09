"""
Evaluation metrics for information retrieval.

All metrics operate on a single query at a time. Aggregation (macro-average)
is done in the evaluation pipeline.

Supports graded relevance (0, 1, 2) for NDCG computation.
"""

import math
import numpy as np
from dataclasses import dataclass


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    qid: int
    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float
    precision_at_3: float
    precision_at_5: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    ndcg_at_3: float
    ndcg_at_5: float


def _hit_rate_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """1 if any relevant document is in top-K, else 0."""
    return 1.0 if any(rid in relevant_ids for rid in retrieved_ids[:k]) else 0.0


def _precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Fraction of top-K that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for rid in top_k if rid in relevant_ids) / k


def _recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Fraction of all relevant docs found in top-K."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for rid in top_k if rid in relevant_ids) / len(relevant_ids)


def _mrr(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def _dcg_at_k(retrieved_ids: list[int], relevance_grades: dict[int, int], k: int) -> float:
    """Discounted Cumulative Gain using graded relevance."""
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        grade = relevance_grades.get(rid, 0)
        # Using the standard formula: (2^rel - 1) / log2(i + 2)
        dcg += (2 ** grade - 1) / math.log2(i + 2)
    return dcg


def _ndcg_at_k(retrieved_ids: list[int], relevance_grades: dict[int, int], k: int) -> float:
    """Normalized DCG: DCG / ideal DCG."""
    dcg = _dcg_at_k(retrieved_ids, relevance_grades, k)

    # Ideal ranking: sort all relevant docs by grade descending
    ideal_grades = sorted(relevance_grades.values(), reverse=True)[:k]
    idcg = 0.0
    for i, grade in enumerate(ideal_grades):
        idcg += (2 ** grade - 1) / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_query_metrics(
    qid: int,
    retrieved_ids: list[int],
    relevance_grades: dict[int, int],
) -> QueryMetrics:
    """Compute all metrics for a single query."""
    # For binary metrics, any chunk with grade >= 1 is "relevant"
    relevant_ids = {cid for cid, grade in relevance_grades.items() if grade >= 1}

    return QueryMetrics(
        qid=qid,
        hit_rate_at_1=_hit_rate_at_k(retrieved_ids, relevant_ids, 1),
        hit_rate_at_3=_hit_rate_at_k(retrieved_ids, relevant_ids, 3),
        hit_rate_at_5=_hit_rate_at_k(retrieved_ids, relevant_ids, 5),
        precision_at_3=_precision_at_k(retrieved_ids, relevant_ids, 3),
        precision_at_5=_precision_at_k(retrieved_ids, relevant_ids, 5),
        recall_at_3=_recall_at_k(retrieved_ids, relevant_ids, 3),
        recall_at_5=_recall_at_k(retrieved_ids, relevant_ids, 5),
        mrr=_mrr(retrieved_ids, relevant_ids),
        ndcg_at_3=_ndcg_at_k(retrieved_ids, relevance_grades, 3),
        ndcg_at_5=_ndcg_at_k(retrieved_ids, relevance_grades, 5),
    )


def aggregate_metrics(query_metrics_list: list[QueryMetrics]) -> dict[str, float]:
    """Macro-average across all queries."""
    if not query_metrics_list:
        return {}

    metric_names = [
        "hit_rate_at_1", "hit_rate_at_3", "hit_rate_at_5",
        "precision_at_3", "precision_at_5",
        "recall_at_3", "recall_at_5",
        "mrr",
        "ndcg_at_3", "ndcg_at_5",
    ]

    aggregated = {}
    for name in metric_names:
        values = [getattr(qm, name) for qm in query_metrics_list]
        aggregated[name] = sum(values) / len(values)

    return aggregated
