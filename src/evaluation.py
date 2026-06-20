import math

import numpy as np


def precision_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(top_k)


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant document found, or 0 if none appear at all."""
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain with binary relevance. Rewards
    relevant documents for appearing early, not just for appearing at all
    somewhere in the top k — a relevant doc at rank 1 contributes more than
    the same doc at rank 5."""
    top_k = ranked_ids[:k]
    dcg = sum(
        (1.0 if doc_id in relevant else 0.0) / math.log2(rank + 1)
        for rank, doc_id in enumerate(top_k, start=1)
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retriever(retriever, eval_set: list[dict], k_values: list[int] = [1, 3, 5]) -> dict:
    """Runs every question in eval_set through the retriever and averages
    each metric across the whole set. Returns a dict keyed like
    'precision@3', 'recall@3', 'mrr', 'ndcg@3'."""
    max_k = max(k_values)
    per_question_metrics = {f"precision@{k}": [] for k in k_values}
    per_question_metrics.update({f"recall@{k}": [] for k in k_values})
    per_question_metrics.update({f"ndcg@{k}": [] for k in k_values})
    per_question_metrics["mrr"] = []

    for item in eval_set:
        relevant = set(item["relevant_docs"])
        results = retriever.retrieve(item["question"], k=max_k)
        ranked_ids = [doc_id for doc_id, _ in results]

        for k in k_values:
            per_question_metrics[f"precision@{k}"].append(precision_at_k(ranked_ids, relevant, k))
            per_question_metrics[f"recall@{k}"].append(recall_at_k(ranked_ids, relevant, k))
            per_question_metrics[f"ndcg@{k}"].append(ndcg_at_k(ranked_ids, relevant, k))
        per_question_metrics["mrr"].append(reciprocal_rank(ranked_ids, relevant))

    return {metric: float(np.mean(values)) for metric, values in per_question_metrics.items()}
