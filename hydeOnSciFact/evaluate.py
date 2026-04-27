from __future__ import annotations

import math
from typing import Dict, List, Set


def _hits_at_k(retrieved_doc_ids: List[str], gold_doc_ids: Set[str], k: int) -> int:
    return sum(1 for did in retrieved_doc_ids[:k] if did in gold_doc_ids)


def recall_at_k(retrieved_doc_ids: List[str], gold_doc_ids: Set[str], k: int) -> float:
    if not gold_doc_ids:
        return 0.0
    return _hits_at_k(retrieved_doc_ids, gold_doc_ids, k) / len(gold_doc_ids)


def precision_at_k(retrieved_doc_ids: List[str], gold_doc_ids: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return _hits_at_k(retrieved_doc_ids, gold_doc_ids, k) / k


def mrr_at_k(retrieved_doc_ids: List[str], gold_doc_ids: Set[str], k: int) -> float:
    for rank, did in enumerate(retrieved_doc_ids[:k], start=1):
        if did in gold_doc_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_doc_ids: List[str], gold_doc_ids: Set[str], k: int) -> float:
    if not gold_doc_ids or k <= 0:
        return 0.0

    dcg = 0.0
    for rank, did in enumerate(retrieved_doc_ids[:k], start=1):
        rel = 1.0 if did in gold_doc_ids else 0.0
        if rel > 0:
            dcg += rel / math.log2(rank + 1)

    ideal_hits = min(len(gold_doc_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_run(
    per_query_retrieved: Dict[str, List[str]],
    qrels: Dict[str, Set[str]],
    ks: List[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    qids = list(per_query_retrieved.keys())
    n = len(qids)

    if n == 0:
        raise ValueError("No queries to evaluate.")

    for k in ks:
        metrics[f"Recall@{k}"] = sum(
            recall_at_k(per_query_retrieved[qid], qrels[qid], k) for qid in qids
        ) / n
        metrics[f"Precision@{k}"] = sum(
            precision_at_k(per_query_retrieved[qid], qrels[qid], k) for qid in qids
        ) / n
        metrics[f"MRR@{k}"] = sum(
            mrr_at_k(per_query_retrieved[qid], qrels[qid], k) for qid in qids
        ) / n
        metrics[f"nDCG@{k}"] = sum(
            ndcg_at_k(per_query_retrieved[qid], qrels[qid], k) for qid in qids
        ) / n

    metrics["num_queries"] = float(n)
    return metrics
