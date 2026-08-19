import time
import numpy as np
from typing import List, Dict, Any

def calculate_mrr(retrieved_selected_flags: List[int]) -> float:
    for rank, flag in enumerate(retrieved_selected_flags, 1):
        if flag == 1:
            return 1.0 / rank
    return 0.0

def evaluate_retrieval_metrics(
    eval_records: List[Dict[str, Any]],
    retriever,
    reranker=None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate Dense FAISS, BM25 Sparse, Hybrid, and Hybrid+Reranker on Recall@1, Recall@5, MRR.
    """
    metrics = {
        "Dense FAISS": {"Recall@1": [], "Recall@5": [], "MRR": []},
        "BM25 Sparse": {"Recall@1": [], "Recall@5": [], "MRR": []},
        "Hybrid": {"Recall@1": [], "Recall@5": [], "MRR": []},
        "Hybrid + Reranker": {"Recall@1": [], "Recall@5": [], "MRR": []}
    }

    evaluated_count = 0

    for rec in eval_records:
        query = rec.get("query", "")
        if not query:
            continue

        # Get candidates
        dense_results, _ = retriever.dense_retriever.retrieve(query, top_k=10)
        sparse_results, _ = retriever.sparse_retriever.retrieve(query, top_k=10)
        hybrid_results, _ = retriever.retrieve(query, candidate_k=20, top_k=10)

        reranked_results = hybrid_results
        if reranker:
            reranked_results, _ = reranker.rerank(query, hybrid_results, top_k=10, enabled=True)

        evaluated_count += 1

        method_results = {
            "Dense FAISS": dense_results,
            "BM25 Sparse": sparse_results,
            "Hybrid": hybrid_results,
            "Hybrid + Reranker": reranked_results
        }

        for method_name, res_list in method_results.items():
            sel_flags = [c.get("is_selected", 0) for c in res_list]
            
            r1 = 1.0 if len(sel_flags) > 0 and sel_flags[0] == 1 else 0.0
            r5 = 1.0 if any(s == 1 for s in sel_flags[:5]) else 0.0
            mrr = calculate_mrr(sel_flags)

            metrics[method_name]["Recall@1"].append(r1)
            metrics[method_name]["Recall@5"].append(r5)
            metrics[method_name]["MRR"].append(mrr)

    summary = {}
    for method_name, m_dict in metrics.items():
        summary[method_name] = {
            "Recall@1": round(float(np.mean(m_dict["Recall@1"])) if m_dict["Recall@1"] else 0.0, 4),
            "Recall@5": round(float(np.mean(m_dict["Recall@5"])) if m_dict["Recall@5"] else 0.0, 4),
            "MRR": round(float(np.mean(m_dict["MRR"])) if m_dict["MRR"] else 0.0, 4),
            "Evaluated_Queries": evaluated_count
        }

    return summary
