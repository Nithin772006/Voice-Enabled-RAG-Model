import numpy as np
from typing import List, Dict, Any, Tuple
from rag_system.retrieval import HybridRetriever
from rag_system.reranker import MultilingualReranker
from rag_system.data_loader import extract_passages


def is_chunk_relevant(cand: Dict[str, Any], val_ex: Dict[str, Any], relevant_texts: set) -> bool:
    """Check if a candidate chunk matches any ground truth relevant passage."""
    meta = cand.get("metadata", {})
    val_qid = val_ex.get("query_id")
    if meta and val_qid is not None and meta.get("query_id") == val_qid:
        if meta.get("is_selected") == 1:
            return True

    c_text = cand.get("text", "").strip()
    if not c_text:
        return False

    for rel_text in relevant_texts:
        rel_clean = rel_text.strip()
        if not rel_clean:
            continue
        if c_text in rel_clean or rel_clean in c_text:
            return True
    return False


def evaluate_retrieval_metrics(
    val_examples: List[Dict[str, Any]],
    retriever: HybridRetriever,
    reranker: MultilingualReranker = None,
    top_k_candidates: int = 20,
    top_k_evidence: int = 5,
) -> Dict[str, float]:
    """
    Evaluate Recall@1, Recall@5, MRR, and Reranked Recall@1/5 on validation samples.
    Uses MSMARCO-XI `is_selected` field for ground truth relevance.
    """
    recalls_at_1 = []
    recalls_at_5 = []
    recalls_at_k = []
    recip_ranks = []

    rerank_recalls_at_1 = []
    rerank_recalls_at_5 = []
    rerank_mrr = []

    for ex in val_examples:
        query = ex.get("query", "")
        if not query:
            continue

        raw_passages, is_selected_labels = extract_passages(ex)
        if not raw_passages or not any(is_selected_labels):
            continue  # Skip queries without ground truth relevant passages

        # Get relevant text set
        relevant_texts = {
            raw_passages[i].strip()
            for i, sel in enumerate(is_selected_labels)
            if sel == 1
        }

        # 1. Candidate Retrieval
        candidates, _ = retriever.retrieve_candidates(query)
        if not candidates:
            recalls_at_1.append(0.0)
            recalls_at_5.append(0.0)
            recalls_at_k.append(0.0)
            recip_ranks.append(0.0)
            continue

        # Check candidate recall using is_chunk_relevant
        cand_matches = [
            1 if is_chunk_relevant(c, ex, relevant_texts) else 0 for c in candidates
        ]

        recalls_at_1.append(1.0 if sum(cand_matches[:1]) > 0 else 0.0)
        recalls_at_5.append(1.0 if sum(cand_matches[:top_k_evidence]) > 0 else 0.0)
        recalls_at_k.append(1.0 if sum(cand_matches) > 0 else 0.0)

        # Candidate MRR
        first_match_rank = 0
        for r, m in enumerate(cand_matches, 1):
            if m == 1:
                first_match_rank = r
                break
        recip_ranks.append(1.0 / first_match_rank if first_match_rank > 0 else 0.0)

        # 2. Reranker Evaluation (if reranker provided)
        if reranker is not None:
            top_evidence, _ = reranker.rerank(query, candidates)
            rerank_matches = [
                1 if is_chunk_relevant(e, ex, relevant_texts) else 0 for e in top_evidence
            ]
            rerank_recalls_at_1.append(1.0 if sum(rerank_matches[:1]) > 0 else 0.0)
            rerank_recalls_at_5.append(
                1.0 if sum(rerank_matches[:top_k_evidence]) > 0 else 0.0
            )

            rr_rank = 0
            for r, m in enumerate(rerank_matches, 1):
                if m == 1:
                    rr_rank = r
                    break
            rerank_mrr.append(1.0 / rr_rank if rr_rank > 0 else 0.0)

    results = {
        "candidate_recall@1": float(np.mean(recalls_at_1)) if recalls_at_1 else 0.0,
        "candidate_recall@5": float(np.mean(recalls_at_5)) if recalls_at_5 else 0.0,
        "candidate_mrr": float(np.mean(recip_ranks)) if recip_ranks else 0.0,
    }

    if reranker is not None and rerank_recalls_at_1:
        results.update({
            "reranked_recall@1": float(np.mean(rerank_recalls_at_1)),
            "reranked_recall@5": float(np.mean(rerank_recalls_at_5)),
            "reranked_mrr": float(np.mean(rerank_mrr)),
        })

    return results


def evaluate_retrieval_table(
    val_examples: List[Dict[str, Any]],
    retriever: HybridRetriever,
    reranker: MultilingualReranker = None,
    top_k: int = 20,
    top_k_evidence: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Evaluate Dense, BM25, Hybrid, and optional Hybrid+Reranker with the same relevance logic."""

    def empty_acc():
        return {"r1": [], "r5": [], "mrr": []}

    acc = {
        "Dense FAISS": empty_acc(),
        "BM25": empty_acc(),
        "Hybrid": empty_acc(),
    }
    if reranker is not None:
        acc["Hybrid + Reranker"] = empty_acc()

    def add_metrics(name: str, candidates: List[Dict[str, Any]], ex: Dict[str, Any], relevant_texts: set):
        matches = [1 if is_chunk_relevant(c, ex, relevant_texts) else 0 for c in candidates]
        acc[name]["r1"].append(1.0 if sum(matches[:1]) > 0 else 0.0)
        acc[name]["r5"].append(1.0 if sum(matches[:top_k_evidence]) > 0 else 0.0)
        rank = 0
        for i, m in enumerate(matches, 1):
            if m:
                rank = i
                break
        acc[name]["mrr"].append(1.0 / rank if rank else 0.0)

    for ex in val_examples:
        query = ex.get("query", "")
        if not query:
            continue
        raw_passages, is_selected_labels = extract_passages(ex)
        if not raw_passages or not any(is_selected_labels):
            continue
        relevant_texts = {
            raw_passages[i].strip()
            for i, sel in enumerate(is_selected_labels)
            if sel == 1 and i < len(raw_passages)
        }

        dense_candidates, _ = retriever.retrieve_dense(query, top_k=top_k)
        bm25_candidates, _ = retriever.retrieve_bm25(query, top_k=top_k)
        hybrid_candidates, _ = retriever.retrieve_candidates(query)

        add_metrics("Dense FAISS", dense_candidates, ex, relevant_texts)
        add_metrics("BM25", bm25_candidates, ex, relevant_texts)
        add_metrics("Hybrid", hybrid_candidates, ex, relevant_texts)

        if reranker is not None:
            top_evidence, _ = reranker.rerank(query, hybrid_candidates)
            add_metrics("Hybrid + Reranker", top_evidence, ex, relevant_texts)

    results = {}
    for name, vals in acc.items():
        results[name] = {
            "Recall@1": float(np.mean(vals["r1"])) if vals["r1"] else 0.0,
            "Recall@5": float(np.mean(vals["r5"])) if vals["r5"] else 0.0,
            "MRR": float(np.mean(vals["mrr"])) if vals["mrr"] else 0.0,
            "Evaluated": float(len(vals["r1"])),
        }
    return results


def calibrate_guardrail_threshold(
    val_examples: List[Dict[str, Any]],
    retriever: HybridRetriever,
    reranker: MultilingualReranker,
    thresholds: List[float] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Calibrate reranker score threshold on validation data.
    Reports false accept/refusal rates and prefers high precision over high recall.
    """
    thresholds = thresholds or [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    y_true = []
    y_scores = []

    for ex in val_examples[:200]:  # Use validation subset for fast calibration
        query = ex.get("query", "")
        if not query:
            continue

        raw_passages, is_selected_labels = extract_passages(ex)
        if not raw_passages:
            continue

        relevant_texts = {
            raw_passages[i].strip()
            for i, sel in enumerate(is_selected_labels)
            if sel == 1 and i < len(raw_passages)
        }

        candidates, _ = retriever.retrieve_candidates(query)
        if not candidates:
            continue

        top_evidence, _ = reranker.rerank(query, candidates)
        if not top_evidence:
            continue

        top_item = top_evidence[0]
        score = top_item.get("reranker_score", 0.0)
        is_rel = 1 if is_chunk_relevant(top_item, ex, relevant_texts) else 0

        y_scores.append(score)
        y_true.append(is_rel)

    if not y_scores:
        return 0.65, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "false_accept_rate": 0.0, "false_refusal_rate": 0.0}

    best_thresh = 0.65
    best_key = (-1.0, -1.0, 0.0)
    best_stats = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "false_accept_rate": 0.0, "false_refusal_rate": 0.0}

    for thresh in thresholds:
        tp = sum(1 for s, t in zip(y_scores, y_true) if s >= thresh and t == 1)
        fp = sum(1 for s, t in zip(y_scores, y_true) if s >= thresh and t == 0)
        fn = sum(1 for s, t in zip(y_scores, y_true) if s < thresh and t == 1)
        tn = sum(1 for s, t in zip(y_scores, y_true) if s < thresh and t == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        false_accept_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_refusal_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        score_key = (precision, f1, -false_accept_rate)
        if score_key > best_key:
            best_key = score_key
            best_thresh = float(thresh)
            best_stats = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "false_accept_rate": float(false_accept_rate),
                "false_refusal_rate": float(false_refusal_rate),
            }

    return best_thresh, best_stats
