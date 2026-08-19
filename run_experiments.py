import os
import time
from rag_system.config import TRAIN_FILE, VAL_FILE, DEFAULT_N_TRAIN_ROWS, INDEX_CACHE_DIR
from rag_system.data_loader import load_msmarco_samples, extract_passages
from rag_system.chunking import OriginalPassageChunker, FixedSizeChunker, SentenceAwareChunker, ParagraphStructureChunker
from rag_system.indexing import DenseIndexer, SparseIndexer
from rag_system.retrieval import HybridRetriever
from rag_system.reranker import MultilingualReranker
from rag_system.guardrails import ConfidenceGuardrail
from rag_system.eval import evaluate_retrieval_metrics, evaluate_retrieval_table, calibrate_guardrail_threshold
from rag_system.benchmark import benchmark_latency, print_latency_report


def run_chunking_experiments():
    print("=" * 70)
    print("MSMARCO-XI CHUNKING STRATEGY BENCHMARK & EVALUATION")
    print("=" * 70)

    # 1. Load Data
    print(f"\nLoading prototype dataset from train file ({DEFAULT_N_TRAIN_ROWS} rows)...")
    train_samples = load_msmarco_samples(TRAIN_FILE, max_rows=DEFAULT_N_TRAIN_ROWS)
    print(f"Loaded {len(train_samples)} training rows.")

    print(f"\nLoading validation dataset from val file...")
    val_samples = load_msmarco_samples(VAL_FILE, max_rows=300)
    print(f"Loaded {len(val_samples)} validation rows.")

    # Prepare raw documents from train only. Validation remains held out.
    documents = []

    def add_samples_to_docs(samples, source_name):
        for idx, row in enumerate(samples):
            query = row.get("query", "")
            answer = row.get("answer", "")
            q_id = row.get("query_id", f"{source_name}_{idx}")
            translated_passages, selected_labels = extract_passages(row)
            for i, p_text in enumerate(translated_passages):
                if not p_text:
                    continue
                is_sel = selected_labels[i] if i < len(selected_labels) else 0
                documents.append({
                    "text": p_text,
                    "query": query,
                    "answer": answer,
                    "query_id": q_id,
                    "passage_idx": i,
                    "is_selected": is_sel,
                })

    add_samples_to_docs(train_samples, "train")

    print(f"Total raw passages in indexed corpus: {len(documents)}")

    strategies = {
        "Original": OriginalPassageChunker(),
        "Fixed": FixedSizeChunker(chunk_size=100, overlap=20),
        "Sentence": SentenceAwareChunker(max_words=100),
        "Paragraph": ParagraphStructureChunker(max_words=120),
    }

    experiment_results = []
    reranker = MultilingualReranker()
    guardrail = ConfidenceGuardrail()

    for name, chunker in strategies.items():
        print("\n" + "=" * 70)
        print(f"RUNNING EXPERIMENT FOR CHUNKING STRATEGY: {name.upper()}")
        print("=" * 70)

        # Chunk corpus
        chunked_docs = chunker.chunk_documents(documents)
        chunk_texts = [d["text"] for d in chunked_docs]
        print(f"Generated {len(chunk_texts)} chunks.")

        # Build index
        print("Building Dense FAISS index...")
        dense_indexer = DenseIndexer()
        dense_indexer.build_index(chunk_texts, batch_size=32)

        print("Building Sparse BM25 index...")
        sparse_indexer = SparseIndexer()
        sparse_indexer.build_index(chunk_texts)

        retriever = HybridRetriever(
            dense_indexer, sparse_indexer, chunked_docs
        )

        # Evaluation metrics
        print("\nEvaluating retrieval quality on validation dataset...")
        eval_metrics = evaluate_retrieval_metrics(
            val_samples, retriever, reranker=reranker, top_k_evidence=5
        )
        method_table = evaluate_retrieval_table(
            val_samples, retriever, reranker=reranker, top_k_evidence=5
        )
        print("\nRetrieval method comparison:")
        print(f"{'Metric':<10} | {'Dense FAISS':<12} | {'BM25':<12} | {'Hybrid':<12} | {'Hybrid + Reranker':<18}")
        print("-" * 75)
        for metric in ["Recall@1", "Recall@5", "MRR"]:
            print(
                f"{metric:<10} | "
                f"{method_table['Dense FAISS'][metric]:<12.4f} | "
                f"{method_table['BM25'][metric]:<12.4f} | "
                f"{method_table['Hybrid'][metric]:<12.4f} | "
                f"{method_table['Hybrid + Reranker'][metric]:<18.4f}"
            )

        # Latency Benchmark
        val_queries = [s["query"] for s in val_samples if s.get("query")]
        lat_results = benchmark_latency(
            val_queries, retriever, reranker, guardrail, n_runs=15
        )

        tot_lat = lat_results.get("Total Retrieval", lat_results.get("total_retrieval_ms", {"P50": 0.0, "P70": 0.0, "P100": 0.0}))

        exp_row = {
            "Strategy": name,
            "Recall@1": eval_metrics.get("candidate_recall@1", 0.0),
            "Recall@5": eval_metrics.get("candidate_recall@5", 0.0),
            "Rerank_Recall@5": eval_metrics.get("reranked_recall@5", 0.0),
            "MRR": eval_metrics.get("candidate_mrr", 0.0),
            "Rerank_MRR": eval_metrics.get("reranked_mrr", 0.0),
            "P50": tot_lat["P50"],
            "P70": tot_lat["P70"],
            "P100": tot_lat["P100"],
        }
        experiment_results.append(exp_row)

    # Threshold calibration report
    print("\n" + "=" * 70)
    print("GUARDRAIL THRESHOLD CALIBRATION ON VALIDATION DATA")
    print("=" * 70)
    calib_thresh, calib_stats = calibrate_guardrail_threshold(
        val_samples, retriever, reranker
    )
    print(f"Calibrated Reranker Threshold: {calib_thresh:.4f}")
    print(
        f"Precision: {calib_stats['precision']:.4f} | "
        f"Recall: {calib_stats['recall']:.4f} | "
        f"F1: {calib_stats['f1']:.4f} | "
        f"False Accept: {calib_stats['false_accept_rate']:.4f} | "
        f"False Refusal: {calib_stats['false_refusal_rate']:.4f}"
    )

    # Summary table
    print("\n" + "=" * 70)
    print("CHUNKING EXPERIMENT COMPARISON MATRIX")
    print("=" * 70)
    header = f"{'Strategy':<10} | {'Recall@1':<9} | {'Recall@5':<9} | {'Rerank_R@5':<11} | {'MRR':<8} | {'P50 (ms)':<9} | {'P70 (ms)':<9} | {'P100 (ms)':<9}"
    print(header)
    print("-" * len(header))

    for r in experiment_results:
        print(
            f"{r['Strategy']:<10} | {r['Recall@1']:<9.4f} | {r['Recall@5']:<9.4f} | {r['Rerank_Recall@5']:<11.4f} | {r['MRR']:<8.4f} | {r['P50']:<9.2f} | {r['P70']:<9.2f} | {r['P100']:<9.2f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    run_chunking_experiments()
