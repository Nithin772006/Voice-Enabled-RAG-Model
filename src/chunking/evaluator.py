import time
from typing import List, Dict, Any
from src.chunking.fixed_chunker import FixedSizeChunker
from src.chunking.sentence_chunker import SentenceAwareChunker
from src.chunking.semantic_chunker import SemanticChunker
from src.chunking.metadata_chunker import MetadataAwareChunker
from src.chunking.adaptive_chunker import AdaptiveChunker

def get_chunker(strategy_name: str):
    name = strategy_name.lower().strip()
    if name == "fixed":
        return FixedSizeChunker()
    elif name == "sentence":
        return SentenceAwareChunker()
    elif name == "semantic":
        return SemanticChunker()
    elif name == "metadata":
        return MetadataAwareChunker()
    elif name == "adaptive":
        return AdaptiveChunker()
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy_name}")

def evaluate_chunking_strategies(sample_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compare all 5 chunking strategies on dataset sample records."""
    strategies = ["fixed", "sentence", "semantic", "metadata", "adaptive"]
    report = {}

    for strat_name in strategies:
        chunker = get_chunker(strat_name)
        t_start = time.perf_counter()
        chunks = chunker.chunk_records(sample_records)
        elapsed_sec = time.perf_counter() - t_start

        total_chunks = len(chunks)
        chunk_lens = [len(c["text"]) for c in chunks] if chunks else [0]
        selected_chunks = sum(1 for c in chunks if c.get("is_selected", 0) == 1)

        report[strat_name] = {
            "strategy": strat_name,
            "total_chunks_produced": total_chunks,
            "selected_ground_truth_chunks": selected_chunks,
            "avg_chunk_length_chars": round(sum(chunk_lens) / float(max(1, total_chunks)), 1),
            "min_chunk_length_chars": min(chunk_lens),
            "max_chunk_length_chars": max(chunk_lens),
            "chunking_time_sec": round(elapsed_sec, 4),
            "throughput_chunks_per_sec": round(total_chunks / max(0.0001, elapsed_sec), 1)
        }

    return report
