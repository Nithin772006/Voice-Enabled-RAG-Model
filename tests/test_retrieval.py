import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexing.sparse_index import SparseIndexer
from src.retrieval.sparse import SparseRetriever

TEST_CHUNKS = [
    {
        "chunk_id": "c1",
        "document_id": "d1",
        "text": "இந்தியாவின் தலைநகரம் புது தில்லி ஆகும்.",
        "language": "tam_Taml",
        "original_record_id": "1",
        "passage_idx": 0,
        "is_selected": 1
    },
    {
        "chunk_id": "c2",
        "document_id": "d2",
        "text": "பிரான்ஸ் நாட்டின் தலைநகரம் பாரிஸ் ஆகும்.",
        "language": "tam_Taml",
        "original_record_id": "2",
        "passage_idx": 0,
        "is_selected": 0
    }
]

class TestRetrieval(unittest.TestCase):

    def test_bm25_sparse_retriever(self):
        sparse_indexer = SparseIndexer()
        texts = [c["text"] for c in TEST_CHUNKS]
        sparse_indexer.build_index(texts)

        retriever = SparseRetriever(sparse_indexer, TEST_CHUNKS)
        results, search_ms = retriever.retrieve("இந்தியாவின் தலைநகரம்", top_k=2)

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["chunk_id"], "c1")
        self.assertGreaterEqual(search_ms, 0.0)

if __name__ == "__main__":
    unittest.main()
