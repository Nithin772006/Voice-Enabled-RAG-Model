import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chunking.fixed_chunker import FixedSizeChunker
from src.chunking.sentence_chunker import SentenceAwareChunker
from src.chunking.semantic_chunker import SemanticChunker
from src.chunking.metadata_chunker import MetadataAwareChunker
from src.chunking.adaptive_chunker import AdaptiveChunker

SAMPLE_DOC_META = {
    "document_id": "doc_test_100",
    "original_record_id": "100",
    "language": "tam_Taml",
    "source": "MSMARCO-XI",
    "passage_idx": 0,
    "is_selected": 1,
    "query": "இந்தியாவின் தலைநகரம் என்ன?"
}

SAMPLE_TEXT = "இந்தியக் குடியரசின் தலைநகரம் புது தில்லி ஆகும். இது நாட்டின் முக்கிய அரசியல் மையமாகும். இது வடக்கு இந்தியாவில் அமைந்துள்ளது."

class TestChunking(unittest.TestCase):

    def test_fixed_chunker(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=10)
        chunks = chunker.chunk_text(SAMPLE_TEXT, SAMPLE_DOC_META)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_strategy, "fixed_size")
        self.assertEqual(chunks[0].document_id, "doc_test_100")

    def test_sentence_chunker(self):
        chunker = SentenceAwareChunker(max_chars_per_chunk=100)
        chunks = chunker.chunk_text(SAMPLE_TEXT, SAMPLE_DOC_META)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_strategy, "sentence_aware")

    def test_semantic_chunker(self):
        chunker = SemanticChunker(max_chunk_chars=120)
        chunks = chunker.chunk_text(SAMPLE_TEXT, SAMPLE_DOC_META)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_strategy, "semantic")

    def test_metadata_chunker(self):
        chunker = MetadataAwareChunker(max_chars=120)
        chunks = chunker.chunk_text(SAMPLE_TEXT, SAMPLE_DOC_META)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertIn("[Context:", chunks[0].text)

    def test_adaptive_chunker(self):
        chunker = AdaptiveChunker(short_threshold=200)
        chunks = chunker.chunk_text(SAMPLE_TEXT, SAMPLE_DOC_META)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertIn("adaptive", chunks[0].chunk_strategy)

if __name__ == "__main__":
    unittest.main()
