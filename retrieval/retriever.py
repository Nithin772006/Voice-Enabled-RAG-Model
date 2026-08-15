import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Reconfigure stdout to support UTF-8 on Windows terminal/console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

logger = logging.getLogger("QdrantRetriever")

class QdrantRetriever:
    """Connects to the local Qdrant database to perform real vector similarity search using BGE-M3."""
    def __init__(
        self, 
        db_path: str = "data/qdrant_db", 
        collection_name: str = "tamil_rag_chunks",
        model_name: str = "BAAI/bge-m3"
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self.collection_name = collection_name
        
        logger.info(f"Connecting retriever to Qdrant at {self.db_path}")
        self.client = QdrantClient(path=str(self.db_path))
        
        logger.info(f"Loading query encoder model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        logger.info("Retriever initialization completed.")

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Performs a real vector similarity search on Qdrant.
        Encodes the query using BGE-M3 and retrieves the top_k most similar chunks.
        """
        logger.info(f"Retrieving context for query: {repr(query)}")
        
        # 1. Generate real query embedding vector
        # normalize_embeddings=True matches the COSINE distance parameter configuration in Qdrant
        query_vector = self.model.encode(query, normalize_embeddings=True).tolist()
        
        # 2. Search Qdrant using the real embedding
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        )
        
        # 3. Format results to standard schema
        retrieved = []
        for rank, hit in enumerate(results.points, 1):
            payload = hit.payload or {}
            retrieved.append({
                "rank": rank,
                "score": round(hit.score if hit.score is not None else 0.0, 4),
                "chunk_id": payload.get("chunk_id", f"unknown_{hit.id}"),
                "language": payload.get("language", "unknown"),
                "text": payload.get("text", "")
            })
            
        logger.info(f"Retrieved {len(retrieved)} real chunks from Qdrant.")
        return retrieved

if __name__ == "__main__":
    # Diagnostic test for real search
    logging.basicConfig(level=logging.INFO)
    
    # Instantiate the real retriever
    retriever = QdrantRetriever()
    
    # We query for company definition (first clean chunk in database)
    test_query = "ஒரு நிறுவனம் என்பது என்ன?"
    results = retriever.retrieve(test_query, top_k=2)
    
    print("\n--- REAL SEARCH RETRIEVED CHUNKS ---")
    for res in results:
        print(f"Rank {res['rank']} | Score: {res['score']} | Chunk ID: {res['chunk_id']}")
        print(f"Text: {res['text']}\n")
