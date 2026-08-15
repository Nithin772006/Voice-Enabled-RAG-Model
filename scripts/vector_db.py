import logging
import sys
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Reconfigure stdout to support UTF-8 on Windows terminal/console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("VectorDatabaseSetup")

def init_qdrant_local(db_path: str = "data/qdrant_db") -> QdrantClient:
    """Initializes and returns a Qdrant client running in local persistent storage mode.
    This creates files inside the specified path, providing full persistence
    without needing a running Docker container.
    """
    db_dir = Path(db_path).resolve()
    db_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Connecting to local Qdrant database at: {db_dir}")
    return QdrantClient(path=str(db_dir))

def create_collection_if_not_exists(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """Creates a collection with specified parameters if it doesn't already exist."""
    # Check if collection exists
    exists = client.collection_exists(collection_name=collection_name)
    if exists:
        logger.info(f"Collection '{collection_name}' already exists. Skipping creation.")
        return
        
    logger.info(f"Creating collection '{collection_name}' (dim: {vector_size}, metric: COSINE)...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )
    logger.info(f"Collection '{collection_name}' created successfully.")

if __name__ == "__main__":
    # Test local connection and collection creation
    client = init_qdrant_local("data/qdrant_db")
    
    # We use 1024 dimensions to match our recommended BAAI/bge-m3 embeddings
    create_collection_if_not_exists(
        client=client,
        collection_name="tamil_rag_chunks",
        vector_size=1024
    )
    
    # Inspect collections list
    collections = client.get_collections()
    logger.info(f"Active collections in database: {[col.name for col in collections.collections]}")
