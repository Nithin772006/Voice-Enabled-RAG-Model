import torch
from dataclasses import dataclass
from pathlib import Path

@dataclass
class EmbeddingConfig:
    # Model configuration
    # BAAI/bge-m3 is recommended for multilingual Indic retrieval due to its state-of-the-art performance
    model_name: str = "BAAI/bge-m3"
    embedding_dim: int = 1024  # Standard dimension size for BGE-M3
    
    # Input/Output paths
    input_chunks_file: Path = Path("data/chunks/ta_chunks.jsonl")
    output_dir: Path = Path("data/embeddings")
    output_file_name: str = "ta_embeddings.parquet"
    
    # Inference parameters
    batch_size: int = 32  # Balanced default; tuned based on GPU memory
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpointing parameters (to resume long jobs)
    checkpoint_interval: int = 5000  # Save progress every 5000 records
    
    def __post_init__(self) -> None:
        self.input_chunks_file = Path(self.input_chunks_file).resolve()
        self.output_dir = Path(self.output_dir).resolve()
        
        # Ensure target output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @property
    def output_embeddings_path(self) -> Path:
        return self.output_dir / self.output_file_name
