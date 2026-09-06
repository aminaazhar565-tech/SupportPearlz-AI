from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

@dataclass(frozen=True)
class Settings:
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    temperature: float = float(os.getenv("TEMPERATURE", "0"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))
    score_threshold: float = float(os.getenv("SCORE_THRESHOLD", "0.25"))
    knowledge_base: Path = ROOT / "data" / "knowledge_base"
    vector_store: Path = ROOT / "data" / "vector_store"
    log_dir: Path = ROOT / "logs"

settings = Settings()
settings.vector_store.mkdir(parents=True, exist_ok=True)
settings.log_dir.mkdir(parents=True, exist_ok=True)

def masked_config(api_key: str | None = None) -> dict:
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    masked = f"{key[:5]}...{key[-4:]}" if len(key) > 12 else ("set" if key else "not set")
    return {
        "openai_model": settings.openai_model,
        "embedding_model": settings.embedding_model,
        "temperature": settings.temperature,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "retrieval_k": settings.retrieval_k,
        "score_threshold": settings.score_threshold,
        "openai_api_key": masked,
    }

if __name__ == "__main__":
    print(masked_config())
