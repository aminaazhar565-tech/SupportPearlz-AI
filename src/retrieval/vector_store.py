from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import settings

COLLECTION = "supportpearlz"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def build_vector_store(chunks):
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(settings.vector_store),
        collection_name=COLLECTION,
    )

def load_vector_store():
    return Chroma(
        persist_directory=str(settings.vector_store),
        embedding_function=get_embeddings(),
        collection_name=COLLECTION,
    )

def index_exists() -> bool:
    return settings.vector_store.exists() and any(settings.vector_store.iterdir())
