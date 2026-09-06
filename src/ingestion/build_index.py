import argparse
from src.config import settings
from src.ingestion.loaders import load_knowledge_base
from src.ingestion.chunking import split_documents
from src.retrieval.vector_store import build_vector_store

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if args.rebuild and settings.vector_store.exists():
        import shutil
        shutil.rmtree(settings.vector_store)
        settings.vector_store.mkdir(parents=True, exist_ok=True)

    docs, skipped = load_knowledge_base(settings.knowledge_base)
    chunks = split_documents(docs)
    build_vector_store(chunks)

    print(f"Loaded documents: {len(docs)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Skipped: {len(skipped)}")
    for item in skipped:
        print(f" - {item['file']}: {item['reason']}")

if __name__ == "__main__":
    main()
