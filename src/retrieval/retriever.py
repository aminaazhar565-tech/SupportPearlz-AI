from src.retrieval.vector_store import load_vector_store

def retrieve(query: str, k: int, threshold: float):
    store = load_vector_store()
    # Chroma returns distance; smaller is better. Convert to a simple relevance score.
    results = store.similarity_search_with_score(query, k=k)
    cleaned = []
    for doc, distance in results:
        score = max(0.0, 1.0 - float(distance))
        doc.metadata["retrieval_score"] = round(score, 4)
        if score >= threshold:
            cleaned.append(doc)
    return cleaned
