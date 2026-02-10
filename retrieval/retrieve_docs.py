from pathlib import Path
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DOCS_DIR = Path("data/docs")


def load_documents() -> Tuple[List[str], List[str]]:
    """
    Loads all markdown documents from data/docs.

    Returns:
        docs: list of document text
        doc_names: list of filenames (same order as docs)
    """
    docs = []
    doc_names = []

    for doc_path in DOCS_DIR.glob("*.md"):
        text = doc_path.read_text()
        docs.append(text)
        doc_names.append(doc_path.name)

    return docs, doc_names


def retrieve_relevant_docs(
    query: str,
    top_k: int = 2
) -> List[str]:
    """
    Retrieves top_k most relevant documents for a given query
    using TF-IDF + cosine similarity.

    Args:
        query: DevOps request text
        top_k: number of docs to retrieve

    Returns:
        List of document texts
    """
    docs, _ = load_documents()

    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(docs)
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, doc_vectors)[0]
    top_indices = similarities.argsort()[::-1][:top_k]

    return [docs[i] for i in top_indices]
