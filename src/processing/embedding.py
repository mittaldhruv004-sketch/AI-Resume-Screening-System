"""Chunk-aware semantic embeddings with a local TF-IDF fallback."""
from typing import List
import numpy as np

_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDER = None


def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDER = SentenceTransformer(_MODEL_ID)
        except Exception:
            _EMBEDDER = "tfidf"
    return _EMBEDDER


def _chunks(text: str, words_per_chunk: int = 100) -> List[str]:
    words = (text or "").split()
    return [" ".join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)] or [""]


def embed_texts(embedder, texts: List[str]) -> np.ndarray:
    if embedder != "tfidf":
        vectors = []
        for text in texts:
            chunks = _chunks(text)
            chunk_vectors = embedder.encode(chunks, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            vector = np.asarray(chunk_vectors, dtype=np.float32).mean(axis=0)
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.vstack(vectors).astype(np.float32)
    from sklearn.feature_extraction.text import TfidfVectorizer
    # TF-IDF fallback is deterministic and works offline; chunk text is not silently truncated.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=12000, stop_words="english")
    try:
        return vectorizer.fit_transform(texts).toarray().astype(np.float32)
    except ValueError:
        return np.zeros((len(texts), 1), dtype=np.float32)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    return normalized @ normalized.T
