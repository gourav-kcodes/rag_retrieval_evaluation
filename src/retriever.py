"""
Two retrievers, sharing a common interface, so the evaluation harness can
compare them directly rather than just trusting one method blindly.

TF-IDF + cosine similarity is the classic vector-space baseline. BM25 is a
generally stronger keyword-ranking function that accounts for document
length and term-frequency saturation (see data/corpus.json's own BM25 entry
for the full explanation).
"""

import re
from abc import ABC, abstractmethod

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace. Deliberately simple —
    no stemming or stopword removal, so retrieval quality reflects the
    ranking function itself, not extra preprocessing tricks."""
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever(ABC):
    name: str

    @abstractmethod
    def fit(self, corpus: list[dict]) -> None:
        ...

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Returns up to k (doc_id, score) pairs, sorted by score descending."""
        ...


class TfidfRetriever(Retriever):
    name = "tfidf"

    def fit(self, corpus: list[dict]) -> None:
        self.doc_ids = [doc["doc_id"] for doc in corpus]
        texts = [doc["title"] + ". " + doc["text"] for doc in corpus]
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        self.doc_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_matrix)[0]
        ranked = sorted(zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]


class BM25Retriever(Retriever):
    name = "bm25"

    def fit(self, corpus: list[dict]) -> None:
        self.doc_ids = [doc["doc_id"] for doc in corpus]
        tokenized_docs = [tokenize(doc["title"] + ". " + doc["text"]) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_docs)

    def retrieve(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]
