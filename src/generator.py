"""
Two interchangeable answer generators behind a common interface.

ExtractiveGenerator needs nothing beyond what's already installed - it picks
the most relevant sentence(s) out of the retrieved documents themselves, so
the answer is always a real excerpt, not a paraphrase. It can't hallucinate
in the usual sense, but it also can't synthesize across documents or phrase
things naturally.
"""

import os
import re
from abc import ABC, abstractmethod

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.retriever import tokenize


def split_sentences(text: str) -> list[str]:
    """Simple period-based sentence splitter. The corpus here was written
    with clean sentence boundaries, so this is sufficient without pulling in
    a full sentence-tokenization library."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


class Generator(ABC):
    name: str

    @abstractmethod
    def generate(self, query: str, retrieved_docs: list[dict]) -> str:
        ...


class ExtractiveGenerator(Generator):
    name = "extractive"

    def generate(self, query: str, retrieved_docs: list[dict], top_n: int = 2) -> str:
        sentences = []
        for doc in retrieved_docs:
            sentences.extend(split_sentences(doc["text"]))

        if not sentences:
            return "No relevant context was retrieved."

        vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        sentence_matrix = vectorizer.fit_transform(sentences)
        query_vec = vectorizer.transform([query])
        scores = cosine_similarity(query_vec, sentence_matrix)[0]

        ranked = sorted(zip(sentences, scores), key=lambda pair: pair[1], reverse=True)
        top_sentences = [s for s, score in ranked[:top_n]]
        return " ".join(top_sentences)


class ClaudeGenerator(Generator):
    name = "claude"

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        import anthropic  # imported lazily so it's only required if this backend is used

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ClaudeGenerator requires the ANTHROPIC_API_KEY environment variable to be set."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, query: str, retrieved_docs: list[dict]) -> str:
        context = "\n\n".join(f"[{doc['title']}]\n{doc['text']}" for doc in retrieved_docs)
        prompt = (
            "Answer the question using only the context below. "
            "If the context doesn't contain the answer, say so explicitly rather than guessing.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer in 1-3 concise sentences."
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


def get_generator(backend: str = "extractive") -> Generator:
    if backend == "extractive":
        return ExtractiveGenerator()
    if backend == "claude":
        return ClaudeGenerator()
    raise ValueError(f"Unknown generator backend: {backend}")
