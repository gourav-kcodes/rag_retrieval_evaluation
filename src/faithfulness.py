"""
Two things are being measured here, and they are not the same question:

1. Correctness — does the generated answer match the reference answer?
   Scored with token-level F1 overlap (the same style of metric used by the
   SQuAD benchmark), which rewards getting the right content words without
   requiring an exact string match.

2. Faithfulness — is the generated answer actually supported by the
   retrieved context, or did the model add things that aren't there?
   This matters separately from correctness: a generator could produce a
   correct-sounding answer that isn't actually grounded in what was
   retrieved, which is a hallucination even if the final answer happens to
   be right.

The lexical faithfulness score here is a proxy, not a perfect hallucination
detector — see the Limitations section in the README for exactly what it
can and can't catch. An optional LLM-judge faithfulness score is also
provided for a more semantic (but slower, API-dependent) check.
"""

import os
import re
from collections import Counter

from src.retriever import tokenize

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "then", "than", "so", "of", "to", "in", "on",
    "at", "for", "with", "by", "from", "as", "it", "its", "this", "that",
    "these", "those", "what", "which", "who", "how", "why", "do", "does",
    "did", "can", "could", "will", "would", "should", "not", "no", "into",
    "about", "over", "between", "also", "such", "their", "there", "each",
}


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def answer_correctness(prediction: str, reference: str) -> float:
    """Token-level F1 between a generated answer and the reference answer."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def lexical_faithfulness(answer: str, context_docs: list[dict]) -> float:
    """Fraction of the answer's substantive (non-stopword) words that appear
    somewhere in the retrieved context. A low score reliably flags
    unsupported content; a high score does not guarantee the answer isn't
    fabricated, since it can be fooled by an answer that reuses real context
    words in a combination the context never actually states."""
    context_text = " ".join(doc["text"] for doc in context_docs)
    context_tokens = set(tokenize(context_text))

    answer_tokens = tokenize(answer)
    content_tokens = [t for t in answer_tokens if t not in STOPWORDS and len(t) > 2]

    if not content_tokens:
        return 1.0  # no substantive claim was made, so there's nothing to be unfaithful about

    supported = sum(1 for t in content_tokens if t in context_tokens)
    return supported / len(content_tokens)


def llm_judge_faithfulness(query: str, answer: str, context_docs: list[dict]) -> float | None:
    """Asks Claude to rate whether the answer is supported by the context.
    Returns None (rather than raising) if no API key is configured, so
    callers can skip this check gracefully instead of crashing the whole
    evaluation run over an optional feature."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    context = "\n\n".join(doc["text"] for doc in context_docs)
    prompt = (
        "You are checking whether an answer is fully supported by the given context.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer: {answer}\n\n"
        "Reply with exactly one word: SUPPORTED if every claim in the answer is backed by "
        "the context, PARTIAL if some of it is unsupported, or UNSUPPORTED if the answer "
        "contradicts or invents information not in the context."
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    verdict = response.content[0].text.strip().upper()
    return {"SUPPORTED": 1.0, "PARTIAL": 0.5, "UNSUPPORTED": 0.0}.get(verdict)
