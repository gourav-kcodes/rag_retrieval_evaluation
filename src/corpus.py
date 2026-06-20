import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_corpus() -> list[dict]:
    """Returns a list of {doc_id, title, text} dicts."""
    with open(DATA_DIR / "corpus.json") as f:
        return json.load(f)


def load_eval_set() -> list[dict]:
    """Returns a list of {qid, question, relevant_docs, reference_answer} dicts."""
    with open(DATA_DIR / "eval_set.json") as f:
        return json.load(f)


def corpus_lookup(corpus: list[dict]) -> dict:
    """doc_id -> document dict, for quick access after retrieval returns IDs."""
    return {doc["doc_id"]: doc for doc in corpus}
