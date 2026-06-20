"""
Sanity checks for the retrieval metrics, faithfulness scoring, and
retrievers — using small, hand-constructed examples where the correct
answer can be verified independently, the same philosophy as the metric
formulas being written out explicitly rather than imported as a black box.
"""

import math

from src.evaluation import precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k
from src.faithfulness import answer_correctness, lexical_faithfulness
from src.retriever import TfidfRetriever, BM25Retriever

RANKED = ["a", "b", "c", "d", "e"]
RELEVANT = {"b", "d"}


def test_precision_at_k():
    assert precision_at_k(RANKED, RELEVANT, 1) == 0.0  # 'a' is not relevant
    assert math.isclose(precision_at_k(RANKED, RELEVANT, 3), 1 / 3)  # only 'b' in top 3
    assert precision_at_k(RANKED, RELEVANT, 5) == 0.4  # 'b' and 'd' out of 5


def test_recall_at_k():
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.0
    assert recall_at_k(RANKED, RELEVANT, 3) == 0.5  # found 1 of 2 relevant docs
    assert recall_at_k(RANKED, RELEVANT, 5) == 1.0  # found both


def test_recall_with_no_relevant_docs_is_zero_not_error():
    assert recall_at_k(RANKED, set(), 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, RELEVANT) == 0.5  # first relevant ('b') is at rank 2
    assert reciprocal_rank(RANKED, {"z"}) == 0.0  # nothing relevant ever appears


def test_ndcg_perfect_ranking_is_one():
    perfect_order = ["b", "d", "a", "c", "e"]  # both relevant docs ranked first
    assert math.isclose(ndcg_at_k(perfect_order, RELEVANT, 3), 1.0)


def test_ndcg_matches_independent_calculation():
    # b (relevant) at rank 2, nothing else relevant in top 3
    dcg = 1 / math.log2(3)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)  # ideal: both relevant docs in ranks 1-2
    expected = dcg / idcg
    assert math.isclose(ndcg_at_k(RANKED, RELEVANT, 3), expected)


def test_ndcg_no_relevant_docs_is_zero():
    assert ndcg_at_k(RANKED, set(), 3) == 0.0


def test_answer_correctness_identical_strings():
    assert answer_correctness("the cat sat on the mat", "the cat sat on the mat") == 1.0


def test_answer_correctness_disjoint_strings():
    assert answer_correctness("completely unrelated content", "totally different words here") == 0.0


def test_answer_correctness_partial_overlap_is_between_zero_and_one():
    score = answer_correctness("the model overfit the training data", "overfitting happens with too little data")
    assert 0.0 < score < 1.0


def test_lexical_faithfulness_fully_grounded_answer():
    context = [{"text": "Overfitting happens when a model learns noise in the training data."}]
    answer = "Overfitting happens when a model learns noise in the training data."
    assert lexical_faithfulness(answer, context) == 1.0


def test_lexical_faithfulness_fabricated_content_scores_lower():
    context = [{"text": "Overfitting happens when a model learns noise in the training data."}]
    fabricated = "Overfitting was first described by quantum physicists studying neural galaxies."
    score = lexical_faithfulness(fabricated, context)
    assert score < 0.5


def test_retrievers_rank_the_obviously_relevant_document_first():
    corpus = [
        {"doc_id": "d1", "title": "Penguins", "text": "Penguins are flightless birds that live in Antarctica and swim well."},
        {"doc_id": "d2", "title": "Volcanoes", "text": "Volcanoes form when magma rises through cracks in the Earth's crust."},
        {"doc_id": "d3", "title": "Penguins again", "text": "Emperor penguins are the largest penguin species and breed on sea ice."},
    ]
    query = "How do penguins survive in Antarctica?"

    for Retriever in (TfidfRetriever, BM25Retriever):
        retriever = Retriever()
        retriever.fit(corpus)
        top_result = retriever.retrieve(query, k=1)[0][0]
        assert top_result in {"d1", "d3"}, f"{Retriever.__name__} ranked an unrelated volcano doc first"
