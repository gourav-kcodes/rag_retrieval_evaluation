"""
End-to-end evaluation: builds both retrievers, scores them against the
labeled question set, then generates answers (extractive always, Claude if
ANTHROPIC_API_KEY is set) and scores those for correctness and faithfulness.

Run from the project root:
    python run_evaluation.py
"""

import json
import os

import pandas as pd

from src.corpus import load_corpus, load_eval_set, corpus_lookup
from src.retriever import TfidfRetriever, BM25Retriever
from src.evaluation import evaluate_retriever
from src.generator import ExtractiveGenerator, ClaudeGenerator
from src.faithfulness import answer_correctness, lexical_faithfulness, llm_judge_faithfulness
from src.visualize import retriever_comparison_plot, per_k_plot, generation_quality_plot, difficulty_breakdown_plot

RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
K_VALUES = [1, 3, 5]
BEST_RETRIEVER_K = 3  # how many docs feed into generation


def evaluate_retrieval(corpus, eval_set):
    retrievers = {"tfidf": TfidfRetriever(), "bm25": BM25Retriever()}
    results = {}
    difficulty_breakdown = {}

    for name, retriever in retrievers.items():
        retriever.fit(corpus)
        results[name] = evaluate_retriever(retriever, eval_set, k_values=K_VALUES)
        print(f"\n{name.upper()} retrieval results (all {len(eval_set)} questions):")
        for metric, value in results[name].items():
            print(f"  {metric:14s} {value:.3f}")

        difficulty_breakdown[name] = {}
        for difficulty in ["easy", "hard"]:
            subset = [item for item in eval_set if item["difficulty"] == difficulty]
            difficulty_breakdown[name][difficulty] = evaluate_retriever(retriever, subset, k_values=K_VALUES)

        print(f"  -- easy (n={sum(1 for i in eval_set if i['difficulty']=='easy')}) vs "
              f"hard (n={sum(1 for i in eval_set if i['difficulty']=='hard')}), key metrics --")
        for metric in ["precision@3", "recall@3", "ndcg@3", "mrr"]:
            easy_val = difficulty_breakdown[name]["easy"][metric]
            hard_val = difficulty_breakdown[name]["hard"][metric]
            print(f"  {metric:14s} easy={easy_val:.3f}   hard={hard_val:.3f}")

    retriever_comparison_plot(results, f"{FIGURES_DIR}/retriever_comparison.png")
    per_k_plot(results, K_VALUES, f"{FIGURES_DIR}/precision_recall_by_k.png")
    difficulty_breakdown_plot(difficulty_breakdown, f"{FIGURES_DIR}/difficulty_breakdown.png")

    pd.DataFrame(results).T.to_csv(f"{RESULTS_DIR}/retrieval_metrics.csv")
    with open(f"{RESULTS_DIR}/retrieval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    difficulty_rows = []
    for name in retrievers:
        for difficulty in ["easy", "hard"]:
            row = {"retriever": name, "difficulty": difficulty, **difficulty_breakdown[name][difficulty]}
            difficulty_rows.append(row)
    pd.DataFrame(difficulty_rows).to_csv(f"{RESULTS_DIR}/retrieval_metrics_by_difficulty.csv", index=False)

    # pick the better retriever (by ndcg@3) to feed the generation step
    best_name = max(results, key=lambda name: results[name]["ndcg@3"])
    return retrievers[best_name], best_name


def evaluate_generation(corpus, eval_set, retriever, retriever_name):
    doc_lookup = corpus_lookup(corpus)
    use_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))

    backends = {"extractive": ExtractiveGenerator()}
    if use_claude:
        try:
            backends["claude"] = ClaudeGenerator()
            print("\nANTHROPIC_API_KEY found — running Claude generation backend too.")
        except Exception as e:
            print(f"\nCould not initialize Claude backend ({e}); continuing with extractive only.")
    else:
        print("\nNo ANTHROPIC_API_KEY set — running extractive backend only.")

    backend_results = {}
    all_rows = []

    for backend_name, generator in backends.items():
        correctness_scores, faithfulness_scores = [], []

        for item in eval_set:
            retrieved = retriever.retrieve(item["question"], k=BEST_RETRIEVER_K)
            retrieved_docs = [doc_lookup[doc_id] for doc_id, _ in retrieved]

            answer = generator.generate(item["question"], retrieved_docs)
            correctness = answer_correctness(answer, item["reference_answer"])
            faithfulness = lexical_faithfulness(answer, retrieved_docs)

            correctness_scores.append(correctness)
            faithfulness_scores.append(faithfulness)

            row = {
                "qid": item["qid"],
                "backend": backend_name,
                "question": item["question"],
                "generated_answer": answer,
                "reference_answer": item["reference_answer"],
                "correctness": correctness,
                "lexical_faithfulness": faithfulness,
            }

            if backend_name == "claude":
                judge_score = llm_judge_faithfulness(item["question"], answer, retrieved_docs)
                row["llm_judge_faithfulness"] = judge_score

            all_rows.append(row)

        backend_results[backend_name] = {
            "correctness": sum(correctness_scores) / len(correctness_scores),
            "lexical_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
        }
        print(f"\n{backend_name.upper()} generation results (retrieval: {retriever_name}, k={BEST_RETRIEVER_K}):")
        for metric, value in backend_results[backend_name].items():
            print(f"  {metric:22s} {value:.3f}")

    pd.DataFrame(all_rows).to_csv(f"{RESULTS_DIR}/generation_details.csv", index=False)
    pd.DataFrame(backend_results).T.to_csv(f"{RESULTS_DIR}/generation_metrics.csv")
    generation_quality_plot(backend_results, f"{FIGURES_DIR}/generation_quality.png")

    return backend_results


def main():
    corpus = load_corpus()
    eval_set = load_eval_set()
    print(f"Corpus: {len(corpus)} documents. Eval set: {len(eval_set)} labeled questions.")

    best_retriever, best_name = evaluate_retrieval(corpus, eval_set)
    print(f"\nUsing {best_name.upper()} (higher nDCG@3) to feed the generation step.")
    evaluate_generation(corpus, eval_set, best_retriever, best_name)

    print("\nDone. Results saved to results/, figures saved to results/figures/")


if __name__ == "__main__":
    main()
