"""Plots for the evaluation results."""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10


def retriever_comparison_plot(results_by_retriever: dict, save_path: str) -> None:
    """Grouped bar chart comparing retrievers across the headline metrics."""
    metrics = ["precision@3", "recall@3", "ndcg@3", "mrr"]
    retrievers = list(results_by_retriever.keys())

    x = np.arange(len(metrics))
    width = 0.8 / len(retrievers)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, retriever_name in enumerate(retrievers):
        values = [results_by_retriever[retriever_name][m] for m in metrics]
        ax.bar(x + i * width, values, width, label=retriever_name.upper())

    ax.set_xticks(x + width * (len(retrievers) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval Quality: TF-IDF vs. BM25")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def per_k_plot(results_by_retriever: dict, k_values: list[int], save_path: str) -> None:
    """How precision and recall change as k grows, for each retriever."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for retriever_name, results in results_by_retriever.items():
        precisions = [results[f"precision@{k}"] for k in k_values]
        recalls = [results[f"recall@{k}"] for k in k_values]
        axes[0].plot(k_values, precisions, marker="o", label=retriever_name.upper())
        axes[1].plot(k_values, recalls, marker="o", label=retriever_name.upper())

    axes[0].set_title("Precision@k")
    axes[1].set_title("Recall@k")
    for ax in axes:
        ax.set_xlabel("k")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_xticks(k_values)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def difficulty_breakdown_plot(difficulty_results: dict, save_path: str) -> None:
    """Shows the same metric for each retriever, split by easy (lexically
    aligned) vs hard (paraphrased) questions — this is where the real
    difference between retrievers, and the real limitation of lexical
    retrieval generally, actually shows up."""
    metrics = ["precision@3", "recall@3", "ndcg@3", "mrr"]
    retrievers = list(difficulty_results.keys())

    fig, axes = plt.subplots(1, len(retrievers), figsize=(6 * len(retrievers), 4.5), sharey=True)
    if len(retrievers) == 1:
        axes = [axes]

    x = np.arange(len(metrics))
    width = 0.35

    for ax, retriever_name in zip(axes, retrievers):
        easy_vals = [difficulty_results[retriever_name]["easy"][m] for m in metrics]
        hard_vals = [difficulty_results[retriever_name]["hard"][m] for m in metrics]
        ax.bar(x - width / 2, easy_vals, width, label="Easy (lexically aligned)", color="#4c72b0")
        ax.bar(x + width / 2, hard_vals, width, label="Hard (paraphrased)", color="#dd8452")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.05)
        ax.set_title(retriever_name.upper())
        ax.legend()

    axes[0].set_ylabel("Score")
    fig.suptitle("Retrieval Quality Collapses on Paraphrased Questions")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)



def generation_quality_plot(results_by_backend: dict, save_path: str) -> None:
    """Mean correctness and faithfulness per generation backend."""
    backends = list(results_by_backend.keys())
    metrics = ["correctness", "lexical_faithfulness"]

    x = np.arange(len(metrics))
    width = 0.8 / len(backends)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, backend in enumerate(backends):
        values = [results_by_backend[backend][m] for m in metrics]
        ax.bar(x + i * width, values, width, label=backend)

    ax.set_xticks(x + width * (len(backends) - 1) / 2)
    ax.set_xticklabels(["Correctness (vs. reference)", "Lexical faithfulness (vs. context)"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Generation Quality by Backend")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
