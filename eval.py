"""Retrieval evaluation — label matching with matryoshka dimension support."""

import json

import numpy as np

from config import MATRYOSHKA_DIMS, RESULTS_DIR, TOP_K_VALUES


def _truncate_and_normalize(embs: np.ndarray, dim: int) -> np.ndarray:
    """Truncate embeddings to `dim` dimensions and L2-normalize."""
    trunc = embs[:, :dim].copy()
    norms = np.linalg.norm(trunc, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return trunc / norms


def _cosine_topk(
    queries: np.ndarray, corpus: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return top-k indices and similarities for each query (cosine similarity).

    Args:
        queries: (n_queries, dim) L2-normalized
        corpus: (n_corpus, dim) L2-normalized
        k: number of results

    Returns:
        indices: (n_queries, k)
        similarities: (n_queries, k)
    """
    sims = queries @ corpus.T  # (n_queries, n_corpus)
    k = min(k, sims.shape[1])
    top_idx = np.argpartition(-sims, k, axis=1)[:, :k]
    # Sort the top-k by descending similarity
    rows = np.arange(len(queries))[:, None]
    top_sims = sims[rows, top_idx]
    sort_order = np.argsort(-top_sims, axis=1)
    top_idx = top_idx[rows, sort_order]
    top_sims = top_sims[rows, sort_order]
    return top_idx, top_sims


def eval_label_retrieval(
    datasets: dict[str, dict],
    embeddings: dict[str, dict[str, np.ndarray]],
    dim: int,
) -> dict:
    """Per-dataset label retrieval using truncated embeddings (matryoshka).

    For each dataset, queries test embeddings against the same dataset's training
    embeddings and checks if top-k neighbors share the query label.
    """
    results = {}

    for name, ds in datasets.items():
        test_embs = _truncate_and_normalize(embeddings[name]["test"], dim)
        train_embs = _truncate_and_normalize(embeddings[name]["train"], dim)
        y_test = ds["y_test"]
        y_train = ds["y_train"]
        n_test = len(y_test)

        top_idx, _ = _cosine_topk(test_embs, train_embs, max(TOP_K_VALUES))

        metrics = {}
        for k in TOP_K_VALUES:
            correct = 0
            for i in range(n_test):
                retrieved_labels = [str(y_train[top_idx[i, j]]) for j in range(k)]
                if str(y_test[i]) in retrieved_labels:
                    correct += 1
            metrics[f"P@{k}"] = round(correct / n_test, 4)

        metrics["1nn_accuracy"] = metrics["P@1"]
        results[name] = metrics
        print(f"  {name} (dim={dim}): {metrics}")

    return results


def eval_global_label_retrieval(
    datasets: dict[str, dict],
    embeddings: dict[str, dict[str, np.ndarray]],
    dim: int,
) -> dict:
    """Global label retrieval: query each test sample against ALL training data.

    Checks if the top-k nearest neighbors (from any dataset) share the query label.
    """
    # Build global training corpus
    all_train_embs = []
    all_train_labels = []
    all_train_datasets = []
    for name, ds in datasets.items():
        train_embs = embeddings[name]["train"]
        all_train_embs.append(train_embs)
        for lbl in ds["y_train"]:
            all_train_labels.append(str(lbl))
            all_train_datasets.append(name)

    global_train = np.concatenate(all_train_embs, axis=0)
    global_train_norm = _truncate_and_normalize(global_train, dim)

    results = {}
    for name, ds in datasets.items():
        test_embs = _truncate_and_normalize(embeddings[name]["test"], dim)
        y_test = ds["y_test"]
        n_test = len(y_test)

        top_idx, _ = _cosine_topk(test_embs, global_train_norm, max(TOP_K_VALUES))

        metrics = {}
        for k in TOP_K_VALUES:
            correct = 0
            for i in range(n_test):
                retrieved_labels = [all_train_labels[top_idx[i, j]] for j in range(k)]
                if str(y_test[i]) in retrieved_labels:
                    correct += 1
            metrics[f"P@{k}"] = round(correct / n_test, 4)

        metrics["1nn_accuracy"] = metrics["P@1"]
        results[name] = metrics
        print(f"  {name} (dim={dim}, global): {metrics}")

    return results


def run_eval(
    datasets: dict[str, dict],
    ts_embeddings: dict[str, dict[str, np.ndarray]],
    mm_embeddings: dict[str, dict[str, np.ndarray]],
) -> dict:
    """Run all evaluations across matryoshka dimensions and save results."""
    all_results = {}

    for dim in MATRYOSHKA_DIMS:
        print(f"\n=== Label Retrieval — TS-only (dim={dim}) ===")
        all_results[f"label_retrieval_ts_only_dim{dim}"] = eval_label_retrieval(
            datasets, ts_embeddings, dim
        )

        print(f"\n=== Label Retrieval — Multi-modal (dim={dim}) ===")
        all_results[f"label_retrieval_multimodal_dim{dim}"] = eval_label_retrieval(
            datasets, mm_embeddings, dim
        )

        print(f"\n=== Global Label Retrieval — TS-only (dim={dim}) ===")
        all_results[f"global_label_retrieval_ts_only_dim{dim}"] = (
            eval_global_label_retrieval(datasets, ts_embeddings, dim)
        )

        print(f"\n=== Global Label Retrieval — Multi-modal (dim={dim}) ===")
        all_results[f"global_label_retrieval_multimodal_dim{dim}"] = (
            eval_global_label_retrieval(datasets, mm_embeddings, dim)
        )

    # Comparison summary at full dimension
    full_dim = MATRYOSHKA_DIMS[-1]
    print(f"\n=== TS-only vs Multi-modal — 1-NN Accuracy (dim={full_dim}) ===")
    for name in datasets:
        ts_acc = all_results[f"label_retrieval_ts_only_dim{full_dim}"][name][
            "1nn_accuracy"
        ]
        mm_acc = all_results[f"label_retrieval_multimodal_dim{full_dim}"][name][
            "1nn_accuracy"
        ]
        delta = round(mm_acc - ts_acc, 4)
        sign = "+" if delta >= 0 else ""
        print(f"  {name:25s}  TS={ts_acc:.4f}  MM={mm_acc:.4f}  Δ={sign}{delta}")

    # Save
    output_path = RESULTS_DIR / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return all_results
