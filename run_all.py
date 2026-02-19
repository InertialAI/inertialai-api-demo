"""CLI entry point: embed → store → eval."""

import argparse

from datasets import load_all_datasets
from embed import embed_all_datasets
from eval import run_eval
from store import store_all


def main():
    parser = argparse.ArgumentParser(description="InertialAI API Guide — full pipeline")
    parser.add_argument(
        "--embed-only", action="store_true", help="Only embed and store (skip eval)"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evals (assumes embeddings exist)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Re-embed even if cache exists"
    )
    args = parser.parse_args()

    use_cache = not args.no_cache

    # Load datasets
    print("Loading datasets...")
    datasets = load_all_datasets()

    if not args.eval_only:
        # Embed
        print("\nEmbedding (TS-only)...")
        ts_embeddings = embed_all_datasets(datasets, mode="ts_only", use_cache=use_cache)

        print("\nEmbedding (Multi-modal)...")
        mm_embeddings = embed_all_datasets(
            datasets, mode="multimodal", use_cache=use_cache
        )

        # Store
        print("\nStoring in ChromaDB...")
        store_all(datasets, ts_embeddings, mode="ts_only")
        store_all(datasets, mm_embeddings, mode="multimodal")
    else:
        # Load cached embeddings for eval
        from config import EMBEDDINGS_DIR

        import numpy as np

        ts_embeddings = {}
        mm_embeddings = {}
        for name in datasets:
            ts_embeddings[name] = {
                "train": np.load(EMBEDDINGS_DIR / f"{name}_train_ts_only.npy"),
                "test": np.load(EMBEDDINGS_DIR / f"{name}_test_ts_only.npy"),
            }
            mm_embeddings[name] = {
                "train": np.load(EMBEDDINGS_DIR / f"{name}_train_multimodal.npy"),
                "test": np.load(EMBEDDINGS_DIR / f"{name}_test_multimodal.npy"),
            }

    if args.embed_only:
        print("\nDone (embed-only mode).")
        return

    # Eval
    print("\nRunning evaluations...")
    run_eval(datasets, ts_embeddings, mm_embeddings)
    print("\nDone.")


if __name__ == "__main__":
    main()
