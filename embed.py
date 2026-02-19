"""Embedding pipeline — TS-only and multi-modal modes."""

import numpy as np
from tqdm import tqdm

from client import embed, get_client
from config import BATCH_SIZE, EMBEDDINGS_DIR


def _ts_to_list(X_sample) -> list[list[float]]:
    """Convert an aeon sample (n_channels, seq_len) to list-of-lists."""
    return [channel.tolist() for channel in X_sample]


def embed_dataset(
    dataset: dict,
    split: str,
    mode: str = "ts_only",
    use_cache: bool = True,
) -> np.ndarray:
    """Embed one split of a dataset. Returns (n_samples, embed_dim) array.

    Args:
        dataset: Dict from datasets.load_dataset().
        split: "train" or "test".
        mode: "ts_only" or "multimodal".
        use_cache: If True, load from disk when available.
    """
    name = dataset["name"]
    cache_path = EMBEDDINGS_DIR / f"{name}_{split}_{mode}.npy"

    if use_cache and cache_path.exists():
        return np.load(cache_path)

    X = dataset[f"X_{split}"]
    description = dataset["description"]
    client = get_client()

    inputs = []
    for i in range(len(X)):
        ts = _ts_to_list(X[i])
        if mode == "multimodal":
            inputs.append({"time_series": ts, "text": description})
        else:
            inputs.append({"time_series": ts})

    # Embed with progress bar
    all_embeddings: list[list[float]] = []
    for start in tqdm(
        range(0, len(inputs), BATCH_SIZE),
        desc=f"{name}/{split}/{mode}",
        unit="batch",
    ):
        batch = inputs[start : start + BATCH_SIZE]
        batch_embs = embed(batch, client=client)
        all_embeddings.extend(batch_embs)

    arr = np.array(all_embeddings)
    np.save(cache_path, arr)
    return arr


def embed_all_datasets(
    datasets: dict[str, dict],
    mode: str = "ts_only",
    use_cache: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """Embed all datasets for both splits. Returns {name: {train: arr, test: arr}}."""
    results = {}
    for name, ds in datasets.items():
        results[name] = {
            "train": embed_dataset(ds, "train", mode=mode, use_cache=use_cache),
            "test": embed_dataset(ds, "test", mode=mode, use_cache=use_cache),
        }
    return results
