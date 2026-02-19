"""ChromaDB storage utilities for embeddings."""

import chromadb
import numpy as np

from config import CHROMA_DIR


def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def store_embeddings(
    chroma_client: chromadb.PersistentClient,
    collection_name: str,
    embeddings: np.ndarray,
    dataset_name: str,
    labels: np.ndarray,
    split: str,
) -> chromadb.Collection:
    """Store embeddings in a ChromaDB collection."""
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{dataset_name}_{split}_{i}" for i in range(len(embeddings))]
    metadatas = [
        {"dataset": dataset_name, "label": str(labels[i]), "split": split, "index": i}
        for i in range(len(embeddings))
    ]

    # ChromaDB has batch limits; insert in chunks
    batch_size = 5000
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end],
        )

    return collection


def store_all(
    datasets: dict[str, dict],
    embeddings: dict[str, dict[str, np.ndarray]],
    mode: str,
) -> None:
    """Store all dataset embeddings into ChromaDB collections."""
    chroma = get_chroma_client()

    for name, ds in datasets.items():
        for split in ("train", "test"):
            collection_name = f"{mode}_{split}"
            store_embeddings(
                chroma,
                collection_name,
                embeddings[name][split],
                name,
                ds[f"y_{split}"],
                split,
            )
            n = len(embeddings[name][split])
            print(f"  Stored {n} vectors → {collection_name} ({name})")


def query_collection(
    collection_name: str,
    query_embeddings: np.ndarray,
    top_k: int = 5,
    where: dict | None = None,
) -> dict:
    """Query a ChromaDB collection. Returns ChromaDB results dict."""
    chroma = get_chroma_client()
    collection = chroma.get_collection(collection_name)

    kwargs = {
        "query_embeddings": query_embeddings.tolist(),
        "n_results": top_k,
        "include": ["metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    return collection.query(**kwargs)
