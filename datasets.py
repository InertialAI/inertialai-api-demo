"""Download and load UCR datasets via the aeon library."""

from aeon.datasets import load_classification

from config import DATASETS


def load_dataset(name: str) -> dict:
    """Load a single UCR dataset. Returns train/test splits and metadata."""
    X_train, y_train = load_classification(name, split="train")
    X_test, y_test = load_classification(name, split="test")

    return {
        "name": name,
        "description": DATASETS[name]["description"],
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def load_all_datasets() -> dict[str, dict]:
    """Load all configured UCR datasets. Returns {name: dataset_dict}."""
    results = {}
    for name in DATASETS:
        print(f"Loading {name}...")
        results[name] = load_dataset(name)
    return results
