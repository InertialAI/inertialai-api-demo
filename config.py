"""Constants, dataset configs, and API settings."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

RESULTS_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# API
API_BASE_URL = "https://inertialai.com/api/v1"
EMBEDDING_MODEL = "inertial-embed-alpha"
BATCH_SIZE = 64

# Datasets — chosen for diversity, small size, and clear domain descriptions
DATASETS = {
    "GunPoint": {
        "description": (
            "Motion sensor data capturing the difference between two actions: "
            "drawing a gun from a hip-mounted holster and simply pointing a finger. "
            "Recorded from a hand-mounted motion sensor."
        ),
    },
    "ECG200": {
        "description": (
            "Electrocardiogram recordings distinguishing normal heartbeats "
            "from those exhibiting myocardial infarction (heart attack) patterns."
        ),
    },
    "ItalyPowerDemand": {
        "description": (
            "Electric power demand measurements from Italy, distinguishing "
            "winter months (October–March) from summer months (April–September)."
        ),
    },
    "SyntheticControl": {
        "description": (
            "Synthetic control chart time series covering six patterns: normal, "
            "cyclic, increasing trend, decreasing trend, upward shift, and downward shift."
        ),
    },
    "Wafer": {
        "description": (
            "Semiconductor manufacturing sensor data from wafer fabrication, "
            "classifying normal production runs versus abnormal ones."
        ),
    },
}

# Eval settings
TOP_K_VALUES = [1, 5]
MATRYOSHKA_DIMS = [64, 128, 256, 512, 768, 1024]
