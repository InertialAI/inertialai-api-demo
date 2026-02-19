# InertialAI API Guide

Practical examples and benchmarks for the [InertialAI](https://inertialai.com) time-series embedding API. Demonstrates retrieval quality on real datasets from the UCR Time Series Archive.

## What's Inside

- **Embedding pipeline** — TS-only and multi-modal (time series + text) modes
- **Vector search** — ChromaDB-backed retrieval with cosine similarity
- **Benchmarks** — Label retrieval accuracy and cross-dataset matching across 5 UCR datasets
- **Interactive demo** — Streamlit app for exploring query results visually

## Datasets

| Dataset | Domain | Classes |
|---------|--------|---------|
| GunPoint | Motion sensor | 2 (draw vs point) |
| ECG200 | Electrocardiogram | 2 (normal vs MI) |
| ItalyPowerDemand | Power grid | 2 (winter vs summer) |
| SyntheticControl | Synthetic | 6 control chart patterns |
| Wafer | Semiconductor | 2 (normal vs abnormal) |

## Quick Start

```bash
# Clone and set up
git clone https://github.com/inertialai/api-guide.git
cd api-guide
pip install -r requirements.txt

# Add your API key
cp .env.example .env
# Edit .env with your InertialAI API key

# Run the full pipeline: download → embed → store → evaluate
python run_all.py

# Launch the interactive demo
streamlit run app.py
```

## CLI Options

```bash
python run_all.py              # Full pipeline
python run_all.py --embed-only # Embed and store only (skip eval)
python run_all.py --eval-only  # Run evals only (requires cached embeddings)
python run_all.py --no-cache   # Force re-embedding
```

## API Usage

The InertialAI embedding API is accessed via the OpenAI SDK with a custom base URL:

```python
from openai import OpenAI
import json

client = OpenAI(
    base_url="https://inertialai.com/api/v1",
    api_key="your-key-here",
)

# Time-series only
response = client.embeddings.create(
    model="inertial-embed-alpha",
    input=json.dumps([{"time_series": [[1.0, 2.0, 3.0, 4.0, 5.0]]}]),
)

# Multi-modal (time series + text)
response = client.embeddings.create(
    model="inertial-embed-alpha",
    input=json.dumps([{
        "time_series": [[1.0, 2.0, 3.0, 4.0, 5.0]],
        "text": "Accelerometer data from a walking trial.",
    }]),
)

embedding = response.data[0].embedding
```

## Project Structure

```
config.py      — Constants, dataset configs, API settings
client.py      — InertialAI API wrapper (OpenAI SDK)
datasets.py    — Download & load UCR datasets via aeon
embed.py       — Embed datasets (TS-only and multi-modal)
store.py       — ChromaDB storage utilities
eval.py        — Retrieval evals (label + dataset matching)
run_all.py     — CLI entry point
app.py         — Streamlit interactive demo
```

## Evaluation Metrics

- **P@1 / P@5** — Precision at K: fraction of test queries where a correct label appears in the top-K retrieved training samples
- **1-NN Accuracy** — Same as P@1; the nearest neighbour's label matches the query
- **Dataset P@K** — Cross-dataset: fraction of top-K results from the same source dataset
- **TS-only vs Multi-modal** — Side-by-side comparison showing the effect of adding text context
