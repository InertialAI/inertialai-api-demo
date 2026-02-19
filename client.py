"""InertialAI API wrapper using the OpenAI SDK."""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from config import API_BASE_URL, BATCH_SIZE, EMBEDDING_MODEL

load_dotenv()

MAX_RETRIES = 5
RETRY_BACKOFF = 2  # seconds, doubled each retry


def get_client() -> OpenAI:
    api_key = os.environ.get("INERTIALAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "INERTIALAI_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(base_url=API_BASE_URL, api_key=api_key, timeout=120)


def _embed_with_retry(client: OpenAI, batch: list[dict]) -> list[list[float]]:
    """Call the embeddings API with exponential backoff on transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=json.dumps(batch),
            )
            return [d.embedding for d in response.data]
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"\n  Retry {attempt + 1}/{MAX_RETRIES} after error: {type(e).__name__}. Waiting {wait}s...")
            time.sleep(wait)


def embed(inputs: list[dict], client: OpenAI | None = None) -> list[list[float]]:
    """Embed a list of input dicts via the InertialAI API.

    Each dict can contain:
      - {"time_series": [[...], ...]}                      — time-series only
      - {"text": "..."}                                    — text only
      - {"time_series": [[...], ...], "text": "..."}       — multi-modal

    The entire input list is JSON-serialised as a single string per the API docs.
    """
    if client is None:
        client = get_client()

    all_embeddings: list[list[float]] = []

    for start in range(0, len(inputs), BATCH_SIZE):
        batch = inputs[start : start + BATCH_SIZE]
        batch_embeddings = _embed_with_retry(client, batch)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings
