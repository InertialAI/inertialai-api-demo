"""Streamlit demo — InertialAI Time-Series Embeddings.

Two views:
  1. Data Annotation — auto-label unlabeled time series via nearest-neighbor search
  2. Embedding Explorer — query individual samples and retrieve nearest neighbors
"""

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.manifold import TSNE

from config import EMBEDDINGS_DIR, MATRYOSHKA_DIMS
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
ASSETS = Path(__file__).parent / "assets"

st.set_page_config(
    page_title="InertialAI Embedding Demo",
    page_icon=str(ASSETS / "favicon.ico"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Colour palette — burgundy & cream
# ---------------------------------------------------------------------------
BURGUNDY = "#6B1D32"
BURGUNDY_LIGHT = "#8B2848"
BURGUNDY_FAINT = "#F5E6EB"
CREAM = "#FFFAF6"
CREAM_DARK = "#FFF0E8"
INK = "#1e293b"
MUTED = "#6B5E57"
CARD_BG = "#FFFAF6"
BORDER = "#E8DCD4"
SUCCESS = "#16a34a"
DANGER = "#dc2626"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {INK};
    }}

    .block-container {{ padding-top: 1.5rem; max-width: 1200px; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stSidebar"] {{ background: {CREAM}; }}

    .step-box {{
        background: {CREAM};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 1rem 1.25rem;
        height: 100%;
    }}
    .step-number {{
        font-size: 1.3rem;
        font-weight: 700;
        color: {BURGUNDY};
        margin-bottom: 0.3rem;
    }}
    .step-title {{
        font-size: 0.9rem;
        font-weight: 600;
        color: {INK};
        margin-bottom: 0.25rem;
    }}
    .step-desc {{
        font-size: 0.82rem;
        color: {MUTED};
        line-height: 1.5;
    }}

    .scenario-box {{
        background: {BURGUNDY_FAINT};
        border: 1px solid {BORDER};
        border-left: 4px solid {BURGUNDY};
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 1rem;
        font-size: 0.88rem;
        color: {INK};
        line-height: 1.5;
    }}

    .ood-box {{
        background: {BURGUNDY_FAINT};
        border: 1px solid {BORDER};
        border-left: 4px solid {BURGUNDY_LIGHT};
        border-radius: 8px;
        padding: 0.75rem 1.1rem;
        margin-bottom: 1.25rem;
        font-size: 0.85rem;
        color: {INK};
        line-height: 1.5;
    }}

    .divider {{
        border: none;
        border-top: 1px solid {BORDER};
        margin: 1.5rem 0;
    }}

    .metric-card {{
        background: {CREAM};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
    }}
    .metric-card .value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {INK};
        line-height: 1.2;
    }}
    .metric-card .label {{
        font-size: 0.72rem;
        font-weight: 500;
        color: {MUTED};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }}

    .result-card {{
        background: {CREAM};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 0.6rem 0.85rem;
        margin-bottom: 0.4rem;
    }}
    .result-card.match {{ border-left: 3px solid {SUCCESS}; }}
    .result-card.no-match {{ border-left: 3px solid {DANGER}; }}

    .tag {{
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 600;
    }}
    .tag-match {{ background: #dcfce7; color: {SUCCESS}; }}
    .tag-miss  {{ background: #fee2e2; color: {DANGER}; }}
    .tag-label {{ background: {BURGUNDY_FAINT}; color: {INK}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASET_INFO = {
    "SyntheticControl": {
        "domain": "Manufacturing / QC",
        "description": (
            "Synthetic control chart time series representing six common process patterns "
            "encountered in statistical quality control: normal operation, cyclic behavior, "
            "increasing trend, decreasing trend, upward shift, and downward shift."
        ),
        "task": "Classify each control chart pattern into one of six process categories.",
        "annotation_scenario": (
            "A manufacturing plant generates thousands of control chart traces daily. "
            "Quality engineers have labeled a small reference set. The remaining traces "
            "need automatic classification to flag anomalies."
        ),
        "class_names": {
            "1": "Normal",
            "2": "Cyclic",
            "3": "Increasing Trend",
            "4": "Decreasing Trend",
            "5": "Upward Shift",
            "6": "Downward Shift",
        },
    },
    "ECG200": {
        "domain": "Healthcare",
        "description": (
            "Electrocardiogram (ECG) recordings from cardiac patients. Each sample "
            "is a single heartbeat waveform captured via chest electrodes. The dataset "
            "distinguishes normal heartbeats from those exhibiting signs of myocardial "
            "infarction (heart attack)."
        ),
        "task": "Classify each ECG heartbeat as Normal or Myocardial Infarction.",
        "annotation_scenario": (
            "A hospital has a library of ECG recordings that need to be labeled for a "
            "cardiac monitoring system. A small set has been reviewed by cardiologists. "
            "The goal is to automatically annotate the remaining recordings so they can "
            "be routed to the right clinical workflow."
        ),
        "class_names": {"-1": "Myocardial Infarction", "1": "Normal"},
    },
    "ItalyPowerDemand": {
        "domain": "Energy",
        "description": (
            "Daily electricity consumption profiles from Italy. Each sample is a 24-point "
            "load curve representing power demand across a single day. The dataset captures "
            "seasonal variation between winter (higher heating load, Oct-Mar) and summer "
            "demand patterns (Apr-Sep)."
        ),
        "task": "Classify each daily load curve as Winter demand or Summer demand.",
        "annotation_scenario": (
            "An energy trading firm needs to tag historical load curves by season for risk "
            "modeling. Only a small fraction have been labeled by analysts. The rest must be "
            "annotated automatically so the full dataset can power downstream models."
        ),
        "class_names": {"1": "Winter", "2": "Summer"},
    },
    "GunPoint": {
        "domain": "Motion / IMU",
        "description": (
            "Wrist-mounted motion sensor recordings of two gestures. Subjects either draw "
            "a replica gun from a hip holster or simply point their index finger forward. "
            "The time series captures the centroid of the right hand's motion trajectory."
        ),
        "task": "Classify each motion recording as a Gun Draw or Point gesture.",
        "annotation_scenario": (
            "A wearable tech company is building a gesture recognition library. Thousands of "
            "motion capture recordings need labels before a classifier can be trained. A small "
            "labeled reference set exists -- the goal is to annotate the rest automatically."
        ),
        "class_names": {"1": "Gun Draw", "2": "Point"},
    },
    "Wafer": {
        "domain": "Semiconductor",
        "description": (
            "Sensor data from semiconductor wafer fabrication. Each time series is a "
            "process measurement from a single wafer run, classified as normal production "
            "or abnormal (defective process)."
        ),
        "task": "Classify each wafer process trace as Normal or Abnormal.",
        "annotation_scenario": (
            "A semiconductor fab produces thousands of wafer traces per day. A small set "
            "has been labeled by process engineers. The rest must be annotated automatically "
            "for defect detection."
        ),
        "class_names": {"-1": "Abnormal", "1": "Normal"},
    },
}

# Color palette for charts — lead with burgundy tones
PALETTE = [
    "#6B1D32",
    "#0ea5e9",
    "#14b8a6",
    "#f59e0b",
    "#ef4444",
    "#22c55e",
    "#8b5cf6",
    "#ec4899",
    "#84cc16",
    "#06b6d4",
]

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color=INK),
)


def get_class_colors(classes: list[str]) -> dict[str, str]:
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(sorted(classes))}


def class_display_name(ds_key: str, cls) -> str:
    names = DATASET_INFO.get(ds_key, {}).get("class_names", {})
    return names.get(str(cls), str(cls))


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def load_and_normalize(path, dim: int | None = None) -> np.ndarray:
    embs = np.load(path).astype(np.float64)
    if dim is not None:
        embs = embs[:, :dim]
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-12)


def annotate_nearest_neighbor(
    labeled_embs: np.ndarray,
    labeled_labels: np.ndarray,
    unlabeled_embs: np.ndarray,
) -> np.ndarray:
    sims = unlabeled_embs @ labeled_embs.T
    nearest = np.argmax(sims, axis=1)
    return labeled_labels[nearest]


def cosine_topk(
    queries: np.ndarray,
    corpus: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    sims = queries @ corpus.T
    k = min(k, sims.shape[1])
    top_idx = np.argpartition(-sims, k, axis=1)[:, :k]
    rows = np.arange(len(queries))[:, None]
    top_sims = sims[rows, top_idx]
    sort_order = np.argsort(-top_sims, axis=1)
    return top_idx[rows, sort_order], top_sims[rows, sort_order]


@st.cache_data(show_spinner="Computing 2D projection...")
def run_tsne(embs_bytes: bytes, n: int, d: int) -> np.ndarray:
    embs = np.frombuffer(embs_bytes, dtype=np.float64).reshape(n, d)
    perp = min(30, max(5, n // 5))
    return TSNE(
        n_components=2,
        perplexity=perp,
        random_state=42,
        init="pca",
        learning_rate="auto",
    ).fit_transform(embs)


@st.cache_data(show_spinner="Loading dataset...")
def cached_load_dataset(name: str) -> dict:
    return load_dataset(name)


@st.cache_data(show_spinner="Splitting labeled / unlabeled...")
def split_labeled_unlabeled(
    ds_key: str,
    _y_train: np.ndarray,
    _y_test: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select `fraction` of total samples as labeled, evenly balanced across classes.

    Returns (labeled_indices, unlabeled_indices) into the combined array
    where combined = concat(train, test).
    """
    y_all = np.concatenate([_y_train, _y_test])
    y_all_str = np.array([str(l) for l in y_all])
    classes = sorted(set(y_all_str))
    n_total = len(y_all)
    n_labeled_total = max(len(classes), int(round(n_total * fraction)))
    per_class = max(1, n_labeled_total // len(classes))

    rng = np.random.default_rng(42)
    labeled_idx = []
    for cls in classes:
        cls_indices = np.where(y_all_str == cls)[0]
        chosen = rng.choice(
            cls_indices, size=min(per_class, len(cls_indices)), replace=False
        )
        labeled_idx.extend(chosen.tolist())

    labeled_idx = np.array(sorted(labeled_idx))
    all_idx = np.arange(n_total)
    unlabeled_idx = np.setdiff1d(all_idx, labeled_idx)
    return labeled_idx, unlabeled_idx


def ts_line_chart(
    sample: np.ndarray,
    color: str = BURGUNDY,
    height: int = 200,
    title: str = "",
) -> go.Figure:
    fig = go.Figure()
    for ch in range(sample.shape[0]):
        fig.add_trace(
            go.Scatter(
                y=sample[ch],
                mode="lines",
                line=dict(width=1.5, color=color),
                name=f"Ch {ch}" if sample.shape[0] > 1 else "",
                showlegend=sample.shape[0] > 1,
            )
        )
    layout = {
        **PLOTLY_THEME,
        "height": height,
        "margin": dict(l=30, r=10, t=30 if title else 10, b=10),
        "showlegend": sample.shape[0] > 1,
        "xaxis": dict(showgrid=False),
        "yaxis": dict(showgrid=True, gridcolor=BORDER),
    }
    if title:
        layout["title"] = dict(text=title, font=dict(size=12))
    fig.update_layout(**layout)
    return fig


def metric_card(value: str, label: str) -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="value">{value}</div>'
        f'<div class="label">{label}</div>'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Detect which datasets have cached embeddings
# ---------------------------------------------------------------------------
available_datasets = []
for ds_name in DATASET_INFO:
    train_path = EMBEDDINGS_DIR / f"{ds_name}_train_ts_only.npy"
    test_path = EMBEDDINGS_DIR / f"{ds_name}_test_ts_only.npy"
    if train_path.exists() and test_path.exists():
        available_datasets.append(ds_name)

if not available_datasets:
    st.error("No embeddings found. Run `python run_all.py` to generate them.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    # Logo
    logo_path = ASSETS / "logo.webp"
    if logo_path.exists():
        logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()
        st.markdown(
            f'<div style="text-align:center;padding:0.5rem 0 1rem;">'
            f'<img src="data:image/webp;base64,{logo_b64}" '
            f'style="max-width:180px;" alt="InertialAI">'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'**<span style="color:{BURGUNDY}">InertialAI</span> Demo**',
            unsafe_allow_html=True,
        )

    dataset_name = st.selectbox(
        "Dataset",
        available_datasets,
        format_func=lambda d: f"{d}  ({DATASET_INFO[d]['domain']})",
    )

    info = DATASET_INFO[dataset_name]

    st.markdown(f"**{info['domain']}**")
    st.caption(info["description"])

    st.markdown("---")

    matryoshka_dim = st.select_slider(
        "Embedding dimension",
        options=MATRYOSHKA_DIMS,
        value=256,
        help=(
            "Truncate embeddings to this many dimensions. The Matryoshka property "
            "means shorter embeddings still retain useful structure."
        ),
    )

    labeled_pct = st.slider(
        "Labeled fraction",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        format="%d%%",
        help="Percentage of total samples used as expert-labeled references for annotation.",
    )

    top_k = st.slider("Top-K (Explorer)", min_value=1, max_value=10, value=5)

    # Dataset stats
    ds = cached_load_dataset(dataset_name)
    st.markdown("---")
    total = len(ds["X_train"]) + len(ds["X_test"])
    classes = sorted(
        set(str(l) for l in ds["y_train"]) | set(str(l) for l in ds["y_test"])
    )
    st.markdown(f"**{total:,} samples** ({total // len(classes)} per class)")
    st.markdown(
        "**Classes:** "
        + ", ".join(class_display_name(dataset_name, c) for c in classes)
    )

# ---------------------------------------------------------------------------
# Load embeddings & build labeled/unlabeled split
# ---------------------------------------------------------------------------
train_embs = load_and_normalize(
    EMBEDDINGS_DIR / f"{dataset_name}_train_ts_only.npy",
    matryoshka_dim,
)
test_embs = load_and_normalize(
    EMBEDDINGS_DIR / f"{dataset_name}_test_ts_only.npy",
    matryoshka_dim,
)
y_train = ds["y_train"]
y_test = ds["y_test"]

# Combine train + test into a single pool, then split 10% labeled / 90% unlabeled
all_embs = np.vstack([train_embs, test_embs])
X_all = np.concatenate([ds["X_train"], ds["X_test"]])
y_all = np.concatenate([y_train, y_test])
y_all_str = np.array([str(l) for l in y_all])

classes = sorted(set(y_all_str))
class_colors = get_class_colors(classes)

labeled_idx, unlabeled_idx = split_labeled_unlabeled(
    dataset_name,
    y_train,
    y_test,
    labeled_pct / 100,
)

labeled_embs = all_embs[labeled_idx]
labeled_labels = y_all[labeled_idx]
labeled_labels_str = y_all_str[labeled_idx]
unlabeled_embs = all_embs[unlabeled_idx]
unlabeled_labels_true = y_all[unlabeled_idx]
unlabeled_labels_true_str = y_all_str[unlabeled_idx]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f'<div style="font-size:1.7rem;font-weight:700;color:{BURGUNDY};">'
    "InertialAI Embedding Demo - Version 0"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<div style="font-size:0.95rem;color:{MUTED};margin-bottom:0.5rem;">'
    "Automatically label time-series data using a small set of labeled examples — no model training required."
    "</div>",
    unsafe_allow_html=True,
)

# OOD notice
st.markdown(
    '<div class="ood-box">'
    "<strong>Zero-shot generalization.</strong> "
    "None of these datasets were seen during training. All results are purely "
    "out-of-distribution — the model receives raw time series and returns embeddings with no task-specific fine-tuning."
    "</div>",
    unsafe_allow_html=True,
)

# Model explanation
with st.expander("How it works"):
    st.markdown(
        """
The InertialAI API converts raw time-series data into embedding vectors. Similar signals produce nearby vectors; dissimilar signals produce distant ones.

Once data is embedded, downstream tasks require no model training — just vector math:

- **Data annotation** — find the nearest labeled embedding and assign its label
- **Similarity search** — retrieve the closest matching signals from a corpus
- **Anomaly detection** — flag samples that are far from any known cluster

**Matryoshka embeddings** can be truncated to smaller dimensions (64 → 1024) with minimal accuracy loss, letting you trade speed and storage for precision.
"""
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_annotate, tab_explore = st.tabs(
    ["Sample Use Case: Data Annotation", "Embedding Explorer"]
)

# ═══════════════════════════════════════════════════════════════════════════
# Tab 1 -- Data Annotation
# ═══════════════════════════════════════════════════════════════════════════
with tab_annotate:

    # -- How it works ---------------------------------
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(
            """<div class="step-box">
              <div class="step-number">1.</div>
              <div class="step-title">Label a small reference set</div>
              <div class="step-desc">
                A domain expert labels a small, class-balanced sample of time series.
                Each sample is embedded into a high-dimensional vector via the InertialAI API.
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """<div class="step-box">
              <div class="step-number">2.</div>
              <div class="step-title">Embed the rest</div>
              <div class="step-desc">
                All remaining unlabeled samples are embedded into the same vector space.
                No retraining or fine-tuning — one API call per sample.
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """<div class="step-box">
              <div class="step-number">3.</div>
              <div class="step-title">Assign labels by proximity</div>
              <div class="step-desc">
                Each unlabeled sample inherits the label of its nearest neighbor
                in the reference set. No classifier to train — just a cosine similarity lookup.
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -- Task description ---------------------------------------------------
    st.markdown(f"**Task:** {info['task']}")

    # -- Run annotation -----------------------------------------------------
    predicted = annotate_nearest_neighbor(labeled_embs, labeled_labels, unlabeled_embs)
    correct_mask = np.array(
        [str(p) == str(t) for p, t in zip(predicted, unlabeled_labels_true)]
    )
    accuracy = correct_mask.mean()
    n_correct = int(correct_mask.sum())
    n_wrong = int((~correct_mask).sum())

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(
        metric_card(f"{len(labeled_idx):,}", f"Expert-labeled ({labeled_pct}%)"),
        unsafe_allow_html=True,
    )
    m2.markdown(
        metric_card(
            f"{len(unlabeled_idx):,}", f"Auto-annotated ({100 - labeled_pct}%)"
        ),
        unsafe_allow_html=True,
    )
    m3.markdown(
        metric_card(f"{accuracy:.1%}", f"Accuracy (dim {matryoshka_dim})"),
        unsafe_allow_html=True,
    )
    m4.markdown(
        metric_card(f"{n_correct:,} / {n_wrong:,}", "Correct / Errors"),
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # -- 2D Embedding Scatter -----------------------------------------------
    st.markdown(f"**Embedding space** (dim={matryoshka_dim}, projected via t-SNE)")
    st.caption(
        "Stars = expert-labeled reference samples. "
        "Circles = auto-annotated samples (correct). "
        "Crosses = annotation errors."
    )

    MAX_UNLABELED_VIZ = 400
    viz_unl_idx = np.arange(len(unlabeled_embs))
    if len(unlabeled_embs) > MAX_UNLABELED_VIZ:
        rng = np.random.default_rng(42)
        viz_unl_idx = rng.choice(len(unlabeled_embs), MAX_UNLABELED_VIZ, replace=False)

    viz_unlabeled = unlabeled_embs[viz_unl_idx]
    viz_correct = correct_mask[viz_unl_idx]
    viz_y_unl_str = unlabeled_labels_true_str[viz_unl_idx]

    all_embs_viz = np.vstack([labeled_embs, viz_unlabeled])
    coords = run_tsne(
        all_embs_viz.tobytes(),
        len(all_embs_viz),
        all_embs_viz.shape[1],
    )
    n_labeled = len(labeled_embs)
    labeled_coords = coords[:n_labeled]
    unlabeled_coords = coords[n_labeled:]

    fig_scatter = go.Figure()

    # Correctly annotated unlabeled points (by class)
    for cls in classes:
        mask = viz_correct & (viz_y_unl_str == cls)
        if not mask.any():
            continue
        name = class_display_name(dataset_name, cls)
        fig_scatter.add_trace(
            go.Scatter(
                x=unlabeled_coords[mask, 0],
                y=unlabeled_coords[mask, 1],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=7,
                    color=class_colors[cls],
                    opacity=0.5,
                ),
                name=f"{name} -- annotated",
                hovertemplate=f"<b>{name}</b><br>Auto-annotated (correct)<extra></extra>",
            )
        )

    # Annotation errors
    wrong_mask = ~viz_correct
    if wrong_mask.any():
        fig_scatter.add_trace(
            go.Scatter(
                x=unlabeled_coords[wrong_mask, 0],
                y=unlabeled_coords[wrong_mask, 1],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=9,
                    color=DANGER,
                    opacity=0.85,
                    line=dict(width=2),
                ),
                name="Annotation error",
                hovertemplate="<b>Annotation error</b><extra></extra>",
            )
        )

    # Expert-labeled points (stars, rendered on top)
    for cls in classes:
        mask = labeled_labels_str == cls
        if not mask.any():
            continue
        name = class_display_name(dataset_name, cls)
        fig_scatter.add_trace(
            go.Scatter(
                x=labeled_coords[mask, 0],
                y=labeled_coords[mask, 1],
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=13,
                    color=class_colors[cls],
                    opacity=1.0,
                    line=dict(width=0.5, color="white"),
                ),
                name=f"{name} -- labeled",
                hovertemplate=f"<b>{name}</b><br>Expert-labeled<extra></extra>",
            )
        )

    fig_scatter.update_layout(
        **PLOTLY_THEME,
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    )
    st.plotly_chart(fig_scatter, width="stretch")

    # -- Example time series per class --------------------------------------
    with st.expander("Example time series per class"):
        n_examples = 3
        for cls in classes:
            name = class_display_name(dataset_name, cls)
            color = class_colors[cls]
            cls_mask = y_all_str == cls
            indices = np.where(cls_mask)[0][:n_examples]

            st.markdown(f"**{name}**")
            cols = st.columns(n_examples)
            for i, idx in enumerate(indices):
                with cols[i]:
                    st.plotly_chart(
                        ts_line_chart(
                            X_all[idx],
                            color=color,
                            height=120,
                        ),
                        width="stretch",
                    )

    # -- Matryoshka dimension accuracy (collapsed at bottom) ----------------
    with st.expander("Accuracy across embedding dimensions (Matryoshka)"):
        st.caption(
            "1-NN annotation accuracy at each embedding dimension. "
            "Smaller dimensions are faster and cheaper to store — "
            "the Matryoshka property means you lose little accuracy by truncating."
        )

        @st.cache_data(show_spinner="Computing accuracies across dimensions...")
        def compute_dim_accuracies(
            ds_key: str,
            _labeled_idx_bytes: bytes,
            _unlabeled_idx_bytes: bytes,
        ) -> pd.DataFrame:
            train_raw = np.load(EMBEDDINGS_DIR / f"{ds_key}_train_ts_only.npy").astype(
                np.float64
            )
            test_raw = np.load(EMBEDDINGS_DIR / f"{ds_key}_test_ts_only.npy").astype(
                np.float64
            )
            all_raw = np.vstack([train_raw, test_raw])
            d = cached_load_dataset(ds_key)
            y = np.concatenate([d["y_train"], d["y_test"]])
            li = np.frombuffer(_labeled_idx_bytes, dtype=np.intp)
            ui = np.frombuffer(_unlabeled_idx_bytes, dtype=np.intp)
            rows = []
            for dim in MATRYOSHKA_DIMS:
                embs_dim = all_raw[:, :dim].copy()
                embs_dim /= np.maximum(
                    np.linalg.norm(embs_dim, axis=1, keepdims=True),
                    1e-12,
                )
                pred = annotate_nearest_neighbor(embs_dim[li], y[li], embs_dim[ui])
                acc = np.mean([str(p) == str(t) for p, t in zip(pred, y[ui])])
                rows.append({"Dimension": dim, "Accuracy": acc})
            return pd.DataFrame(rows)

        df_dims = compute_dim_accuracies(
            dataset_name,
            labeled_idx.tobytes(),
            unlabeled_idx.tobytes(),
        )

        fig_dims = px.bar(
            df_dims,
            x="Dimension",
            y="Accuracy",
            text=df_dims["Accuracy"].apply(lambda v: f"{v:.1%}"),
            color_discrete_sequence=[BURGUNDY],
        )
        fig_dims.update_layout(
            **PLOTLY_THEME,
            height=300,
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(type="category", title="Embedding Dimension"),
            yaxis=dict(
                range=[0, 1.05],
                gridcolor=BORDER,
                title="1-NN Accuracy",
            ),
        )
        fig_dims.update_traces(textposition="outside", textfont_size=11)
        st.plotly_chart(fig_dims, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════
# Tab 2 -- Embedding Explorer
# ═══════════════════════════════════════════════════════════════════════════
with tab_explore:

    st.markdown("**Nearest neighbor search**")
    st.caption(
        f"Select a sample and retrieve its top-{top_k} most similar matches "
        f"using cosine similarity in the {matryoshka_dim}-d embedding space."
    )

    col_query, col_results = st.columns([1, 2], gap="large")

    with col_query:
        sample_idx = st.number_input(
            "Sample index",
            min_value=0,
            max_value=len(X_all) - 1,
            value=0,
            step=1,
        )
        sample = X_all[sample_idx]
        label = y_all[sample_idx]
        st.plotly_chart(
            ts_line_chart(
                sample,
                height=220,
                title=f"{dataset_name} [{sample_idx}]",
            ),
            width="stretch",
        )

    # Retrieve from full dataset (exclude self)
    query_emb = all_embs[sample_idx : sample_idx + 1]

    corpus_mask = np.ones(len(all_embs), dtype=bool)
    corpus_mask[sample_idx] = False
    corpus = all_embs[corpus_mask]
    corpus_indices = np.where(corpus_mask)[0]

    top_idx, top_sims = cosine_topk(query_emb, corpus, top_k)

    with col_results:
        st.markdown(f"**Top-{top_k} results** (dim={matryoshka_dim})")
        result_cols = st.columns(min(top_k, 3))
        for rank in range(top_k):
            real_idx = corpus_indices[top_idx[0, rank]]
            ret_label = str(y_all[real_idx])
            ret_name = class_display_name(dataset_name, ret_label)
            match = ret_label == str(label)

            col = result_cols[rank % len(result_cols)]
            with col:
                sc = "match" if match else "no-match"
                tc = "tag-match" if match else "tag-miss"
                tt = "Match" if match else "Mismatch"
                st.markdown(
                    f'<div class="result-card {sc}">'
                    f"<strong>#{rank + 1}</strong> "
                    f'<span class="tag {tc}">{tt}</span> '
                    f"{ret_name}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    ts_line_chart(X_all[real_idx], height=160),
                    width="stretch",
                )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # -- t-SNE overview (full dataset) --------------------------------------
    st.markdown(f"**Embedding space** (dim={matryoshka_dim}, projected via t-SNE)")
    st.caption(
        "Colored by true class label. Well-separated clusters mean the model captures the signal structure needed to distinguish classes."
    )

    MAX_EXPLORE_VIZ = 500
    if len(all_embs) > MAX_EXPLORE_VIZ:
        rng_e = np.random.default_rng(42)
        explore_idx = rng_e.choice(len(all_embs), MAX_EXPLORE_VIZ, replace=False)
        explore_idx.sort()
    else:
        explore_idx = np.arange(len(all_embs))

    explore_embs = all_embs[explore_idx]
    explore_coords = run_tsne(
        explore_embs.tobytes(),
        len(explore_embs),
        explore_embs.shape[1],
    )
    explore_labels = y_all_str[explore_idx]

    fig_explore = go.Figure()

    for cls in classes:
        mask = explore_labels == cls
        if not mask.any():
            continue
        name = class_display_name(dataset_name, cls)
        fig_explore.add_trace(
            go.Scatter(
                x=explore_coords[mask, 0],
                y=explore_coords[mask, 1],
                mode="markers",
                marker=dict(size=7, color=class_colors[cls], opacity=0.65),
                name=name,
                hovertemplate=f"<b>{name}</b><extra></extra>",
            )
        )

    fig_explore.update_layout(
        **PLOTLY_THEME,
        height=500,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    )
    st.plotly_chart(fig_explore, width="stretch")
