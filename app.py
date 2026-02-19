"""Streamlit interactive demo for InertialAI embeddings."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.manifold import TSNE

from config import DATASETS, EMBEDDINGS_DIR, MATRYOSHKA_DIMS
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Page config & global styles
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="InertialAI Embedding Explorer",
    layout="wide",
)

# Soft neutral palette
INK = "#1e293b"
MUTED = "#64748b"
CARD_BG = "#f8fafc"
BORDER = "#e2e8f0"
BG = "#ffffff"
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

    header[data-testid="stHeader"] {{ background: transparent; }}
    .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }}

    .section-title {{
        font-weight: 700;
        font-size: 1.35rem;
        color: {INK};
        margin-bottom: 0.2rem;
    }}
    .section-caption {{
        font-size: 0.85rem;
        color: {MUTED};
        margin-bottom: 1.25rem;
    }}

    .metric-card {{
        background: {CARD_BG};
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
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 0.6rem 0.85rem;
        margin-bottom: 0.4rem;
    }}
    .result-card.match {{ border-left: 3px solid {SUCCESS}; }}
    .result-card.no-match {{ border-left: 3px solid {DANGER}; }}

    .section-divider {{
        border: none;
        border-top: 1px solid {BORDER};
        margin: 2rem 0 1.5rem 0;
    }}

    .tag {{
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 600;
    }}
    .tag-match {{ background: #dcfce7; color: {SUCCESS}; }}
    .tag-miss  {{ background: #fee2e2; color: {DANGER}; }}
    .tag-label {{ background: #f1f5f9; color: {INK}; }}

    [data-testid="stSidebar"] {{
        background: {CARD_BG};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color=INK),
    margin=dict(l=40, r=20, t=40, b=30),
)

# Soft, distinguishable palette for datasets
DATASET_COLORS = ["#6366f1", "#0ea5e9", "#14b8a6", "#f59e0b", "#8b5cf6"]

# Mode comparison: two clearly different but non-alarming tones
MODE_COLORS = {"TS-only": "#6366f1", "Multi-modal": "#0ea5e9"}


def truncate_and_normalize(embs: np.ndarray, dim: int) -> np.ndarray:
    trunc = embs[:, :dim].copy()
    norms = np.linalg.norm(trunc, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return trunc / norms


def cosine_topk(
    queries: np.ndarray, corpus: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    sims = queries @ corpus.T
    k = min(k, sims.shape[1])
    top_idx = np.argpartition(-sims, k, axis=1)[:, :k]
    rows = np.arange(len(queries))[:, None]
    top_sims = sims[rows, top_idx]
    sort_order = np.argsort(-top_sims, axis=1)
    top_idx = top_idx[rows, sort_order]
    top_sims = top_sims[rows, sort_order]
    return top_idx, top_sims


def metric_card(value: str, label: str) -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="value">{value}</div>'
        f'<div class="label">{label}</div>'
        f"</div>"
    )


def section_header(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def ts_line_chart(sample: np.ndarray, height: int = 200, title: str = "") -> go.Figure:
    fig = go.Figure()
    for ch in range(sample.shape[0]):
        fig.add_trace(
            go.Scatter(
                y=sample[ch],
                mode="lines",
                name=f"Ch {ch}",
                line=dict(width=1.5, color="#6366f1"),
            )
        )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=height,
        title=dict(text=title, font=dict(size=12)),
        showlegend=sample.shape[0] > 1,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=BORDER),
    )
    return fig


def compute_pk(
    test_embs: np.ndarray,
    corpus_embs: np.ndarray,
    y_test,
    y_corpus,
    k: int,
) -> float:
    """Precision@K — fraction of test queries where the correct label appears in top-k."""
    top_idx, _ = cosine_topk(test_embs, corpus_embs, k)
    correct = 0
    for i in range(len(y_test)):
        retrieved = [str(y_corpus[top_idx[i, j]]) for j in range(min(k, top_idx.shape[1]))]
        if str(y_test[i]) in retrieved:
            correct += 1
    return round(correct / len(y_test), 4)


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="section-title" style="font-size:1.7rem;">InertialAI Embedding Explorer</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-caption">Explore time-series embeddings with matryoshka dimension control</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="section-title" style="font-size:1.05rem;">Settings</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    dataset_name = st.selectbox("Dataset", list(DATASETS.keys()))
    mode = st.radio("Embedding mode", ["ts_only", "multimodal"], horizontal=True)
    matryoshka_dim = st.select_slider(
        "Matryoshka dimension",
        options=MATRYOSHKA_DIMS,
        value=1024,
        help="Truncate embeddings to this many dimensions before computing similarity",
    )
    top_k = st.slider("Top-K results", min_value=1, max_value=10, value=5)

    st.markdown("---")
    ds = load_dataset(dataset_name)
    st.markdown(f"**{dataset_name}**")
    st.caption(ds["description"])
    st.markdown(
        f"Train **{len(ds['X_train'])}** &middot; Test **{len(ds['X_test'])}** samples"
    )

# ---------------------------------------------------------------------------
# Load embeddings
# ---------------------------------------------------------------------------
emb_train_path = EMBEDDINGS_DIR / f"{dataset_name}_train_{mode}.npy"
emb_test_path = EMBEDDINGS_DIR / f"{dataset_name}_test_{mode}.npy"

if not emb_train_path.exists() or not emb_test_path.exists():
    st.error(
        f"Embeddings not found for **{dataset_name}** / **{mode}**. "
        "Run `python run_all.py` first."
    )
    st.stop()

raw_train_embs = np.load(emb_train_path)
raw_test_embs = np.load(emb_test_path)

# ---------------------------------------------------------------------------
# 1. Query & Retrieve
# ---------------------------------------------------------------------------
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
section_header("Query & Retrieve")

col_query, col_results = st.columns([1, 2], gap="large")

with col_query:
    split = st.radio("Split", ["test", "train"], horizontal=True)
    X = ds[f"X_{split}"]
    y = ds[f"y_{split}"]
    sample_idx = st.number_input(
        "Sample index", min_value=0, max_value=len(X) - 1, value=0, step=1
    )
    sample = X[sample_idx]
    label = y[sample_idx]

    st.markdown(
        f'<span class="tag tag-label">Label: {label}</span>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        ts_line_chart(sample, height=220, title=f"{dataset_name} [{split}][{sample_idx}]"),
        use_container_width=True,
    )

# Retrieve
raw_embs = raw_test_embs if split == "test" else raw_train_embs
query_emb = truncate_and_normalize(raw_embs[sample_idx : sample_idx + 1], matryoshka_dim)
corpus_emb = truncate_and_normalize(raw_train_embs, matryoshka_dim)

if split == "train":
    mask = np.ones(len(corpus_emb), dtype=bool)
    mask[sample_idx] = False
    corpus_for_query = corpus_emb[mask]
    train_indices = np.where(mask)[0]
else:
    corpus_for_query = corpus_emb
    train_indices = np.arange(len(corpus_emb))

top_idx, top_sims = cosine_topk(query_emb, corpus_for_query, top_k)

with col_results:
    st.markdown(
        f"**Top-{top_k} results** &mdash; within *{dataset_name}* &middot; dim={matryoshka_dim}"
    )
    result_cols = st.columns(min(top_k, 3))
    for rank in range(top_k):
        real_idx = train_indices[top_idx[0, rank]]
        ret_label = str(ds["y_train"][real_idx])
        sim = top_sims[0, rank]
        match = ret_label == str(label)

        col = result_cols[rank % len(result_cols)]
        with col:
            sc = "match" if match else "no-match"
            tc = "tag-match" if match else "tag-miss"
            tt = "Match" if match else "Mismatch"
            st.markdown(
                f'<div class="result-card {sc}">'
                f"<strong>#{rank + 1}</strong> &nbsp;"
                f'<span class="tag {tc}">{tt}</span> &nbsp;'
                f"label={ret_label} &nbsp; sim={sim:.4f}"
                f"</div>",
                unsafe_allow_html=True,
            )
            ret_sample = ds["X_train"][real_idx]
            st.plotly_chart(ts_line_chart(ret_sample, height=180), use_container_width=True)

# ---------------------------------------------------------------------------
# 2. Matryoshka Dimension — Retrieval Accuracy (P@1 & P@5)
# ---------------------------------------------------------------------------
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
section_header(
    "Matryoshka Dimension &mdash; Retrieval Accuracy",
    "P@1 and P@5 label-matching accuracy across embedding dimensions",
)

eval_scope = st.radio(
    "Evaluation scope",
    ["Within dataset", "Global (all datasets)"],
    horizontal=True,
    help="Within dataset: same-dataset retrieval. Global: retrieve from all training data.",
)


@st.cache_data(show_spinner="Computing retrieval accuracies across dimensions...")
def compute_matryoshka_accuracies(emb_mode: str, scope: str) -> pd.DataFrame:
    rows = []
    dataset_names = list(DATASETS.keys())

    if scope == "Global (all datasets)":
        all_train_raw = []
        all_labels = []
        for dn in dataset_names:
            p = EMBEDDINGS_DIR / f"{dn}_train_{emb_mode}.npy"
            if not p.exists():
                continue
            all_train_raw.append(np.load(p))
            d = load_dataset(dn)
            all_labels.extend([str(l) for l in d["y_train"]])
        if not all_train_raw:
            return pd.DataFrame()
        global_raw = np.concatenate(all_train_raw, axis=0)
        global_labels = np.array(all_labels)

        for dim in MATRYOSHKA_DIMS:
            corpus = truncate_and_normalize(global_raw, dim)
            for dn in dataset_names:
                tp = EMBEDDINGS_DIR / f"{dn}_test_{emb_mode}.npy"
                if not tp.exists():
                    continue
                test_embs = truncate_and_normalize(np.load(tp), dim)
                d = load_dataset(dn)
                p1 = compute_pk(test_embs, corpus, d["y_test"], global_labels, 1)
                p5 = compute_pk(test_embs, corpus, d["y_test"], global_labels, 5)
                rows.append({"Dimension": dim, "Dataset": dn, "P@1": p1, "P@5": p5})
    else:
        for dim in MATRYOSHKA_DIMS:
            for dn in dataset_names:
                trp = EMBEDDINGS_DIR / f"{dn}_train_{emb_mode}.npy"
                tep = EMBEDDINGS_DIR / f"{dn}_test_{emb_mode}.npy"
                if not trp.exists() or not tep.exists():
                    continue
                d = load_dataset(dn)
                train_embs = truncate_and_normalize(np.load(trp), dim)
                test_embs = truncate_and_normalize(np.load(tep), dim)
                p1 = compute_pk(test_embs, train_embs, d["y_test"], d["y_train"], 1)
                p5 = compute_pk(test_embs, train_embs, d["y_test"], d["y_train"], 5)
                rows.append({"Dimension": dim, "Dataset": dn, "P@1": p1, "P@5": p5})

    return pd.DataFrame(rows)


df_mat = compute_matryoshka_accuracies(mode, eval_scope)

if not df_mat.empty:
    df_melted = df_mat.melt(
        id_vars=["Dimension", "Dataset"],
        value_vars=["P@1", "P@5"],
        var_name="Metric",
        value_name="Accuracy",
    )

    tab_p1, tab_p5 = st.tabs(["P@1 (1-NN Accuracy)", "P@5"])

    for tab, metric in [(tab_p1, "P@1"), (tab_p5, "P@5")]:
        with tab:
            df_m = df_melted[df_melted["Metric"] == metric]
            fig = px.bar(
                df_m,
                x="Dimension",
                y="Accuracy",
                color="Dataset",
                barmode="group",
                text_auto=".3f",
                color_discrete_sequence=DATASET_COLORS,
            )
            fig.update_layout(
                **PLOTLY_LAYOUT,
                height=400,
                yaxis=dict(range=[0, 1.05], gridcolor=BORDER, title=metric),
                xaxis=dict(type="category", title="Embedding Dimension"),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, font=dict(size=11),
                ),
            )
            fig.update_traces(textposition="outside", textfont_size=10)
            st.plotly_chart(fig, use_container_width=True)

    # Accuracy tables
    st.markdown("**Full results**")
    tbl = df_mat.copy()
    tbl["Dimension"] = tbl["Dimension"].astype(str)
    pivot_p1 = tbl.pivot(index="Dataset", columns="Dimension", values="P@1")
    pivot_p5 = tbl.pivot(index="Dataset", columns="Dimension", values="P@5")

    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.caption("P@1")
        st.dataframe(
            pivot_p1.style.format("{:.4f}").background_gradient(
                cmap="Blues", axis=None, vmin=0.5, vmax=1.0
            ),
            use_container_width=True,
        )
    with tcol2:
        st.caption("P@5")
        st.dataframe(
            pivot_p5.style.format("{:.4f}").background_gradient(
                cmap="Blues", axis=None, vmin=0.5, vmax=1.0
            ),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# 3. TS-only vs Multi-modal Comparison (P@1 & P@5)
# ---------------------------------------------------------------------------
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
section_header(
    "TS-only vs Multi-modal",
    f"Label retrieval accuracy comparison at dim={matryoshka_dim}",
)

comp_scope = st.radio(
    "Comparison scope",
    ["Within dataset", "Global (all datasets)"],
    horizontal=True,
    key="comp_scope",
)


@st.cache_data(show_spinner="Computing mode comparison...")
def compute_mode_comparison(dim: int, scope: str) -> pd.DataFrame:
    rows = []
    dataset_names = list(DATASETS.keys())

    for m, m_label in [("ts_only", "TS-only"), ("multimodal", "Multi-modal")]:
        if scope == "Global (all datasets)":
            all_train = []
            all_labels = []
            for dn in dataset_names:
                p = EMBEDDINGS_DIR / f"{dn}_train_{m}.npy"
                if not p.exists():
                    continue
                all_train.append(np.load(p))
                d = load_dataset(dn)
                all_labels.extend([str(l) for l in d["y_train"]])
            if not all_train:
                continue
            global_raw = np.concatenate(all_train, axis=0)
            global_labels = np.array(all_labels)
            corpus = truncate_and_normalize(global_raw, dim)

            for dn in dataset_names:
                tp = EMBEDDINGS_DIR / f"{dn}_test_{m}.npy"
                if not tp.exists():
                    continue
                test_embs = truncate_and_normalize(np.load(tp), dim)
                d = load_dataset(dn)
                p1 = compute_pk(test_embs, corpus, d["y_test"], global_labels, 1)
                p5 = compute_pk(test_embs, corpus, d["y_test"], global_labels, 5)
                rows.append({"Dataset": dn, "Mode": m_label, "P@1": p1, "P@5": p5})
        else:
            for dn in dataset_names:
                trp = EMBEDDINGS_DIR / f"{dn}_train_{m}.npy"
                tep = EMBEDDINGS_DIR / f"{dn}_test_{m}.npy"
                if not trp.exists() or not tep.exists():
                    continue
                d = load_dataset(dn)
                train_embs = truncate_and_normalize(np.load(trp), dim)
                test_embs = truncate_and_normalize(np.load(tep), dim)
                p1 = compute_pk(test_embs, train_embs, d["y_test"], d["y_train"], 1)
                p5 = compute_pk(test_embs, train_embs, d["y_test"], d["y_train"], 5)
                rows.append({"Dataset": dn, "Mode": m_label, "P@1": p1, "P@5": p5})

    return pd.DataFrame(rows)


df_comp = compute_mode_comparison(matryoshka_dim, comp_scope)

if not df_comp.empty:
    tab_cp1, tab_cp5 = st.tabs(["P@1 (1-NN Accuracy)", "P@5"])

    for tab, metric in [(tab_cp1, "P@1"), (tab_cp5, "P@5")]:
        with tab:
            fig_comp = px.bar(
                df_comp,
                x="Dataset",
                y=metric,
                color="Mode",
                barmode="group",
                color_discrete_map=MODE_COLORS,
                text_auto=".3f",
            )
            fig_comp.update_layout(
                **PLOTLY_LAYOUT,
                height=380,
                yaxis=dict(range=[0, 1.05], gridcolor=BORDER, title=metric),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5,
                ),
            )
            fig_comp.update_traces(textposition="outside", textfont_size=11)
            st.plotly_chart(fig_comp, use_container_width=True)

    # Delta table
    pivot_comp = df_comp.pivot(index="Dataset", columns="Mode", values=["P@1", "P@5"])

    if ("P@1", "TS-only") in pivot_comp.columns and ("P@1", "Multi-modal") in pivot_comp.columns:
        delta_df = pd.DataFrame({
            "TS P@1": pivot_comp[("P@1", "TS-only")],
            "MM P@1": pivot_comp[("P@1", "Multi-modal")],
            "P@1 Delta": pivot_comp[("P@1", "Multi-modal")] - pivot_comp[("P@1", "TS-only")],
            "TS P@5": pivot_comp[("P@5", "TS-only")],
            "MM P@5": pivot_comp[("P@5", "Multi-modal")],
            "P@5 Delta": pivot_comp[("P@5", "Multi-modal")] - pivot_comp[("P@5", "TS-only")],
        })
        st.dataframe(
            delta_df.style.format("{:.4f}").map(
                lambda v: f"color: {SUCCESS}" if isinstance(v, float) and v > 0 else (
                    f"color: {DANGER}" if isinstance(v, float) and v < 0 else ""
                ),
                subset=["P@1 Delta", "P@5 Delta"],
            ),
            use_container_width=True,
        )

        # Summary metric cards
        mcols = st.columns(4)
        avg_ts1 = delta_df["TS P@1"].mean()
        avg_mm1 = delta_df["MM P@1"].mean()
        avg_ts5 = delta_df["TS P@5"].mean()
        avg_mm5 = delta_df["MM P@5"].mean()
        with mcols[0]:
            st.markdown(metric_card(f"{avg_ts1:.3f}", "Avg TS P@1"), unsafe_allow_html=True)
        with mcols[1]:
            st.markdown(metric_card(f"{avg_mm1:.3f}", "Avg MM P@1"), unsafe_allow_html=True)
        with mcols[2]:
            st.markdown(metric_card(f"{avg_ts5:.3f}", "Avg TS P@5"), unsafe_allow_html=True)
        with mcols[3]:
            st.markdown(metric_card(f"{avg_mm5:.3f}", "Avg MM P@5"), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 4. Embedding Space (t-SNE)
# ---------------------------------------------------------------------------
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
section_header(
    "Embedding Space (t-SNE)",
    f"Training embeddings projected to 2D at dim={matryoshka_dim}",
)

tsne_col1, tsne_col2 = st.columns([1, 3])

with tsne_col1:
    perplexity = st.slider("Perplexity", min_value=5, max_value=50, value=30)
    color_by = st.radio("Color by", ["Dataset", "Label", "Dataset + Label"])

scatter_embs = []
scatter_labels = []
scatter_datasets = []

for ds_name in DATASETS:
    emb_file = EMBEDDINGS_DIR / f"{ds_name}_train_{mode}.npy"
    if not emb_file.exists():
        continue
    embs = np.load(emb_file)
    ds_data = load_dataset(ds_name)
    labels_arr = ds_data["y_train"]
    scatter_embs.append(embs)
    for i in range(len(embs)):
        scatter_datasets.append(ds_name)
        scatter_labels.append(str(labels_arr[i]))

if scatter_embs:
    all_scatter_embs = np.concatenate(scatter_embs, axis=0)
    trunc_embs = truncate_and_normalize(all_scatter_embs, matryoshka_dim)

    @st.cache_data(show_spinner="Running t-SNE...")
    def compute_tsne(embs_bytes: bytes, n_samples: int, dim: int, perp: int) -> np.ndarray:
        embs = np.frombuffer(embs_bytes, dtype=np.float64).reshape(n_samples, dim)
        tsne = TSNE(
            n_components=2, perplexity=perp, random_state=42,
            init="pca", learning_rate="auto",
        )
        return tsne.fit_transform(embs)

    coords = compute_tsne(trunc_embs.tobytes(), len(trunc_embs), matryoshka_dim, perplexity)

    df_scatter = pd.DataFrame({
        "t-SNE 1": coords[:, 0],
        "t-SNE 2": coords[:, 1],
        "Dataset": scatter_datasets,
        "Label": scatter_labels,
        "Dataset + Label": [f"{d} / {l}" for d, l in zip(scatter_datasets, scatter_labels)],
    })

    with tsne_col2:
        fig_scatter = px.scatter(
            df_scatter,
            x="t-SNE 1", y="t-SNE 2",
            color=color_by,
            hover_data=["Dataset", "Label"],
            opacity=0.7,
            color_discrete_sequence=DATASET_COLORS + ["#ec4899", "#84cc16", "#06b6d4", "#a78bfa"],
        )
        fig_scatter.update_traces(marker=dict(size=5))
        fig_scatter.update_layout(
            **PLOTLY_LAYOUT,
            height=550,
            legend=dict(orientation="v", font=dict(size=10)),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("No embeddings found. Run `python run_all.py` to generate them.")
