#!/usr/bin/env python3
"""Streamlit explorer for federated phenotype embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from phenotype_groups import CATEGORY_COLORS, classify_trait, display_name, load_labels as load_label_csv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDINGS = ROOT / "data" / "federated" / "global_phenotype_embeddings.npz"
DEFAULT_LABELS = ROOT / "data" / "pgs" / "phenotype_labels.csv"

SHARED_EDGE_COLOR = "rgba(232,238,247,0.22)"
NONGENETIC_EDGE_COLOR = "rgba(59,130,246,0.78)"
GENETIC_EDGE_COLOR = "rgba(244,162,97,0.85)"


@st.cache_data(show_spinner=False)
def load_labels(path: str) -> dict[str, str]:
    return load_label_csv(path)


@st.cache_data(show_spinner=False)
def load_embeddings(path: str, mtime: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blob = np.load(path)
    phenotype_ids = np.asarray(blob["phenotype_ids"]).astype(str)
    nongenetic = np.asarray(blob["nongenetic_phenotype_embeddings"], dtype=np.float32)
    genetic = np.asarray(blob["genetic_phenotype_embeddings"], dtype=np.float32)
    return phenotype_ids, nongenetic, genetic


@st.cache_data(show_spinner=False)
def cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / np.maximum(norms, 1e-8)
    similar = unit @ unit.T
    np.fill_diagonal(similar, 0.0)
    return similar.astype(np.float32)


def threshold_edges(similarity: np.ndarray, threshold: float, visible: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    visible_idx = np.flatnonzero(visible)
    if visible_idx.size == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([], dtype=np.float32)
    sub = similarity[np.ix_(visible_idx, visible_idx)]
    ii, jj = np.triu_indices(visible_idx.size, k=1)
    keep = sub[ii, jj] >= threshold
    return visible_idx[ii[keep]], visible_idx[jj[keep]], sub[ii[keep], jj[keep]]


def _base_layout(title: str, height: int = 540) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=18, color="#e8eef7"), x=0.02, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#121826",
        font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", color="#d7deea"),
        margin=dict(l=20, r=20, t=56, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            x=0,
            bgcolor="rgba(18,24,38,0.85)",
            font=dict(size=11),
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        hovermode="closest",
        height=height,
    )


@st.cache_data(show_spinner="Laying out network…")
def force_positions(
    nodes: tuple[int, ...],
    edges: tuple[tuple[int, int], ...],
    weights: tuple[float, ...],
    seed: int,
) -> dict[int, tuple[float, float]]:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from((i, j, w) for (i, j), w in zip(edges, weights))
    n_nodes = max(len(nodes), 1)
    positions = nx.spring_layout(
        graph,
        seed=seed,
        weight="weight",
        iterations=80,
        k=1.05 / np.sqrt(n_nodes),
    )
    return {int(node): (float(xy[0]), float(xy[1])) for node, xy in positions.items()}


def _empty_graph(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**_base_layout(title))
    fig.add_annotation(
        text="No phenotype pairs pass the similarity threshold.",
        showarrow=False,
        font=dict(size=14, color="#93a4bf"),
    )
    return fig


def _pack_edges(positions: dict[int, tuple[float, float]], edges: list[tuple[int, int]]) -> tuple[list[float | None], list[float | None]]:
    xs: list[float | None] = []
    ys: list[float | None] = []
    for i, j in edges:
        if i not in positions or j not in positions:
            continue
        x0, y0 = positions[i]
        x1, y1 = positions[j]
        xs.extend((x0, x1, None))
        ys.extend((y0, y1, None))
    return xs, ys


def _edge_trace(xs: list[float | None], ys: list[float | None], color: str, name: str, width: float, showlegend: bool) -> go.Scatter:
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(width=width, color=color),
        hoverinfo="skip",
        name=name,
        legendgroup=name,
        showlegend=showlegend,
    )


def _add_nodes(
    fig: go.Figure,
    names: list[str],
    categories: list[str],
    phenotype_ids: list[str],
    connected: np.ndarray,
    positions: dict[int, tuple[float, float]],
) -> None:
    connected_set = {int(idx) for idx in connected.tolist()}
    for category, color in CATEGORY_COLORS.items():
        node_ids = [
            idx
            for idx, category_name in enumerate(categories)
            if category_name == category and idx in connected_set
        ]
        if not node_ids:
            continue
        fig.add_trace(
            go.Scatter(
                x=[positions[idx][0] for idx in node_ids],
                y=[positions[idx][1] for idx in node_ids],
                mode="markers",
                name=category,
                marker=dict(size=11, color=color, line=dict(width=0.6, color="#0b1018"), opacity=0.95),
                text=[names[idx] for idx in node_ids],
                customdata=np.stack(
                    (
                        np.array([phenotype_ids[idx] for idx in node_ids]),
                        np.array([categories[idx] for idx in node_ids]),
                    ),
                    axis=1,
                ),
                hovertemplate="<b>%{text}</b><br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>",
            )
        )


def graph_layout(
    similarity: np.ndarray,
    threshold: float,
    visible: np.ndarray,
    seed: int,
) -> tuple[list[tuple[int, int]], np.ndarray, dict[int, tuple[float, float]]]:
    left, right, weights = threshold_edges(similarity, threshold, visible)
    connected = np.unique(np.concatenate([left, right])) if left.size else np.array([], dtype=np.int64)
    if connected.size == 0:
        return [], connected, {}
    edges = [(int(i), int(j)) for i, j in zip(left, right)]
    positions = force_positions(
        tuple(int(n) for n in connected.tolist()),
        tuple(edges),
        tuple(round(float(w), 6) for w in weights),
        seed,
    )
    return edges, connected, positions


def graph_figure(
    names: list[str],
    categories: list[str],
    phenotype_ids: list[str],
    similarity: np.ndarray,
    threshold: float,
    title: str,
    visible: np.ndarray,
    seed: int,
) -> tuple[go.Figure, int, int, dict[int, tuple[float, float]], np.ndarray]:
    edges, connected, positions = graph_layout(similarity, threshold, visible, seed)
    n_edges = len(edges)
    n_nodes = int(connected.size)
    if n_nodes == 0:
        return _empty_graph(f"{title}  ·  no edges ≥ {threshold:.2f}"), n_edges, n_nodes, positions, connected

    fig = go.Figure()
    fig.add_trace(_edge_trace(*_pack_edges(positions, edges), SHARED_EDGE_COLOR, "Edges", 0.8, False))
    _add_nodes(fig, names, categories, phenotype_ids, connected, positions)
    fig.update_layout(**_base_layout(f"{title}  ·  {n_nodes:,} nodes  ·  {n_edges:,} edges ≥ {threshold:.2f}"))
    return fig, n_edges, n_nodes, positions, connected


def difference_graph_figure(
    names: list[str],
    categories: list[str],
    phenotype_ids: list[str],
    nongenetic_sim: np.ndarray,
    genetic_sim: np.ndarray,
    threshold: float,
    visible: np.ndarray,
    positions: dict[int, tuple[float, float]],
    genetic_connected: np.ndarray,
) -> tuple[go.Figure, int, int, int]:
    n_left, n_right, _ = threshold_edges(nongenetic_sim, threshold, visible)
    g_left, g_right, _ = threshold_edges(genetic_sim, threshold, visible)
    nongenetic_edges = {tuple(sorted((int(i), int(j)))) for i, j in zip(n_left, n_right)}
    genetic_edges = {tuple(sorted((int(i), int(j)))) for i, j in zip(g_left, g_right)}
    shared = sorted(genetic_edges & nongenetic_edges)
    only_genetic = sorted(genetic_edges - nongenetic_edges)
    only_nongenetic = sorted(
        edge for edge in (nongenetic_edges - genetic_edges)
        if edge[0] in positions and edge[1] in positions
    )

    fig = go.Figure()
    if genetic_connected.size == 0:
        fig = _empty_graph("Genetic network with edge differences  ·  no edges")
        return fig, 0, 0, 0

    fig.add_trace(_edge_trace(*_pack_edges(positions, shared), SHARED_EDGE_COLOR, "Shared", 0.8, True))
    fig.add_trace(_edge_trace(*_pack_edges(positions, only_nongenetic), NONGENETIC_EDGE_COLOR, "Nongenetic only", 1.05, True))
    fig.add_trace(_edge_trace(*_pack_edges(positions, only_genetic), GENETIC_EDGE_COLOR, "Genetic only", 1.05, True))
    _add_nodes(fig, names, categories, phenotype_ids, genetic_connected, positions)
    fig.update_layout(
        **_base_layout(
            f"Genetic network with edge differences  ·  {len(shared):,} shared  ·  "
            f"{len(only_nongenetic):,} clinical-only  ·  {len(only_genetic):,} genetic-only",
            height=620,
        )
    )
    return fig, len(shared), len(only_nongenetic), len(only_genetic)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
        html, body, [class*="css"] { font-family: "IBM Plex Sans", sans-serif; }
        .stApp { background: radial-gradient(1200px 600px at 10% -10%, #1c2740 0%, #0b1018 45%); }
        [data-testid="stSidebar"] { background: #101826; border-right: 1px solid #243049; }
        h1 { letter-spacing: -0.03em; font-weight: 600 !important; }
        .hero-kicker { color: #8fb3d9; text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.78rem; }
        .metric-row { display: flex; gap: 12px; margin: 0.4rem 0 1.2rem; }
        .metric-card {
            flex: 1; background: rgba(22, 32, 51, 0.85); border: 1px solid #2a3b58;
            border-radius: 14px; padding: 14px 16px;
        }
        .metric-card .label { color: #93a4bf; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
        .metric-card .value { color: #f4f7fb; font-size: 1.35rem; font-weight: 600; margin-top: 4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Phenotype embedding explorer", layout="wide", page_icon="🧬")
    inject_css()

    st.markdown('<div class="hero-kicker">Federated collaborative filtering</div>', unsafe_allow_html=True)
    st.title("Phenotype embedding explorer")
    st.caption(
        "Clinical and genetic similarity networks for the same phenotypes. "
        "The comparison graph reuses the genetic layout and recolors edges that differ."
    )

    with st.sidebar:
        st.header("Controls")
        embeddings_path = st.text_input("Embeddings", str(DEFAULT_EMBEDDINGS))
        labels_path = st.text_input("Phenotype labels", str(DEFAULT_LABELS))
        threshold = st.slider("Edge similarity threshold", 0.0, 1.0, 0.65, 0.01)
        search = st.text_input("Highlight phenotype", placeholder="e.g. heart, diabetes")
        st.caption("The difference graph uses the genetic layout. White edges are shared, blue are clinical-only, orange are genetic-only.")

    phenotype_ids, nongenetic, genetic = load_embeddings(
        embeddings_path, Path(embeddings_path).stat().st_mtime
    )
    labels = load_labels(labels_path)
    names = [display_name(labels.get(pid, pid)) for pid in phenotype_ids]
    categories = [classify_trait(labels.get(pid, pid)) for pid in phenotype_ids]

    counts = {name: categories.count(name) for name in CATEGORY_COLORS}
    selected_categories = st.sidebar.multiselect(
        "Trait groups",
        options=list(CATEGORY_COLORS),
        default=[name for name, count in counts.items() if count and name != "Other"],
        format_func=lambda name: f"{name} ({counts.get(name, 0)})",
    )
    visible = np.array([category in set(selected_categories) for category in categories], dtype=bool)
    if search.strip():
        query = search.strip().lower()
        visible &= np.array(
            [query in name.lower() or query in pid.lower() or query in cat.lower()
             for name, pid, cat in zip(names, phenotype_ids, categories)],
            dtype=bool,
        )
        if not visible.any():
            st.warning("No phenotypes match that search in the selected groups.")
            visible = np.array([category in set(selected_categories) for category in categories], dtype=bool)

    nongenetic_sim = cosine_similarity(nongenetic)
    genetic_sim = cosine_similarity(genetic)

    nongenetic_graph, n_nongenetic_edges, n_nongenetic_nodes, _, _ = graph_figure(
        names, categories, phenotype_ids.tolist(), nongenetic_sim, threshold,
        "Nongenetic similarity network", visible, seed=0,
    )
    genetic_graph, n_genetic_edges, n_genetic_nodes, genetic_positions, genetic_connected = graph_figure(
        names, categories, phenotype_ids.tolist(), genetic_sim, threshold,
        "Genetic similarity network", visible, seed=1,
    )

    difference_graph, n_shared, n_only_clinical, n_only_genetic = difference_graph_figure(
        names, categories, phenotype_ids.tolist(), nongenetic_sim, genetic_sim,
        threshold, visible, genetic_positions, genetic_connected,
    )

    n_visible = int(visible.sum())
    st.markdown(
        f"""
        <div class="metric-row">
          <div class="metric-card"><div class="label">Phenotypes shown</div><div class="value">{n_visible} / {len(names)}</div></div>
          <div class="metric-card"><div class="label">Shared edges</div><div class="value">{n_shared:,}</div></div>
          <div class="metric-card"><div class="label">Clinical-only</div><div class="value">{n_only_clinical:,}</div></div>
          <div class="metric-card"><div class="label">Genetic-only</div><div class="value">{n_only_genetic:,}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.plotly_chart(nongenetic_graph, width="stretch")
        st.caption(f"{n_nongenetic_nodes:,} connected phenotypes, {n_nongenetic_edges:,} edges.")
    with right:
        st.plotly_chart(genetic_graph, width="stretch")
        st.caption(f"{n_genetic_nodes:,} connected phenotypes, {n_genetic_edges:,} edges.")

    st.plotly_chart(difference_graph, width="stretch")
    st.caption(
        "Same layout as the genetic graph. White edges are in both networks, "
        "blue edges are clinical-only, and orange edges are genetic-only."
    )


if __name__ == "__main__":
    main()
